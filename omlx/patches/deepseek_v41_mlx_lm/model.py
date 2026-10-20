# SPDX-License-Identifier: Apache-2.0
"""``mlx_lm.models.deepseek_v41``: oMLX's DeepSeek V4.1 for a cluster rank.

The worker's progressive loader calls ``load_pipeline_stage`` or
``load_tensor_shard`` instead of mlx-lm's loader: the checkpoint format
(packed projections, Engram tables) is oMLX's own. The planner reads
``pipeline_stage_starts`` because layers share compressed KV and index state
with earlier source layers.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx

from omlx.patches.deepseek_v41.config import ModelConfig
from omlx.patches.deepseek_v41.language import LanguageModel
from omlx.patches.deepseek_v41.loading import load
from omlx.patches.deepseek_v41.model import Model as _LoadedModel

ModelArgs = ModelConfig
# Read by the planner, which would otherwise refuse a checkpoint with a vision
# config. LanguageModel inherits PipelineMixin.pipeline, the method the worker
# replaces with the planner's split.
SUPPORTS_PIPELINE = True

__all__ = [
    "LanguageModel",
    "Model",
    "ModelArgs",
    "load_pipeline_stage",
    "load_tensor_shard",
    "pipeline_stage_starts",
]


def pipeline_stage_starts(config: dict[str, Any]) -> tuple[int, ...]:
    return ModelArgs.from_dict(config).pipeline_stage_starts()


class Model(_LoadedModel):
    """A loaded stage, presented the way mlx-lm serves a model."""

    # Worker ranks may advance their stage without the vocabulary projection.
    _omlx_supports_rank_zero_logits = True

    def __call__(
        self,
        inputs: mx.array,
        cache: Any = None,
        skip_logits: bool = False,
    ) -> mx.array:
        return self.language_model(inputs, cache=cache, skip_logits=skip_logits)

    @property
    def model(self) -> LanguageModel:
        """mlx-lm and the worker read the pipeline stage from ``model.model``."""

        return self.language_model

    @property
    def layers(self) -> list:
        return self.language_model.pipeline_layers

    @property
    def _omlx_output_vocab_size(self) -> int:
        return int(self.config.vocab_size)

    def make_cache(self) -> list:
        return self.language_model.make_cache()


def load_pipeline_stage(model_path: Any, group: Any) -> Model:
    """This rank's stage of a converted checkpoint, text model only."""

    model, _ = load(
        model_path, stage=lambda language_model: language_model.pipeline(group)
    )
    model.__class__ = Model
    return model


def load_tensor_shard(model_path: Any, group: Any) -> Model:
    """This rank's tensor-parallel slice of a converted checkpoint, text model only."""

    # Every rank reads every Engram row, so the tables stay on SSD.
    model, _ = load(model_path, tensor=group, engram_ssd_offload=True)
    model.__class__ = Model
    return model
