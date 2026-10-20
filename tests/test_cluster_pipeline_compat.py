# SPDX-License-Identifier: Apache-2.0

import json
import os
import socket
import subprocess
import sys

from omlx.cluster.pipeline_compat import (
    install_pipeline_compatibility,
    pipeline_assignment_is_honored,
)
from omlx.cluster.planner import PipelineAssignment


def _assignment():
    return (
        PipelineAssignment(
            node_id="local",
            rank=0,
            start_layer=0,
            end_layer=2,
            layer_weight_bytes=2,
            fixed_weight_bytes=1,
            reserve_bytes=1,
            capacity_bytes=8,
        ),
    )


def _model_config(tmp_path, model_type):
    model = tmp_path / model_type
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"model_type": model_type}))
    return model


def test_standard_pipeline_mixin_has_an_explicit_assignment_contract(tmp_path):
    model = _model_config(tmp_path, "deepseek_v3")

    assert not pipeline_assignment_is_honored(model)
    with install_pipeline_compatibility(_assignment()):
        assert pipeline_assignment_is_honored(model)


def test_thin_qwen_moe_wrapper_inherits_the_pipeline_contract(tmp_path):
    model = _model_config(tmp_path, "qwen3_5_moe")

    assert not pipeline_assignment_is_honored(model)
    with install_pipeline_compatibility(_assignment()):
        assert pipeline_assignment_is_honored(model)


def test_nemotron_compatibility_has_an_explicit_assignment_contract(tmp_path):
    model = _model_config(tmp_path, "nemotron_h")

    assert not pipeline_assignment_is_honored(model)
    with install_pipeline_compatibility(_assignment()):
        assert pipeline_assignment_is_honored(model)


def test_deepseek_v32_inherits_the_pipeline_contract(tmp_path):
    model = _model_config(tmp_path, "deepseek_v32")

    with install_pipeline_compatibility(_assignment()):
        assert pipeline_assignment_is_honored(model)


def test_minimax_declares_its_wrapped_assigned_stage_contract(tmp_path):
    from omlx.utils.model_loading import maybe_apply_pre_load_patches

    model = _model_config(tmp_path, "minimax_m3_vl")
    maybe_apply_pre_load_patches(str(model))

    assert pipeline_assignment_is_honored(model)


def test_glm5_next_inherits_the_pipeline_contract(tmp_path):
    from omlx.utils.model_loading import maybe_apply_pre_load_patches

    model = _model_config(tmp_path, "glm5_next")
    maybe_apply_pre_load_patches(str(model))

    assert not pipeline_assignment_is_honored(model)
    with install_pipeline_compatibility(_assignment()):
        assert pipeline_assignment_is_honored(model)


_GLM5_NEXT_TEXT = dict(
    model_type="glm5_next_text",
    hidden_size=256,
    intermediate_size=256,
    moe_intermediate_size=128,
    num_hidden_layers=5,
    num_attention_heads=4,
    num_key_value_heads=4,
    vocab_size=1024,
    layer_types=["linear_attention"] * 3
    + ["deepseek_sparse_attention", "linear_attention"],
    mlp_layer_types=["dense"] * 3 + ["sparse", "sparse"],
    indexer_types=["full"] * 5,
    first_k_dense_replace=3,
    n_routed_experts=8,
    n_shared_experts=1,
    num_experts_per_tok=4,
    kv_lora_rank=64,
    q_lora_rank=128,
    qk_head_dim=256,
    qk_nope_head_dim=64,
    qk_rope_head_dim=0,
    v_head_dim=64,
    index_head_dim=64,
    index_n_heads=4,
    index_topk=64,
    index_kpool=4,
    linear_num_heads=4,
    linear_head_dim=64,
    linear_conv_kernel_dim=4,
    hc_mult=4,
    num_nextn_predict_layers=0,
    routed_scaling_factor=2.5,
    max_position_embeddings=4096,
    rms_norm_eps=1e-5,
    linear_attn_config=dict(
        gate_lower_bound=-5.0, head_dim=64, num_heads=4, short_conv_kernel_size=4
    ),
)


def test_glm5_next_sanitize_maps_hf_and_mlx_names_to_the_text_model():
    import mlx.core as mx
    from mlx.utils import tree_flatten

    from omlx.patches.glm5_next_mlx_lm import apply_glm5_next_mlx_lm_patch

    assert apply_glm5_next_mlx_lm_patch()
    from mlx_lm.models import glm5_next

    model = glm5_next.Model(
        glm5_next.ModelArgs.from_dict(
            {"model_type": "glm5_next", "text_config": _GLM5_NEXT_TEXT}
        )
    )
    norm, head, vision = mx.ones((256,)), mx.ones((1024, 256)), mx.ones((4,))
    hf = model.sanitize(
        {
            "model.language_model.norm.weight": norm,
            "lm_head.weight": head,
            "model.visual.patch_embed.proj.weight": vision,
        }
    )
    converted = model.sanitize(
        {
            "language_model.model.norm.weight": norm,
            "language_model.lm_head.weight": head,
            "vision_model.patch_embed.proj.weight": vision,
        }
    )

    # The rank loads with strict=False, so a wrong name would stay random.
    expected = {"language_model.model.norm.weight", "language_model.lm_head.weight"}
    assert set(hf) == set(converted) == expected
    assert expected <= {name for name, _ in tree_flatten(model.parameters())}


# One rank of the GLM-5.3 test below: a tiny random model, run by the vendored
# single-node forward and then as rank 1 [0, 3) -> rank 0 [3, 5).
_GLM5_NEXT_RANK = """
import json
import sys
import mlx.core as mx
group = mx.distributed.init(backend="ring", strict=True)
from omlx.patches.glm5_next_mlx_lm import apply_glm5_next_mlx_lm_patch
assert apply_glm5_next_mlx_lm_patch()
from mlx_lm.models import glm5_next
from mlx_vlm.models.glm5_next import language
from omlx.cluster.planner import PipelineAssignment, install_unequal_pipeline_plan

text = json.loads(sys.argv[1])
args = glm5_next.ModelArgs.from_dict({"model_type": "glm5_next", "text_config": text})
plan = (
    PipelineAssignment("a", 0, 3, 5, 1, 0, 0, 1),
    PipelineAssignment("b", 1, 0, 3, 1, 0, 0, 1),
)
mx.random.seed(7)
prompt = mx.random.randint(0, 1024, (1, 265))
# LayerPipeline prefill, a short block, then two deferred one-token decodes.
steps = [prompt[:, :260], prompt[:, 260:263], prompt[:, 263:264], prompt[:, 264:]]

def run(forward, make_cache):
    # One evaluation per step, as in serving, keeps collectives in order.
    cache = make_cache()
    outs = []
    for chunk in steps:
        outs.append(forward(chunk, cache))
        mx.eval(outs[-1])
    return outs

def compare(embed_dtype):
    model = glm5_next.Model(args)
    embed = model.language_model.model.embed_tokens
    embed.weight = embed.weight.astype(embed_dtype)
    mx.eval(model.parameters())
    lm = model.language_model
    lm.model.__class__ = language.Glm5NextModel
    reference = run(lambda x, c: lm(x, cache=c).logits, lm.make_cache)
    lm.model.__class__ = glm5_next.PipelinedTextModel
    with install_unequal_pipeline_plan(plan):
        model.model.pipeline(group)
    got = run(lambda x, c: model(x, cache=c), model.make_cache)
    return reference, got

reference, got = compare(mx.float32)
assert all(mx.array_equal(r, g).item() for r, g in zip(reference, got))
# The layers return fp32 while recv_like takes the bf16 embedding dtype.
reference, got = compare(mx.bfloat16)
assert all(mx.isfinite(g).all().item() for g in got)
assert all(
    mx.array_equal(mx.argmax(r[:, -1], -1), mx.argmax(g[:, -1], -1)).item()
    for r, g in zip(reference, got)
)
print(json.dumps({"rank": group.rank(), "ok": True}), flush=True)
"""


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_pipelined_glm5_next_forward_matches_single_process(tmp_path):
    hostfile = tmp_path / "ring.json"
    hostfile.write_text(
        json.dumps([[f"127.0.0.1:{_free_port()}"], [f"127.0.0.1:{_free_port()}"]])
    )
    procs = [
        subprocess.Popen(
            [sys.executable, "-c", _GLM5_NEXT_RANK, json.dumps(_GLM5_NEXT_TEXT)],
            env=dict(os.environ, MLX_RANK=str(rank), MLX_HOSTFILE=str(hostfile)),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for rank in (0, 1)
    ]
    outputs = []
    for proc in procs:
        try:
            out, _ = proc.communicate(timeout=300)
        except subprocess.TimeoutExpired:
            for p in procs:
                p.kill()
            raise
        outputs.append((proc.returncode, out))
    for rank, (code, out) in enumerate(outputs):
        assert code == 0, f"rank {rank} failed:\n{out[-2000:]}"
        assert '"ok": true' in out
