# SPDX-License-Identifier: Apache-2.0
"""GGUF model files and the model folders oMLX derives from them.

oMLX serves a GGUF only through an external engine. Each GGUF model gets a
generated folder under ``<base path>/gguf`` with a config.json derived from
the file's header and links to the file and its vision projectors, so
discovery, settings and the engines handle it like any other model folder.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

logger = logging.getLogger(__name__)

GGUF_MAGIC = b"GGUF"
SPLIT_RE = re.compile(
    r"^(?P<prefix>.+)-(?P<index>\d{5})-of-(?P<count>\d{5})\.gguf$", re.IGNORECASE
)
# Vision projector precisions in the order a GGUF is paired with them.
PROJECTOR_PREFERENCE = ("bf16", "f16", "f32")

# Header sanity bounds: well above real files, far below a corrupt length.
_MAX_KEYS = 1 << 20
_MAX_STRING = 64 << 20
_MAX_ARRAY = 1 << 26

_SCALARS = {
    0: "<B",
    1: "<b",
    2: "<H",
    3: "<h",
    4: "<I",
    5: "<i",
    6: "<f",
    7: "<?",
    10: "<Q",
    11: "<q",
    12: "<d",
}
_STRING = 8
_ARRAY = 9

# Hugging Face model types of the GGUF architectures oMLX names like their
# safetensors checkpoints; any other architecture keeps its GGUF name.
_HF_MODEL_TYPES = {"qwen35": "qwen3_5", "qwen35moe": "qwen3_5_moe"}

# The generated folder's files besides the links to the source files.
CONFIG_FILE = "config.json"
_CHAT_TEMPLATE_FILE = "chat_template.jinja"
_GENERATION_CONFIG_FILE = "generation_config.json"


class GgufError(ValueError):
    """Raised for a file that is not a readable GGUF."""


@dataclass(frozen=True)
class ArrayInfo:
    """An array value read without its elements."""

    length: int


@dataclass(frozen=True)
class GgufModel:
    """One GGUF model in a folder: its parts and the projectors beside it."""

    model_id: str
    parts: tuple[Path, ...]
    projectors: tuple[Path, ...]

    @property
    def path(self) -> Path:
        return self.parts[0]


def file_kind(name: str) -> str:
    """'mmproj', 'imatrix' or 'gguf' for a .gguf file name."""
    base = name.rsplit("/", 1)[-1].lower()
    if "mmproj" in base:
        return "mmproj"
    if "imatrix" in base:
        return "imatrix"
    return "gguf"


def projector_rank(name: str) -> tuple[int, str]:
    """Sort key putting the projector a GGUF pairs with first."""
    base = name.rsplit("/", 1)[-1].lower()
    for rank, tag in enumerate(PROJECTOR_PREFERENCE):
        if re.search(rf"(^|[-_.]){tag}([-_.]|$)", base):
            return rank, name
    return len(PROJECTOR_PREFERENCE), name


def _root_ggufs(folder: Path) -> list[Path]:
    try:
        entries = sorted(folder.iterdir())
    except OSError:
        return []
    return [
        path
        for path in entries
        if path.name.lower().endswith(".gguf")
        and not path.name.startswith(".")
        and path.is_file()
    ]


def find_models(folder: Path) -> list[GgufModel]:
    """The GGUF models at the root of ``folder``.

    A split GGUF is one model named by its prefix; projectors and imatrix
    files are not models, and subfolders (split BF16 copies, MTP heads) are
    not scanned.
    """
    files = _root_ggufs(folder)
    projectors = tuple(
        sorted(
            (path for path in files if file_kind(path.name) == "mmproj"),
            key=lambda path: projector_rank(path.name),
        )
    )
    split_parts: dict[str, list[Path]] = {}
    models: list[GgufModel] = []
    for path in files:
        if file_kind(path.name) != "gguf":
            continue
        split = SPLIT_RE.match(path.name)
        if split:
            split_parts.setdefault(split["prefix"], []).append(path)
            continue
        models.append(GgufModel(path.name[: -len(".gguf")], (path,), projectors))
    for prefix, parts in split_parts.items():
        count = int(SPLIT_RE.match(parts[0].name)["count"])
        if len(parts) != count:
            logger.warning(
                "Skipping incomplete split GGUF %s in %s: %d of %d parts",
                prefix,
                folder,
                len(parts),
                count,
            )
            continue
        models.append(GgufModel(prefix, tuple(parts), projectors))
    return sorted(models, key=lambda model: model.model_id)


def is_gguf_folder(folder: Path) -> bool:
    """True for a folder of GGUF models rather than a safetensors checkpoint."""
    try:
        if any(folder.glob("*.safetensors")):
            return False
    except OSError:
        return False
    return any(file_kind(path.name) == "gguf" for path in _root_ggufs(folder))


# -- header -------------------------------------------------------------------


class _Reader:
    def __init__(self, stream: BinaryIO):
        self._stream = stream

    def take(self, size: int) -> bytes:
        data = self._stream.read(size)
        if len(data) != size:
            raise GgufError("truncated GGUF header")
        return data

    def scalar(self, fmt: str):
        return struct.unpack(fmt, self.take(struct.calcsize(fmt)))[0]

    def length(self, bound: int) -> int:
        value = self.scalar("<Q")
        if value > bound:
            raise GgufError("GGUF header length out of range")
        return value

    def string(self) -> str:
        return self.take(self.length(_MAX_STRING)).decode("utf-8", "replace")

    def skip_string(self) -> None:
        self._stream.seek(self.length(_MAX_STRING), os.SEEK_CUR)

    def value(self, kind: int, keep: bool):
        if kind in _SCALARS:
            return self.scalar(_SCALARS[kind])
        if kind == _STRING:
            return self.string()
        if kind != _ARRAY:
            raise GgufError(f"unknown GGUF value type {kind}")
        item = self.scalar("<I")
        count = self.length(_MAX_ARRAY)
        if keep:
            return [self.value(item, keep=True) for _ in range(count)]
        if item == _STRING:
            for _ in range(count):
                self.skip_string()
        elif item in _SCALARS:
            self._stream.seek(struct.calcsize(_SCALARS[item]) * count, os.SEEK_CUR)
        else:
            raise GgufError(f"unsupported GGUF array of type {item}")
        return ArrayInfo(count)


def read_metadata(
    path: Path, keep_arrays: frozenset[str] = frozenset()
) -> dict[str, Any]:
    """Key-value metadata of a GGUF header.

    Arrays read as ``ArrayInfo`` unless their key is in ``keep_arrays``,
    which spares reading a 250K-entry vocabulary when only its size matters.
    """
    with open(path, "rb") as stream:
        reader = _Reader(stream)
        if reader.take(4) != GGUF_MAGIC:
            raise GgufError(f"{path.name} is not a GGUF file")
        version = reader.scalar("<I")
        if version < 2:
            raise GgufError(f"{path.name} uses GGUF version {version}")
        reader.scalar("<Q")  # tensor count
        count = reader.length(_MAX_KEYS)
        metadata: dict[str, Any] = {}
        for _ in range(count):
            key = reader.string()
            kind = reader.scalar("<I")
            metadata[key] = reader.value(kind, keep=key in keep_arrays)
    return metadata


def _int(value) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def derive_config(
    metadata: dict[str, Any], projector: dict[str, Any] | None = None
) -> dict[str, Any]:
    """A config.json for a GGUF model from its header, in the Hugging Face layout."""
    arch = metadata.get("general.architecture")
    if not isinstance(arch, str) or not arch:
        raise GgufError("GGUF header names no architecture")

    def field(name: str) -> int | None:
        return _int(metadata.get(f"{arch}.{name}"))

    layers = field("block_count")
    if layers is not None:
        # MTP layers are not part of the model the engines run.
        layers -= field("nextn_predict_layers") or 0
    tokens = metadata.get("tokenizer.ggml.tokens")
    text = {
        "hidden_size": field("embedding_length"),
        "num_hidden_layers": layers,
        "max_position_embeddings": field("context_length"),
        "num_attention_heads": field("attention.head_count"),
        "num_key_value_heads": field("attention.head_count_kv"),
        "head_dim": field("attention.key_length"),
        "full_attention_interval": field("full_attention_interval"),
        "num_experts": field("expert_count"),
        "num_experts_per_tok": field("expert_used_count"),
        "vocab_size": tokens.length if isinstance(tokens, ArrayInfo) else None,
    }
    hf_type = _HF_MODEL_TYPES.get(arch)
    text = {key: value for key, value in text.items() if value is not None}
    text["model_type"] = f"{hf_type}_text" if hf_type else arch
    config: dict[str, Any] = {"model_type": hf_type or arch, "text_config": text}
    if projector is not None:
        vision = {"model_type": projector.get("general.architecture") or "clip"}
        if isinstance(projector.get("clip.projector_type"), str):
            vision["projector_type"] = projector["clip.projector_type"]
        config["vision_config"] = vision
    return config


def _generation_config(metadata: dict[str, Any]) -> dict[str, Any]:
    names = {"temp": "temperature", "top_p": "top_p", "top_k": "top_k"}
    config = {}
    for key, name in names.items():
        value = metadata.get(f"general.sampling.{key}")
        if isinstance(value, float):
            config[name] = round(value, 6)
        elif _int(value) is not None:
            config[name] = value
    return config


# -- generated folders ----------------------------------------------------------


def _folder_name(model: GgufModel) -> str:
    """Name that changes whenever the model's files change."""
    digest = hashlib.sha1()
    for path in (*model.parts, *model.projectors):
        stat = path.stat()
        digest.update(f"{path.resolve()}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode())
    return f"{model.model_id}-{digest.hexdigest()[:12]}"


def model_size(model: GgufModel) -> int:
    """Bytes of the parts and the projector an engine loads beside them."""
    size = sum(part.stat().st_size for part in model.parts)
    if model.projectors:
        size += model.projectors[0].stat().st_size
    return size


def prepare_model_folder(model: GgufModel, root: Path) -> Path:
    """The generated folder for ``model``, built on first use."""
    folder = root / _folder_name(model)
    if (folder / CONFIG_FILE).is_file():
        return folder
    metadata = read_metadata(model.path)
    projector = read_metadata(model.projectors[0]) if model.projectors else None
    config = derive_config(metadata, projector)
    config["omlx_gguf"] = {
        "source": str(model.path),
        "parts": [part.name for part in model.parts],
        "projectors": [path.name for path in model.projectors],
    }
    partial = root / f".{folder.name}.partial"
    shutil.rmtree(partial, ignore_errors=True)
    partial.mkdir(parents=True)
    for path in (*model.parts, *model.projectors):
        (partial / path.name).symlink_to(path.resolve())
    (partial / CONFIG_FILE).write_text(json.dumps(config, indent=2))
    template = metadata.get("tokenizer.chat_template")
    if isinstance(template, str) and template.strip():
        (partial / _CHAT_TEMPLATE_FILE).write_text(template)
    generation = _generation_config(metadata)
    if generation:
        (partial / _GENERATION_CONFIG_FILE).write_text(json.dumps(generation))
    try:
        partial.rename(folder)
    except OSError:
        # Another discovery built it first.
        shutil.rmtree(partial, ignore_errors=True)
    return folder


def prune_model_folders(root: Path, keep: set[str]) -> None:
    """Remove generated folders, by name, whose model is gone or changed."""
    try:
        entries = list(root.iterdir())
    except OSError:
        return
    for entry in entries:
        if entry.is_dir() and entry.name not in keep:
            shutil.rmtree(entry, ignore_errors=True)


def source_info(model_path: str | Path) -> dict[str, Any] | None:
    """The ``omlx_gguf`` record of a generated folder, or None."""
    try:
        config = json.loads((Path(model_path) / CONFIG_FILE).read_text())
    except (OSError, ValueError):
        return None
    info = config.get("omlx_gguf") if isinstance(config, dict) else None
    return info if isinstance(info, dict) else None
