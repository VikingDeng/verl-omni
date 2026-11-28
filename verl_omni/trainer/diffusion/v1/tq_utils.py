# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import dataclasses
import logging
import os
from collections import Counter
from collections.abc import Iterable
from typing import Any, Literal

import numpy as np
import torch
import transfer_queue as tq
from tensordict import TensorDict
from transfer_queue import KVBatchMeta
from verl.protocol import DataProto
from verl.utils import tensordict_utils as tu

from verl_omni.trainer.diffusion.diffusion_trainer_utils import MEDIA_ARTIFACT_FIELD_PREFIX
from verl_omni.workers.config.diffusion.model import DIFFUSION_TQ_KEYS

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


def diffusion_persisted_tq_fields(
    algorithm: Literal["policy_gradient", "direct_preference"],
) -> list[str]:
    """Return trainer-computed fields persisted after a diffusion update.

    ``old_log_probs`` is not persisted: every consumer recomputes it from the
    actor (or falls back to ``rollout_log_probs`` in bypass mode), so writing
    the [batch, num_timesteps] tensor back would be dead TQ traffic.
    """
    if algorithm == "policy_gradient":
        return ["advantages", "returns", "sample_level_scores", "sample_level_rewards", "reward_extra_info"]
    if algorithm == "direct_preference":
        return ["sample_level_scores", "sample_level_rewards", "reward_extra_info"]
    raise ValueError(f"Unsupported diffusion trainer algorithm: {algorithm}")


# Fields consumed by the post-step rollout-dump read. Selecting them
# explicitly keeps the dump from re-pulling the trainer-only compute payloads
# (all_latents, prompt_embeds, ...) which dominate row size. Absent optional
# fields are silently skipped by TransferQueue's field selection. The score
# fields are used when the driver has no sidecar scores for the dump.
DIFFUSION_DUMP_TQ_FIELDS = (
    "prompts",
    "responses",
    "rm_scores",
    "sample_level_scores",
    "reward_model",
    "data_source",
    "extra_info",
    "raw_prompt",
    "__num_turns__",
    "uid",
    "extra_fields",
    "tool_extra_fields",
    "image",
    "video",
    "output",
    "audio",
    "audio_sample_rate",
    "media_kind",
)


def diffusion_dump_tq_fields(batch_meta: KVBatchMeta) -> list[str]:
    """Return the rollout-dump projection, including every stored media artifact.

    Artifact names are adapter-defined (``DiffusionIOSpec``) and their
    declarations ride ``extra_fields``; ``validate_artifacts`` raises on any
    declared artifact whose ``media_artifact__<name>`` tensor is missing from
    the read, so the names come from the stored schema, not a fixed list.
    """
    stored = (
        tq.get_client()
        .kv_retrieve_meta(keys=list(batch_meta.keys), partition_id=batch_meta.partition_id, create=False)
        .field_names
    )
    return [*DIFFUSION_DUMP_TQ_FIELDS, *(name for name in stored if name.startswith(MEDIA_ARTIFACT_FIELD_PREFIX))]


def diffusion_metric_tq_fields(
    algorithm: Literal["policy_gradient", "direct_preference"],
) -> list[str]:
    """Return fields persisted and consumed by diffusion metric computation."""
    if algorithm == "policy_gradient":
        return [
            "sample_level_rewards",
            "sample_level_scores",
            "advantages",
            "returns",
            "uid",
            "extra_fields",
            "reward_extra_info",
        ]
    if algorithm == "direct_preference":
        return ["sample_level_rewards", "sample_level_scores", "uid", "extra_fields", "reward_extra_info"]
    raise ValueError(f"Unsupported diffusion trainer algorithm: {algorithm}")


def _unwrap_non_tensor_item(item: Any) -> Any:
    """Unwrap common non-tensor wrappers (e.g. NonTensorData) to raw values."""
    value = item
    # Some TransferQueue paths return wrapper objects with a ``data`` attribute.
    # Unwrap a few levels defensively until reaching a plain python value.
    for _ in range(4):
        if isinstance(value, str | bytes | bytearray | dict | list | tuple | np.ndarray | torch.Tensor):
            break
        data_attr = getattr(value, "data", None)
        if data_attr is None or data_attr is value:
            break
        value = data_attr
    if isinstance(value, np.generic):
        return value.item()
    return value


def _to_object_array(value: Any) -> np.ndarray:
    """Normalize non-tensor TransferQueue values to object ndarray."""
    if isinstance(value, np.ndarray) and value.dtype == object:
        items = value.tolist()
    elif isinstance(value, np.ndarray):
        items = value.tolist()
    elif isinstance(value, str | bytes | dict):
        items = [value]
    else:
        try:
            items = list(value)
        except TypeError:
            items = [value]
    items = [_unwrap_non_tensor_item(item) for item in items]
    arr = np.empty(len(items), dtype=object)
    arr[:] = items
    return arr


def _stack_field(value: Any, padding: float = 0.0) -> torch.Tensor | None:
    """Normalize a TransferQueue-returned field into a stacked tensor.

    Diffusion worker outputs are pre-padded to fixed shapes (prompt_length for
    token ids, fixed C/H/W for images, max_prompt_embed_length for embeds), so
    most fields come back already stacked. Variable-length fallbacks are padded
    so the downstream diffusion compute path receives uniform batch tensors.
    """
    if value is None:
        return None
    if isinstance(value, torch.Tensor) and value.is_nested:
        return value.to_padded_tensor(padding=padding)
    if isinstance(value, torch.Tensor):
        return value
    if hasattr(value, "to_padded_tensor"):
        return value.to_padded_tensor(padding=padding)
    if isinstance(value, list | tuple):
        tensors = [v if isinstance(v, torch.Tensor) else torch.as_tensor(v) for v in value]
        if not tensors:
            return None
        return torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True, padding_value=padding)
    return torch.as_tensor(value)


def diffusion_tq_batch_to_dataproto(
    batch_meta: KVBatchMeta,
    pad_token_id: int = 0,
    select_fields: list[str] | None = None,
) -> DataProto:
    """Read TQ rows and assemble a diffusion ``DataProto``.

    Args:
        batch_meta: ``KVBatchMeta`` returned by ``ReplayBuffer.sample``.
        pad_token_id: Padding token id for variable-length prompt token tensors.
        select_fields: Optional TQ fields to retrieve. ``None`` preserves the
            full-payload behavior required by training and validation.

    Returns:
        ``DataProto`` whose ``batch`` carries the retrieved diffusion tensors and
        whose ``non_tensor_batch`` carries row metadata and ``extra_fields``.
    """
    sort_idx = sort_diffusion_tq_keys(list(batch_meta.keys))
    keys = [batch_meta.keys[i] for i in sort_idx]
    partition_id = batch_meta.partition_id

    data = tq.kv_batch_get(
        keys=keys,
        partition_id=partition_id,
        select_fields=select_fields,
    )

    batch_dict: dict[str, torch.Tensor] = {}
    non_tensor_batch: dict[str, Any] = {}
    for field, value in data.items():
        padding = float(pad_token_id) if field == "prompts" else 0.0
        stacked = None
        try:
            stacked = _stack_field(value, padding=padding)
        except (TypeError, ValueError, RuntimeError):
            # Non-numeric/object fields are forwarded via non_tensor_batch.
            stacked = None

        if stacked is not None and isinstance(stacked, torch.Tensor):
            batch_dict[field] = stacked
            continue
        non_tensor_batch[field] = _to_object_array(value)

    # Unpack extra_fields dict rows into top-level non_tensor_batch keys so the
    # diffusion compute path can read min/max_global_steps and reward_extra_info.
    extra_fields_arr = non_tensor_batch.pop("extra_fields", None)
    if extra_fields_arr is not None:
        for i, extra in enumerate(extra_fields_arr.tolist()):
            if not isinstance(extra, dict):
                continue
            for k, v in extra.items():
                if k not in non_tensor_batch:
                    non_tensor_batch[k] = np.full(len(extra_fields_arr), None, dtype=object)
                non_tensor_batch[k][i] = v

    batch = TensorDict(batch_dict, batch_size=len(keys))
    return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)


def _pack_reward_extra_info(data: DataProto, expected_rows: int) -> list[dict[str, Any]] | None:
    reward_extra_keys = list(data.meta_info.get("reward_extra_keys") or [])
    existing = data.non_tensor_batch.get("reward_extra_info")
    if not reward_extra_keys and existing is None:
        return None

    columns = []
    for key in reward_extra_keys:
        if key not in data.non_tensor_batch:
            raise ValueError(f"reward_extra_info: missing non_tensor_batch column {key!r}")
        column = data.non_tensor_batch[key]
        if len(column) != expected_rows:
            raise ValueError(f"reward_extra_info column {key!r}: expected {expected_rows} rows, got {len(column)}")
        columns.append(column.tolist())

    if existing is not None and len(existing) != expected_rows:
        raise ValueError(f"reward_extra_info: expected {expected_rows} rows, got {len(existing)}")

    rows = (
        [dict(info) if info is not None else {} for info in existing]
        if existing is not None
        else [{} for _ in range(expected_rows)]
    )
    if reward_extra_keys:
        for row, values in zip(rows, zip(*columns, strict=True), strict=True):
            row.update(zip(reward_extra_keys, values, strict=True))
    return rows


def put_dataproto_fields_to_tq(
    batch_meta: KVBatchMeta,
    data: DataProto,
    fields: list[str],
) -> None:
    """Write selected tensors and trainer-computed reward diagnostics back to TransferQueue.

    Args:
        batch_meta: ``KVBatchMeta`` whose ``keys`` and ``partition_id`` target
            the rows to update.
        data: ``DataProto`` containing the computed tensor fields.
        fields: Fields to persist. Absent fields are skipped unless
            ``reward_extra_info`` is requested and ``reward_extra_keys`` declares
            diagnostic columns. Each declared column must have one entry per batch key.

    Raises:
        ValueError: A declared diagnostic column is missing or has the wrong length.
    """
    output: dict[str, Any] = {}
    for field in fields:
        if data.batch is not None and field in data.batch:
            output[field] = data.batch[field]
        elif field == "reward_extra_info":
            reward_extra_info = _pack_reward_extra_info(data, len(batch_meta.keys))
            if reward_extra_info is not None:
                output[field] = reward_extra_info
    if not output:
        return
    sort_idx = sort_diffusion_tq_keys(list(batch_meta.keys))
    sorted_keys = [batch_meta.keys[i] for i in sort_idx]
    tq.kv_batch_put(
        keys=sorted_keys,
        partition_id=batch_meta.partition_id,
        fields=tu.get_tensordict(output),
    )


def _snapshot_diagnostic_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {k: _snapshot_diagnostic_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_snapshot_diagnostic_value(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_snapshot_diagnostic_value(v) for v in value)
    return value


@dataclasses.dataclass(slots=True)
class _SidecarRow:
    tensors: dict[str, torch.Tensor]
    uid: Any | None
    reward_extra_info: dict[str, Any] | None


class DiffusionStepSidecar:
    """Keep detached metric rows until the driver clears the step's queue keys."""

    def __init__(self) -> None:
        self._rows: dict[tuple[str, str], _SidecarRow] = {}

    def record(
        self,
        batch_meta: KVBatchMeta,
        data: DataProto,
        algorithm: Literal["policy_gradient", "direct_preference"],
    ) -> None:
        """Retain metric tensors and metadata for the sampled queue rows."""
        n_keys = len(batch_meta.keys)
        if n_keys == 0:
            return
        data_rows = data[:n_keys] if len(data) > n_keys else data
        if len(data_rows) != n_keys:
            raise ValueError(f"Diffusion step sidecar expected {n_keys} rows, got {len(data_rows)}")

        if DIFFUSION_TQ_KEYS in data_rows.non_tensor_batch:
            keys = [str(key) for key in data_rows.non_tensor_batch[DIFFUSION_TQ_KEYS].tolist()]
            if Counter(keys) != Counter(batch_meta.keys):
                raise ValueError("Diffusion step sidecar row keys do not match the sampled queue keys")
        else:
            keys = [batch_meta.keys[i] for i in sort_diffusion_tq_keys(list(batch_meta.keys))]

        batch = data_rows.batch
        fields = (
            {
                name: batch[name]
                for name in diffusion_persisted_tq_fields(algorithm)
                if name != "reward_extra_info" and name in batch
            }
            if batch is not None
            else {}
        )
        uids = data_rows.non_tensor_batch.get("uid")
        reward_extra_info = _pack_reward_extra_info(data_rows, n_keys)
        self._rows.update(
            {
                (batch_meta.partition_id, key): _SidecarRow(
                    tensors={name: tensor[i].detach().cpu().clone() for name, tensor in fields.items()},
                    uid=uids[i] if uids is not None else None,
                    reward_extra_info=(
                        _snapshot_diagnostic_value(reward_extra_info[i]) if reward_extra_info is not None else None
                    ),
                )
                for i, key in enumerate(keys)
            }
        )

    def build_metrics_dataproto(self, batch_meta: KVBatchMeta) -> DataProto | None:
        """Assemble complete metric rows in canonical queue key order."""
        keys = [(batch_meta.partition_id, key) for key in batch_meta.keys]
        if not keys or any(
            key not in self._rows or "sample_level_rewards" not in self._rows[key].tensors for key in keys
        ):
            return None

        rows = [self._rows[keys[i]] for i in sort_diffusion_tq_keys(list(batch_meta.keys))]
        tensors = {}
        for name in rows[0].tensors:
            if all(name in row.tensors for row in rows):
                values = [row.tensors[name] for row in rows]
                tensors[name] = torch.stack(values) if values[0].ndim == 0 else _stack_field(values)

        non_tensors = {}
        if any(row.uid is not None for row in rows):
            non_tensors["uid"] = np.fromiter((row.uid for row in rows), dtype=object, count=len(rows))
        if any(row.reward_extra_info is not None for row in rows):
            non_tensors["reward_extra_info"] = np.fromiter(
                (dict(row.reward_extra_info) if row.reward_extra_info is not None else None for row in rows),
                dtype=object,
                count=len(rows),
            )
        return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors)

    def get_sample_level_scores(self, batch_meta: KVBatchMeta) -> list[float] | None:
        """Return all row scores in the requested key order when available."""
        keys = [(batch_meta.partition_id, key) for key in batch_meta.keys]
        if not keys or any(
            key not in self._rows or "sample_level_scores" not in self._rows[key].tensors for key in keys
        ):
            return None
        return [float(self._rows[key].tensors["sample_level_scores"].sum().item()) for key in keys]

    def evict(self, partition_id: str, keys: Iterable[str]) -> None:
        """Discard metric rows after their queue rows are cleared."""
        for key in keys:
            self._rows.pop((partition_id, key), None)

    def clear(self) -> None:
        """Discard every retained metric row."""
        self._rows.clear()

    def __len__(self) -> int:
        return len(self._rows)


def sort_diffusion_tq_keys(keys: list[str]) -> list[int]:
    """Return indices that sort TransferQueue keys in rollout order.

    Args:
        keys: TransferQueue row keys from ``KVBatchMeta.keys``.

    Returns:
        Permutation indices that reorder ``keys`` by ``(uid, rollout, output)``.
    """
    return sorted(range(len(keys)), key=lambda i: _parse_tq_key(keys[i]))


def _parse_tq_key(key: str) -> tuple[str, int, int]:
    """Split a ``{uid}_{session}_{output}`` TransferQueue key."""
    parts = key.rsplit("_", 2)
    if len(parts) == 3:
        try:
            return parts[0], int(parts[1]), int(parts[2])
        except ValueError:
            return key, 0, 0
    return key, 0, 0


def canonicalize_diffusion_tq_meta(batch_meta: KVBatchMeta) -> KVBatchMeta:
    """Return a copy of ``batch_meta`` with rows in v0 rollout order.

    Upstream ``_materialize_batch`` emits trajectory keys in TransferQueue
    ``kv_list`` iteration order, which is storage-arbitrary and differs per
    run, while the v0 trainer produced rows prompt-major in dataset order
    (``prompt_index * rollout.n + session``). ``prompt_index`` is unique only
    within one generation batch, so rows carry ``(gen_batch_seq,
    prompt_index)`` — the per-run generation-batch number plus the position
    inside it — which stays unique across every path that can mix batches in
    one sample (DAPO speculative refills, incomplete-group replacement,
    multi-chunk generation dispatches). Rows whose tag predates the pair fall
    back to uid order, which is deterministic but not dataset order.
    """
    if len(batch_meta.keys) < 2:
        return batch_meta

    def sort_key(i: int) -> tuple:
        uid, session, output = _parse_tq_key(batch_meta.keys[i])
        tag = batch_meta.tags[i]
        prompt_index = tag.get("prompt_index") if isinstance(tag, dict) else None
        gen_batch_seq = tag.get("gen_batch_seq") if isinstance(tag, dict) else None
        if prompt_index is None or gen_batch_seq is None:
            return (1, uid, session, output)
        return (0, int(gen_batch_seq), int(prompt_index), session, output)

    perm = sorted(range(len(batch_meta.keys)), key=sort_key)
    # replace() carries fields/extra_info over so future consumers reading
    # them (e.g. a kv_batch_get driven by batch_meta.fields) see real values.
    return dataclasses.replace(
        batch_meta,
        keys=[batch_meta.keys[i] for i in perm],
        tags=[batch_meta.tags[i] for i in perm],
    )
