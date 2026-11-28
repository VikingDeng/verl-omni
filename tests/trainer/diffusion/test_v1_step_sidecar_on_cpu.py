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

import gc
import weakref
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from transfer_queue import KVBatchMeta
from verl import DataProto
from verl.utils import tensordict_utils as tu

from verl_omni.trainer.diffusion.v1 import tq_utils
from verl_omni.trainer.diffusion.v1 import trainer_base as trainer_base_module
from verl_omni.trainer.diffusion.v1.trainer_sync import PolicyGradientDiffusionTrainerV1Sync
from verl_omni.workers.config.diffusion.model import DIFFUSION_TQ_KEYS


@pytest.fixture
def queue(monkeypatch):
    rows = {}

    def put(*, keys, partition_id, fields):
        columns = {field: tu.get(fields, field) for field in fields.keys()}
        for i, key in enumerate(keys):
            row = rows.setdefault((partition_id, key), {})
            row.update({name: deepcopy(column[i]) for name, column in columns.items()})

    def get(*, keys, partition_id, select_fields=None):
        selected = [rows[partition_id, key] for key in keys]
        fields = selected[0] if select_fields is None else select_fields
        columns = {}
        for field in fields:
            if all(field in row for row in selected):
                values = [row[field] for row in selected]
                columns[field] = torch.stack(values) if isinstance(values[0], torch.Tensor) else values
        return tu.get_tensordict(columns)

    def retrieve_meta(*, keys, partition_id, create):
        assert create is False
        return SimpleNamespace(field_names=set.intersection(*(set(rows[partition_id, key]) for key in keys)))

    def clear(*, keys, partition_id):
        for key in keys:
            del rows[partition_id, key]

    put, get, clear = Mock(side_effect=put), Mock(side_effect=get), Mock(side_effect=clear)
    monkeypatch.setattr(tq_utils.tq, "kv_batch_put", put)
    monkeypatch.setattr(tq_utils.tq, "kv_batch_get", get)
    monkeypatch.setattr(tq_utils.tq, "get_client", lambda: SimpleNamespace(kv_retrieve_meta=retrieve_meta))
    monkeypatch.setattr(tq_utils.tq, "kv_clear", clear)
    return SimpleNamespace(put=put, get=get, clear=clear)


def _trainer(*, direct_preference=False):
    trainer = object.__new__(PolicyGradientDiffusionTrainerV1Sync)
    trainer.config = OmegaConf.create(
        {
            "algorithm": {"adv_estimator": "flow_grpo", "global_std": True},
            "actor_rollout_ref": {"actor": {"ppo_mini_batch_size": 1}, "rollout": {"n": 2}},
            "data": {"train_batch_size": 2},
            "global_profiler": {"steps": None, "profile_continuous_steps": False},
            "trainer": {
                "project_name": "sidecar",
                "experiment_name": "test",
                "logger": "console",
                "val_before_train": False,
                "save_freq": -1,
                "test_freq": -1,
                "total_epochs": 1,
            },
        }
    )
    trainer.trainer_mode = "sync"
    trainer.parameter_sync_step = 1
    trainer.global_steps = 0
    trainer._nonfinite_grad_streak = 0
    trainer._training_watchdog = None
    trainer._is_direct_preference = direct_preference
    trainer._loss_fn = None
    trainer._has_old_adapter = False
    trainer._teacher_one_step_off = False
    trainer.use_rm = False
    trainer.use_reference_policy = False
    trainer.use_teacher_policy = False
    trainer.worker_side_trajectory = False
    trainer.skip_old_log_prob = False
    trainer.reward_loop_manager = SimpleNamespace(reward_loop_worker_handles=object())
    trainer.actor_rollout_wg = SimpleNamespace()
    trainer.resource_pool_manager = SimpleNamespace(get_n_gpus=lambda: 1)
    trainer.tokenizer = SimpleNamespace(
        pad_token_id=0,
        batch_decode=lambda prompts, skip_special_tokens: [str(prompt.item()) for prompt in prompts],
    )
    trainer._compute_old_log_prob = Mock(
        side_effect=lambda data: DataProto.from_dict(
            tensors={"old_log_probs": -torch.ones_like(data.batch["all_timesteps"])}
        )
    )
    trainer._update_actor = Mock(return_value=DataProto(meta_info={"metrics": {"actor/grad_norm": 2.0}}))
    trainer._dump_generations = Mock()
    return trainer


def _seed_rollouts(queue, keys, values, *, partition="train", scored=True, streaming_extras=None):
    tags = [
        {"response_shape": (3, 2, 2), "is_padding": False, "min_global_steps": 0, "max_global_steps": 0} for _ in keys
    ]
    prompts = torch.tensor(values, dtype=torch.long).reshape(-1, 1)
    fields = {
        "uid": [key.rsplit("_", 2)[0] for key in keys],
        "prompts": prompts,
        "responses": prompts.reshape(-1, 1, 1, 1).expand(-1, 3, 2, 2).to(torch.uint8),
        "all_timesteps": torch.tensor([[1.0, 0.5]]).repeat(len(keys), 1),
        "all_latents": torch.zeros(len(keys), 3, 2, 2),
    }
    if scored:
        fields["rm_scores"] = torch.tensor(values, dtype=torch.float32).reshape(-1, 1)
    if streaming_extras is not None:
        fields["extra_fields"] = [{"reward_extra_info": row} for row in streaming_extras]
    queue.put(keys=keys, partition_id=partition, fields=tu.get_tensordict(fields))
    return KVBatchMeta(partition_id=partition, keys=keys, tags=tags)


def _compute_metrics(trainer, meta, metrics=None):
    metrics = {} if metrics is None else metrics.copy()
    trainer._compute_metrics(meta, metrics, {"gen": 1.0, "step": 2.0}, global_steps=1, epoch=0)
    return metrics


@pytest.mark.parametrize("algorithm", ["policy_gradient", "direct_preference"])
def test_step_metrics_match_transfer_queue_round_trip(queue, algorithm):
    direct_preference = algorithm == "direct_preference"
    trainer = _trainer(direct_preference=direct_preference)
    meta = _seed_rollouts(
        queue,
        ["b_0_0", "a_1_0", "a_0_0", "b_1_0"],
        [2, 4, 1, 9],
        streaming_extras=[{"ocr": 0.5}, {"ocr": 0.25}, {"ocr": None}, {"ocr": 1.0}],
    )
    source = trainer._read_training_batch(meta)
    if direct_preference:
        trainer._prepare_actor_batch = Mock(side_effect=lambda data, rewards: data.select_idxs([0, 1]))
        source.batch["rm_scores"] = source.batch["rm_scores"].squeeze(-1)
        source.batch["old_log_probs"] = -torch.ones_like(source.batch["all_timesteps"])
    queue.put.reset_mock()
    queue.get.reset_mock()

    train_metrics = {}
    if direct_preference:
        result = trainer._train_direct_preference_batch(train_metrics, {}, meta, source)
        trainer._compute_old_log_prob.assert_not_called()
        assert len(trainer._update_actor.call_args.args[0]) == 2
        persisted = source
    else:
        result = trainer._train_sampled_batch(train_metrics, {}, meta, data=source)
        trainer._compute_old_log_prob.assert_called_once()
        persisted = trainer._update_actor.call_args.args[0]
    assert result is meta
    metrics = _compute_metrics(trainer, meta, train_metrics)
    queue.put.assert_not_called()
    queue.get.assert_not_called()

    snapshot = trainer._step_sidecar.build_metrics_dataproto(meta)
    tq_utils.put_dataproto_fields_to_tq(meta, persisted, tq_utils.diffusion_persisted_tq_fields(algorithm))
    fallback_trainer = _trainer(direct_preference=direct_preference)
    assert metrics == _compute_metrics(fallback_trainer, meta, train_metrics)
    fields = tq_utils.diffusion_metric_tq_fields(algorithm)
    queue.get.assert_called_once_with(
        keys=["a_0_0", "a_1_0", "b_0_0", "b_1_0"],
        partition_id="train",
        select_fields=fields,
    )
    round_trip = tq_utils.diffusion_tq_batch_to_dataproto(meta, select_fields=fields)
    assert set(snapshot.batch.keys()) == set(round_trip.batch.keys())
    assert "old_log_probs" not in snapshot.batch
    torch.testing.assert_close(snapshot.batch, round_trip.batch, rtol=0, atol=0)
    assert snapshot.non_tensor_batch["uid"].tolist() == ["a", "a", "b", "b"]
    assert (
        snapshot.non_tensor_batch["reward_extra_info"].tolist()
        == round_trip.non_tensor_batch["reward_extra_info"].tolist()
    )
    assert metrics["perf/total_num_images"] == 4
    assert metrics["critic/rewards/mean"] == 4.0
    assert metrics["critic/rewards/group_size"] == 2.0
    assert metrics["critic/ocr/mean"] == pytest.approx(1.75 / 3)
    assert ("critic/advantages/mean" in metrics) is not direct_preference
    assert ("critic/returns/mean" in metrics) is not direct_preference
    if direct_preference:
        trainer._log_rollout_data(meta, {}, "/tmp/rollouts")
        assert trainer._dump_generations.call_args.kwargs["scores"] == [1.0, 4.0, 2.0, 9.0]


def test_parameter_sync_cycle_combines_rows_for_metrics_and_dump(queue):
    trainer = _trainer()
    trainer.parameter_sync_step = 2
    trainer.prepare_step = lambda: {}
    trainer.on_sample_end = lambda: None
    metas = iter(
        [
            _seed_rollouts(queue, ["z_1_0", "z_0_0"], [9, 7]),
            _seed_rollouts(queue, ["a_1_0", "a_0_0"], [5, 3]),
        ]
    )
    trainer._sample_training_batch = Mock(side_effect=lambda batch_size: (next(metas), {}))
    queue.put.reset_mock()

    metrics = {}
    combined = trainer.step(metrics, {})
    assert combined.keys == ["z_1_0", "z_0_0", "a_1_0", "a_0_0"]
    assert [call.args for call in trainer._sample_training_batch.call_args_list] == [(1,), (1,)]
    assert queue.get.call_count == 2
    assert _compute_metrics(trainer, combined, metrics)["critic/rewards/mean"] == 6.0
    assert queue.get.call_count == 2
    queue.put.assert_not_called()

    snapshot = trainer._step_sidecar.build_metrics_dataproto(combined)
    assert snapshot.non_tensor_batch["uid"].tolist() == ["a", "a", "z", "z"]
    assert snapshot.batch["sample_level_scores"].flatten().tolist() == [3, 5, 7, 9]

    trainer._log_rollout_data(combined, {}, "/tmp/rollouts")
    dumped = trainer._dump_generations.call_args.kwargs
    assert queue.get.call_count == 3
    assert "rm_scores" not in queue.get.call_args.kwargs["select_fields"]
    assert "sample_level_scores" not in queue.get.call_args.kwargs["select_fields"]
    assert dumped["inputs"] == ["3", "5", "7", "9"]
    assert dumped["outputs"][:, 0, 0, 0].tolist() == [3, 5, 7, 9]
    assert dumped["scores"] == [3, 5, 7, 9]
    assert dumped["reward_extra_infos_dict"]["uid"] == ["a_0_0", "a_1_0", "z_0_0", "z_1_0"]


def test_balance_padding_is_excluded_from_sidecar(queue):
    trainer = _trainer()
    trainer.actor_rollout_wg._query_dispatch_info = lambda mesh: [0, 1, 2, 3]
    meta = _seed_rollouts(queue, ["c_1_0", "c_0_0", "a_0_0"], [8, 4, 2])

    trainer._train_sampled_batch({}, {}, meta)

    actor_data = trainer._update_actor.call_args.args[0]
    assert len(actor_data) == 4
    torch.testing.assert_close(actor_data.batch["advantages"][-1], torch.zeros(2))
    snapshot = trainer._step_sidecar.build_metrics_dataproto(meta)
    assert len(trainer._step_sidecar) == 3
    assert len(snapshot) == 3
    assert snapshot.batch["sample_level_scores"].flatten().tolist() == [2, 4, 8]
    assert _compute_metrics(trainer, meta)["critic/rewards/mean"] == pytest.approx(14 / 3)


@pytest.mark.parametrize(
    ("column", "message"),
    [
        (None, "missing non_tensor_batch column 'reward/clap/errors'"),
        ([0], "column 'reward/clap/errors': expected 2 rows, got 1"),
    ],
)
def test_sidecar_validates_declared_reward_extra_columns(column, message):
    trainer = _trainer()
    name = "reward/clap/errors"
    meta = KVBatchMeta(partition_id="train", keys=["a_0_0", "b_0_0"], tags=[{}, {}])
    scores = torch.tensor([[2.0], [4.0]])
    data = DataProto.from_dict(
        tensors={"sample_level_scores": scores, "sample_level_rewards": scores},
        meta_info={"reward_extra_keys": [name]},
    )
    if column is not None:
        data.non_tensor_batch[name] = np.array(column, dtype=object)

    with pytest.raises(ValueError, match=message):
        trainer._record_step_sidecar(meta, data, "direct_preference")


def test_colocated_rewards_merge_streaming_diagnostics_and_dump_scores(queue):
    trainer = _trainer()
    name = "reward/clap/errors"
    meta = _seed_rollouts(
        queue, ["p_1_0", "p_0_0"], [7, 5], scored=False, streaming_extras=[{"ocr": 0.75}, {"ocr": 0.25}]
    )
    reward = DataProto.from_dict(
        tensors={"rm_scores": torch.tensor([[13.0], [27.0]])},
        non_tensors={name: np.array([1, None], dtype=object)},
        meta_info={"reward_extra_keys": [name]},
    )
    trainer.use_rm = True
    trainer.reward_loop_manager = SimpleNamespace(
        reward_loop_worker_handles=None, compute_rm_score=Mock(return_value=reward)
    )
    queue.put.reset_mock()

    trainer._train_sampled_batch({}, {}, meta)
    metrics = _compute_metrics(trainer, meta)

    queue.put.assert_not_called()
    assert queue.get.call_count == 1
    assert metrics["critic/ocr/mean"] == pytest.approx(0.5)
    assert metrics[f"critic/{name}/count"] == 1
    assert metrics[f"critic/{name}/mean"] == pytest.approx(0.5)
    snapshot = trainer._step_sidecar.build_metrics_dataproto(meta)
    assert snapshot.non_tensor_batch["reward_extra_info"].tolist() == [
        {"ocr": 0.25, name: 1},
        {"ocr": 0.75, name: None},
    ]
    assert "reward_extra_keys" not in snapshot.meta_info

    trainer._log_rollout_data(meta, {}, "/tmp/rollouts")
    dumped = trainer._dump_generations.call_args.kwargs
    assert dumped["scores"] == [13.0, 27.0]
    assert dumped["inputs"] == ["5", "7"]
    assert dumped["reward_extra_infos_dict"]["uid"] == ["p_0_0", "p_1_0"]


def test_worker_side_projected_rows_use_explicit_keys(queue):
    trainer = _trainer()
    trainer.worker_side_trajectory = True
    meta = _seed_rollouts(queue, ["p_1_0", "p_0_0"], [9, 3])
    data = trainer._read_training_batch(meta)
    assert data.non_tensor_batch[DIFFUSION_TQ_KEYS].tolist() == ["p_0_0", "p_1_0"]
    data = data.select_idxs([1, 0])
    assert "all_latents" not in data.batch

    trainer._train_sampled_batch({}, {}, meta, data=data)

    snapshot = trainer._step_sidecar.build_metrics_dataproto(meta)
    assert DIFFUSION_TQ_KEYS not in snapshot.non_tensor_batch
    assert snapshot.batch["sample_level_scores"].flatten().tolist() == [3, 9]
    assert trainer._step_sidecar.get_sample_level_scores(meta) == [9, 3]

    data.non_tensor_batch[DIFFUSION_TQ_KEYS][0] = "p_0_0"
    with pytest.raises(ValueError, match="row keys do not match"):
        trainer._step_sidecar.record(meta, data, "policy_gradient")
    assert trainer._step_sidecar.get_sample_level_scores(meta) == [9, 3]


def test_sidecar_requires_complete_keys_and_evicts_by_partition():
    sidecar = tq_utils.DiffusionStepSidecar()
    train = KVBatchMeta(partition_id="train", keys=["p_0_0"], tags=[{}])
    other = KVBatchMeta(partition_id="other", keys=["p_0_0"], tags=[{}])
    incomplete = KVBatchMeta(partition_id="train", keys=["p_0_0", "p_1_0"], tags=[{}, {}])
    empty = KVBatchMeta(partition_id="train", keys=[], tags=[])
    data = DataProto.from_dict(tensors={"sample_level_scores": torch.tensor([1.0])})
    sidecar.record(empty, data, "direct_preference")
    sidecar.record(train, data, "direct_preference")
    sidecar.record(other, data, "direct_preference")

    for meta in (empty, incomplete):
        assert sidecar.build_metrics_dataproto(meta) is None
        assert sidecar.get_sample_level_scores(meta) is None
    assert len(sidecar) == 2

    sidecar.evict(train.partition_id, train.keys)
    assert sidecar.build_metrics_dataproto(train) is None
    assert sidecar.get_sample_level_scores(other) == [1.0]
    assert len(sidecar) == 1


def test_sidecar_metrics_pad_variable_length_timestep_rows():
    trainer = _trainer()
    for key, reward, num_timesteps in [("z_0_0", 3.0, 3), ("a_0_0", 2.0, 2)]:
        meta = KVBatchMeta(partition_id="train", keys=[key], tags=[{}])
        scores = torch.tensor([[reward]])
        data = DataProto.from_dict(
            tensors={"sample_level_scores": scores, "sample_level_rewards": scores.expand(1, num_timesteps)}
        )
        trainer._record_step_sidecar(meta, data, "policy_gradient")

    meta = KVBatchMeta(partition_id="train", keys=["z_0_0", "a_0_0"], tags=[{}, {}])
    snapshot = trainer._step_sidecar.build_metrics_dataproto(meta)

    torch.testing.assert_close(snapshot.batch["sample_level_rewards"], torch.tensor([[2.0, 2.0, 0.0], [3.0, 3.0, 3.0]]))
    assert trainer._step_sidecar.get_sample_level_scores(meta) == [3.0, 2.0]


def test_sidecar_snapshots_metrics_without_retaining_trajectory(queue):
    trainer = _trainer()
    meta = _seed_rollouts(queue, ["p_0_0", "p_1_0"], [1, 3])
    data = trainer._read_training_batch(meta)
    trajectory = weakref.ref(data.batch["all_latents"])
    trainer._train_sampled_batch({}, {}, meta, data=data)

    actor_data = trainer._update_actor.call_args.args[0]
    metric_fields = ["advantages", "returns", "sample_level_scores", "sample_level_rewards"]
    expected = {name: actor_data.batch[name].clone() for name in metric_fields}
    for name in metric_fields:
        actor_data.batch[name].zero_()
    trainer._update_actor.reset_mock()
    trainer._compute_old_log_prob.reset_mock()
    del data, actor_data
    gc.collect()

    assert trajectory() is None
    snapshot = trainer._step_sidecar.build_metrics_dataproto(meta)
    assert set(snapshot.batch.keys()) == set(metric_fields)
    for name, value in expected.items():
        torch.testing.assert_close(snapshot.batch[name], value, rtol=0, atol=0)


def _configure_fit(trainer, monkeypatch, *, steps):
    trainer.total_training_steps = trainer.steps_per_epoch = steps
    trainer.checkpoint_manager = SimpleNamespace(update_weights=Mock(), sleep_replicas=Mock())
    trainer._init_profile_state()
    trainer._reissue_inflight_prompts = Mock()
    trainer._shutdown_dump_executor = Mock()
    trainer._shutdown_dataloaders = Mock()
    trainer.prepare_step = lambda: {}
    logger = Mock()
    monkeypatch.setattr(trainer_base_module, "Tracking", lambda **kwargs: logger)
    monkeypatch.setattr(trainer_base_module, "init_skip_manager", Mock())
    monkeypatch.setattr(trainer_base_module, "log_skip_summary", Mock())
    return logger


def test_fit_evicts_step_rows_after_queue_clear(queue, monkeypatch):
    trainer = _trainer()
    logger = _configure_fit(trainer, monkeypatch, steps=3)
    metas = [_seed_rollouts(queue, [f"p{step}_0_0", f"p{step}_1_0"], [step, step + 1]) for step in range(1, 4)]
    trainer._sample_training_batch = Mock(side_effect=[(meta, {}) for meta in metas])
    clear_rows = queue.clear.side_effect
    sidecar_sizes_at_clear = []
    sidecar_sizes_at_log = []

    def clear(*, keys, partition_id):
        sidecar_sizes_at_clear.append(len(trainer._step_sidecar))
        clear_rows(keys=keys, partition_id=partition_id)

    queue.clear.side_effect = clear
    logger.log.side_effect = lambda **kwargs: sidecar_sizes_at_log.append(len(trainer._step_sidecar))

    trainer.fit(object())

    assert sidecar_sizes_at_clear == [2, 2, 2]
    assert sidecar_sizes_at_log == [0, 0, 0]
    assert [call.kwargs["keys"] for call in queue.clear.call_args_list] == [meta.keys for meta in metas]


def test_fit_clears_sidecar_when_post_step_metrics_raise(queue, monkeypatch):
    trainer = _trainer()
    _configure_fit(trainer, monkeypatch, steps=1)
    meta = _seed_rollouts(queue, ["p_0_0", "p_1_0"], [1, 3])
    trainer._sample_training_batch = Mock(return_value=(meta, {}))

    def fail_metrics(*args):
        assert len(trainer._step_sidecar) == 2
        raise RuntimeError("metric failure")

    trainer._compute_metrics = fail_metrics

    with pytest.raises(RuntimeError, match="metric failure"):
        trainer.fit(object())

    assert len(trainer._step_sidecar) == 0
    queue.clear.assert_not_called()


def test_standalone_rollout_dump_falls_back_to_queue_scores(queue):
    trainer = _trainer()
    meta = _seed_rollouts(queue, ["p_0_0", "p_1_0"], [3, 8])
    queue.put(
        keys=meta.keys,
        partition_id=meta.partition_id,
        fields=tu.get_tensordict({"sample_level_scores": torch.tensor([[13.0], [28.0]])}),
    )

    trainer._log_rollout_data(meta, {}, "/tmp/rollouts")

    assert trainer._dump_generations.call_args.kwargs["scores"] == [13.0, 28.0]
    assert trainer._dump_generations.call_args.kwargs["reward_extra_infos_dict"]["uid"] == meta.keys
