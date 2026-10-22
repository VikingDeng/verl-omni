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
"""CPU tests for the UnifiedReward parser."""

from types import SimpleNamespace

import pytest
import torch

from verl_omni.utils.reward_score.unified_reward import _parse_unified_reward_scores, compute_score_unified_reward


def test_unified_reward_requires_all_labeled_axes():
    assert _parse_unified_reward_scores("Alignment Score: 4\nCoherence Score: 5") == {}
    assert _parse_unified_reward_scores("Alignment Score: 4\nCoherence Score: 5\nStyle Score: 3") == {
        "alignment": 4.0,
        "coherence": 5.0,
        "style": 3.0,
    }


@pytest.mark.asyncio
async def test_unified_reward_system_prompt_is_optional(monkeypatch):
    from verl_omni.utils.reward_score import unified_reward

    requests = []

    async def chat_complete(router_address, chat_complete_request, session=None):
        requests.append(chat_complete_request)
        content = "Alignment Score (1-5): 4\nCoherence Score (1-5): 5\nStyle Score (1-5): 3"
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    monkeypatch.setattr(unified_reward, "_chat_complete", chat_complete)
    image = torch.zeros(3, 8, 8, dtype=torch.uint8)
    extra_info = {"media_kind": "image"}

    await compute_score_unified_reward(
        data_source="test",
        solution_image=image,
        ground_truth="a cat",
        extra_info=extra_info,
        reward_router_address="judge",
        system_prompt="A",
    )
    await compute_score_unified_reward(
        data_source="test",
        solution_image=image,
        ground_truth="a cat",
        extra_info=extra_info,
        reward_router_address="judge",
    )

    assert requests[0]["messages"][0] == {"role": "system", "content": "A"}
    assert [message["role"] for message in requests[0]["messages"]] == ["system", "user"]
    assert requests[1]["messages"] == [requests[0]["messages"][1]]
