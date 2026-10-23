# -*- coding: utf-8 -*-
"""
Unified Reward Function - 统一路由
根据 problem_type 自动调用对应的 reward 模块
"""

import logging
from typing import Any, Dict, List

from rewards import boolean as bool_reward
from rewards import grounding as grounding_reward
from rewards import llava as llava_reward
from rewards import math_task as math_reward
from rewards import multiple_choice as mc_reward
from rewards import numerical as num_reward
from rewards import ocr as ocr_reward
from rewards import open_ended as oe_reward
from rewards.utils import normalize_response, preprocess_ground_truth


logger = logging.getLogger(__name__)

REWARD_NAME = "unified"
REWARD_TYPE = "batch"

# 任务类型到 reward 模块的映射
REWARD_MAPPING = {
    # 数学类
    "math": math_reward,
    "mathematics": math_reward,
    "perception": math_reward,
    "reasoning": math_reward,
    # 选择题类
    "multiple choice": mc_reward,
    "multiple_choice": mc_reward,
    # 数值类
    "numerical": num_reward,
    "number": num_reward,
    "regression": num_reward,
    # 开放式
    "open-ended": oe_reward,
    "open_ended": oe_reward,
    "video qa": oe_reward,
    "video description": oe_reward,
    "free-form": oe_reward,
    # OCR
    "ocr": ocr_reward,
    # 布尔
    "boolean": bool_reward,
    "binary classification": bool_reward,
    # LLaVA Critic
    "llava": llava_reward,
    "critic": llava_reward,
    # 定位类（特殊处理）
    "spatial grounding": "spatial",
    "temporal grounding": "temporal",
    "spatial-temporal grounding": "spatial-temporal",
}


def _zero_score() -> Dict[str, float]:
    return {
        "overall": 0.0,
        "accuracy": 0.0,
        "format": 0.0,
        "length_penalty": 0.0,
    }


def _compute_group(reward_target: Any, inputs: List[Dict[str, Any]], **kwargs) -> List[Dict[str, float]]:
    if reward_target == "spatial":
        return grounding_reward.spatial_compute_score(inputs, **kwargs)
    if reward_target == "temporal":
        return grounding_reward.temporal_compute_score(inputs, **kwargs)
    if reward_target == "spatial-temporal":
        return grounding_reward.spatial_temporal_compute_score(inputs, **kwargs)
    return reward_target.compute_score(inputs, **kwargs)


def compute_score(reward_inputs: List[Dict[str, Any]], **kwargs) -> List[Dict[str, float]]:
    """
    统一的 reward 计算入口
    根据每个样本的 problem_type 自动路由到对应的 reward 函数
    """
    results = [_zero_score() for _ in reward_inputs]
    grouped_inputs = {}

    for idx, inp in enumerate(reward_inputs):
        try:
            problem_type = (inp.get("problem_type", "") or "").lower().strip()
            reward_target = REWARD_MAPPING.get(problem_type, oe_reward)
            response = inp.get("response", "")
            response = normalize_response(response)
            ground_truth = preprocess_ground_truth(inp.get("ground_truth", ""))
            inp_processed = {**inp, "response": response, "ground_truth": ground_truth}
            grouped_inputs.setdefault(reward_target, []).append((idx, inp_processed))
        except Exception as e:
            logger.error(f"Error preparing reward for sample {idx}: {e}")

    for reward_target, indexed_inputs in grouped_inputs.items():
        indices, inputs = zip(*indexed_inputs)
        try:
            scores = _compute_group(reward_target, list(inputs), **kwargs)
            if len(scores) != len(inputs):
                raise ValueError(f"Reward returned {len(scores)} scores for {len(inputs)} inputs")
            for idx, score in zip(indices, scores):
                results[idx] = score
        except Exception as group_error:
            logger.error(f"Error computing reward batch: {group_error}")
            # Preserve per-sample fault isolation for reward implementations
            # that cannot process one malformed item in an otherwise valid batch.
            for idx, inp in indexed_inputs:
                try:
                    results[idx] = _compute_group(reward_target, [inp], **kwargs)[0]
                except Exception as sample_error:
                    logger.error(f"Error computing reward for sample {idx}: {sample_error}")

    return results
