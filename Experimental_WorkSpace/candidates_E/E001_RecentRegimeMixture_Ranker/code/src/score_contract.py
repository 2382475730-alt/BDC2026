from __future__ import annotations

import torch


PRIMARY_SCORE_KEY = "ranking_scores"


def primary_scores(outputs: dict[str, torch.Tensor]) -> torch.Tensor:
    """Return the only score tensor allowed for R5 Top-k training and inference."""
    return outputs[PRIMARY_SCORE_KEY]
