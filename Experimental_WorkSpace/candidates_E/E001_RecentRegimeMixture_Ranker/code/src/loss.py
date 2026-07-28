from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

from score_contract import primary_scores


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor | None) -> torch.Tensor:
    if weights is None:
        return values.mean()
    weights = weights.to(device=values.device, dtype=values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1e-12)


class ListwiseRankingLoss(nn.Module):
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = float(temperature)

    def forward(self, scores: torch.Tensor, relevance: torch.Tensor, mask: torch.Tensor, day_weights: torch.Tensor | None = None) -> torch.Tensor:
        valid_days = mask.sum(dim=-1) > 1
        if not bool(valid_days.any()):
            return scores.sum() * 0.0
        pred = F.log_softmax(scores.masked_fill(~mask.bool(), -1e9)[valid_days] / self.temperature, dim=-1)
        target = F.softmax(relevance.masked_fill(~mask.bool(), -1e9)[valid_days] / self.temperature, dim=-1)
        losses = -(target * pred).sum(dim=-1)
        selected_weights = day_weights[valid_days] if day_weights is not None else None
        return _weighted_mean(losses, selected_weights)


class PairwiseBoundaryLoss(nn.Module):
    def __init__(self, margin: float = 0.05, top_k: int = 5, bottom_k: int = 20):
        super().__init__()
        self.margin, self.top_k, self.bottom_k = float(margin), int(top_k), int(bottom_k)

    def forward(self, scores: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        losses: list[torch.Tensor] = []
        for day_scores, day_labels, day_mask in zip(scores, labels, mask.bool()):
            valid_scores, valid_labels = day_scores[day_mask], day_labels[day_mask]
            if valid_scores.numel() <= 1:
                continue
            order = torch.argsort(valid_labels, descending=True)
            top = order[:min(self.top_k, order.numel())]
            bottom_count = min(self.bottom_k, max(1, order.numel() - top.numel()))
            bottom = order[-bottom_count:]
            losses.append(F.relu(self.margin - (valid_scores[top, None] - valid_scores[None, bottom])).mean())
        return torch.stack(losses).mean() if losses else scores.sum() * 0.0


def straight_through_topk_weights(
    scores: torch.Tensor, mask: torch.Tensor, top_k: int = 2, temperature: float = 0.05,
) -> torch.Tensor:
    """Hard equal Top-K forward pass with a soft boundary surrogate gradient."""
    if scores.shape != mask.shape:
        raise ValueError("scores and mask must have identical shapes")
    result = torch.zeros_like(scores)
    tau = max(float(temperature), 1e-6)
    for row, (day_scores, day_mask) in enumerate(zip(scores, mask.bool())):
        valid = day_scores[day_mask]
        if valid.numel() == 0:
            continue
        k = min(max(1, int(top_k)), valid.numel())
        hard = torch.zeros_like(valid)
        top_indices = torch.topk(valid, k=k, largest=True, sorted=False).indices
        hard[top_indices] = 1.0 / float(k)
        if k == valid.numel():
            soft = torch.full_like(valid, 1.0 / float(k))
        else:
            boundary = torch.topk(valid, k=k, largest=True).values[-1].detach()
            soft = torch.sigmoid((valid - boundary) / tau)
            soft = soft / soft.sum().clamp_min(1e-12)
        result[row, day_mask] = hard + soft - soft.detach()
    return result


# Stable public name retained for tests and copied R5 utilities.
differentiable_topk_weights = straight_through_topk_weights


@dataclass
class LossBreakdown:
    total: torch.Tensor
    ranking: torch.Tensor
    pairwise: torch.Tensor
    portfolio_return: torch.Tensor
    topk_utility: torch.Tensor
    robust_penalty: torch.Tensor
    recent_aux: torch.Tensor
    objective_mode: str


class PortfolioAwareRankingLoss(nn.Module):
    def __init__(
        self,
        temperature: float = 1.0,
        pairwise_weight: float = 0.25,
        pairwise_margin: float = 0.05,
        portfolio_return_weight: float = 0.0,
        portfolio_volatility_weight: float = 0.0,
        objective_mode: str = "topk_aligned",
        topk_k: int = 2,
        topk_temperature: float = 0.05,
        topk_return_weight: float = 0.05,
        robust_penalty_weight: float = 0.01,
        robust_downside_temperature: float = 0.02,
        recent_aux_weight: float = 0.0,
        recent_half_life: int = 252,
        **_: object,
    ):
        super().__init__()
        if objective_mode != "topk_aligned":
            raise ValueError("E_R1 permits only the train/inference-aligned Top-K objective")
        if int(recent_half_life) not in {126, 252}:
            raise ValueError("recent_half_life is frozen to 126 or 252 trading days")
        self.listwise = ListwiseRankingLoss(temperature)
        self.pairwise = PairwiseBoundaryLoss(pairwise_margin)
        self.pairwise_weight = float(pairwise_weight)
        self.topk_k = int(topk_k)
        self.topk_temperature = float(topk_temperature)
        self.topk_return_weight = float(topk_return_weight + portfolio_return_weight)
        self.robust_penalty_weight = float(robust_penalty_weight)
        self.robust_downside_temperature = float(robust_downside_temperature)
        self.recent_aux_weight = float(recent_aux_weight)
        self.recent_half_life = int(recent_half_life)
        self.objective_mode = objective_mode
        self.unused_portfolio_volatility_weight = float(portfolio_volatility_weight)

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        labels: torch.Tensor,
        relevance: torch.Tensor,
        mask: torch.Tensor,
        recency_weights: torch.Tensor | None = None,
    ) -> LossBreakdown:
        scores = primary_scores(outputs)
        ranking = self.listwise(scores, relevance, mask)
        pairwise = self.pairwise(scores, labels, mask)
        weights = straight_through_topk_weights(scores, mask, self.topk_k, self.topk_temperature)
        daily_returns = (weights * labels.masked_fill(~mask.bool(), 0.0)).sum(dim=-1)
        portfolio_return = daily_returns.mean()
        tau = max(self.robust_downside_temperature, 1e-6)
        robust_penalty = (tau * F.softplus(-daily_returns / tau)).mean()
        if self.recent_aux_weight > 0.0 and "recent_residual" in outputs:
            recent_aux = self.listwise(outputs["recent_residual"], relevance, mask, recency_weights)
        else:
            recent_aux = scores.sum() * 0.0
        total = (
            ranking + self.pairwise_weight * pairwise
            - self.topk_return_weight * portfolio_return
            + self.robust_penalty_weight * robust_penalty
            + self.recent_aux_weight * recent_aux
        )
        return LossBreakdown(
            total=total, ranking=ranking.detach(), pairwise=pairwise.detach(),
            portfolio_return=portfolio_return.detach(), topk_utility=portfolio_return.detach(),
            robust_penalty=robust_penalty.detach(), recent_aux=recent_aux.detach(),
            objective_mode=self.objective_mode,
        )
