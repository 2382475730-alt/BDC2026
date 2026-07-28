from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_softmax(values: torch.Tensor, mask: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    masked = values.masked_fill(~mask.bool(), -1e9)
    return torch.softmax(masked / max(float(temperature), 1e-6), dim=-1)


class ListwiseRankingLoss(nn.Module):
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, scores: torch.Tensor, relevance: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        valid_days = mask.sum(dim=-1) > 1
        if not bool(valid_days.any()):
            return scores.sum() * 0.0
        masked_scores = scores.masked_fill(~mask.bool(), -1e9)
        masked_relevance = relevance.masked_fill(~mask.bool(), -1e9)
        pred_log_probs = F.log_softmax(masked_scores[valid_days] / self.temperature, dim=-1)
        target_probs = F.softmax(masked_relevance[valid_days] / self.temperature, dim=-1)
        return -(target_probs * pred_log_probs).sum(dim=-1).mean()


class PairwiseBoundaryLoss(nn.Module):
    def __init__(self, margin: float = 0.05, top_k: int = 5, bottom_k: int = 20):
        super().__init__()
        self.margin = margin
        self.top_k = top_k
        self.bottom_k = bottom_k

    def forward(self, scores: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        losses: list[torch.Tensor] = []
        for day_scores, day_labels, day_mask in zip(scores, labels, mask):
            valid_scores = day_scores[day_mask]
            valid_labels = day_labels[day_mask]
            if valid_scores.numel() <= 1:
                continue
            order = torch.argsort(valid_labels, descending=True)
            top = order[: min(self.top_k, order.numel())]
            bottom = order[-min(self.bottom_k, max(1, order.numel() - top.numel())) :]
            gap = valid_scores[top].unsqueeze(1) - valid_scores[bottom].unsqueeze(0)
            losses.append(F.relu(self.margin - gap).mean())
        if not losses:
            return scores.sum() * 0.0
        return torch.stack(losses).mean()


def differentiable_portfolio_weights(
    logits: torch.Tensor,
    mask: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    return masked_softmax(logits, mask=mask, temperature=temperature)


@dataclass
class LossBreakdown:
    total: torch.Tensor
    ranking: torch.Tensor
    pairwise: torch.Tensor
    portfolio_return: torch.Tensor
    portfolio_volatility: torch.Tensor


class PortfolioAwareRankingLoss(nn.Module):
    def __init__(
        self,
        temperature: float = 1.0,
        pairwise_weight: float = 0.25,
        pairwise_margin: float = 0.05,
        portfolio_return_weight: float = 0.0,
        portfolio_volatility_weight: float = 0.0,
    ):
        super().__init__()
        self.listwise = ListwiseRankingLoss(temperature=temperature)
        self.pairwise = PairwiseBoundaryLoss(margin=pairwise_margin)
        self.pairwise_weight = pairwise_weight
        self.portfolio_return_weight = portfolio_return_weight
        self.portfolio_volatility_weight = portfolio_volatility_weight
        self.temperature = temperature

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        labels: torch.Tensor,
        relevance: torch.Tensor,
        mask: torch.Tensor,
    ) -> LossBreakdown:
        scores = outputs["ranking_scores"]
        logits = outputs["portfolio_logits"]
        ranking = self.listwise(scores, relevance, mask)
        pairwise = self.pairwise(scores, labels, mask)

        weights = differentiable_portfolio_weights(logits, mask=mask, temperature=self.temperature)
        daily_returns = (weights * labels.masked_fill(~mask.bool(), 0.0)).sum(dim=-1)
        portfolio_return = daily_returns.mean()
        portfolio_volatility = daily_returns.std(unbiased=False) if daily_returns.numel() > 1 else daily_returns.sum() * 0.0

        total = (
            ranking
            + self.pairwise_weight * pairwise
            - self.portfolio_return_weight * portfolio_return
            + self.portfolio_volatility_weight * portfolio_volatility
        )
        return LossBreakdown(
            total=total,
            ranking=ranking.detach(),
            pairwise=pairwise.detach(),
            portfolio_return=portfolio_return.detach(),
            portfolio_volatility=portfolio_volatility.detach(),
        )
