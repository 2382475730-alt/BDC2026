from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_softmax(values: torch.Tensor, mask: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    masked = values.masked_fill(~mask.bool(), -1e9)
    weights = torch.softmax(masked / max(float(temperature), 1e-6), dim=-1)
    weights = weights * mask.to(dtype=weights.dtype)
    return weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)


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


class PredictedBoundaryLoss(nn.Module):
    """Push true top-k names above the model's highest-scoring hard negatives."""

    def __init__(
        self,
        margin: float = 0.05,
        top_k: int = 5,
        hard_negative_k: int = 15,
    ):
        super().__init__()
        self.margin = float(margin)
        self.top_k = int(top_k)
        self.hard_negative_k = int(hard_negative_k)

    def forward(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        valid_mask = (
            torch.ones_like(scores, dtype=torch.bool)
            if mask is None
            else mask.bool()
        )
        losses: list[torch.Tensor] = []
        for day_scores, day_labels, day_mask in zip(scores, labels, valid_mask):
            valid_scores = day_scores[day_mask]
            valid_labels = day_labels[day_mask]
            selected_count = min(self.top_k, int(valid_scores.numel()))
            if selected_count <= 0 or valid_scores.numel() <= selected_count:
                continue

            true_order = torch.argsort(valid_labels, descending=True)
            true_top = true_order[:selected_count]
            true_top_mask = torch.zeros_like(valid_scores, dtype=torch.bool)
            true_top_mask[true_top] = True

            predicted_order = torch.argsort(valid_scores, descending=True)
            hard_negative_order = predicted_order[~true_top_mask[predicted_order]]
            hard_negative_count = min(
                self.hard_negative_k,
                int(hard_negative_order.numel()),
            )
            if hard_negative_count <= 0:
                continue
            hard_negatives = hard_negative_order[:hard_negative_count]
            gaps = (
                valid_scores[true_top].unsqueeze(1)
                - valid_scores[hard_negatives].unsqueeze(0)
            )
            losses.append(F.relu(self.margin - gaps).mean())
        if not losses:
            return scores.sum() * 0.0
        return torch.stack(losses).mean()


def differentiable_portfolio_weights(
    logits: torch.Tensor,
    mask: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    return masked_softmax(logits, mask=mask, temperature=temperature)


class _StraightThroughHardWeights(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        hard: torch.Tensor,
        soft: torch.Tensor,
    ) -> torch.Tensor:
        return hard

    @staticmethod
    def backward(
        ctx,
        gradient: torch.Tensor,
    ) -> tuple[None, torch.Tensor]:
        return None, gradient


def differentiable_topk_weights(
    scores: torch.Tensor,
    k: int = 5,
    temperature: float = 0.10,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Exact hard top-k weights in the forward pass with softmax gradients."""

    if scores.ndim != 2:
        raise ValueError("scores must have shape [batch, stocks]")
    valid = torch.ones_like(scores, dtype=torch.bool) if mask is None else mask.bool()
    hard = torch.zeros_like(scores)
    for row_index in range(scores.shape[0]):
        valid_index = torch.nonzero(valid[row_index], as_tuple=False).flatten()
        selected_count = min(max(int(k), 0), int(valid_index.numel()))
        if selected_count == 0:
            continue
        local_scores = scores[row_index, valid_index]
        local_top = torch.topk(local_scores, k=selected_count).indices
        selected_index = valid_index[local_top]
        hard[row_index, selected_index] = 1.0 / float(selected_count)
    soft = masked_softmax(scores, mask=valid, temperature=temperature)
    return _StraightThroughHardWeights.apply(hard, soft)


@dataclass
class LossBreakdown:
    total: torch.Tensor
    ranking: torch.Tensor
    pairwise: torch.Tensor
    portfolio_return: torch.Tensor
    portfolio_volatility: torch.Tensor
    topk_return: torch.Tensor
    downside: torch.Tensor
    cvar20: torch.Tensor
    positive_tail: torch.Tensor


class HStabilityLoss(nn.Module):
    """H-series loss aligned to the final top-5 equal policy."""

    def __init__(
        self,
        ranking_weight: float = 1.0,
        temperature: float = 1.0,
        pairwise_weight: float = 0.25,
        pairwise_margin: float = 0.05,
        topk_return_weight: float = 0.05,
        downside_weight: float = 0.08,
        downside_temperature: float = 0.02,
        cvar_weight: float = 0.04,
        cvar_alpha: float = 0.20,
        positive_tail_weight: float = 0.02,
        positive_tail_cap: float = 0.08,
        topk: int = 5,
        topk_temperature: float = 0.10,
        boundary_mode: str = "label_bottom",
        boundary_hard_negative_k: int = 15,
    ):
        super().__init__()
        self.listwise = ListwiseRankingLoss(temperature=temperature)
        if boundary_mode == "label_bottom":
            self.pairwise = PairwiseBoundaryLoss(
                margin=pairwise_margin,
                top_k=topk,
            )
        elif boundary_mode == "predicted_hard":
            self.pairwise = PredictedBoundaryLoss(
                margin=pairwise_margin,
                top_k=topk,
                hard_negative_k=boundary_hard_negative_k,
            )
        else:
            raise ValueError(f"unsupported boundary_mode: {boundary_mode}")
        self.boundary_mode = boundary_mode
        self.ranking_weight = float(ranking_weight)
        self.pairwise_weight = float(pairwise_weight)
        self.topk_return_weight = float(topk_return_weight)
        self.downside_weight = float(downside_weight)
        self.downside_temperature = float(downside_temperature)
        self.cvar_weight = float(cvar_weight)
        self.cvar_alpha = float(cvar_alpha)
        self.positive_tail_weight = float(positive_tail_weight)
        self.positive_tail_cap = float(positive_tail_cap)
        self.topk = int(topk)
        self.topk_temperature = float(topk_temperature)

    def component_weights(self) -> dict[str, float]:
        return {
            "ranking": self.ranking_weight,
            "pairwise": self.pairwise_weight,
            "topk_return": self.topk_return_weight,
            "downside": self.downside_weight,
            "cvar20": self.cvar_weight,
            "positive_tail": self.positive_tail_weight,
        }

    def component_terms(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor | None = None,
        relevance: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        valid = torch.ones_like(scores, dtype=torch.bool) if mask is None else mask.bool()
        target_relevance = labels if relevance is None else relevance
        ranking = self.listwise(scores, target_relevance, valid)
        pairwise = self.pairwise(scores, labels, valid)
        weights = differentiable_topk_weights(
            scores,
            k=self.topk,
            temperature=self.topk_temperature,
            mask=valid,
        )
        selected_returns = (
            weights * labels.masked_fill(~valid, 0.0)
        ).sum(dim=-1)
        topk_return = -selected_returns.mean()
        downside = (
            F.softplus(-selected_returns / max(self.downside_temperature, 1e-6))
            * self.downside_temperature
        ).mean()
        tail_count = max(1, int(torch.ceil(
            torch.tensor(selected_returns.numel() * self.cvar_alpha)
        ).item()))
        worst_returns = torch.topk(
            selected_returns,
            k=min(tail_count, selected_returns.numel()),
            largest=False,
        ).values
        cvar20 = F.relu(-worst_returns).mean()
        positive_tail = F.relu(
            selected_returns - self.positive_tail_cap
        ).square().mean()
        return {
            "ranking": ranking,
            "pairwise": pairwise,
            "topk_return": topk_return,
            "downside": downside,
            "cvar20": cvar20,
            "positive_tail": positive_tail,
        }

    def forward(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor | None = None,
        relevance: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        terms = self.component_terms(
            scores,
            labels,
            mask=mask,
            relevance=relevance,
        )
        weights = self.component_weights()
        total = sum(weights[name] * value for name, value in terms.items())
        return total, {
            **{name: value.detach() for name, value in terms.items()},
            "total": total.detach(),
        }


class PortfolioAwareRankingLoss(nn.Module):
    def __init__(
        self,
        temperature: float = 1.0,
        pairwise_weight: float = 0.25,
        pairwise_margin: float = 0.05,
        portfolio_return_weight: float = 0.0,
        portfolio_volatility_weight: float = 0.0,
        objective_mode: str = "legacy",
        ranking_weight: float = 1.0,
        topk_return_weight: float = 0.05,
        downside_weight: float = 0.08,
        downside_temperature: float = 0.02,
        cvar_weight: float = 0.04,
        cvar_alpha: float = 0.20,
        positive_tail_weight: float = 0.02,
        positive_tail_cap: float = 0.08,
        topk: int = 5,
        topk_temperature: float = 0.10,
        boundary_hard_negative_k: int = 15,
    ):
        super().__init__()
        self.listwise = ListwiseRankingLoss(temperature=temperature)
        self.pairwise = PairwiseBoundaryLoss(margin=pairwise_margin)
        self.pairwise_weight = pairwise_weight
        self.portfolio_return_weight = portfolio_return_weight
        self.portfolio_volatility_weight = portfolio_volatility_weight
        self.temperature = temperature
        self.objective_mode = objective_mode
        boundary_mode = (
            "predicted_hard"
            if objective_mode == "predicted_boundary_stability"
            else "label_bottom"
        )
        self.h_stability = HStabilityLoss(
            ranking_weight=ranking_weight,
            temperature=temperature,
            pairwise_weight=pairwise_weight,
            pairwise_margin=pairwise_margin,
            topk_return_weight=topk_return_weight,
            downside_weight=downside_weight,
            downside_temperature=downside_temperature,
            cvar_weight=cvar_weight,
            cvar_alpha=cvar_alpha,
            positive_tail_weight=positive_tail_weight,
            positive_tail_cap=positive_tail_cap,
            topk=topk,
            topk_temperature=topk_temperature,
            boundary_mode=boundary_mode,
            boundary_hard_negative_k=boundary_hard_negative_k,
        )

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        labels: torch.Tensor,
        relevance: torch.Tensor,
        mask: torch.Tensor,
    ) -> LossBreakdown:
        scores = outputs["ranking_scores"]
        if self.objective_mode in {
            "topk_stability",
            "predicted_boundary_stability",
        }:
            total, parts = self.h_stability(
                scores,
                labels,
                mask=mask,
                relevance=relevance,
            )
            zero = total.detach() * 0.0
            return LossBreakdown(
                total=total,
                ranking=parts["ranking"],
                pairwise=parts["pairwise"],
                portfolio_return=-parts["topk_return"],
                portfolio_volatility=zero,
                topk_return=parts["topk_return"],
                downside=parts["downside"],
                cvar20=parts["cvar20"],
                positive_tail=parts["positive_tail"],
            )
        if self.objective_mode != "legacy":
            raise ValueError(f"unsupported objective_mode: {self.objective_mode}")
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
            topk_return=(-portfolio_return).detach(),
            downside=portfolio_return.detach() * 0.0,
            cvar20=portfolio_return.detach() * 0.0,
            positive_tail=portfolio_return.detach() * 0.0,
        )

    def diagnostic_score_gradients(
        self,
        outputs: dict[str, torch.Tensor],
        labels: torch.Tensor,
        relevance: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict[str, dict[str, float]]:
        if self.objective_mode not in {
            "topk_stability",
            "predicted_boundary_stability",
        }:
            raise ValueError(
                "component gradient diagnostics require an H stability objective"
            )
        scores = outputs["ranking_scores"]
        terms = self.h_stability.component_terms(
            scores,
            labels,
            mask=mask,
            relevance=relevance,
        )
        weights = self.h_stability.component_weights()
        raw: dict[str, float] = {}
        weighted: dict[str, float] = {}
        score_gradient_l2: dict[str, float] = {}
        for name, value in terms.items():
            weighted_value = weights[name] * value
            gradient = torch.autograd.grad(
                weighted_value,
                scores,
                retain_graph=True,
                allow_unused=True,
            )[0]
            raw[name] = float(value.detach().cpu())
            weighted[name] = float(weighted_value.detach().cpu())
            score_gradient_l2[name] = (
                float(gradient.detach().norm().cpu())
                if gradient is not None
                else 0.0
            )
        return {
            "raw": raw,
            "weighted": weighted,
            "score_gradient_l2": score_gradient_l2,
        }
