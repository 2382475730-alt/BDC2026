from __future__ import annotations

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float, max_len: int = 256):
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div[:pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.dropout(values + self.pe[:, :values.size(1)])


class TemporalStockEncoder(nn.Module):
    def __init__(self, input_dim: int, d_model: int, nhead: int, layers: int, feedforward: int, dropout: float, max_len: int):
        super().__init__()
        self.projection = nn.Linear(input_dim, d_model)
        self.position = PositionalEncoding(d_model, dropout, max_len)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=feedforward,
            dropout=dropout, batch_first=True, activation="gelu", norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.pool = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.Tanh(), nn.Linear(d_model // 2, 1))

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        batch, stocks, length, width = sequence.shape
        values = sequence.reshape(batch * stocks, length, width)
        encoded = self.encoder(self.position(self.projection(values)))
        weights = torch.softmax(self.pool(encoded), dim=1)
        return (encoded * weights).sum(dim=1).reshape(batch, stocks, -1)


class MarketSequenceEncoder(nn.Module):
    def __init__(self, input_dim: int, d_model: int, dropout: float, temporal: bool):
        super().__init__()
        self.temporal = temporal
        if temporal:
            self.gru = nn.GRU(input_dim, d_model, num_layers=1, batch_first=True)
            self.post = nn.Sequential(nn.LayerNorm(d_model), nn.Dropout(dropout))
        else:
            self.static = nn.Sequential(
                nn.Linear(input_dim, d_model), nn.LayerNorm(d_model), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(d_model, d_model), nn.LayerNorm(d_model),
            )

    def forward(self, market_sequence: torch.Tensor) -> torch.Tensor:
        if self.temporal:
            _, hidden = self.gru(market_sequence)
            return self.post(hidden[-1])
        return self.static(market_sequence[:, -1])


class CrossStockBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        attended, _ = self.attention(values, values, values, key_padding_mask=~mask.bool(), need_weights=False)
        return self.norm(values + self.dropout(attended))


class RecentRegimeMixtureRanker(nn.Module):
    """Stable alpha plus a market-gated recent residual alpha.

    `ranking_scores == stable_score + gate * recent_residual` is an exact API
    contract and the only score used by both the training Top-K objective and
    inference ranking.
    """
    def __init__(self, stock_input_dim: int, market_input_dim: int, sequence_length: int, recent_length: int, config: dict):
        super().__init__()
        self.variant = str(config.get("variant", "e3_mixture"))
        if self.variant not in {"e1_static", "e2_market_gru", "e3_mixture"}:
            raise ValueError(f"unsupported model variant: {self.variant}")
        self.recent_length = int(recent_length)
        d_model = int(config.get("d_model", 128))
        nhead = int(config.get("nhead", 4))
        dropout = float(config.get("dropout", 0.1))
        feedforward = int(config.get("dim_feedforward", d_model * 2))
        self.stable_encoder = TemporalStockEncoder(
            stock_input_dim, d_model, nhead, int(config.get("num_layers", 2)),
            feedforward, dropout, sequence_length,
        )
        temporal_market = self.variant in {"e2_market_gru", "e3_mixture"}
        self.market_encoder = MarketSequenceEncoder(market_input_dim, d_model, dropout, temporal=temporal_market)
        self.stable_fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout),
        )
        self.stable_cross_stock = CrossStockBlock(d_model, nhead, dropout)
        self.stable_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, 1),
        )
        if self.variant == "e3_mixture":
            self.recent_encoder = TemporalStockEncoder(
                stock_input_dim, d_model, nhead, int(config.get("recent_num_layers", 1)),
                feedforward, dropout, recent_length,
            )
            self.recent_fusion = nn.Sequential(
                nn.Linear(d_model * 2, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout),
            )
            self.recent_cross_stock = CrossStockBlock(d_model, nhead, dropout)
            self.recent_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model // 2, 1),
            )
            initial_gate = min(max(float(config.get("initial_gate", 0.2)), 1e-4), 1.0 - 1e-4)
            gate_bias = math.log(initial_gate / (1.0 - initial_gate))
            self.gate_head = nn.Linear(d_model, 1)
            nn.init.zeros_(self.gate_head.weight)
            nn.init.constant_(self.gate_head.bias, gate_bias)
        self._initialize()
        if self.variant == "e3_mixture":
            # Preserve the stable expert at initialization.
            nn.init.zeros_(self.recent_head[-1].weight)
            nn.init.zeros_(self.recent_head[-1].bias)
            initial_gate = min(max(float(config.get("initial_gate", 0.2)), 1e-4), 1.0 - 1e-4)
            nn.init.zeros_(self.gate_head.weight)
            nn.init.constant_(self.gate_head.bias, math.log(initial_gate / (1.0 - initial_gate)))

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        sequence = batch["stock_sequence"]
        market_sequence = batch["market_sequence"]
        mask = batch["mask"].bool()
        market = self.market_encoder(market_sequence)
        stable = self.stable_encoder(sequence)
        market_per_stock = market.unsqueeze(1).expand(-1, sequence.size(1), -1)
        stable = self.stable_cross_stock(self.stable_fusion(torch.cat([stable, market_per_stock], dim=-1)), mask)
        stable_score = self.stable_head(stable).squeeze(-1)
        if self.variant == "e3_mixture":
            recent = self.recent_encoder(sequence[:, :, -self.recent_length:, :])
            recent = self.recent_cross_stock(self.recent_fusion(torch.cat([recent, market_per_stock], dim=-1)), mask)
            recent_residual = self.recent_head(recent).squeeze(-1)
            gate = torch.sigmoid(self.gate_head(market)).expand(-1, sequence.size(1))
        else:
            recent_residual = torch.zeros_like(stable_score)
            gate = torch.zeros_like(stable_score)
        ranking_scores = stable_score + gate * recent_residual
        ranking_scores = ranking_scores.masked_fill(~mask, -1e9)
        return {
            "ranking_scores": ranking_scores,
            "portfolio_logits": ranking_scores,
            "stable_score": stable_score.masked_fill(~mask, -1e9),
            "recent_residual": recent_residual.masked_fill(~mask, 0.0),
            "gate": gate.masked_fill(~mask, 0.0),
            "embeddings": stable,
        }


# The E lineage intentionally changes the class but keeps this alias so the
# copied R5 entrypoint can load code snapshots without import ambiguity.
MarketIntelligenceRanker = RecentRegimeMixtureRanker
