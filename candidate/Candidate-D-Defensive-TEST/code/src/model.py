from __future__ import annotations

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float, max_len: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.dropout(values + self.pe[:, : values.size(1)])


class MarketEncoder(nn.Module):
    def __init__(self, input_dim: int, d_model: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, market_features: torch.Tensor) -> torch.Tensor:
        return self.net(market_features)


class FactorEncoder(nn.Module):
    def __init__(self, input_dim: int, d_model: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

    def forward(self, factor_features: torch.Tensor) -> torch.Tensor:
        return self.net(factor_features)


class StockTemporalEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        max_len: int,
    ):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.position = PositionalEncoding(d_model, dropout=dropout, max_len=max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.attention_pool = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, stock_sequence: torch.Tensor) -> torch.Tensor:
        batch, stocks, seq_len, feature_dim = stock_sequence.shape
        flat = stock_sequence.reshape(batch * stocks, seq_len, feature_dim)
        encoded = self.input_projection(flat)
        encoded = self.position(encoded)
        encoded = self.encoder(encoded)
        weights = torch.softmax(self.attention_pool(encoded), dim=1)
        pooled = torch.sum(encoded * weights, dim=1)
        return pooled.reshape(batch, stocks, -1)


class MarketConditionedFusion(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float, use_factor_encoder: bool):
        super().__init__()
        input_width = d_model * (3 if use_factor_encoder else 2)
        self.projection = nn.Sequential(
            nn.Linear(input_width, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.cross_stock = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        stock_embedding: torch.Tensor,
        market_embedding: torch.Tensor,
        mask: torch.Tensor,
        factor_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        market_context = market_embedding.unsqueeze(1).expand(-1, stock_embedding.size(1), -1)
        parts = [stock_embedding, market_context]
        if factor_embedding is not None:
            parts.append(factor_embedding)
        fused = self.projection(torch.cat(parts, dim=-1))
        attended, _ = self.cross_stock(fused, fused, fused, key_padding_mask=~mask.bool())
        return self.norm(fused + self.dropout(attended))


class MarketIntelligenceRanker(nn.Module):
    def __init__(
        self,
        stock_input_dim: int,
        market_input_dim: int,
        factor_input_dim: int,
        sequence_length: int,
        config: dict,
    ):
        super().__init__()
        d_model = int(config["d_model"])
        dropout = float(config.get("dropout", 0.1))
        nhead = int(config.get("nhead", 4))
        self.use_factor_encoder = bool(config.get("use_factor_encoder", False))

        self.stock_encoder = StockTemporalEncoder(
            input_dim=stock_input_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=int(config.get("num_layers", 2)),
            dim_feedforward=int(config.get("dim_feedforward", d_model * 2)),
            dropout=dropout,
            max_len=sequence_length,
        )
        self.market_encoder = MarketEncoder(market_input_dim, d_model=d_model, dropout=dropout)
        self.factor_encoder = (
            FactorEncoder(factor_input_dim, d_model=d_model, dropout=dropout)
            if self.use_factor_encoder
            else None
        )
        self.fusion = MarketConditionedFusion(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            use_factor_encoder=self.use_factor_encoder,
        )
        self.ranking_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        self.portfolio_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        stock_sequence = batch["stock_sequence"]
        market_features = batch["market_features"]
        factor_features = batch["factor_features"]
        mask = batch["mask"].bool()

        stock_embedding = self.stock_encoder(stock_sequence)
        market_embedding = self.market_encoder(market_features)
        factor_embedding = self.factor_encoder(factor_features) if self.factor_encoder is not None else None
        fused = self.fusion(stock_embedding, market_embedding, mask=mask, factor_embedding=factor_embedding)

        ranking_scores = self.ranking_head(fused).squeeze(-1).masked_fill(~mask, -1e9)
        portfolio_logits = self.portfolio_head(fused).squeeze(-1).masked_fill(~mask, -1e9)
        return {
            "ranking_scores": ranking_scores,
            "portfolio_logits": portfolio_logits,
            "embeddings": fused,
        }
