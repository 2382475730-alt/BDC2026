import json
sequence_length = 60
feature_num = '158+39'
config = json.loads(r'''
{
    "seed": 42,
    "sequence_length": 60,
    "d_model": 256,
    "nhead": 4,
    "num_layers": 3,
    "dim_feedforward": 512,
    "batch_size": 4,
    "num_epochs": 30,
    "learning_rate": 2e-05,
    "dropout": 0.1,
    "feature_num": "158+39",
    "max_grad_norm": 5.0,
    "pairwise_weight": 0.5,
    "boundary_margin": 0.1,
    "global_ce_weight": 0.03,
    "score_l2_weight": 0.0001,
    "bottom_k": 20,
    "base_weight": 1.0,
    "top5_weight": 2.0,
    "use_grad_clip": true,
    "early_stop_patience": 5,
    "ensemble_second_best": true,
    "inference_cross_sectional_norm": true,
    "run_train_rerank_search": false,
    "rerank_pool_size": 150,
    "rerank_return_5_weight": -3.0,
    "rerank_volume_change_weight": 0.8,
    "rerank_volatility_10_weight": -0.3,
    "rerank_extra_weights": {
        "atr_14": 1.0,
        "open_close_spread": 1.5,
        "volume_ratio": -1.0,
        "_gate_volume_ratio_5": 0.0
    },
    "output_top_k": 5,
    "weight_mode": "regime_volume_gate_c69_else_c4",
    "rank_weights": [
        0.05,
        0.15,
        0.5,
        0.15,
        0.15
    ],
    "softmax_temperature": 0.03,
    "min_weight": 0.05,
    "max_weight": 0.5,
    "rerank_search": {
        "return_5": [
            -1.5,
            -1.0,
            -0.6,
            -0.3,
            0.0,
            0.3
        ],
        "volume_change": [
            -0.2,
            0.0,
            0.2,
            0.4,
            0.6,
            0.8
        ],
        "volatility_10": [
            -0.8,
            -0.6,
            -0.4,
            -0.2,
            0.0,
            0.2
        ],
        "pool_size": [
            10,
            20,
            50
        ],
        "inference_cross_sectional_norm": [
            false,
            true
        ],
        "ensemble_second_best": [
            false,
            true
        ],
        "output_top_k": [
            1,
            2,
            3,
            5
        ],
        "weight_mode": [
            "floor_capped_softmax",
            "capped_softmax",
            "equal",
            "softmax"
        ],
        "softmax_temperature": [
            0.7,
            1.0,
            1.5,
            2.0
        ],
        "min_weight": [
            0.0,
            0.005,
            0.01
        ],
        "max_weight": [
            0.4,
            0.5,
            0.6,
            0.7,
            0.75,
            1.0
        ]
    },
    "rerank_acceptance_override": {
        "enabled": true,
        "pool_size": 150,
        "inference_cross_sectional_norm": true,
        "ensemble_second_best": true,
        "rerank_return_5_weight": -3.0,
        "rerank_volume_change_weight": 0.8,
        "rerank_volatility_10_weight": -0.3,
        "rerank_extra_weights": {
            "atr_14": 1.0,
            "open_close_spread": 1.5,
            "volume_ratio": -1.0,
            "_gate_volume_ratio_5": 0.0
        },
        "output_top_k": 5,
        "weight_mode": "regime_volume_gate_c69_else_c4",
        "rank_weights": [
            0.05,
            0.15,
            0.5,
            0.15,
            0.15
        ],
        "softmax_temperature": 0.03,
        "min_weight": 0.05,
        "max_weight": 0.5,
        "manual_selected_ranks": [
            1,
            2,
            12,
            15,
            17
        ],
        "regime_gate_feature": "_gate_volume_ratio_5",
        "regime_volume_ratio_scaled_threshold": 1.111401,
        "regime_manual_selected_ranks": [
            1,
            2,
            12,
            15,
            17
        ],
        "regime_manual_rank_weights": [
            0.05,
            0.15,
            0.5,
            0.15,
            0.15
        ],
        "regime_c4_temperature": 0.03,
        "regime_c4_min_weight": 0.05,
        "regime_c4_max_weight": 0.5
    },
    "output_dir": "./model/60_158+39",
    "data_path": "./data",
    "train_filename": "train.csv",
    "manual_selected_ranks": [
        1,
        2,
        12,
        15,
        17
    ],
    "regime_gate_feature": "_gate_volume_ratio_5",
    "regime_volume_ratio_scaled_threshold": 1.111401,
    "regime_manual_selected_ranks": [
        1,
        2,
        12,
        15,
        17
    ],
    "regime_manual_rank_weights": [
        0.05,
        0.15,
        0.5,
        0.15,
        0.15
    ],
    "regime_c4_temperature": 0.03,
    "regime_c4_min_weight": 0.05,
    "regime_c4_max_weight": 0.5
}
''')
