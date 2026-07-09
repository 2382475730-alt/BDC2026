# 配置参数
sequence_length = 60
feature_num = '158+39'
config = {
    'seed': 42,
    'sequence_length': sequence_length,   # 使用过去60个交易日的数据（排序任务可以用稍短的序列）
    'd_model': 256,          # Transformer输入维度
    'nhead': 4,             # 注意力头数量
    'num_layers': 3,        # Transformer层数
    'dim_feedforward': 512, # 前馈网络维度
    'batch_size': 4,        # 排序任务batch_size可以小一些，因为每个batch包含更多股票
    'num_epochs': 30,       # early stopping keeps runtime bounded
    'learning_rate': 2e-5,
    'dropout': 0.1,
    'feature_num': feature_num,
    'max_grad_norm': 5.0,

    'pairwise_weight': 0.5, # top/bottom boundary pairwise loss weight
    'boundary_margin': 0.1,
    'global_ce_weight': 0.03,
    'score_l2_weight': 1e-4,
    'bottom_k': 20,
    'base_weight': 1.0, # 非top-k样本权重
    'top5_weight': 2.0, # top-5样本权重（应大于base_weight）
    'use_grad_clip': True,
    'early_stop_patience': 5,
    'ensemble_second_best': True,
    'inference_cross_sectional_norm': True,
    'run_train_rerank_search': False,
    'rerank_pool_size': 150,
    'rerank_return_5_weight': -3.0,
    'rerank_volume_change_weight': 0.8,
    'rerank_volatility_10_weight': -0.3,
    'rerank_extra_weights': {
        'atr_14': 1.0,
        'open_close_spread': 1.5,
        'volume_ratio': -1.0,
    },
    'output_top_k': 5,
    'weight_mode': 'floor_capped_softmax',
    'softmax_temperature': 0.03,
    'min_weight': 0.05,
    'max_weight': 0.5,
    'rerank_search': {
        'return_5': [-1.5, -1.0, -0.6, -0.3, 0.0, 0.3],
        'volume_change': [-0.2, 0.0, 0.2, 0.4, 0.6, 0.8],
        'volatility_10': [-0.8, -0.6, -0.4, -0.2, 0.0, 0.2],
        'pool_size': [10, 20, 50],
        'inference_cross_sectional_norm': [False, True],
        'ensemble_second_best': [False, True],
        'output_top_k': [1, 2, 3, 5],
        'weight_mode': ['floor_capped_softmax', 'capped_softmax', 'equal', 'softmax'],
        'softmax_temperature': [0.7, 1.0, 1.5, 2.0],
        'min_weight': [0.0, 0.005, 0.01],
        'max_weight': [0.4, 0.5, 0.6, 0.7],
    },
    'rerank_acceptance_override': {
        'enabled': True,
        'pool_size': 150,
        'inference_cross_sectional_norm': True,
        'ensemble_second_best': True,
        'rerank_return_5_weight': -3.0,
        'rerank_volume_change_weight': 0.8,
        'rerank_volatility_10_weight': -0.3,
        'rerank_extra_weights': {
            'atr_14': 1.0,
            'open_close_spread': 1.5,
            'volume_ratio': -1.0,
        },
        'output_top_k': 5,
        'weight_mode': 'floor_capped_softmax',
        'softmax_temperature': 0.03,
        'min_weight': 0.05,
        'max_weight': 0.5,
    },

    'output_dir': f'./model/{sequence_length}_{feature_num}',
    'data_path': './data',
    'train_filename': 'train.csv',

    # Validation framework controls. The default remains the original C4 path.
    'validation_mode': 'legacy',
    'walk_forward_output_dir': './walk_forward_runs',
    'walk_forward_folds': [
        ('fold1', '2024-01-02', '2024-06-28'),
        ('fold2', '2024-07-01', '2024-12-31'),
        ('fold3', '2025-01-02', '2025-06-30'),
        ('fold4', '2025-07-01', '2025-12-31'),
        ('fold5', '2026-01-19', '2026-03-12'),
    ],
    'walk_forward_run_name': 'C4_frozen_walkforward',
    'walk_forward_score_mode': 'base_and_rerank',
    'same_update_steps': False,
    'sample_weight_mode': None,
    'duplicate_recent_samples': True,
    'recent_weight_start_date': '2024-01-01',
    'recent_sample_multiplier': 2,
    'recent_sample_weight': 2.0,
    'enable_top5_weight_fix': False,
    'use_positional_encoding': True,
}
