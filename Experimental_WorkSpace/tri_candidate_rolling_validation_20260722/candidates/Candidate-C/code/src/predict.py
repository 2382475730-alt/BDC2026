import os
import multiprocessing as mp
import json

import joblib
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from config import config
from model import StockTransformer
from utils import engineer_features_39, engineer_features_158plus39


feature_cloums_map = {
	'39': [
		'instrument', '开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌额', '换手率', '涨跌幅',
		'sma_5', 'sma_20', 'ema_12', 'ema_26', 'rsi', 'macd', 'macd_signal', 'volume_change', 'obv',
		'volume_ma_5', 'volume_ma_20', 'volume_ratio', 'kdj_k', 'kdj_d', 'kdj_j', 'boll_mid', 'boll_std',
		'atr_14', 'ema_60', 'volatility_10', 'volatility_20', 'return_1', 'return_5', 'return_10',
		'high_low_spread', 'open_close_spread', 'high_close_spread', 'low_close_spread'
	],
	'158+39': [
		'instrument', '开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌额', '换手率', '涨跌幅',
		'KMID', 'KLEN', 'KMID2', 'KUP', 'KUP2', 'KLOW', 'KLOW2', 'KSFT', 'KSFT2', 'OPEN0', 'HIGH0', 'LOW0',
		'VWAP0', 'ROC5', 'ROC10', 'ROC20', 'ROC30', 'ROC60', 'MA5', 'MA10', 'MA20', 'MA30', 'MA60', 'STD5',
		'STD10', 'STD20', 'STD30', 'STD60', 'BETA5', 'BETA10', 'BETA20', 'BETA30', 'BETA60', 'RSQR5', 'RSQR10',
		'RSQR20', 'RSQR30', 'RSQR60', 'RESI5', 'RESI10', 'RESI20', 'RESI30', 'RESI60', 'MAX5', 'MAX10', 'MAX20',
		'MAX30', 'MAX60', 'MIN5', 'MIN10', 'MIN20', 'MIN30', 'MIN60', 'QTLU5', 'QTLU10', 'QTLU20', 'QTLU30',
		'QTLU60', 'QTLD5', 'QTLD10', 'QTLD20', 'QTLD30', 'QTLD60', 'RANK5', 'RANK10', 'RANK20', 'RANK30',
		'RANK60', 'RSV5', 'RSV10', 'RSV20', 'RSV30', 'RSV60', 'IMAX5', 'IMAX10', 'IMAX20', 'IMAX30', 'IMAX60',
		'IMIN5', 'IMIN10', 'IMIN20', 'IMIN30', 'IMIN60', 'IMXD5', 'IMXD10', 'IMXD20', 'IMXD30', 'IMXD60',
		'CORR5', 'CORR10', 'CORR20', 'CORR30', 'CORR60', 'CORD5', 'CORD10', 'CORD20', 'CORD30', 'CORD60',
		'CNTP5', 'CNTP10', 'CNTP20', 'CNTP30', 'CNTP60', 'CNTN5', 'CNTN10', 'CNTN20', 'CNTN30', 'CNTN60',
		'CNTD5', 'CNTD10', 'CNTD20', 'CNTD30', 'CNTD60', 'SUMP5', 'SUMP10', 'SUMP20', 'SUMP30', 'SUMP60',
		'SUMN5', 'SUMN10', 'SUMN20', 'SUMN30', 'SUMN60', 'SUMD5', 'SUMD10', 'SUMD20', 'SUMD30', 'SUMD60',
		'VMA5', 'VMA10', 'VMA20', 'VMA30', 'VMA60', 'VSTD5', 'VSTD10', 'VSTD20', 'VSTD30', 'VSTD60', 'WVMA5',
		'WVMA10', 'WVMA20', 'WVMA30', 'WVMA60', 'VSUMP5', 'VSUMP10', 'VSUMP20', 'VSUMP30', 'VSUMP60', 'VSUMN5',
		'VSUMN10', 'VSUMN20', 'VSUMN30', 'VSUMN60', 'VSUMD5', 'VSUMD10', 'VSUMD20', 'VSUMD30', 'VSUMD60',
		'sma_5', 'sma_20', 'ema_12', 'ema_26', 'rsi', 'macd', 'macd_signal', 'volume_change', 'obv',
		'volume_ma_5', 'volume_ma_20', 'volume_ratio', 'kdj_k', 'kdj_d', 'kdj_j', 'boll_mid', 'boll_std',
		'atr_14', 'ema_60', 'volatility_10', 'volatility_20', 'return_1', 'return_5', 'return_10',
		'high_low_spread', 'open_close_spread', 'high_close_spread', 'low_close_spread'
	]
}

feature_engineer_func_map = {
	'39': engineer_features_39,
	'158+39': engineer_features_158plus39,
}


def preprocess_predict_data(df, stockid2idx):
	assert config['feature_num'] in feature_engineer_func_map, f"Unsupported feature_num: {config['feature_num']}"
	feature_engineer = feature_engineer_func_map[config['feature_num']]
	feature_columns = feature_cloums_map[config['feature_num']]

	df = df.copy()
	df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)
	groups = [group for _, group in df.groupby('股票代码', sort=False)]
	if len(groups) == 0:
		raise ValueError('输入数据为空，无法预测')

	num_processes = min(10, mp.cpu_count())
	print('cpus!!!!!!!!!!!!!!!!!!',mp.cpu_count())
	with mp.Pool(processes=num_processes) as pool:
		processed_list = list(tqdm(pool.imap(feature_engineer, groups), total=len(groups), desc='预测集特征工程'))

	processed = pd.concat(processed_list).reset_index(drop=True)
	processed['instrument'] = processed['股票代码'].map(stockid2idx)
	processed = processed.dropna(subset=['instrument']).copy()
	processed['instrument'] = processed['instrument'].astype(np.int64)
	processed['日期'] = pd.to_datetime(processed['日期'])
	volume_col = feature_columns[5]
	if volume_col in processed.columns:
		processed = processed.sort_values(['股票代码', '日期']).reset_index(drop=True)
		processed['_gate_volume_ratio_5'] = processed.groupby('股票代码')[volume_col].transform(
			lambda values: values.astype(float) / values.astype(float).shift(5)
		)
		processed['_gate_volume_ratio_5'] = processed['_gate_volume_ratio_5'].replace([np.inf, -np.inf], np.nan).fillna(0.0)

	return processed, feature_columns


def build_inference_sequences(data, features, sequence_length, stock_ids, latest_date):
	sequences, sequence_stock_ids = [], []
	for stock_id in stock_ids:
		stock_history = data[
			(data['股票代码'] == stock_id) &
			(data['日期'] <= latest_date)
		].sort_values('日期').tail(sequence_length)

		if len(stock_history) == sequence_length:
			sequences.append(stock_history[features].values.astype(np.float32))
			sequence_stock_ids.append(stock_id)

	if len(sequences) == 0:
		raise ValueError('没有可用于预测的股票序列，请检查数据与 sequence_length')

	return np.asarray(sequences, dtype=np.float32), sequence_stock_ids


def cross_sectional_normalize_np(sequences, eps=1e-8):
	if sequences.shape[0] <= 1:
		return np.zeros_like(sequences, dtype=np.float32)
	mean = sequences.mean(axis=0, keepdims=True)
	std = sequences.std(axis=0, keepdims=True)
	return ((sequences - mean) / (std + eps)).astype(np.float32)


def zscore_1d(values, eps=1e-8):
	values = np.asarray(values, dtype=np.float32)
	std = values.std()
	if values.size == 0 or std < eps:
		return np.zeros_like(values, dtype=np.float32)
	return ((values - values.mean()) / (std + eps)).astype(np.float32)


_LATEST_GATE_STATE = {}


def softmax_np(values, temperature=1.0):
	values = np.asarray(values, dtype=np.float64)
	temperature = max(float(temperature), 1e-6)
	values = values / temperature
	values = values - values.max()
	exp_values = np.exp(values)
	return (exp_values / exp_values.sum()).astype(np.float64)


def cap_weights(weights, max_weight):
	weights = np.asarray(weights, dtype=np.float64)
	if weights.sum() <= 0:
		weights = np.full(len(weights), 1.0 / len(weights), dtype=np.float64)
	else:
		weights = weights / weights.sum()
	max_weight = max(float(max_weight), 1.0 / len(weights))
	if max_weight >= 1.0:
		return weights

	active = np.ones(len(weights), dtype=bool)
	capped = np.zeros(len(weights), dtype=np.float64)
	remaining = 1.0
	base = weights.copy()
	for _ in range(len(weights) + 1):
		if not active.any():
			break
		active_idx = np.flatnonzero(active)
		active_base = base[active]
		if active_base.sum() <= 0:
			allocation = np.full(len(active_idx), remaining / len(active_idx), dtype=np.float64)
		else:
			allocation = remaining * active_base / active_base.sum()
		over_cap = allocation > max_weight + 1e-12
		if not over_cap.any():
			capped[active_idx] = allocation
			remaining = 0.0
			break
		capped_idx = active_idx[over_cap]
		capped[capped_idx] = max_weight
		active[capped_idx] = False
		remaining = 1.0 - capped.sum()

	if remaining > 1e-10 and active.any():
		active_idx = np.flatnonzero(active)
		capped[active_idx] += remaining / len(active_idx)
	capped = np.minimum(capped, max_weight)
	return capped / capped.sum()


def floor_cap_weights(weights, min_weight, max_weight):
	weights = np.asarray(weights, dtype=np.float64)
	n = len(weights)
	if n == 0:
		return weights
	min_weight = max(float(min_weight), 0.0)
	max_weight = min(max(float(max_weight), min_weight), 1.0)
	if min_weight * n > 1.0 + 1e-12:
		raise ValueError('min_weight is infeasible for selected portfolio size')
	if max_weight * n < 1.0 - 1e-12:
		raise ValueError('max_weight is infeasible for selected portfolio size')
	residual = 1.0 - min_weight * n
	if residual <= 1e-12:
		return np.full(n, min_weight, dtype=np.float64)
	if weights.sum() <= 0:
		base = np.full(n, 1.0 / n, dtype=np.float64)
	else:
		base = weights / weights.sum()
	result = np.full(n, min_weight, dtype=np.float64)
	active = np.ones(n, dtype=bool)
	remaining = residual
	residual_cap = max_weight - min_weight
	for _ in range(n + 1):
		if not active.any() or remaining <= 1e-12:
			break
		active_idx = np.flatnonzero(active)
		active_base = base[active]
		if active_base.sum() <= 0:
			allocation = np.full(len(active_idx), remaining / len(active_idx), dtype=np.float64)
		else:
			allocation = remaining * active_base / active_base.sum()
		over_cap = allocation > residual_cap + 1e-12
		if not over_cap.any():
			result[active_idx] += allocation
			break
		capped_idx = active_idx[over_cap]
		result[capped_idx] = max_weight
		active[capped_idx] = False
		remaining = 1.0 - result.sum()
	result = np.minimum(result, max_weight)
	return result / result.sum()


def select_output_portfolio(
	candidate_indices,
	final_scores,
	sequence_stock_ids,
	output_top_k,
	weight_mode,
	temperature,
	min_weight,
	max_weight,
	rank_weights=None,
):
	output_top_k = max(1, min(int(output_top_k), len(candidate_indices), 5))
	order = np.argsort(final_scores)[::-1]
	if weight_mode == 'top1_full':
		selected_order = order[:1]
		weights = np.asarray([1.0], dtype=np.float64)
	elif weight_mode == 'equal':
		selected_order = order[:output_top_k]
		weights = np.full(len(selected_order), 1.0 / len(selected_order), dtype=np.float64)
	elif weight_mode == 'manual_rank_weights':
		selected_order = order[:output_top_k]
		if rank_weights is None:
			rank_weights = config.get('rank_weights')
		raw_weights = np.asarray(rank_weights if rank_weights is not None else [], dtype=np.float64)
		if raw_weights.size < len(selected_order):
			raw_weights = np.pad(raw_weights, (0, len(selected_order) - raw_weights.size), constant_values=0.0)
		weights = np.maximum(raw_weights[:len(selected_order)], 0.0)
		if weights.sum() <= 0:
			weights = np.full(len(selected_order), 1.0 / len(selected_order), dtype=np.float64)
		else:
			weights = weights / weights.sum()
			keep = weights > 1e-12
			if keep.any():
				selected_order = selected_order[keep]
				weights = weights[keep] / weights[keep].sum()
	elif weight_mode == 'manual_selected_rank_weights':
		raw_ranks = config.get('manual_selected_ranks', list(range(1, output_top_k + 1)))
		selected_positions = []
		for raw_rank in raw_ranks:
			position = int(raw_rank) - 1
			if 0 <= position < len(order):
				selected_positions.append(position)
			if len(selected_positions) >= output_top_k:
				break
		if len(selected_positions) < output_top_k:
			seen = set(selected_positions)
			for position in range(len(order)):
				if position not in seen:
					selected_positions.append(position)
					seen.add(position)
				if len(selected_positions) >= output_top_k:
					break
		selected_order = order[np.asarray(selected_positions[:output_top_k], dtype=np.int64)]
		if rank_weights is None:
			rank_weights = config.get('rank_weights')
		raw_weights = np.asarray(rank_weights if rank_weights is not None else [], dtype=np.float64)
		if raw_weights.size < len(selected_order):
			raw_weights = np.pad(raw_weights, (0, len(selected_order) - raw_weights.size), constant_values=0.0)
		weights = np.maximum(raw_weights[:len(selected_order)], 0.0)
		if weights.sum() <= 0:
			weights = np.full(len(selected_order), 1.0 / len(selected_order), dtype=np.float64)
		else:
			weights = weights / weights.sum()
	elif weight_mode == 'regime_volume_gate_c69_else_c4':
		gate_feature = config.get('regime_gate_feature', 'volume_ratio')
		gate_value = float(_LATEST_GATE_STATE.get(gate_feature, {}).get('mean', 0.0))
		gate_threshold = float(config.get('regime_volume_ratio_scaled_threshold', 0.167712))
		use_manual = gate_value <= gate_threshold
		if use_manual:
			raw_ranks = config.get('regime_manual_selected_ranks', config.get('manual_selected_ranks', list(range(1, output_top_k + 1))))
			selected_positions = []
			for raw_rank in raw_ranks:
				position = int(raw_rank) - 1
				if 0 <= position < len(order):
					selected_positions.append(position)
				if len(selected_positions) >= output_top_k:
					break
			if len(selected_positions) < output_top_k:
				seen = set(selected_positions)
				for position in range(len(order)):
					if position not in seen:
						selected_positions.append(position)
						seen.add(position)
					if len(selected_positions) >= output_top_k:
						break
			selected_order = order[np.asarray(selected_positions[:output_top_k], dtype=np.int64)]
			raw_weights = np.asarray(
				config.get('regime_manual_rank_weights', config.get('rank_weights', [])),
				dtype=np.float64,
			)
			if raw_weights.size < len(selected_order):
				raw_weights = np.pad(raw_weights, (0, len(selected_order) - raw_weights.size), constant_values=0.0)
			weights = np.maximum(raw_weights[:len(selected_order)], 0.0)
			if weights.sum() <= 0:
				weights = np.full(len(selected_order), 1.0 / len(selected_order), dtype=np.float64)
			else:
				weights = weights / weights.sum()
		else:
			selected_order = order[:output_top_k]
			weights = floor_cap_weights(
				softmax_np(final_scores[selected_order], temperature=float(config.get('regime_c4_temperature', temperature))),
				float(config.get('regime_c4_min_weight', min_weight)),
				float(config.get('regime_c4_max_weight', 0.5)),
			)
	elif weight_mode == 'capped_softmax':
		selected_order = order[:output_top_k]
		weights = cap_weights(softmax_np(final_scores[selected_order], temperature=temperature), max_weight)
	elif weight_mode == 'floor_capped_softmax':
		selected_order = order[:output_top_k]
		weights = floor_cap_weights(
			softmax_np(final_scores[selected_order], temperature=temperature),
			min_weight,
			max_weight,
		)
	else:
		selected_order = order[:output_top_k]
		weights = softmax_np(final_scores[selected_order], temperature=temperature)
		min_weight = max(float(min_weight), 0.0)
		if min_weight > 0 and len(weights) > 1:
			keep = weights >= min_weight
			if not keep.any():
				keep[np.argmax(weights)] = True
			selected_order = selected_order[keep]
			weights = weights[keep]
			weights = weights / weights.sum()

	top_indices = candidate_indices[selected_order]
	top_stocks = [sequence_stock_ids[i] for i in top_indices]
	return top_stocks, weights


def latest_feature_values(processed, sequence_stock_ids, latest_date, feature_name):
	if feature_name not in processed.columns:
		return np.zeros(len(sequence_stock_ids), dtype=np.float32)
	latest_rows = processed[processed['日期'] == latest_date]
	latest_rows = latest_rows.drop_duplicates('股票代码', keep='last').set_index('股票代码')
	values = []
	for stock_id in sequence_stock_ids:
		if stock_id in latest_rows.index:
			values.append(latest_rows.at[stock_id, feature_name])
		else:
			values.append(0.0)
	cleaned = np.nan_to_num(np.asarray(values, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
	if feature_name == config.get('regime_gate_feature', 'volume_ratio') and cleaned.size > 0:
		_LATEST_GATE_STATE[feature_name] = {
			'mean': float(cleaned.mean()),
			'median': float(np.median(cleaned)),
		}
	return cleaned


def predict_scores_from_checkpoints(model_paths, x, input_dim, stock_count, device):
	score_list = []
	for model_path in model_paths:
		model = StockTransformer(input_dim=input_dim, config=config, num_stocks=stock_count)
		model.load_state_dict(torch.load(model_path, map_location=device))
		model.to(device)
		model.eval()
		with torch.no_grad():
			score_list.append(model(x).squeeze(0).detach().cpu().numpy())
	return np.mean(score_list, axis=0)


def load_rerank_config(output_dir):
	rerank_config_path = os.path.join(output_dir, 'rerank_config.json')
	if not os.path.exists(rerank_config_path):
		return {}
	with open(rerank_config_path, 'r', encoding='utf-8') as f:
		return json.load(f)


def get_inference_setting(rerank_config, key, default):
	return rerank_config.get(key, config.get(key, default))


def main():
	data_file = os.path.join(config['data_path'], config.get('train_filename', 'train.csv'))
	model_path = os.path.join(config['output_dir'], 'best_model.pth')
	scaler_path = os.path.join(config['output_dir'], 'scaler.pkl')
	output_path = os.path.join('./output/', 'result.csv')

	if not os.path.exists(model_path):
		raise FileNotFoundError(f'未找到模型文件: {model_path}')
	if not os.path.exists(scaler_path):
		raise FileNotFoundError(f'未找到Scaler文件: {scaler_path}')

	raw_df = pd.read_csv(data_file, dtype={'股票代码': str})
	raw_df['股票代码'] = raw_df['股票代码'].astype(str).str.zfill(6)
	raw_df['日期'] = pd.to_datetime(raw_df['日期'])
	latest_date = raw_df['日期'].max()

	stock_ids = sorted(raw_df['股票代码'].unique())
	stockid2idx = {sid: idx for idx, sid in enumerate(stock_ids)}

	processed, features = preprocess_predict_data(raw_df, stockid2idx)
	processed[features] = processed[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)

	scaler = joblib.load(scaler_path)
	processed[features] = scaler.transform(processed[features])

	sequence_length = config['sequence_length']
	sequences_np, sequence_stock_ids = build_inference_sequences(
		processed,
		features,
		sequence_length,
		stock_ids,
		latest_date,
	)

	if torch.cuda.is_available():
		device = torch.device('cuda')
	elif torch.backends.mps.is_available():
		device = torch.device('mps')
	else:
		device = torch.device('cpu')

	rerank_config = load_rerank_config(config['output_dir'])

	if get_inference_setting(rerank_config, 'inference_cross_sectional_norm', True):
		sequences_np = cross_sectional_normalize_np(sequences_np)
	x = torch.from_numpy(sequences_np).unsqueeze(0).to(device)  # [1, N, L, F]

	model_paths = [model_path]
	second_model_path = os.path.join(config['output_dir'], 'second_best_model.pth')
	if get_inference_setting(rerank_config, 'ensemble_second_best', True) and os.path.exists(second_model_path):
		model_paths.append(second_model_path)
	scores = predict_scores_from_checkpoints(model_paths, x, len(features), len(stock_ids), device)

	if len(sequence_stock_ids) < 5:
		raise ValueError(f'可预测股票不足5只，当前仅有 {len(sequence_stock_ids)} 只')

	pool_size = min(get_inference_setting(rerank_config, 'pool_size', config.get('rerank_pool_size', 20)), len(sequence_stock_ids))
	candidate_indices = np.argsort(scores)[::-1][:pool_size]
	return_5 = latest_feature_values(processed, sequence_stock_ids, latest_date, 'return_5')[candidate_indices]
	volume_change = latest_feature_values(processed, sequence_stock_ids, latest_date, 'volume_change')[candidate_indices]
	volatility_10 = latest_feature_values(processed, sequence_stock_ids, latest_date, 'volatility_10')[candidate_indices]
	final_scores = (
		zscore_1d(scores[candidate_indices])
		+ get_inference_setting(rerank_config, 'rerank_return_5_weight', 0.3) * zscore_1d(return_5)
		+ get_inference_setting(rerank_config, 'rerank_volume_change_weight', 0.2) * zscore_1d(volume_change)
		+ get_inference_setting(rerank_config, 'rerank_volatility_10_weight', 0.2) * zscore_1d(volatility_10)
	)
	for feature_name, feature_weight in get_inference_setting(rerank_config, 'rerank_extra_weights', {}).items():
		extra_values = latest_feature_values(processed, sequence_stock_ids, latest_date, feature_name)[candidate_indices]
		final_scores = final_scores + float(feature_weight) * zscore_1d(extra_values)

	top5, weights = select_output_portfolio(
		candidate_indices,
		final_scores,
		sequence_stock_ids,
		get_inference_setting(rerank_config, 'output_top_k', config.get('output_top_k', 5)),
		get_inference_setting(rerank_config, 'weight_mode', config.get('weight_mode', 'softmax')),
		get_inference_setting(rerank_config, 'softmax_temperature', config.get('softmax_temperature', 1.0)),
		get_inference_setting(rerank_config, 'min_weight', config.get('min_weight', 0.0)),
		get_inference_setting(rerank_config, 'max_weight', config.get('max_weight', 1.0)),
		get_inference_setting(rerank_config, 'rank_weights', config.get('rank_weights')),
	)
	output_df = pd.DataFrame({
		'stock_id': top5,
		'weight': weights,
	})
	os.makedirs(os.path.dirname(output_path), exist_ok=True)
	output_df.to_csv(output_path, index=False, encoding='utf-8')

	print(f'预测日期: {latest_date.date()}')
	print(f'参与排序股票数: {len(sequence_stock_ids)}')
	print(f'重排序候选数: {pool_size}')
	print(f'使用模型数: {len(model_paths)}')
	print(f'结果已写入: {output_path}')


if __name__ == '__main__':
	mp.set_start_method('spawn', force=True)
	main()
