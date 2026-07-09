import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from tensorboardX import SummaryWriter
from config import config
from model import StockTransformer
from utils import engineer_features_39, engineer_features_158plus39
from utils import create_ranking_dataset_vectorized
import joblib
import os
import json
import multiprocessing as mp
import random
import shutil
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

feature_cloums_map = {
    '39': ['instrument','开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌额', '换手率', '涨跌幅','sma_5', 'sma_20', 'ema_12', 'ema_26', 'rsi', 'macd', 'macd_signal', 'volume_change', 'obv','volume_ma_5', 'volume_ma_20', 'volume_ratio', 'kdj_k', 'kdj_d', 'kdj_j', 'boll_mid', 'boll_std', 'atr_14', 'ema_60', 'volatility_10', 'volatility_20', 'return_1', 'return_5', 'return_10',  'high_low_spread', 'open_close_spread', 'high_close_spread', 'low_close_spread'],

    '158+39': ['instrument','开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌额', '换手率', '涨跌幅','KMID', 'KLEN', 'KMID2', 'KUP', 'KUP2', 'KLOW', 'KLOW2', 'KSFT', 'KSFT2', 'OPEN0', 'HIGH0', 'LOW0', 'VWAP0', 'ROC5', 'ROC10', 'ROC20', 'ROC30', 'ROC60', 'MA5', 'MA10', 'MA20', 'MA30', 'MA60', 'STD5', 'STD10', 'STD20', 'STD30', 'STD60', 'BETA5', 'BETA10', 'BETA20', 'BETA30', 'BETA60', 'RSQR5', 'RSQR10', 'RSQR20', 'RSQR30', 'RSQR60', 'RESI5', 'RESI10', 'RESI20', 'RESI30', 'RESI60', 'MAX5', 'MAX10', 'MAX20', 'MAX30', 'MAX60', 'MIN5', 'MIN10', 'MIN20', 'MIN30', 'MIN60', 'QTLU5', 'QTLU10', 'QTLU20', 'QTLU30', 'QTLU60', 'QTLD5', 'QTLD10', 'QTLD20', 'QTLD30', 'QTLD60', 'RANK5', 'RANK10', 'RANK20', 'RANK30', 'RANK60', 'RSV5', 'RSV10', 'RSV20', 'RSV30', 'RSV60', 'IMAX5', 'IMAX10', 'IMAX20', 'IMAX30', 'IMAX60', 'IMIN5', 'IMIN10', 'IMIN20', 'IMIN30', 'IMIN60', 'IMXD5', 'IMXD10', 'IMXD20', 'IMXD30', 'IMXD60', 'CORR5', 'CORR10', 'CORR20', 'CORR30', 'CORR60', 'CORD5', 'CORD10', 'CORD20', 'CORD30', 'CORD60', 'CNTP5', 'CNTP10', 'CNTP20', 'CNTP30', 'CNTP60', 'CNTN5', 'CNTN10', 'CNTN20', 'CNTN30', 'CNTN60', 'CNTD5', 'CNTD10', 'CNTD20', 'CNTD30', 'CNTD60', 'SUMP5', 'SUMP10', 'SUMP20', 'SUMP30', 'SUMP60', 'SUMN5', 'SUMN10', 'SUMN20', 'SUMN30', 'SUMN60', 'SUMD5', 'SUMD10', 'SUMD20', 'SUMD30', 'SUMD60', 'VMA5', 'VMA10', 'VMA20', 'VMA30', 'VMA60', 'VSTD5', 'VSTD10', 'VSTD20', 'VSTD30', 'VSTD60', 'WVMA5', 'WVMA10', 'WVMA20', 'WVMA30', 'WVMA60', 'VSUMP5', 'VSUMP10', 'VSUMP20', 'VSUMP30', 'VSUMP60', 'VSUMN5', 'VSUMN10', 'VSUMN20', 'VSUMN30', 'VSUMN60', 'VSUMD5', 'VSUMD10', 'VSUMD20', 'VSUMD30', 'VSUMD60','sma_5', 'sma_20', 'ema_12', 'ema_26', 'rsi', 'macd', 'macd_signal', 'volume_change', 'obv', 'volume_ma_5', 'volume_ma_20', 'volume_ratio', 'kdj_k', 'kdj_d', 'kdj_j', 'boll_mid', 'boll_std', 'atr_14', 'ema_60', 'volatility_10', 'volatility_20', 'return_1', 'return_5', 'return_10',  'high_low_spread', 'open_close_spread', 'high_close_spread', 'low_close_spread']
}
feature_engineer_func_map = {
    '39': engineer_features_39,
    '158+39': engineer_features_158plus39
}


def cross_sectional_normalize_torch(sequences, eps=1e-8):
    """Normalize [num_stocks, seq_len, features] across stocks for one day."""
    if sequences.size(0) <= 1:
        return torch.zeros_like(sequences)
    mean = sequences.mean(dim=0, keepdim=True)
    std = sequences.std(dim=0, unbiased=False, keepdim=True)
    return (sequences - mean) / (std + eps)


def _build_label_and_clean(processed, drop_small_open=True):
    """统一构建标签并清洗无效样本。"""
    processed['open_t1'] = processed.groupby('股票代码')['开盘'].shift(-1)
    processed['open_t5'] = processed.groupby('股票代码')['开盘'].shift(-5)

    # 过滤无效开盘价，避免收益率极端爆炸
    if drop_small_open:
        processed = processed[processed['open_t1'] > 1e-4]

    processed['label'] = (processed['open_t5'] - processed['open_t1']) / (processed['open_t1'] + 1e-12)
    processed = processed.dropna(subset=['label'])

    processed.drop(columns=['open_t1', 'open_t5'], inplace=True)
    return processed


def _preprocess_common(df, stockid2idx, desc, drop_small_open=True):
    assert config['feature_num'] in feature_engineer_func_map, f"Unsupported feature_num: {config['feature_num']}"
    assert stockid2idx is not None, "stockid2idx 不能为空"
    feature_engineer = feature_engineer_func_map[config['feature_num']]
    feature_columns = feature_cloums_map[config['feature_num']]

    # 保证时序正确，避免 shift 标签错位
    df = df.copy()
    df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)

    print(f"正在使用多进程进行{desc}...")
    groups = [group for _, group in df.groupby('股票代码', sort=False)]
    if len(groups) == 0:
        raise ValueError(f"{desc}输入为空，无法继续")

    num_processes = min(10, mp.cpu_count())
    with mp.Pool(processes=num_processes) as pool:
        processed_list = list(tqdm(pool.imap(feature_engineer, groups), total=len(groups), desc=desc))

    processed = pd.concat(processed_list).reset_index(drop=True)

    # 映射股票索引，并剔除映射失败样本
    processed['instrument'] = processed['股票代码'].map(stockid2idx)
    processed = processed.dropna(subset=['instrument']).copy()
    processed['instrument'] = processed['instrument'].astype(np.int64)

    processed = _build_label_and_clean(processed, drop_small_open=drop_small_open)
    return processed, feature_columns


# 数据预处理函数
def preprocess_data(df, is_train=True, stockid2idx=None):
    if not is_train:
        return _preprocess_common(df, stockid2idx, desc="特征工程", drop_small_open=False)
    return _preprocess_common(df, stockid2idx, desc="特征工程", drop_small_open=True)


def preprocess_val_data(df, stockid2idx=None):
    # 验证集与训练集保持同口径，避免 label 分布漂移
    return _preprocess_common(df, stockid2idx, desc="验证集特征工程", drop_small_open=True)


# 加权的排序损失函数
class WeightedRankingLoss(nn.Module):
    """Top-k listwise + top/bottom boundary ranking loss."""

    def __init__(
        self,
        temperature=1.0,
        k=5,
        weight_factor=2.0,
        pairwise_weight=0.7,
        base_weight=1.0,
        margin=0.2,
        global_ce_weight=0.05,
        score_l2_weight=1e-4,
        bottom_k=20,
    ):
        super(WeightedRankingLoss, self).__init__()
        self.temperature = temperature
        self.k = k
        self.weight_factor = weight_factor
        self.pairwise_weight = pairwise_weight
        self.base_weight = base_weight
        self.margin = margin
        self.global_ce_weight = global_ce_weight
        self.score_l2_weight = score_l2_weight
        self.bottom_k = bottom_k
        self.last_boundary_pair_count = 0

    def _listwise_ce(self, pred, target):
        pred_probs = F.softmax(pred / self.temperature, dim=1)
        target_probs = F.softmax(target / self.temperature, dim=1)
        return -(target_probs * torch.log(pred_probs + 1e-12)).sum(dim=1).mean()

    def forward(self, y_pred, y_true):
        """
        y_pred: [batch, num_items]
        y_true: [batch, num_items], cross-sectional normalized forward returns
        """
        _, num_items = y_true.size()
        if num_items <= 1:
            return y_pred.sum() * 0.0

        k = min(self.k, num_items)
        sorted_indices = torch.argsort(y_true, dim=1, descending=True)
        top_indices = sorted_indices[:, :k]

        top_pred = torch.gather(y_pred, 1, top_indices)
        top_true = torch.gather(y_true, 1, top_indices)
        topk_listwise = self._listwise_ce(top_pred, top_true)

        boundary_pairwise = y_pred.sum() * 0.0
        self.last_boundary_pair_count = 0
        if num_items > k:
            bottom_k = min(self.bottom_k, num_items - k)
            bottom_indices = sorted_indices[:, -bottom_k:]
            bottom_pred = torch.gather(y_pred, 1, bottom_indices)
            pred_gap = top_pred.unsqueeze(2) - bottom_pred.unsqueeze(1)
            self.last_boundary_pair_count = pred_gap.numel()
            boundary_pairwise = F.relu(self.margin - pred_gap).mean()

        global_ce = self._listwise_ce(y_pred, y_true)
        score_l2 = y_pred.pow(2).mean()

        return (
            topk_listwise
            + self.pairwise_weight * boundary_pairwise
            + self.global_ce_weight * global_ce
            + self.score_l2_weight * score_l2
        )

def calculate_ranking_metrics(y_pred, y_true, masks, k=5):
    """计算新的评估指标：Top 5 收益之和，以及与理论最高值和随机值的比值"""
    batch_size = y_pred.size(0)
    
    # Metrics accumulators
    pred_return_sum_list = []
    max_return_sum_list = []
    random_return_sum_list = []
    ratio_pred_list = []
    ratio_random_list = []
    final_score_list = []
    
    for i in range(batch_size):
        mask = masks[i]
        valid_indices = mask.nonzero().squeeze()
        
        if valid_indices.numel() < k:
            continue
            
        valid_pred = y_pred[i][valid_indices]
        valid_true = y_true[i][valid_indices] # This is the 5-day return
        
        # 1. Predicted Top 5
        _, pred_indices = torch.topk(valid_pred, k)
        pred_top_returns = valid_true[pred_indices]
        pred_return_sum = pred_top_returns.sum().item()
        
        # 2. True Top 5 (Theoretical Max)
        _, true_indices = torch.topk(valid_true, k)
        true_top_returns = valid_true[true_indices]
        max_return_sum = true_top_returns.sum().item()
        
        # 3. Random 5 (Expected Value)
        # Expected sum = 5 * mean(all valid returns)
        random_return_sum = k * valid_true.mean().item()
        
        # 计算每个样本的比例与稳定化 final_score
        ratio_pred = pred_return_sum / (max_return_sum + 1e-12) if abs(max_return_sum) > 1e-9 else 0.0
        ratio_random = random_return_sum / (max_return_sum + 1e-12) if abs(max_return_sum) > 1e-9 else 0.0
        denominator = max_return_sum - random_return_sum
        final_score = (pred_return_sum - random_return_sum) / (denominator + 1e-12) if abs(denominator) > 1e-6 else 0.0
        
        pred_return_sum_list.append(pred_return_sum)
        max_return_sum_list.append(max_return_sum)
        random_return_sum_list.append(random_return_sum)
        ratio_pred_list.append(ratio_pred)
        ratio_random_list.append(ratio_random)
        final_score_list.append(final_score)
        
    metrics = {
        'pred_return_sum': np.mean(pred_return_sum_list) if pred_return_sum_list else 0.0,
        'max_return_sum': np.mean(max_return_sum_list) if max_return_sum_list else 0.0,
        'random_return_sum': np.mean(random_return_sum_list) if random_return_sum_list else 0.0,
    }
    
    # 比值用逐样本均值，降低极端日影响
    metrics['ratio_pred'] = np.mean(ratio_pred_list) if ratio_pred_list else 0.0
    metrics['ratio_random'] = np.mean(ratio_random_list) if ratio_random_list else 0.0
    metrics['final_score'] = np.mean(final_score_list) if final_score_list else 0.0
    
    return metrics

class RankingDataset(torch.utils.data.Dataset):
    """排序数据集类"""
    def __init__(self, sequences, targets, relevance_scores, stock_indices):
        self.sequences = sequences
        self.targets = targets
        self.relevance_scores = relevance_scores
        self.stock_indices = stock_indices
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return {
            'sequences': torch.FloatTensor(self.sequences[idx]),  # [num_stocks, seq_len, features]
            'targets': torch.FloatTensor(self.targets[idx]),      # [num_stocks] 真实涨跌幅
            'relevance': torch.FloatTensor(self.relevance_scores[idx]),  # [num_stocks] normalized returns
            'stock_indices': torch.LongTensor(self.stock_indices[idx])  # [num_stocks] 股票索引
        }

def collate_fn(batch):
    """自定义collate函数处理变长序列"""
    sequences = [item['sequences'] for item in batch]
    targets = [item['targets'] for item in batch]
    relevance = [item['relevance'] for item in batch]
    stock_indices = [item['stock_indices'] for item in batch]
    
    # 找到最大股票数量
    max_stocks = max(seq.size(0) for seq in sequences)
    
    # Padding到相同长度
    padded_sequences = []
    padded_targets = []
    padded_relevance = []
    padded_stock_indices = []
    masks = []
    
    for seq, tgt, rel, stock_idx in zip(sequences, targets, relevance, stock_indices):
        num_stocks = seq.size(0)
        seq_len = seq.size(1)
        feature_dim = seq.size(2)
        
        seq = cross_sectional_normalize_torch(seq)

        # 创建padding
        if num_stocks < max_stocks:
            pad_size = max_stocks - num_stocks
            seq_pad = torch.zeros(pad_size, seq_len, feature_dim)
            tgt_pad = torch.zeros(pad_size)
            rel_pad = torch.zeros(pad_size)
            stock_pad = torch.zeros(pad_size, dtype=torch.long)
            
            seq = torch.cat([seq, seq_pad], dim=0)
            tgt = torch.cat([tgt, tgt_pad], dim=0)
            rel = torch.cat([rel, rel_pad], dim=0)
            stock_idx = torch.cat([stock_idx, stock_pad], dim=0)
        
        # 创建mask标记有效位置
        mask = torch.ones(max_stocks)
        mask[num_stocks:] = 0
        
        padded_sequences.append(seq)
        padded_targets.append(tgt)
        padded_relevance.append(rel)
        padded_stock_indices.append(stock_idx)
        masks.append(mask)
    
    return {
        'sequences': torch.stack(padded_sequences),      # [batch, max_stocks, seq_len, features]
        'targets': torch.stack(padded_targets),          # [batch, max_stocks]
        'relevance': torch.stack(padded_relevance),      # [batch, max_stocks]
        'stock_indices': torch.stack(padded_stock_indices),  # [batch, max_stocks]
        'masks': torch.stack(masks)                      # [batch, max_stocks]
    }

# 排序训练函数
def train_ranking_model(model, dataloader, criterion, optimizer, device, epoch, writer):
    model.train()
    total_loss = 0
    total_metrics = {}
    local_step = 0
    
    for batch in tqdm(dataloader, desc=f"Training Epoch {epoch+1}"):
        sequences = batch['sequences'].to(device)    # [batch, max_stocks, seq_len, features]
        targets = batch['targets'].to(device)        # [batch, max_stocks] 真实涨跌幅
        relevance = batch['relevance'].to(device)    # [batch, max_stocks] 预处理的相关性得分
        masks = batch['masks'].to(device)            # [batch, max_stocks] 有效位置mask
        
        optimizer.zero_grad()
        
        # 模型预测
        outputs = model(sequences)  # [batch, max_stocks] 预测分数
        
        # 应用mask，只考虑有效股票
        masked_outputs = outputs * masks + (1 - masks) * (-1e9)  # 无效位置设为很小的值
        masked_targets = targets * masks
        masked_relevance = relevance.float() * masks  # 使用预处理好的相关性得分
        
        # 计算损失（只对有效股票计算）
        batch_loss = None
        batch_size = sequences.size(0)
        
        for i in range(batch_size):
            mask = masks[i]
            valid_indices = mask.nonzero().squeeze()
            
            if valid_indices.numel() == 0:
                continue
                
            if valid_indices.dim() == 0:
                valid_indices = valid_indices.unsqueeze(0)
            
            # 获取有效股票的预测值和预处理好的相关性得分
            valid_pred = masked_outputs[i][valid_indices]
            valid_relevance = masked_relevance[i][valid_indices]
            
            if len(valid_pred) > 1:
                # 直接使用预处理好的相关性得分，无需重新计算
                loss = criterion(valid_pred.unsqueeze(0), valid_relevance.unsqueeze(0))
                batch_loss = batch_loss + loss if isinstance(batch_loss, torch.Tensor) else loss
        
        if batch_loss is not None:
            batch_loss = batch_loss / batch_size
            batch_loss.backward()
            if config.get('use_grad_clip', True):
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
                if writer:
                    writer.add_scalar('train/grad_norm', grad_norm, global_step=epoch*len(dataloader)+local_step)
            optimizer.step()
            
            total_loss += batch_loss.item()
            
            # 计算评估指标
            with torch.no_grad():
                metrics = calculate_ranking_metrics(masked_outputs, masked_targets, masks, k=5)
                for k, v in metrics.items():
                    if k not in total_metrics:
                        total_metrics[k] = 0
                    total_metrics[k] += v
            
            local_step += 1
            if writer:
                writer.add_scalar('train/loss', batch_loss.item(), global_step=epoch*len(dataloader)+local_step)
                for k, v in metrics.items():
                    writer.add_scalar(f'train/{k}', v, global_step=epoch*len(dataloader)+local_step)
    
    # 计算平均指标
    if local_step > 0:
        for k in total_metrics:
            total_metrics[k] /= local_step
    
    return total_loss / len(dataloader) if len(dataloader) > 0 else 0, total_metrics

def evaluate_ranking_model(model, dataloader, criterion, device, writer, epoch, split_name='eval'):
    model.eval()
    total_loss = 0
    total_metrics = {}
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"{split_name} Epoch {epoch+1}"):
            sequences = batch['sequences'].to(device)
            targets = batch['targets'].to(device)
            relevance = batch['relevance'].to(device)
            masks = batch['masks'].to(device)
            
            # 模型预测
            outputs = model(sequences)
            
            # 应用mask
            masked_outputs = outputs * masks + (1 - masks) * (-1e9)
            masked_targets = targets * masks
            masked_relevance = relevance * masks
            
            # 计算损失
            batch_loss = None
            batch_size = sequences.size(0)
            
            for i in range(batch_size):
                mask = masks[i]
                valid_indices = mask.nonzero().squeeze()
                
                if valid_indices.numel() == 0:
                    continue
                    
                if valid_indices.dim() == 0:
                    valid_indices = valid_indices.unsqueeze(0)
                
                valid_pred = masked_outputs[i][valid_indices]
                valid_relevance = masked_relevance[i][valid_indices]
                
                if len(valid_pred) > 1:
                    loss = criterion(valid_pred.unsqueeze(0), valid_relevance.unsqueeze(0))
                    batch_loss = batch_loss + loss if batch_loss is not None else loss
            
            if batch_loss is not None:
                batch_loss = batch_loss / batch_size
                total_loss += batch_loss.item()
            
            # 计算评估指标
            metrics = calculate_ranking_metrics(masked_outputs, masked_targets, masks, k=5)
            for k, v in metrics.items():
                if k not in total_metrics:
                    total_metrics[k] = 0
                total_metrics[k] += v
            
            num_batches += 1
    
    # 计算平均指标
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    for k in total_metrics:
        total_metrics[k] /= num_batches
    
    if writer:
        writer.add_scalar(f'{split_name}/loss', avg_loss, global_step=epoch)
        for k, v in total_metrics.items():
            writer.add_scalar(f'{split_name}/{k}', v, global_step=epoch)
    
    return avg_loss, total_metrics


def average_metric_dicts(metric_dicts):
    if not metric_dicts:
        return {}
    keys = set().union(*(metrics.keys() for metrics in metric_dicts))
    return {key: float(np.mean([metrics.get(key, 0.0) for metrics in metric_dicts])) for key in keys}


def compute_robust_validation_score(window_scores):
    if not window_scores:
        return 0.0
    scores = [float(score) for score in window_scores]
    return 0.7 * float(np.mean(scores)) + 0.3 * float(np.min(scores))


def split_dataset_into_windows(dataset, num_windows=3):
    if len(dataset) == 0:
        return []
    window_count = min(num_windows, len(dataset))
    indices = np.array_split(np.arange(len(dataset)), window_count)
    return [Subset(dataset, idx.tolist()) for idx in indices if len(idx) > 0]


def evaluate_validation_windows(model, val_loaders, criterion, device, writer, epoch):
    window_losses = []
    window_metrics = []

    for idx, loader in enumerate(val_loaders, start=1):
        split_name = f'val_w{idx}'
        loss, metrics = evaluate_ranking_model(model, loader, criterion, device, writer, epoch, split_name=split_name)
        window_losses.append(loss)
        window_metrics.append(metrics)

    window_final_scores = [metrics.get('final_score', 0.0) for metrics in window_metrics]
    summary = {
        'loss': float(np.mean(window_losses)) if window_losses else 0.0,
        'metrics': average_metric_dicts(window_metrics),
        'window_losses': window_losses,
        'window_metrics': window_metrics,
        'window_final_scores': window_final_scores,
        'mean_final_score': float(np.mean(window_final_scores)) if window_final_scores else 0.0,
        'min_final_score': float(np.min(window_final_scores)) if window_final_scores else 0.0,
        'robust_val_score': compute_robust_validation_score(window_final_scores),
    }

    if writer:
        writer.add_scalar('val_robust/loss', summary['loss'], global_step=epoch)
        writer.add_scalar('val_robust/mean_final_score', summary['mean_final_score'], global_step=epoch)
        writer.add_scalar('val_robust/min_final_score', summary['min_final_score'], global_step=epoch)
        writer.add_scalar('val_robust/robust_val_score', summary['robust_val_score'], global_step=epoch)
        for k, v in summary['metrics'].items():
            writer.add_scalar(f'val_robust/{k}', v, global_step=epoch)

    return summary


def zscore_np(values, eps=1e-8):
    values = np.asarray(values, dtype=np.float32)
    std = values.std()
    if values.size == 0 or std < eps:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - values.mean()) / (std + eps)).astype(np.float32)


def softmax_np(values, temperature=1.0):
    values = np.asarray(values, dtype=np.float64)
    temperature = max(float(temperature), 1e-6)
    values = values / temperature
    values = values - values.max()
    exp_values = np.exp(values)
    return exp_values / exp_values.sum()


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


def select_rerank_portfolio(
    candidate_indices,
    final_scores,
    output_top_k,
    weight_mode,
    softmax_temperature,
    min_weight,
    max_weight,
):
    output_top_k = max(1, min(int(output_top_k), len(candidate_indices), 5))
    order = np.argsort(final_scores)[::-1]
    if weight_mode == 'top1_full':
        selected_order = order[:1]
        weights = np.asarray([1.0], dtype=np.float64)
    elif weight_mode == 'equal':
        selected_order = order[:output_top_k]
        weights = np.full(len(selected_order), 1.0 / len(selected_order), dtype=np.float64)
    elif weight_mode == 'capped_softmax':
        selected_order = order[:output_top_k]
        weights = cap_weights(softmax_np(final_scores[selected_order], temperature=softmax_temperature), max_weight)
    elif weight_mode == 'floor_capped_softmax':
        selected_order = order[:output_top_k]
        weights = floor_cap_weights(
            softmax_np(final_scores[selected_order], temperature=softmax_temperature),
            min_weight,
            max_weight,
        )
    else:
        selected_order = order[:output_top_k]
        weights = softmax_np(final_scores[selected_order], temperature=softmax_temperature)
        min_weight = max(float(min_weight), 0.0)
        if min_weight > 0 and len(weights) > 1:
            keep = weights >= min_weight
            if not keep.any():
                keep[np.argmax(weights)] = True
            selected_order = selected_order[keep]
            weights = weights[keep]
            weights = weights / weights.sum()

    return candidate_indices[selected_order], weights


def float_list(values):
    return [float(value) for value in values]


def load_models_for_scoring(model_paths, input_dim, num_stocks, device):
    models = []
    for model_path in model_paths:
        model = StockTransformer(input_dim=input_dim, config=config, num_stocks=num_stocks)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        models.append(model)
    return models


def configured_rerank_extra_feature_names():
    names = set(config.get('rerank_extra_weights', {}).keys())
    override = config.get('rerank_acceptance_override', {})
    names.update(override.get('rerank_extra_weights', {}).keys())
    for extra_weights in config.get('rerank_search', {}).get('extra_weights', []):
        if isinstance(extra_weights, dict):
            names.update(extra_weights.keys())
    return sorted(names)


def collect_validation_rerank_records(
    val_window_datasets,
    features,
    model_paths,
    input_dim,
    num_stocks,
    device,
    normalize_features,
):
    rerank_features = ['return_5', 'volume_change', 'volatility_10'] + configured_rerank_extra_feature_names()
    feature_indices = {name: features.index(name) for name in rerank_features if name in features}
    models = load_models_for_scoring(model_paths, input_dim, num_stocks, device)
    window_records = []

    with torch.no_grad():
        for window_dataset in val_window_datasets:
            records = []
            for item in window_dataset:
                seq = item['sequences'].float()
                if normalize_features:
                    seq = cross_sectional_normalize_torch(seq)
                x = seq.unsqueeze(0).to(device)
                scores = np.mean(
                    [model(x).squeeze(0).detach().cpu().numpy() for model in models],
                    axis=0,
                )
                raw_seq = item['sequences'].numpy()
                record = {
                    'scores': scores,
                    'targets': item['targets'].numpy(),
                }
                for feature_name, feature_index in feature_indices.items():
                    record[feature_name] = raw_seq[:, -1, feature_index]
                records.append(record)
            window_records.append(records)

    return window_records


def score_rerank_records(
    window_records,
    pool_size,
    return_5_weight,
    volume_change_weight,
    volatility_10_weight,
    output_top_k=5,
    weight_mode='softmax',
    softmax_temperature=1.0,
    min_weight=0.0,
    max_weight=1.0,
    extra_weights=None,
):
    window_scores = []
    extra_weights = extra_weights or {}

    for records in window_records:
        daily_scores = []
        for record in records:
            scores = record['scores']
            targets = record['targets']
            if len(scores) < 1:
                continue
            current_pool = min(pool_size, len(scores))
            candidate_indices = np.argsort(scores)[::-1][:current_pool]
            final_scores = (
                zscore_np(scores[candidate_indices])
                + return_5_weight * zscore_np(record['return_5'][candidate_indices])
                + volume_change_weight * zscore_np(record['volume_change'][candidate_indices])
                + volatility_10_weight * zscore_np(record['volatility_10'][candidate_indices])
            )
            for feature_name, feature_weight in extra_weights.items():
                if feature_name in record:
                    final_scores = final_scores + float(feature_weight) * zscore_np(record[feature_name][candidate_indices])
            top_indices, weights = select_rerank_portfolio(
                candidate_indices,
                final_scores,
                output_top_k,
                weight_mode,
                softmax_temperature,
                min_weight,
                max_weight,
            )
            daily_scores.append(float(np.sum(targets[top_indices] * weights)))
        window_scores.append(float(np.mean(daily_scores)) if daily_scores else 0.0)

    return {
        'window_scores': window_scores,
        'mean_window_score': float(np.mean(window_scores)) if window_scores else 0.0,
        'min_window_score': float(np.min(window_scores)) if window_scores else 0.0,
        'robust_val_score': compute_robust_validation_score(window_scores),
    }


def search_best_rerank_config(
    val_window_datasets,
    features,
    best_model_path,
    second_model_path,
    input_dim,
    num_stocks,
    device,
    output_dir,
):
    search_space = config.get('rerank_search', {})
    if not config.get('run_train_rerank_search', False):
        search_space = {
            'return_5': [config.get('rerank_return_5_weight', 0.3)],
            'volume_change': [config.get('rerank_volume_change_weight', 0.2)],
            'volatility_10': [config.get('rerank_volatility_10_weight', 0.2)],
            'pool_size': [config.get('rerank_pool_size', 20)],
            'inference_cross_sectional_norm': [config.get('inference_cross_sectional_norm', False)],
            'ensemble_second_best': [config.get('ensemble_second_best', False)],
            'output_top_k': [config.get('output_top_k', 5)],
            'weight_mode': [config.get('weight_mode', 'softmax')],
            'softmax_temperature': [config.get('softmax_temperature', 1.0)],
            'min_weight': [config.get('min_weight', 0.0)],
            'max_weight': [config.get('max_weight', 1.0)],
        }
    model_mode_options = []
    for use_ensemble in search_space.get('ensemble_second_best', [config.get('ensemble_second_best', False)]):
        if use_ensemble and not os.path.exists(second_model_path):
            continue
        paths = [best_model_path, second_model_path] if use_ensemble else [best_model_path]
        model_mode_options.append((use_ensemble, paths))

    if not model_mode_options or not val_window_datasets:
        return None

    best_result = None
    record_cache = {}

    for normalize_features in search_space.get(
        'inference_cross_sectional_norm',
        [config.get('inference_cross_sectional_norm', False)],
    ):
        for use_ensemble, model_paths in model_mode_options:
            cache_key = (normalize_features, use_ensemble)
            record_cache[cache_key] = collect_validation_rerank_records(
                val_window_datasets,
                features,
                model_paths,
                input_dim,
                num_stocks,
                device,
                normalize_features,
            )

            for pool_size in search_space.get('pool_size', [config.get('rerank_pool_size', 20)]):
                for return_5_weight in search_space.get('return_5', [config.get('rerank_return_5_weight', 0.3)]):
                    for volume_change_weight in search_space.get(
                        'volume_change',
                        [config.get('rerank_volume_change_weight', 0.2)],
                    ):
                        for volatility_10_weight in search_space.get(
                            'volatility_10',
                            [config.get('rerank_volatility_10_weight', 0.2)],
                        ):
                            for output_top_k in search_space.get('output_top_k', [config.get('output_top_k', 5)]):
                                for weight_mode in search_space.get('weight_mode', [config.get('weight_mode', 'softmax')]):
                                    for softmax_temperature in search_space.get(
                                        'softmax_temperature',
                                        [config.get('softmax_temperature', 1.0)],
                                    ):
                                        for min_weight in search_space.get('min_weight', [config.get('min_weight', 0.0)]):
                                            for max_weight in search_space.get('max_weight', [config.get('max_weight', 1.0)]):
                                                try:
                                                    scored = score_rerank_records(
                                                        record_cache[cache_key],
                                                        pool_size,
                                                        return_5_weight,
                                                        volume_change_weight,
                                                        volatility_10_weight,
                                                        output_top_k,
                                                        weight_mode,
                                                        softmax_temperature,
                                                        min_weight,
                                                        max_weight,
                                                        config.get('rerank_extra_weights', {}),
                                                    )
                                                except ValueError:
                                                    continue
                                                candidate = {
                                                    **scored,
                                                    'pool_size': int(pool_size),
                                                    'inference_cross_sectional_norm': bool(normalize_features),
                                                    'ensemble_second_best': bool(use_ensemble),
                                                    'rerank_return_5_weight': float(return_5_weight),
                                                    'rerank_volume_change_weight': float(volume_change_weight),
                                                    'rerank_volatility_10_weight': float(volatility_10_weight),
                                                    'output_top_k': int(output_top_k),
                                                    'weight_mode': str(weight_mode),
                                                    'softmax_temperature': float(softmax_temperature),
                                                    'min_weight': float(min_weight),
                                                    'max_weight': float(max_weight),
                                                    'rerank_extra_weights': {
                                                        key: float(value)
                                                        for key, value in config.get('rerank_extra_weights', {}).items()
                                                    },
                                                    'metric': 'mean validation weighted return with output controls',
                                                }
                                                if best_result is None or candidate['robust_val_score'] > best_result['robust_val_score']:
                                                    best_result = candidate

    override = config.get('rerank_acceptance_override', {})
    if override.get('enabled'):
        normalize_features = bool(override.get('inference_cross_sectional_norm', False))
        use_ensemble = bool(override.get('ensemble_second_best', False))
        override_extra_weights = {
            key: float(value)
            for key, value in override.get('rerank_extra_weights', config.get('rerank_extra_weights', {})).items()
        }
        if not use_ensemble or os.path.exists(second_model_path):
            cache_key = (normalize_features, use_ensemble)
            if cache_key not in record_cache:
                model_paths = [best_model_path, second_model_path] if use_ensemble else [best_model_path]
                record_cache[cache_key] = collect_validation_rerank_records(
                    val_window_datasets,
                    features,
                    model_paths,
                    input_dim,
                    num_stocks,
                    device,
                    normalize_features,
                )
            scored = score_rerank_records(
                record_cache[cache_key],
                int(override.get('pool_size', config.get('rerank_pool_size', 20))),
                float(override.get('rerank_return_5_weight', config.get('rerank_return_5_weight', 0.3))),
                float(override.get('rerank_volume_change_weight', config.get('rerank_volume_change_weight', 0.2))),
                float(override.get('rerank_volatility_10_weight', config.get('rerank_volatility_10_weight', 0.2))),
                int(override.get('output_top_k', config.get('output_top_k', 5))),
                str(override.get('weight_mode', config.get('weight_mode', 'softmax'))),
                float(override.get('softmax_temperature', config.get('softmax_temperature', 1.0))),
                float(override.get('min_weight', config.get('min_weight', 0.0))),
                float(override.get('max_weight', config.get('max_weight', 1.0))),
                override_extra_weights,
            )
            best_result = {
                **scored,
                'pool_size': int(override.get('pool_size', config.get('rerank_pool_size', 20))),
                'inference_cross_sectional_norm': normalize_features,
                'ensemble_second_best': use_ensemble,
                'rerank_return_5_weight': float(
                    override.get('rerank_return_5_weight', config.get('rerank_return_5_weight', 0.3))
                ),
                'rerank_volume_change_weight': float(
                    override.get('rerank_volume_change_weight', config.get('rerank_volume_change_weight', 0.2))
                ),
                'rerank_volatility_10_weight': float(
                    override.get('rerank_volatility_10_weight', config.get('rerank_volatility_10_weight', 0.2))
                ),
                'output_top_k': int(override.get('output_top_k', config.get('output_top_k', 5))),
                'weight_mode': str(override.get('weight_mode', config.get('weight_mode', 'softmax'))),
                'softmax_temperature': float(
                    override.get('softmax_temperature', config.get('softmax_temperature', 1.0))
                ),
                'min_weight': float(override.get('min_weight', config.get('min_weight', 0.0))),
                'max_weight': float(override.get('max_weight', config.get('max_weight', 1.0))),
                'rerank_extra_weights': override_extra_weights,
                'metric': 'acceptance-gated validation-grid candidate',
            }

    if best_result is not None:
        rerank_config_path = os.path.join(output_dir, 'rerank_config.json')
        with open(rerank_config_path, 'w', encoding='utf-8') as f:
            json.dump(best_result, f, indent=4, ensure_ascii=False)
        print(f"保存最优 rerank 配置: {rerank_config_path}")
        print(f"Rerank robust score: {best_result['robust_val_score']:.6f}")
        print(f"Rerank window scores: {best_result['window_scores']}")

    return best_result


def predict_top_stocks(model, data, features, sequence_length, scaler, stockid2idx, device, top_k=5):
    """
    预测某一天涨幅前top_k的股票
    """
    model.eval()
    
    # 获取最后一天的数据作为预测基础
    latest_date = data['日期'].max()
    
    # 准备预测数据
    day_sequences = []
    day_stock_codes = []
    day_stock_indices = []
    
    for stock_code in data['股票代码'].unique():
        # 获取该股票历史sequence_length天的数据
        stock_history = data[
            (data['股票代码'] == stock_code) & 
            (data['日期'] <= latest_date)
        ].sort_values('日期').tail(sequence_length)
        
        if len(stock_history) == sequence_length:
            seq = stock_history[features].values
            day_sequences.append(seq)
            day_stock_codes.append(stock_code)
            day_stock_indices.append(stockid2idx[stock_code])
    
    if len(day_sequences) == 0:
        return []
    
    # 转换为tensor
    sequences = torch.FloatTensor(np.array(day_sequences)).unsqueeze(0).to(device)  # [1, num_stocks, seq_len, features]
    
    with torch.no_grad():
        # 模型预测
        outputs = model(sequences)  # [1, num_stocks]
        scores = outputs.squeeze().cpu().numpy()  # [num_stocks]
        
        # 获取排名前top_k的股票
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        top_stocks = []
        for idx in top_indices:
            top_stocks.append({
                'stock_code': day_stock_codes[idx],
                'predicted_score': scores[idx],
                'rank': len(top_stocks) + 1
            })
    
    return top_stocks

def save_predictions(top_stocks, output_path):
    """保存预测结果"""
    results = []
    for stock in top_stocks:
        results.append({
            '排名': stock['rank'],
            '股票代码': stock['stock_code'],
            '预测分数': stock['predicted_score']
        })
    
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"预测结果已保存到: {output_path}")


def split_train_val_by_last_month(df, sequence_length):
    """按最后一个月做验证集划分，并为验证集补充序列上下文。"""
    df = df.copy()
    df['日期'] = pd.to_datetime(df['日期'])
    df = df.sort_values(['日期', '股票代码']).reset_index(drop=True)

    last_date = df['日期'].max()
    val_start = (last_date - pd.DateOffset(months=2)).normalize()

    # 验证集需要保留前 sequence_length-1 个交易日作为序列上下文，
    # 这样第一个验证样本的窗口结束日就可以落在 val_start。
    val_context_start = val_start - pd.tseries.offsets.BDay(sequence_length - 1)

    train_df = df[df['日期'] < val_start].copy()
    val_df = df[df['日期'] >= val_context_start].copy()

    if train_df['日期'].nunique() < sequence_length + 5:
        print("短窗口训练模式: 默认训练/验证切分不足以构造60日序列，改用全量训练文件进行训练和内部验证。")
        train_df = df.copy()
        val_df = df.copy()
        val_start = df['日期'].min()

    print(f"全量数据范围: {df['日期'].min().date()} 到 {last_date.date()}")
    print(f"训练集范围: {train_df['日期'].min().date()} 到 {train_df['日期'].max().date()}")
    print(f"验证集目标范围(最后一个月): {val_start.date()} 到 {last_date.date()}")
    print(f"验证集实际取数范围(含序列上下文): {val_df['日期'].min().date()} 到 {val_df['日期'].max().date()}")

    # 恢复为字符串，保持与原流程一致
    train_df['日期'] = train_df['日期'].dt.strftime('%Y-%m-%d')
    val_df['日期'] = val_df['日期'].dt.strftime('%Y-%m-%d')

    return train_df, val_df, val_start

# 主程序
def main():
    set_seed(config.get('seed', 42))
    output_dir = config['output_dir']
    os.makedirs(output_dir,exist_ok=True)
    # 保存在output_dir中保存当前的配置文件，以便复现
    data_path = config['data_path']
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    is_train = True
    writer = SummaryWriter(log_dir=os.path.join(output_dir, 'log')) if is_train else None
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    # 1. 数据加载
    data_file = os.path.join(data_path, config.get('train_filename', 'train.csv'))
    full_df = pd.read_csv(data_file)
    train_df, val_df, val_start = split_train_val_by_last_month(full_df, config['sequence_length'])
    
    # 获取所有股票ID，建立映射
    all_stock_ids = full_df['股票代码'].unique()
    stockid2idx = {sid: idx for idx, sid in enumerate(sorted(all_stock_ids))}
    num_stocks = len(stockid2idx)
    
    # 2. 特征工程与预处理
    train_data, features = preprocess_data(train_df, is_train=True, stockid2idx=stockid2idx)
    val_data, _ = preprocess_val_data(val_df, stockid2idx=stockid2idx)
    
    # 3. 标准化
    scaler = StandardScaler()

    train_data[features] = train_data[features].replace([np.inf, -np.inf], np.nan)
    val_data[features] = val_data[features].replace([np.inf, -np.inf], np.nan)
    # 丢弃nan数据
    train_data = train_data.dropna(subset=features)
    val_data = val_data.dropna(subset=features)
    # 然后再缩放
    train_data[features] = scaler.fit_transform(train_data[features])
    val_data[features] = scaler.transform(val_data[features])
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))

    
    # 4. 创建排序数据集
    train_sequences, train_targets, train_relevance, train_stock_indices = create_ranking_dataset_vectorized(
        train_data,
        features,
        config['sequence_length'],
        ranking_data_path=config.get('train_ranking_data_path')
    )
    val_sequences, val_targets, val_relevance, val_stock_indices = create_ranking_dataset_vectorized(
        val_data,
        features,
        config['sequence_length'],
        ranking_data_path=config.get('val_ranking_data_path'),
        min_window_end_date=val_start.strftime('%Y-%m-%d')
    )

    print(f"训练集样本数: {len(train_sequences)}")
    print(f"验证集样本数: {len(val_sequences)}")
    
    # 5. 创建排序数据集和数据加载器
    train_dataset = RankingDataset(train_sequences, train_targets, train_relevance, train_stock_indices)
    val_dataset = RankingDataset(val_sequences, val_targets, val_relevance, val_stock_indices)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=0,  # 减少worker数量避免内存问题
        pin_memory=False
    )
    
    val_window_datasets = split_dataset_into_windows(val_dataset, num_windows=3)
    val_loaders = [
        DataLoader(
            window_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
            pin_memory=False
        )
        for window_dataset in val_window_datasets
    ]
    print(f"验证窗口数: {len(val_window_datasets)}")
    for idx, window_dataset in enumerate(val_window_datasets, start=1):
        print(f"验证窗口 {idx} 样本数: {len(window_dataset)}")
    
    # 6. 模型初始化
    model = StockTransformer(input_dim=len(features), config=config, num_stocks=num_stocks)
    model.to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # 7. 损失函数和优化器
    criterion = WeightedRankingLoss(
        k=5,
        temperature=1.0,
        weight_factor=config['top5_weight'],
        pairwise_weight=config['pairwise_weight'],
        base_weight=config.get('base_weight', 1.0),
        margin=config.get('boundary_margin', 0.2),
        global_ce_weight=config.get('global_ce_weight', 0.05),
        score_l2_weight=config.get('score_l2_weight', 1e-4),
        bottom_k=config.get('bottom_k', 20)
    )  # 使用加权排序损失
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.2, total_iters=config['num_epochs'])
    
    # 8. 排序模型训练
    if is_train:
        best_score = -float('inf')
        second_best_score = -float('inf')
        best_epoch = -1
        second_best_epoch = -1
        best_summary = None
        second_best_summary = None
        epochs_without_improvement = 0
        patience = config.get('early_stop_patience', 5)
        best_model_path = os.path.join(output_dir, 'best_model.pth')
        second_model_path = os.path.join(output_dir, 'second_best_model.pth')
        rerank_config_path = os.path.join(output_dir, 'rerank_config.json')
        if os.path.exists(second_model_path):
            os.remove(second_model_path)
        if os.path.exists(rerank_config_path):
            os.remove(rerank_config_path)
        
        for epoch in range(config['num_epochs']):
            print(f"\n=== Epoch {epoch+1}/{config['num_epochs']} ===")
            
            # 训练
            train_loss, train_metrics = train_ranking_model(
                model, train_loader, criterion, optimizer, device, epoch, writer
            )
            
            print(f"Train Loss: {train_loss:.4f}")
            for k, v in train_metrics.items():
                print(f"Train {k}: {v:.4f}")
            
            # 验证
            val_summary = evaluate_validation_windows(
                model, val_loaders, criterion, device, writer, epoch
            )
            for idx, (loss, metrics) in enumerate(
                zip(val_summary['window_losses'], val_summary['window_metrics']), start=1
            ):
                print(f"Val W{idx} Loss: {loss:.4f}")
                for k, v in metrics.items():
                    print(f"Val W{idx} {k}: {v:.4f}")
            print(f"Val Mean final_score: {val_summary['mean_final_score']:.4f}")
            print(f"Val Min final_score: {val_summary['min_final_score']:.4f}")
            print(f"Val Robust score: {val_summary['robust_val_score']:.4f}")
            
            # 学习率调度
            scheduler.step()
            if writer:
                writer.add_scalar('train/learning_rate', scheduler.get_last_lr()[0], global_step=epoch)
            

            # 保存最佳模型（基于 robust validation score）
            current_final_score = val_summary['robust_val_score']
            if current_final_score > best_score:
                if best_epoch != -1 and os.path.exists(best_model_path):
                    shutil.copyfile(best_model_path, second_model_path)
                    second_best_score = best_score
                    second_best_epoch = best_epoch
                    second_best_summary = best_summary
                best_score = current_final_score
                best_epoch = epoch + 1
                best_summary = val_summary
                torch.save(model.state_dict(), best_model_path)
                epochs_without_improvement = 0
                print(f"保存最佳模型 - robust score: {best_score:.4f}")
            elif current_final_score > second_best_score:
                second_best_score = current_final_score
                second_best_epoch = epoch + 1
                second_best_summary = val_summary
                torch.save(model.state_dict(), second_model_path)
                epochs_without_improvement += 1
                print(f"保存第二佳模型 - robust score: {second_best_score:.4f}")
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                print(f"Early stopping: {patience} epochs without validation improvement.")
                break
        print(f"\n训练完成！最佳 epoch: {best_epoch}, 最佳 robust score: {best_score:.4f}")
        with open(os.path.join(output_dir, 'final_score.txt'), 'w') as f:
            f.write(f"Best epoch: {best_epoch}\n")
            f.write(f"Best robust_val_score: {best_score:.6f}\n")
            if best_summary:
                f.write(f"Best window_final_scores: {float_list(best_summary['window_final_scores'])}\n")
                f.write(f"Best mean_final_score: {best_summary['mean_final_score']:.6f}\n")
                f.write(f"Best min_final_score: {best_summary['min_final_score']:.6f}\n")
            f.write(f"Second best epoch: {second_best_epoch}\n")
            f.write(f"Second best robust_val_score: {second_best_score:.6f}\n")
            if second_best_summary:
                f.write(f"Second best window_final_scores: {float_list(second_best_summary['window_final_scores'])}\n")
                f.write(f"Second best mean_final_score: {second_best_summary['mean_final_score']:.6f}\n")
                f.write(f"Second best min_final_score: {second_best_summary['min_final_score']:.6f}\n")

        rerank_result = None
        if os.path.exists(best_model_path):
            rerank_result = search_best_rerank_config(
                val_window_datasets,
                features,
                best_model_path,
                second_model_path,
                len(features),
                num_stocks,
                device,
                output_dir,
            )

        if rerank_result:
            with open(os.path.join(output_dir, 'final_score.txt'), 'a', encoding='utf-8') as f:
                f.write(f"Rerank robust_val_score: {rerank_result['robust_val_score']:.6f}\n")
                f.write(f"Rerank window_scores: {rerank_result['window_scores']}\n")
                f.write(f"Rerank pool_size: {rerank_result['pool_size']}\n")
                f.write(f"Rerank inference_cross_sectional_norm: {rerank_result['inference_cross_sectional_norm']}\n")
                f.write(f"Rerank ensemble_second_best: {rerank_result['ensemble_second_best']}\n")
                f.write(f"Rerank return_5 weight: {rerank_result['rerank_return_5_weight']}\n")
                f.write(f"Rerank volume_change weight: {rerank_result['rerank_volume_change_weight']}\n")
                f.write(f"Rerank volatility_10 weight: {rerank_result['rerank_volatility_10_weight']}\n")
                f.write(f"Rerank extra weights: {rerank_result.get('rerank_extra_weights', {})}\n")
                f.write(f"Rerank output_top_k: {rerank_result['output_top_k']}\n")
                f.write(f"Rerank weight_mode: {rerank_result['weight_mode']}\n")
                f.write(f"Rerank softmax_temperature: {rerank_result['softmax_temperature']}\n")
                f.write(f"Rerank min_weight: {rerank_result['min_weight']}\n")
                f.write(f"Rerank max_weight: {rerank_result.get('max_weight', 1.0)}\n")

        if writer:
            writer.close()

        return best_score

if __name__ == "__main__":
    # 多进程保护
    mp.set_start_method('spawn', force=True)
    best_score = main()
    print(f"\n########## 训练完成！最佳 robust score: {best_score:.4f} ##########")
