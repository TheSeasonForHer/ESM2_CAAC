#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基线模型对比实验 (baseline_comparison_v2.py)
对比模型:
  传统基线: ESM2-MLP, ESM2-Context-MLP, ESM2-RF, ESM2-Context-RF
  SOTA基线: ESM2-FineTuned, ESM2-Attention
  本文方法: CAAC
修正内容:
  1. MLP: Adam优化器, 更深的网络(512-256-128), Dropout降至0.3, ReduceLROnPlateau调度
  2. RF: n_estimators=500, max_depth=None, class_weight='balanced'
  3. 新增ESM2-FineTuned: 可训练深度分类头, LayerNorm稳定训练
  4. 新增ESM2-Attention: Multi-head Attention聚合特征, 模拟DeepFRI思想
"""

import os
import sys
import json
import argparse
import logging
import random
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

warnings.filterwarnings('ignore')


# ==================== 配置 ====================

@dataclass
class BaselineConfig:
    features_dir: str = "/home/zjw/zjwdata/3_deep_learning/training_data/curated_v2/features"
    output_dir: str = "/home/zjw/zjwdata/3_deep_learning/training_data/curated_v2/baseline_results_v2"
    caac_report_path: str = "/home/zjw/zjwdata/3_deep_learning/training_data/curated_v2/gnn_model/training_report.json"

    # 所有待对比模型
    models: Tuple[str, ...] = (
        'esm2_mlp', 'esm2_context_mlp',
        'esm2_rf', 'esm2_context_rf',
        'esm2_finetuned',  # 新增: ESM2 + 可训练深度分类头
        'esm2_attention',  # 新增: ESM2 + Attention聚合
    )

    n_folds: int = 5
    seed: int = 42
    device: str = "cuda:0"

    # ========== MLP修正参数 ==========
    mlp_hidden_dims: Tuple[int, ...] = (512, 256, 128)  # 加深: 3层
    mlp_dropout: float = 0.3  # 降低dropout
    mlp_epochs: int = 100
    mlp_batch_size: int = 1024
    mlp_lr: float = 1e-3  # Adam默认lr
    mlp_weight_decay: float = 5e-4
    mlp_early_stop: int = 15
    mlp_optimizer: str = "adam"  # Adam替代SGD

    # ========== RF修正参数 ==========
    rf_n_estimators: int = 500  # 增加树数量
    rf_max_depth: Optional[int] = None  # 不限制深度
    rf_min_samples_split: int = 5  # 降低
    rf_min_samples_leaf: int = 2  # 降低
    rf_class_weight: str = "balanced"  # 处理类别不平衡
    rf_n_jobs: int = -1

    # ========== FineTuned/Attention共享参数 ==========
    ft_hidden_dim: int = 512
    ft_dropout: float = 0.3
    ft_epochs: int = 100
    ft_batch_size: int = 1024
    ft_lr: float = 1e-3
    ft_weight_decay: float = 5e-4
    ft_early_stop: int = 15

    log_level: str = "INFO"

    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(f"{self.output_dir}/fold_results", exist_ok=True)


# ==================== 数据集 ====================

class FeatureDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# ==================== MLP模型 (修正版) ====================

class ESM2MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Tuple[int, ...],
                 num_classes: int, dropout: float = 0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))  # 添加LayerNorm
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_dim, num_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        return logits


# ==================== ESM2-FineTuned (新增SOTA基线) ====================

class ESM2FineTuned(nn.Module):
    """
    ESM2-650M embedding + 可训练深度分类头
    参考ProtTrans/ESM-based function prediction的标准做法
    """

    def __init__(self, input_dim: int = 1280, num_classes: int = 3,
                 hidden_dim: int = 512, dropout: float = 0.3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.67),  # 逐层降低dropout
            nn.Linear(hidden_dim // 2, num_classes)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.classifier(x)


# ==================== ESM2-Attention (新增SOTA基线) ====================

class ESM2AttentionClassifier(nn.Module):
    """
    ESM2 embedding + Multi-head Self-Attention聚合
    模拟DeepFRI的attention思想(简化版,无需结构信息)
    """

    def __init__(self, input_dim: int = 1280, num_classes: int = 3,
                 num_heads: int = 8, hidden_dim: int = 512, dropout: float = 0.3):
        super().__init__()
        # 将1D特征视为seq_len=1的序列, 用attention学习特征交互
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.67),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x: [batch, 1280] -> [batch, 1, hidden_dim]
        x = self.proj(x).unsqueeze(1)  # [batch, 1, hidden_dim]
        attn_out, _ = self.attention(x, x, x)  # self-attention
        x = self.norm1(x + attn_out)  # residual + norm
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)  # residual + norm
        x = x.squeeze(1)  # [batch, hidden_dim]
        return self.classifier(x)


# ==================== 统一深度学习训练器 ====================

class DeepLearningTrainer:
    """
    统一训练器: 支持MLP, FineTuned, Attention
    """

    def __init__(self, config: BaselineConfig, logger: logging.Logger, model_name: str):
        self.config = config
        self.logger = logger
        self.model_name = model_name
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    def create_optimizer(self, model: nn.Module, lr: float, weight_decay: float):
        if self.config.mlp_optimizer == "adam":
            return torch.optim.Adam(
                model.parameters(), lr=lr, weight_decay=weight_decay
            )
        else:
            return torch.optim.SGD(
                model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay
            )

    def train_epoch(self, model, loader, optimizer, criterion, scheduler=None, scaler=None):
        model.train()
        total_loss = 0
        all_preds, all_labels = [], []
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(self.device, non_blocking=True)
            batch_y = batch_y.to(self.device, non_blocking=True)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch_y.cpu().numpy())

        metrics = {
            'loss': total_loss / len(loader),
            'accuracy': accuracy_score(all_labels, all_preds),
            'f1_macro': f1_score(all_labels, all_preds, average='macro', zero_division=0),
        }
        for cls in range(3):
            metrics[f'f1_class_{cls}'] = f1_score(
                all_labels, all_preds, labels=[cls], average='macro', zero_division=0
            )
        return metrics

    def evaluate(self, model, loader, criterion):
        model.eval()
        total_loss = 0
        all_preds, all_labels, all_probs = [], [], []
        with torch.no_grad():
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device, non_blocking=True)
                batch_y = batch_y.to(self.device, non_blocking=True)
                with torch.cuda.amp.autocast():
                    logits = model(batch_x)
                loss = criterion(logits, batch_y)
                total_loss += loss.item()
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)
                all_probs.append(probs)
                all_preds.extend(preds)
                all_labels.extend(batch_y.cpu().numpy())

        metrics = {
            'loss': total_loss / len(loader),
            'accuracy': accuracy_score(all_labels, all_preds),
            'f1_macro': f1_score(all_labels, all_preds, average='macro', zero_division=0),
            'f1_weighted': f1_score(all_labels, all_preds, average='weighted', zero_division=0),
        }
        for cls in range(3):
            metrics[f'precision_class_{cls}'] = precision_score(
                all_labels, all_preds, labels=[cls], average='macro', zero_division=0
            )
            metrics[f'recall_class_{cls}'] = recall_score(
                all_labels, all_preds, labels=[cls], average='macro', zero_division=0
            )
            metrics[f'f1_class_{cls}'] = f1_score(
                all_labels, all_preds, labels=[cls], average='macro', zero_division=0
            )
        cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2])
        return metrics, cm, np.array(all_labels), np.array(all_preds), np.vstack(all_probs)

    def run_fold(self, fold_idx: int, train_dataset, val_dataset, test_dataset,
                 model: nn.Module, epochs: int, batch_size: int, lr: float,
                 weight_decay: float, early_stop: int) -> Dict:
        self.logger.info(f"\n{'=' * 60}")
        self.logger.info(f"{self.model_name} - Fold {fold_idx}/{self.config.n_folds}")
        self.logger.info(f"{'=' * 60}")

        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True, num_workers=8, pin_memory=True,
                                  persistent_workers=True, prefetch_factor=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size,
                                shuffle=True, num_workers=8, pin_memory=True,
                                persistent_workers=True, prefetch_factor=4)
        test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                 shuffle=True, num_workers=8, pin_memory=True,
                                 persistent_workers=True, prefetch_factor=4)

        model = model.to(self.device)
        optimizer = self.create_optimizer(model, lr, weight_decay)
        criterion = nn.CrossEntropyLoss()
        scaler = torch.cuda.amp.GradScaler()  # 新增
        amp_enabled = True  # 新增
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5
        )

        best_val_f1 = 0
        patience_counter = 0
        best_state = None

        for epoch in range(1, epochs + 1):
            train_metrics = self.train_epoch(model, train_loader, optimizer, criterion, scaler=scaler)
            val_metrics, _, _, _, _ = self.evaluate(model, val_loader, criterion)

            # 学习率调度
            scheduler.step(val_metrics['f1_macro'])

            if val_metrics['f1_macro'] > best_val_f1:
                best_val_f1 = val_metrics['f1_macro']
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= early_stop:
                    self.logger.info(f"  早停于epoch {epoch}")
                    break

            if epoch % 20 == 0:
                self.logger.info(
                    f"  Epoch {epoch:3d} | "
                    f"Train F1: {train_metrics['f1_macro']:.4f} | "
                    f"Val F1: {val_metrics['f1_macro']:.4f}"
                )

        model.load_state_dict(best_state)
        test_metrics, test_cm, test_labels, test_preds, test_probs = self.evaluate(
            model, test_loader, criterion
        )

        self.logger.info(f"  Test F1-macro: {test_metrics['f1_macro']:.4f}")
        self.logger.info(f"  Test Accuracy: {test_metrics['accuracy']:.4f}")

        return {
            'fold': fold_idx,
            'best_val_f1': float(best_val_f1),
            'test_metrics': {k: float(v) for k, v in test_metrics.items()},
            'test_cm': test_cm.tolist(),
        }


# ==================== RF训练器 (修正版) ====================

class RFTrainer:
    def __init__(self, config: BaselineConfig, logger: logging.Logger, model_name: str):
        self.config = config
        self.logger = logger
        self.model_name = model_name

    def run_fold(self, fold_idx: int, train_x, train_y, test_x, test_y) -> Dict:
        self.logger.info(f"\n{'=' * 60}")
        self.logger.info(f"{self.model_name} - Fold {fold_idx}/{self.config.n_folds}")
        self.logger.info(f"{'=' * 60}")

        rf = RandomForestClassifier(
            n_estimators=self.config.rf_n_estimators,
            max_depth=self.config.rf_max_depth,
            min_samples_split=self.config.rf_min_samples_split,
            min_samples_leaf=self.config.rf_min_samples_leaf,
            max_features='sqrt',
            class_weight=self.config.rf_class_weight,  # 关键修正: 处理不平衡
            n_jobs=self.config.rf_n_jobs,
            random_state=self.config.seed + fold_idx,
        )

        self.logger.info(f"  训练RF: {train_x.shape[0]} samples, "
                         f"estimators={self.config.rf_n_estimators}, "
                         f"class_weight={self.config.rf_class_weight}")
        rf.fit(train_x, train_y)

        test_preds = rf.predict(test_x)
        test_probs = rf.predict_proba(test_x)

        metrics = {
            'accuracy': accuracy_score(test_y, test_preds),
            'f1_macro': f1_score(test_y, test_preds, average='macro', zero_division=0),
            'f1_weighted': f1_score(test_y, test_preds, average='weighted', zero_division=0),
        }
        for cls in range(3):
            metrics[f'precision_class_{cls}'] = precision_score(
                test_y, test_preds, labels=[cls], average='macro', zero_division=0
            )
            metrics[f'recall_class_{cls}'] = recall_score(
                test_y, test_preds, labels=[cls], average='macro', zero_division=0
            )
            metrics[f'f1_class_{cls}'] = f1_score(
                test_y, test_preds, labels=[cls], average='macro', zero_division=0
            )

        cm = confusion_matrix(test_y, test_preds, labels=[0, 1, 2])

        self.logger.info(f"  Test F1-macro: {metrics['f1_macro']:.4f}")
        self.logger.info(f"  Test Accuracy: {metrics['accuracy']:.4f}")

        return {
            'fold': fold_idx,
            'test_metrics': {k: float(v) for k, v in metrics.items()},
            'test_cm': cm.tolist(),
        }


# ==================== CAAC结果加载器 (不变) ====================

class CAACResultLoader:
    def __init__(self, report_path: str, logger: logging.Logger):
        self.report_path = report_path
        self.logger = logger

    def load(self) -> Dict:
        self.logger.info(f"\n{'=' * 60}")
        self.logger.info("加载CAAC已有结果")
        self.logger.info(f"{'=' * 60}")

        if not os.path.exists(self.report_path):
            raise FileNotFoundError(f"CAAC报告不存在: {self.report_path}")

        with open(self.report_path, 'r') as f:
            report = json.load(f)

        fold_results = report.get('fold_results', [])
        if not fold_results:
            raise ValueError("CAAC报告中无fold_results")

        self.logger.info(f"  加载 {len(fold_results)} 折结果")
        unified_results = []
        for fold in fold_results:
            unified = {
                'fold': fold['fold'],
                'best_val_f1': fold.get('best_val_f1', 0),
                'test_metrics': fold.get('test_metrics', {}),
                'test_cm': fold.get('test_cm', []),
            }
            unified_results.append(unified)

        test_f1s = [r['test_metrics'].get('f1_macro', 0) for r in unified_results]
        test_accs = [r['test_metrics'].get('accuracy', 0) for r in unified_results]

        summary = {
            'model': 'caac',
            'test_f1_macro_mean': float(np.mean(test_f1s)),
            'test_f1_macro_std': float(np.std(test_f1s)),
            'test_accuracy_mean': float(np.mean(test_accs)),
            'test_accuracy_std': float(np.std(test_accs)),
            'fold_results': unified_results,
        }

        for cls in range(3):
            cls_f1s = [r['test_metrics'].get(f'f1_class_{cls}', 0) for r in unified_results]
            summary[f'class_{cls}_f1_mean'] = float(np.mean(cls_f1s))
            summary[f'class_{cls}_f1_std'] = float(np.std(cls_f1s))

        self.logger.info(
            f"  CAAC F1-macro: {summary['test_f1_macro_mean']:.4f} ± {summary['test_f1_macro_std']:.4f}"
        )
        self.logger.info(
            f"  CAAC Accuracy: {summary['test_accuracy_mean']:.4f} ± {summary['test_accuracy_std']:.4f}"
        )

        return summary


# ==================== 主流程 ====================

class BaselineComparisonPipeline:
    def __init__(self, config: BaselineConfig):
        self.config = config
        self.logger = self._setup_logger()
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config.seed)
            # 4090 加速设置
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    def _setup_logger(self) -> logging.Logger:
        os.makedirs(f"{self.config.output_dir}/logs", exist_ok=True)
        logger = logging.getLogger("baseline_comparison_v2")
        logger.setLevel(logging.INFO)
        if logger.handlers:
            logger.handlers.clear()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fh = logging.FileHandler(f"{self.config.output_dir}/logs/baseline_v2_{timestamp}.log")
        fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(fh)
        logger.addHandler(ch)
        return logger

    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
        self.logger.info("加载特征数据...")

        all_features = np.load(f"{self.config.features_dir}/all_features.npy")
        all_labels = np.load(f"{self.config.features_dir}/all_labels.npy")
        metadata = pd.read_csv(f"{self.config.features_dir}/all_metadata.csv")

        esm2_features = all_features[:, :1280]

        self.logger.info(f"总样本: {len(all_features):,}")
        self.logger.info(f"ESM2维度: {esm2_features.shape[1]}")
        self.logger.info(f"融合特征维度: {all_features.shape[1]}")
        self.logger.info(f"标签分布: {dict(zip(*np.unique(all_labels, return_counts=True)))}")

        return esm2_features, all_features, all_labels, metadata

    def create_model(self, model_name: str, input_dim: int) -> nn.Module:
        """根据模型名创建对应的PyTorch模型"""
        if model_name in ('esm2_mlp', 'esm2_context_mlp'):
            return ESM2MLP(
                input_dim=input_dim,
                hidden_dims=self.config.mlp_hidden_dims,
                num_classes=3,
                dropout=self.config.mlp_dropout
            )
        elif model_name == 'esm2_finetuned':
            return ESM2FineTuned(
                input_dim=input_dim,
                num_classes=3,
                hidden_dim=self.config.ft_hidden_dim,
                dropout=self.config.ft_dropout
            )
        elif model_name == 'esm2_attention':
            return ESM2AttentionClassifier(
                input_dim=input_dim,
                num_classes=3,
                num_heads=8,
                hidden_dim=self.config.ft_hidden_dim,
                dropout=self.config.ft_dropout
            )
        else:
            raise ValueError(f"未知PyTorch模型: {model_name}")

    def run_model(self, model_name: str, features: np.ndarray, labels: np.ndarray) -> Dict:
        self.logger.info(f"\n{'#' * 70}")
        self.logger.info(f"# 模型: {model_name}")
        self.logger.info(f"{'#' * 70}")

        skf = StratifiedKFold(n_splits=self.config.n_folds, shuffle=True,
                              random_state=self.config.seed)

        fold_results = []
        fold_idx = 1

        for train_val_idx, test_idx in skf.split(features, labels):
            train_val_x = features[train_val_idx]
            train_val_y = labels[train_val_idx]
            test_x = features[test_idx]
            test_y = labels[test_idx]

            # 内层划分: 从train_val中分出val (4:1)
            inner_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.config.seed)
            train_idx_rel, val_idx_rel = next(inner_skf.split(train_val_x, train_val_y))

            train_x = train_val_x[train_idx_rel]
            train_y = train_val_y[train_idx_rel]
            val_x = train_val_x[val_idx_rel]
            val_y = train_val_y[val_idx_rel]

            self.logger.info(f"  Train: {len(train_x)}, Val: {len(val_x)}, Test: {len(test_x)}")

            if model_name in ('esm2_mlp', 'esm2_context_mlp',
                              'esm2_finetuned', 'esm2_attention'):
                # 深度学习模型统一训练流程
                input_dim = features.shape[1]
                train_dataset = FeatureDataset(train_x, train_y)
                val_dataset = FeatureDataset(val_x, val_y)
                test_dataset = FeatureDataset(test_x, test_y)

                model = self.create_model(model_name, input_dim)

                # 根据模型类型选择训练参数
                if model_name in ('esm2_mlp', 'esm2_context_mlp'):
                    epochs = self.config.mlp_epochs
                    batch_size = self.config.mlp_batch_size
                    lr = self.config.mlp_lr
                    wd = self.config.mlp_weight_decay
                    es = self.config.mlp_early_stop
                else:
                    epochs = self.config.ft_epochs
                    batch_size = self.config.ft_batch_size
                    lr = self.config.ft_lr
                    wd = self.config.ft_weight_decay
                    es = self.config.ft_early_stop

                trainer = DeepLearningTrainer(self.config, self.logger, model_name)
                result = trainer.run_fold(
                    fold_idx, train_dataset, val_dataset, test_dataset,
                    model, epochs, batch_size, lr, wd, es
                )

            elif model_name in ('esm2_rf', 'esm2_context_rf'):
                trainer = RFTrainer(self.config, self.logger, model_name)
                result = trainer.run_fold(fold_idx, train_x, train_y, test_x, test_y)

            else:
                raise ValueError(f"未知模型: {model_name}")

            fold_results.append(result)
            fold_idx += 1

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # 汇总统计
        test_f1s = [r['test_metrics']['f1_macro'] for r in fold_results]
        test_accs = [r['test_metrics']['accuracy'] for r in fold_results]

        summary = {
            'model': model_name,
            'test_f1_macro_mean': float(np.mean(test_f1s)),
            'test_f1_macro_std': float(np.std(test_f1s)),
            'test_accuracy_mean': float(np.mean(test_accs)),
            'test_accuracy_std': float(np.std(test_accs)),
            'fold_results': fold_results,
        }

        for cls in range(3):
            cls_f1s = [r['test_metrics'][f'f1_class_{cls}'] for r in fold_results]
            summary[f'class_{cls}_f1_mean'] = float(np.mean(cls_f1s))
            summary[f'class_{cls}_f1_std'] = float(np.std(cls_f1s))

        self.logger.info(f"\n{'=' * 60}")
        self.logger.info(f"{model_name} 汇总:")
        self.logger.info(
            f"  F1-macro: {summary['test_f1_macro_mean']:.4f} ± {summary['test_f1_macro_std']:.4f}"
        )
        self.logger.info(
            f"  Accuracy: {summary['test_accuracy_mean']:.4f} ± {summary['test_accuracy_std']:.4f}"
        )
        for cls in range(3):
            cls_name = {0: 'Negative', 1: 'Positive', 2: 'Hard'}[cls]
            self.logger.info(
                f"  F1-{cls_name}: {summary[f'class_{cls}_f1_mean']:.4f} ± {summary[f'class_{cls}_f1_std']:.4f}"
            )

        return summary

    def generate_comparison_table(self, all_results: Dict[str, Dict]):
        rows = []

        name_map = {
            'esm2_mlp': 'ESM2-MLP',
            'esm2_context_mlp': 'ESM2-Context-MLP',
            'esm2_rf': 'ESM2-RF',
            'esm2_context_rf': 'ESM2-Context-RF',
            'esm2_finetuned': 'ESM2-FineTuned',
            'esm2_attention': 'ESM2-Attention',
            'caac': 'CAAC (Ours)',
        }

        # 按固定顺序排列
        order = [
            'esm2_mlp', 'esm2_rf',
            'esm2_context_mlp', 'esm2_context_rf',
            'esm2_finetuned', 'esm2_attention',
            'caac'
        ]

        for model_name in order:
            if model_name not in all_results:
                continue
            result = all_results[model_name]
            display_name = name_map.get(model_name, model_name)
            row = {
                'Model': display_name,
                'F1-macro': f"{result['test_f1_macro_mean']:.4f} ± {result['test_f1_macro_std']:.4f}",
                'Accuracy': f"{result['test_accuracy_mean']:.4f} ± {result['test_accuracy_std']:.4f}",
                'F1-Negative': f"{result.get('class_0_f1_mean', 0):.4f} ± {result.get('class_0_f1_std', 0):.4f}",
                'F1-Positive': f"{result.get('class_1_f1_mean', 0):.4f} ± {result.get('class_1_f1_std', 0):.4f}",
                'F1-Hard': f"{result.get('class_2_f1_mean', 0):.4f} ± {result.get('class_2_f1_std', 0):.4f}",
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        csv_path = f"{self.config.output_dir}/baseline_comparison_v2.csv"
        df.to_csv(csv_path, index=False)

        self.logger.info(f"\n{'=' * 80}")
        self.logger.info("基线对比结果 (修正版):")
        self.logger.info(f"{'=' * 80}")
        self.logger.info(f"\n{df.to_string(index=False)}")
        self.logger.info(f"\n保存至: {csv_path}")

        json_path = f"{self.config.output_dir}/baseline_results_v2.json"
        with open(json_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        self.logger.info(f"JSON: {json_path}")

    def run(self):
        self.logger.info("=" * 80)
        self.logger.info("基线模型对比实验 v2 (修正版)")
        self.logger.info(f"开始时间: {datetime.now()}")
        self.logger.info("修正内容: MLP/RF参数优化 + 新增ESM2-FineTuned/Attention SOTA基线")
        self.logger.info("=" * 80)

        esm2_features, all_features, all_labels, metadata = self.load_data()

        all_results = {}

        # 定义各模型使用的特征
        feature_sets = {
            'esm2_mlp': esm2_features,
            'esm2_context_mlp': all_features,
            'esm2_rf': esm2_features,
            'esm2_context_rf': all_features,
            'esm2_finetuned': esm2_features,  # SOTA基线只用ESM2
            'esm2_attention': esm2_features,  # SOTA基线只用ESM2
        }

        # 1. 训练所有基线模型
        for model_name in self.config.models:
            if model_name not in feature_sets:
                continue

            features = feature_sets[model_name]
            result = self.run_model(model_name, features, all_labels)
            all_results[model_name] = result

            with open(f"{self.config.output_dir}/{model_name}_results.json", 'w') as f:
                json.dump(result, f, indent=2)

        # 2. 加载CAAC已有结果
        try:
            caac_loader = CAACResultLoader(self.config.caac_report_path, self.logger)
            caac_result = caac_loader.load()
            all_results['caac'] = caac_result
        except Exception as e:
            self.logger.error(f"加载CAAC结果失败: {e}")
            self.logger.error("请确认CAAC训练已完成并生成training_report.json")
            raise

        # 3. 生成对比表格
        self.generate_comparison_table(all_results)

        self.logger.info(f"\n{'=' * 80}")
        self.logger.info("基线对比实验完成!")
        self.logger.info(f"输出目录: {self.config.output_dir}")
        self.logger.info(f"{'=' * 80}")

        return all_results


# ==================== 命令行入口 ====================

def main():
    parser = argparse.ArgumentParser(description='基线模型对比实验 v2 (修正版)')
    parser.add_argument('--features-dir',
                        default="/home/zjw/zjwdata/3_deep_learning/training_data/curated_v2/features")
    parser.add_argument('--output-dir',
                        default="/home/zjw/zjwdata/3_deep_learning/training_data/curated_v2/baseline_results_v2")
    parser.add_argument('--caac-report',
                        default="/home/zjw/zjwdata/3_deep_learning/training_data/curated_v2/gnn_model/training_report.json")
    parser.add_argument('--models', nargs='+',
                        default=[
                            'esm2_mlp', 'esm2_context_mlp',
                            'esm2_rf', 'esm2_context_rf',
                            'esm2_finetuned', 'esm2_attention',
                        ],
                        help='要训练的基线模型列表（不含CAAC）')
    parser.add_argument('--n-folds', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default="cuda:0")

    args = parser.parse_args()

    config = BaselineConfig(
        features_dir=args.features_dir,
        output_dir=args.output_dir,
        caac_report_path=args.caac_report,
        models=tuple(args.models),
        n_folds=args.n_folds,
        seed=args.seed,
        device=args.device,
    )

    pipeline = BaselineComparisonPipeline(config)
    results = pipeline.run()

    return results


if __name__ == "__main__":
    main()