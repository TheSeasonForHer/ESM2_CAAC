#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模块5: gnn_training.py
功能: 3分类GNN模型训练（负=0, 正=1, 难=2）
      5折交叉验证 + 早停 + TensorBoard + 学习率调度

输入: 模块4输出 (curated_v2/features/)
  - all_features.npy      # (N, 1408)
  - all_labels.npy        # (N,) {0,1,2}
  - all_metadata.csv      # gene_id, category, label, has_context

输出: curated_v2/gnn_model/
  - best_model_fold{k}.pt         # 每折最佳模型
  - final_model.pt                # 全量数据训练的最终模型
  - training_report.json          # 完整训练报告
  - fold_results/                 # 每折详细结果
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
from dataclasses import dataclass, asdict
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

from tqdm import tqdm

warnings.filterwarnings('ignore')


# ==================== 配置类 ====================

@dataclass
class GNNConfig:
    """GNN训练配置"""
    # 输入路径
    features_dir: str = "/home/zjw/zjwdata/3_deep_learning/training_data/curated_v2/features"

    # 输出路径
    output_dir: str = "/home/zjw/zjwdata/3_deep_learning/training_data/curated_v2/gnn_model"

    # 模型参数
    input_dim: int = 1408  # ESM2(1280) + Context(128)
    hidden_dim: int = 512
    num_classes: int = 3  # 0=负, 1=正, 2=难
    num_attention_heads: int = 8
    dropout: float = 0.3
    attention_dropout: float = 0.1

    # 训练参数
    epochs: int = 100
    batch_size: int = 512
    learning_rate: float = 5e-4
    weight_decay: float = 1e-5
    warmup_epochs: int = 5
    early_stopping_patience: int = 15

    # 学习率调度
    lr_scheduler_type: str = "cosine_warmup"
    min_lr: float = 1e-6
    T_0: int = 10

    # 验证策略
    n_folds: int = 5
    validation_split: float = 0.15  # 每折内部再划分val

    # 硬件
    device: str = "cuda:0"

    # 日志
    log_level: str = "INFO"
    tensorboard: bool = True
    log_interval: int = 10
    save_interval: int = 5

    # 随机种子
    seed: int = 42

    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(f"{self.output_dir}/fold_results", exist_ok=True)


# ==================== 日志系统 ====================

def setup_logger(name: str, log_dir: str, level: str = "INFO") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))

    if logger.handlers:
        logger.handlers.clear()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fh = logging.FileHandler(f"{log_dir}/{name}_{timestamp}.log")
    fh.setLevel(getattr(logging, level))
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(getattr(logging, level))

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ==================== 随机种子 ====================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==================== 数据集 ====================

class GeneFeatureDataset(Dataset):
    """基因特征数据集"""

    def __init__(self, features: np.ndarray, labels: np.ndarray, metadata: pd.DataFrame):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        self.metadata = metadata.reset_index(drop=True)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx], idx


# ==================== 模型定义 ====================

class AttentionGNN(nn.Module):
    """
    3分类GNN模型
    结构: Projection -> MultiHeadAttention -> FFN -> Classifier
    """

    def __init__(self, config: GNNConfig):
        super().__init__()
        self.config = config

        # 特征投影
        self.projection = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )

        # 多头自注意力
        self.attention = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            batch_first=True
        )

        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim)
        )

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, config.num_classes)
        )

        # 可学习的残差权重
        self.alpha = nn.Parameter(torch.ones(1))

    def forward(self, x, return_attention=False):
        # x: (batch, input_dim)

        # 投影
        x_proj = self.projection(x)  # (batch, hidden)

        # 添加序列维度用于注意力
        x_seq = x_proj.unsqueeze(1)  # (batch, 1, hidden)

        # 自注意力
        attn_out, attn_weights = self.attention(x_seq, x_seq, x_seq)
        attn_out = attn_out.squeeze(1)  # (batch, hidden)

        # 残差连接
        x_residual = x_proj + self.alpha * attn_out

        # 前馈
        x_ffn = self.ffn(x_residual)

        # 最终残差
        x_final = x_residual + x_ffn

        # 分类
        logits = self.classifier(x_final)

        if return_attention:
            return logits, attn_weights
        return logits


# ==================== 训练器 ====================

class GNNTrainer:
    """GNN训练器（支持5折交叉验证）"""

    def __init__(self, config: GNNConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")

        self.logger.info(f"使用设备: {self.device}")
        if torch.cuda.is_available():
            self.logger.info(f"GPU: {torch.cuda.get_device_name(self.device)}")
            self.logger.info(f"显存: {torch.cuda.get_device_properties(self.device).total_memory / 1e9:.1f} GB")

    def create_optimizer(self, model: nn.Module):
        """创建优化器"""
        return torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.999)
        )

    def create_scheduler(self, optimizer, steps_per_epoch: int):
        """创建学习率调度器"""
        total_steps = steps_per_epoch * self.config.epochs
        warmup_steps = steps_per_epoch * self.config.warmup_epochs

        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            else:
                progress = (step - warmup_steps) / (total_steps - warmup_steps)
                return 0.5 * (1 + np.cos(np.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def train_epoch(self, model, loader, optimizer, criterion, scheduler, epoch, writer=None):
        """训练一个epoch"""
        model.train()
        total_loss = 0
        all_preds = []
        all_labels = []

        pbar = tqdm(loader, desc=f"Train Epoch {epoch}")
        for batch_idx, (batch_x, batch_y, _) in enumerate(pbar):
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch_y.cpu().numpy())

            # TensorBoard记录
            global_step = epoch * len(loader) + batch_idx
            if writer and batch_idx % self.config.log_interval == 0:
                writer.add_scalar('train/batch_loss', loss.item(), global_step)
                writer.add_scalar('train/batch_acc',
                                  accuracy_score(batch_y.cpu().numpy(), preds), global_step)

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        metrics = {
            'loss': total_loss / len(loader),
            'accuracy': accuracy_score(all_labels, all_preds),
            'f1_macro': f1_score(all_labels, all_preds, average='macro'),
            'f1_weighted': f1_score(all_labels, all_preds, average='weighted'),
        }

        # 每类F1
        for cls in range(self.config.num_classes):
            cls_f1 = f1_score(all_labels, all_preds, labels=[cls], average='macro', zero_division=0)
            metrics[f'f1_class_{cls}'] = cls_f1

        return metrics

    def evaluate(self, model, loader, criterion, prefix="val"):
        """评估"""
        model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for batch_x, batch_y, _ in tqdm(loader, desc=prefix):
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                logits = model(batch_x)
                loss = criterion(logits, batch_y)

                total_loss += loss.item()
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)

                all_preds.extend(preds)
                all_labels.extend(batch_y.cpu().numpy())
                all_probs.extend(probs)

        metrics = {
            'loss': total_loss / len(loader),
            'accuracy': accuracy_score(all_labels, all_preds),
            'precision_macro': precision_score(all_labels, all_preds, average='macro', zero_division=0),
            'recall_macro': recall_score(all_labels, all_preds, average='macro', zero_division=0),
            'f1_macro': f1_score(all_labels, all_preds, average='macro', zero_division=0),
            'f1_weighted': f1_score(all_labels, all_preds, average='weighted', zero_division=0),
        }

        # 每类指标
        for cls in range(self.config.num_classes):
            cls_precision = precision_score(all_labels, all_preds, labels=[cls], average='macro', zero_division=0)
            cls_recall = recall_score(all_labels, all_preds, labels=[cls], average='macro', zero_division=0)
            cls_f1 = f1_score(all_labels, all_preds, labels=[cls], average='macro', zero_division=0)
            metrics[f'precision_class_{cls}'] = cls_precision
            metrics[f'recall_class_{cls}'] = cls_recall
            metrics[f'f1_class_{cls}'] = cls_f1

        cm = confusion_matrix(all_labels, all_preds, labels=list(range(self.config.num_classes)))

        return metrics, cm, np.array(all_labels), np.array(all_preds), np.array(all_probs)

    def plot_confusion_matrix(self, cm, fold_idx: int, save_path: str):
        """绘制混淆矩阵"""
        plt.figure(figsize=(8, 6))
        classes = ['Negative(0)', 'Positive(1)', 'Hard(2)']

        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix - Fold {fold_idx}')
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        # 添加数值
        thresh = cm.max() / 2.
        for i, j in np.ndindex(cm.shape):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        self.logger.info(f"混淆矩阵: {save_path}")

    def plot_learning_curves(self, history: Dict, fold_idx: int, save_path: str):
        """绘制学习曲线"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        epochs = range(1, len(history['train_loss']) + 1)

        # Loss
        axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train')
        axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val')
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].legend()

        # Accuracy
        axes[0, 1].plot(epochs, history['train_acc'], 'b-', label='Train')
        axes[0, 1].plot(epochs, history['val_acc'], 'r-', label='Val')
        axes[0, 1].set_title('Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].legend()

        # F1 Macro
        axes[1, 0].plot(epochs, history['train_f1_macro'], 'b-', label='Train')
        axes[1, 0].plot(epochs, history['val_f1_macro'], 'r-', label='Val')
        axes[1, 0].set_title('F1 Macro')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].legend()

        # Learning Rate
        if 'lr' in history:
            axes[1, 1].plot(epochs, history['lr'], 'g-')
            axes[1, 1].set_title('Learning Rate')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_yscale('log')

        plt.suptitle(f'Learning Curves - Fold {fold_idx}')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        self.logger.info(f"学习曲线: {save_path}")

    def run_fold(
            self,
            fold_idx: int,
            train_dataset: GeneFeatureDataset,
            val_dataset: GeneFeatureDataset,
            test_dataset: GeneFeatureDataset,
            full_dataset: GeneFeatureDataset
    ) -> Dict:
        """训练单折"""
        self.logger.info(f"\n{'=' * 60}")
        self.logger.info(f"Fold {fold_idx}/{self.config.n_folds}")
        self.logger.info(f"{'=' * 60}")

        # DataLoader
        train_loader = DataLoader(
            train_dataset, batch_size=self.config.batch_size,
            shuffle=True, num_workers=4, pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.config.batch_size,
            shuffle=False, num_workers=4, pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=self.config.batch_size,
            shuffle=False, num_workers=4, pin_memory=True
        )

        # 模型
        model = AttentionGNN(self.config).to(self.device)

        # 统计参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.info(f"总参数量: {total_params:,}")
        self.logger.info(f"可训练参数量: {trainable_params:,}")

        # 优化器
        optimizer = self.create_optimizer(model)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        scheduler = self.create_scheduler(optimizer, len(train_loader))

        # TensorBoard
        writer = None
        if self.config.tensorboard:
            log_dir = f"{self.config.output_dir}/logs/fold_{fold_idx}"
            writer = SummaryWriter(log_dir)

        # 训练历史
        history = defaultdict(list)
        best_val_f1 = 0
        patience_counter = 0
        best_state = None

        for epoch in range(1, self.config.epochs + 1):
            # 训练
            train_metrics = self.train_epoch(
                model, train_loader, optimizer, criterion, scheduler, epoch, writer
            )

            # 验证
            val_metrics, val_cm, _, _, _ = self.evaluate(model, val_loader, criterion, "Val")

            # 记录历史
            history['train_loss'].append(train_metrics['loss'])
            history['train_acc'].append(train_metrics['accuracy'])
            history['train_f1_macro'].append(train_metrics['f1_macro'])
            history['val_loss'].append(val_metrics['loss'])
            history['val_acc'].append(val_metrics['accuracy'])
            history['val_f1_macro'].append(val_metrics['f1_macro'])
            history['lr'].append(optimizer.param_groups[0]['lr'])

            # TensorBoard
            if writer:
                writer.add_scalars('loss', {'train': train_metrics['loss'], 'val': val_metrics['loss']}, epoch)
                writer.add_scalars('accuracy', {'train': train_metrics['accuracy'], 'val': val_metrics['accuracy']},
                                   epoch)
                writer.add_scalars('f1_macro', {'train': train_metrics['f1_macro'], 'val': val_metrics['f1_macro']},
                                   epoch)

            # 日志
            self.logger.info(
                f"Epoch {epoch:3d} | "
                f"Train Loss: {train_metrics['loss']:.4f} Acc: {train_metrics['accuracy']:.4f} F1: {train_metrics['f1_macro']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']:.4f} F1: {val_metrics['f1_macro']:.4f}"
            )

            # 保存最佳模型
            if val_metrics['f1_macro'] > best_val_f1:
                best_val_f1 = val_metrics['f1_macro']
                patience_counter = 0
                best_state = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_f1': best_val_f1,
                    'config': asdict(self.config)
                }

                # 保存
                save_path = f"{self.config.output_dir}/best_model_fold{fold_idx}.pt"
                torch.save(best_state, save_path)
                self.logger.info(f"  ✓ 保存最佳模型 (Val F1: {best_val_f1:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= self.config.early_stopping_patience:
                    self.logger.info(f"  早停触发于epoch {epoch}")
                    break

            # 定期保存checkpoint
            if epoch % self.config.save_interval == 0:
                ckpt_path = f"{self.config.output_dir}/fold_results/checkpoint_fold{fold_idx}_epoch{epoch}.pt"
                torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()}, ckpt_path)

        # 关闭writer
        if writer:
            writer.close()

        # 加载最佳模型进行测试
        self.logger.info(f"\n加载最佳模型进行测试...")
        model.load_state_dict(best_state['model_state_dict'])

        test_metrics, test_cm, test_labels, test_preds, test_probs = self.evaluate(
            model, test_loader, criterion, "Test"
        )

        self.logger.info(f"Test Results:")
        self.logger.info(f"  Loss: {test_metrics['loss']:.4f}")
        self.logger.info(f"  Accuracy: {test_metrics['accuracy']:.4f}")
        self.logger.info(f"  F1 Macro: {test_metrics['f1_macro']:.4f}")
        self.logger.info(f"  F1 Weighted: {test_metrics['f1_weighted']:.4f}")

        for cls in range(self.config.num_classes):
            self.logger.info(
                f"  Class {cls} - P: {test_metrics[f'precision_class_{cls}']:.4f} "
                f"R: {test_metrics[f'recall_class_{cls}']:.4f} "
                f"F1: {test_metrics[f'f1_class_{cls}']:.4f}"
            )

        # 绘制图表
        fold_result_dir = f"{self.config.output_dir}/fold_results/fold_{fold_idx}"
        os.makedirs(fold_result_dir, exist_ok=True)

        self.plot_confusion_matrix(test_cm, fold_idx, f"{fold_result_dir}/confusion_matrix.png")
        self.plot_learning_curves(history, fold_idx, f"{fold_result_dir}/learning_curves.png")

        # 保存预测结果
        results_df = pd.DataFrame({
            'true_label': test_labels,
            'pred_label': test_preds,
            'prob_0': test_probs[:, 0],
            'prob_1': test_probs[:, 1],
            'prob_2': test_probs[:, 2],
        })
        results_df.to_csv(f"{fold_result_dir}/test_predictions.csv", index=False)

        # 保存详细分类报告
        report = classification_report(
            test_labels, test_preds,
            target_names=['Negative', 'Positive', 'Hard'],
            digits=4
        )
        with open(f"{fold_result_dir}/classification_report.txt", 'w') as f:
            f.write(report)

        return {
            'fold': fold_idx,
            'best_epoch': best_state['epoch'],
            'best_val_f1': float(best_val_f1),
            'test_metrics': {k: float(v) for k, v in test_metrics.items()},
            'test_cm': test_cm.tolist(),
            'history': {k: [float(v) for v in vals] for k, vals in history.items()},
            'result_dir': fold_result_dir,
        }

    def run_final_training(self, full_dataset: GeneFeatureDataset) -> Dict:
        """使用全量数据训练最终模型（用于预测）"""
        self.logger.info(f"\n{'=' * 60}")
        self.logger.info("最终模型训练（全量数据）")
        self.logger.info(f"{'=' * 60}")

        # 划分train/val（用于早停监控）
        n = len(full_dataset)
        val_size = int(n * self.config.validation_split)
        train_size = n - val_size

        indices = list(range(n))
        np.random.shuffle(indices)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]

        train_subset = Subset(full_dataset, train_indices)
        val_subset = Subset(full_dataset, val_indices)

        train_loader = DataLoader(train_subset, batch_size=self.config.batch_size,
                                  shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_subset, batch_size=self.config.batch_size,
                                shuffle=False, num_workers=4, pin_memory=True)

        # 模型
        model = AttentionGNN(self.config).to(self.device)
        optimizer = self.create_optimizer(model)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        scheduler = self.create_scheduler(optimizer, len(train_loader))

        best_val_f1 = 0
        best_state = None
        patience_counter = 0

        for epoch in range(1, self.config.epochs + 1):
            train_metrics = self.train_epoch(
                model, train_loader, optimizer, criterion, scheduler, epoch
            )
            val_metrics, _, _, _, _ = self.evaluate(model, val_loader, criterion, "Val")

            self.logger.info(
                f"Epoch {epoch:3d} | "
                f"Train F1: {train_metrics['f1_macro']:.4f} | "
                f"Val F1: {val_metrics['f1_macro']:.4f}"
            )

            if val_metrics['f1_macro'] > best_val_f1:
                best_val_f1 = val_metrics['f1_macro']
                patience_counter = 0
                best_state = model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= self.config.early_stopping_patience:
                    self.logger.info(f"早停于epoch {epoch}")
                    break

        # 保存最终模型
        model.load_state_dict(best_state)
        final_path = f"{self.config.output_dir}/final_model.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': asdict(self.config),
            'val_f1': best_val_f1,
        }, final_path)

        self.logger.info(f"最终模型: {final_path}")

        return {
            'final_model_path': final_path,
            'best_val_f1': float(best_val_f1),
        }


# ==================== 主流程 ====================

class GNNTrainingPipeline:
    """GNN训练主流程"""

    def __init__(self, config: GNNConfig):
        self.config = config
        self.logger = setup_logger("gnn_training", f"{config.output_dir}/logs", config.log_level)
        set_seed(config.seed)

        self.logger.info("=" * 80)
        self.logger.info("模块5: GNN 3分类模型训练")
        self.logger.info(f"开始时间: {datetime.now()}")
        self.logger.info(f"配置: {asdict(config)}")
        self.logger.info("=" * 80)

    def run(self) -> Dict:
        # 加载数据
        self.logger.info("\n加载特征数据...")

        features = np.load(f"{self.config.features_dir}/all_features.npy")
        labels = np.load(f"{self.config.features_dir}/all_labels.npy")
        metadata = pd.read_csv(f"{self.config.features_dir}/all_metadata.csv")

        self.logger.info(f"特征矩阵: {features.shape}")
        self.logger.info(f"标签分布: {dict(zip(*np.unique(labels, return_counts=True)))}")

        # 创建数据集
        full_dataset = GeneFeatureDataset(features, labels, metadata)

        # 5折交叉验证
        skf = StratifiedKFold(n_splits=self.config.n_folds, shuffle=True, random_state=self.config.seed)

        fold_results = []
        fold_idx = 1

        for train_val_idx, test_idx in skf.split(features, labels):
            self.logger.info(f"\n{'=' * 60}")
            self.logger.info(f"Fold {fold_idx}/{self.config.n_folds}")
            self.logger.info(f"Train+Val: {len(train_val_idx)}, Test: {len(test_idx)}")
            self.logger.info(f"{'=' * 60}")

            # 从train_val中划分train/val
            train_val_features = features[train_val_idx]
            train_val_labels = labels[train_val_idx]

            # 再次划分
            inner_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.config.seed)
            train_idx_rel, val_idx_rel = next(inner_skf.split(train_val_features, train_val_labels))

            train_idx = train_val_idx[train_idx_rel]
            val_idx = train_val_idx[val_idx_rel]

            self.logger.info(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

            # 类别分布
            for name, idx in [('Train', train_idx), ('Val', val_idx), ('Test', test_idx)]:
                dist = np.bincount(labels[idx], minlength=self.config.num_classes)
                self.logger.info(f"  {name}: {dist}")

            # 创建子数据集
            train_dataset = GeneFeatureDataset(features[train_idx], labels[train_idx], metadata.iloc[train_idx])
            val_dataset = GeneFeatureDataset(features[val_idx], labels[val_idx], metadata.iloc[val_idx])
            test_dataset = GeneFeatureDataset(features[test_idx], labels[test_idx], metadata.iloc[test_idx])

            # 训练
            trainer = GNNTrainer(self.config, self.logger)
            result = trainer.run_fold(fold_idx, train_dataset, val_dataset, test_dataset, full_dataset)
            fold_results.append(result)

            # 清理显存
            torch.cuda.empty_cache()

            fold_idx += 1

        # 汇总5折结果
        self.logger.info(f"\n{'=' * 60}")
        self.logger.info("5折交叉验证汇总")
        self.logger.info(f"{'=' * 60}")

        test_f1s = [r['test_metrics']['f1_macro'] for r in fold_results]
        test_accs = [r['test_metrics']['accuracy'] for r in fold_results]

        self.logger.info(f"Test F1 Macro: {np.mean(test_f1s):.4f} ± {np.std(test_f1s):.4f}")
        self.logger.info(f"Test Accuracy: {np.mean(test_accs):.4f} ± {np.std(test_accs):.4f}")

        for cls in range(self.config.num_classes):
            cls_f1s = [r['test_metrics'][f'f1_class_{cls}'] for r in fold_results]
            self.logger.info(f"Class {cls} F1: {np.mean(cls_f1s):.4f} ± {np.std(cls_f1s):.4f}")

        # 最终模型训练（全量数据）
        final_result = GNNTrainer(self.config, self.logger).run_final_training(full_dataset)

        # 保存完整报告
        report = {
            'timestamp': datetime.now().isoformat(),
            'configuration': asdict(self.config),
            'fold_results': fold_results,
            'summary': {
                'test_f1_macro_mean': float(np.mean(test_f1s)),
                'test_f1_macro_std': float(np.std(test_f1s)),
                'test_accuracy_mean': float(np.mean(test_accs)),
                'test_accuracy_std': float(np.std(test_accs)),
            },
            'final_model': final_result,
        }

        report_path = f"{self.config.output_dir}/training_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        self.logger.info(f"\n{'=' * 80}")
        self.logger.info("模块5完成!")
        self.logger.info(f"报告: {report_path}")
        self.logger.info(f"最终模型: {final_result['final_model_path']}")
        self.logger.info(f"{'=' * 80}")

        return report


# ==================== 命令行入口 ====================

def main():
    parser = argparse.ArgumentParser(description='模块5: GNN 3分类模型训练')
    parser.add_argument('--features-dir',
                        default="/home/zjw/zjwdata/3_deep_learning/training_data/curated_v2/features")
    parser.add_argument('--output-dir',
                        default="/home/zjw/zjwdata/3_deep_learning/training_data/curated_v2/gnn_model")
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--hidden-dim', type=int, default=512)
    parser.add_argument('--n-folds', type=int, default=5)
    parser.add_argument('--device', default="cuda:0")
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    config = GNNConfig(
        features_dir=args.features_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        hidden_dim=args.hidden_dim,
        n_folds=args.n_folds,
        device=args.device,
        seed=args.seed,
    )

    pipeline = GNNTrainingPipeline(config)
    results = pipeline.run()

    return results


if __name__ == "__main__":
    main()