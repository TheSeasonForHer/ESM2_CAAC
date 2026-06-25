#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
CAAC Confidence Calibration & Tier Threshold Validation
基于5-Fold CV验证集的精度-召回分析、阈值验证和权重消融实验

输入:
  - 5-Fold CV test predictions: fold_results/fold_*/test_predictions.csv
  - 全部特征: features/all_features.npy (863015, 1408)
  - 全部标签: features/all_labels.npy (863015,)
  - 全部metadata: features/all_metadata.csv (gene_id, category, label, has_context)
  - 正样本参考库: features/positive/esm2_features.npy + gene_ids.csv + metadata
  - GNN模型: gnn_model/best_model_fold*.pt
  - 模型定义: /home/zjw/Projects4_fixed/model_training.py

输出:
  - calibration_results/
    ├── precision_recall_analysis.png      # PR曲线
    ├── confidence_distribution.png         # 置信度分布
    ├── tier_threshold_validation.png       # Tier阈值验证
    ├── weight_ablation.png                 # 权重消融
    ├── calibration_report.json             # 完整报告
    └── calibration_report.md             # Markdown报告(论文用)

作者: CAAC Framework - Reviewer Response
================================================================================
"""

import os
import sys
import re
import json
import argparse
import logging
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# sklearn metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    precision_recall_curve, average_precision_score,
    precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)

# matplotlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ==================== 配置 ====================

@dataclass
class Config:
    """配置类"""
    # 数据路径
    features_dir: str = "/home/zjw/zjwdata/3_deep_learning/training_data/curated_v2/features"
    fold_results_dir: str = "/home/zjw/zjwdata/3_deep_learning/training_data/curated_v2/gnn_model/fold_results"
    training_report: str = "/home/zjw/zjwdata/3_deep_learning/training_data/curated_v2/gnn_model/training_report.json"
    positive_features: str = "/home/zjw/zjwdata/3_deep_learning/training_data/curated_v2/features/positive/esm2_features.npy"
    positive_gene_ids: str = "/home/zjw/zjwdata/3_deep_learning/training_data/curated_v2/features/positive/gene_ids.csv"
    positive_metadata: str = "/home/zjw/zjwdata/3_deep_learning/training_data/curated_v2/positive_samples_info.tsv"
    model_training_path: str = "/home/zjw/Projects4_fixed/model_training.py"

    # 输出路径
    output_dir: str = "/home/zjw/zjwdata/3_deep_learning/training_data/curated_v2/calibration_results"

    # 设备
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    batch_size: int = 512

    # 5-Fold CV参数（必须与训练时一致）
    n_folds: int = 5
    random_state: int = 42

    # 置信度公式参数（当前值）
    consistency_weight: float = 0.8
    excess_similarity_weight: float = 0.2
    similarity_baseline: float = 0.9
    excess_similarity_scale: float = 10.0

    # 邻居检索参数
    faiss_top_k: int = 50
    top_neighbors_for_consensus: int = 5
    min_consensus_ratio: float = 0.4

    # Tier阈值（当前值，用于验证）
    tier1_threshold: float = 0.60
    tier2_threshold: float = 0.40
    tier3_threshold: float = 0.25

    # 权重消融网格
    weight_grid_wc: List[float] = None
    weight_grid_we: List[float] = None

    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(f"{self.output_dir}/logs", exist_ok=True)
        if self.weight_grid_wc is None:
            self.weight_grid_wc = [0.6, 0.7, 0.8, 0.9, 1.0]
        if self.weight_grid_we is None:
            self.weight_grid_we = [0.0, 0.1, 0.2, 0.3, 0.4]


# ==================== 日志 ====================

def setup_logger(name: str, log_dir: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if logger.handlers:
        logger.handlers.clear()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fh = logging.FileHandler(f"{log_dir}/{name}_{timestamp}.log")
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ==================== 主类：校准分析器 ====================

class CAACCalibrator:
    """
    CAAC置信度校准器

    核心功能:
    1. 从5-Fold CV重建完整验证集
    2. 计算每条验证样本的confidence score
    3. 精度-召回分析
    4. Tier阈值验证
    5. 权重消融实验
    """

    def __init__(self, config: Config):
        self.config = config
        self.logger = setup_logger("calibrator", f"{config.output_dir}/logs")

        self.logger.info("=" * 80)
        self.logger.info("CAAC Confidence Calibration & Tier Threshold Validation")
        self.logger.info("=" * 80)

        # 加载数据
        self._load_data()

        # 加载正样本参考库
        self._load_positive_reference()

        # 导入模型
        self._import_model()

    # ---------- 数据加载 ----------

    def _load_data(self):
        """加载全部数据和5-Fold CV结果"""
        self.logger.info("\n[Step 1] 加载数据")

        # 加载全部特征和标签
        self.logger.info("  加载 all_features.npy ...")
        self.all_features = np.load(f"{self.config.features_dir}/all_features.npy")
        self.logger.info(f"    Shape: {self.all_features.shape}")

        self.logger.info("  加载 all_labels.npy ...")
        self.all_labels = np.load(f"{self.config.features_dir}/all_labels.npy")
        self.logger.info(f"    Shape: {self.all_labels.shape}")
        self.logger.info(f"    Distribution: {np.bincount(self.all_labels)}")

        # 加载metadata
        self.logger.info("  加载 all_metadata.csv ...")
        self.all_metadata = pd.read_csv(f"{self.config.features_dir}/all_metadata.csv")
        self.logger.info(f"    Shape: {self.all_metadata.shape}")
        self.logger.info(f"    Columns: {list(self.all_metadata.columns)}")

        # 加载5-Fold test predictions
        self.logger.info("  加载5-Fold test predictions ...")
        self.fold_predictions = {}
        for fold in range(1, self.config.n_folds + 1):
            pred_file = f"{self.config.fold_results_dir}/fold_{fold}/test_predictions.csv"
            self.fold_predictions[fold] = pd.read_csv(pred_file)
            self.logger.info(f"    Fold {fold}: {len(self.fold_predictions[fold]):,} samples")

        # 重建5-Fold划分，获取test indices
        self.logger.info("  重建5-Fold划分 ...")
        self._reconstruct_folds()

    def _reconstruct_folds(self):
        """
        使用StratifiedKFold重建划分，获取每折的test indices
        """
        skf = StratifiedKFold(
            n_splits=self.config.n_folds,
            shuffle=True,
            random_state=self.config.random_state
        )

        self.test_indices = {}  # fold -> test indices
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(self.all_features, self.all_labels), 1):
            self.test_indices[fold_idx] = test_idx
            self.logger.info(f"    Fold {fold_idx}: test indices [{test_idx[0]}...{test_idx[-1]}], "
                           f"n={len(test_idx):,}")

        # 验证一致性
        self._verify_fold_consistency()

    def _verify_fold_consistency(self):
        """验证重建的fold划分与test_predictions一致"""
        self.logger.info("  验证fold一致性 ...")

        for fold in range(1, self.config.n_folds + 1):
            test_idx = self.test_indices[fold]
            pred_df = self.fold_predictions[fold]

            # 检查true_label是否一致
            true_from_data = self.all_labels[test_idx]
            true_from_pred = pred_df['true_label'].values

            if len(true_from_data) != len(true_from_pred):
                self.logger.warning(f"    Fold {fold}: 长度不一致! "
                                  f"data={len(true_from_data)}, pred={len(true_from_pred)}")
                continue

            match = np.all(true_from_data == true_from_pred)
            self.logger.info(f"    Fold {fold}: true_label匹配 = {match}")

            if not match:
                # 尝试找到正确的对应关系
                self.logger.warning(f"    Fold {fold}: 尝试重新对齐...")
                # 可能是最后一条数据被排除
                if len(true_from_pred) < len(true_from_data):
                    true_from_data = true_from_data[:len(true_from_pred)]
                    match = np.all(true_from_data == true_from_pred)
                    self.logger.info(f"    Fold {fold}: 截断后匹配 = {match}")

    def _load_positive_reference(self):
        """加载正样本参考库（用于邻居检索）"""
        self.logger.info("\n[Step 2] 加载正样本参考库")

        # 加载正样本ESM2特征
        self.logger.info("  加载正样本ESM2特征 ...")
        self.pos_features = np.load(self.config.positive_features).astype('float32')
        self.logger.info(f"    Shape: {self.pos_features.shape}")

        # 加载正样本gene_ids
        self.logger.info("  加载正样本gene_ids ...")
        pos_ids_df = pd.read_csv(self.config.positive_gene_ids)
        self.pos_gene_ids = pos_ids_df['gene_id'].tolist()
        self.logger.info(f"    Count: {len(self.pos_gene_ids):,}")

        # 加载正样本metadata（用于功能注释）
        self.logger.info("  加载正样本metadata ...")
        self.pos_metadata = pd.read_csv(
            self.config.positive_metadata,
            sep='\t',
            low_memory=False,
            usecols=['Entry_ID', 'CAZy_Families', 'EC_Numbers']
        )
        self.logger.info(f"    Shape: {self.pos_metadata.shape}")

        # 建立gene_id到metadata索引的映射
        self.pos_geneid_to_metaidx = {}
        for idx, row in self.pos_metadata.iterrows():
            gid = str(row['Entry_ID']).strip()
            self.pos_geneid_to_metaidx[gid] = idx

        # 构建FAISS索引
        self._build_faiss_index()

    def _build_faiss_index(self):
        """构建FAISS索引用于邻居检索"""
        self.logger.info("  构建FAISS索引 ...")

        try:
            import faiss

            dim = self.pos_features.shape[1]
            faiss.normalize_L2(self.pos_features)

            if faiss.get_num_gpus() > 0:
                self.logger.info("    使用FAISS-GPU")
                res = faiss.StandardGpuResources()
                cpu_index = faiss.IndexFlatIP(dim)
                cpu_index.add(self.pos_features)
                self.faiss_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
            else:
                self.logger.info("    使用FAISS-CPU")
                self.faiss_index = faiss.IndexFlatIP(dim)
                self.faiss_index.add(self.pos_features)

            self.faiss_available = True
            self.logger.info(f"    索引构建完成: {self.faiss_index.ntotal:,} vectors")

        except ImportError:
            self.logger.warning("    FAISS未安装，使用sklearn cosine_similarity")
            self.faiss_available = False
            self.faiss_index = None

    def _import_model(self):
        """导入GNN模型定义"""
        self.logger.info("\n[Step 3] 导入模型定义")
        sys.path.insert(0, os.path.dirname(self.config.model_training_path))
        import model_training
        self.AttentionGNN = model_training.AttentionGNN
        self.GNNConfig = model_training.GNNConfig
        self.logger.info("  模型定义导入成功")

    # ---------- 核心：计算Confidence Score ----------

    def compute_confidence_scores(self) -> pd.DataFrame:
        """
        为所有验证样本计算confidence score

        返回DataFrame，包含:
        - gene_id, true_label, pred_label, prob_0/1/2
        - confidence_score, consistency, excess_similarity
        - predicted_cazy, predicted_ec
        """
        self.logger.info("\n[Step 4] 计算Confidence Scores")

        all_results = []

        for fold in range(1, self.config.n_folds + 1):
            self.logger.info(f"\n  处理 Fold {fold}/{self.config.n_folds}")

            # 获取该fold的test indices
            test_idx = self.test_indices[fold]
            pred_df = self.fold_predictions[fold]

            # 处理长度不一致（如果存在）
            if len(test_idx) != len(pred_df):
                self.logger.warning(f"    长度不一致: test_idx={len(test_idx)}, pred_df={len(pred_df)}")
                min_len = min(len(test_idx), len(pred_df))
                test_idx = test_idx[:min_len]

            # 获取test样本的ESM2特征（前1280维）
            test_features = self.all_features[test_idx][:, :1280].astype('float32')
            test_gene_ids = self.all_metadata.iloc[test_idx]['gene_id'].values

            self.logger.info(f"    Test features: {test_features.shape}")

            # FAISS检索邻居
            self.logger.info("    FAISS检索邻居 ...")
            neighbor_indices, neighbor_sims = self._search_neighbors(test_features)

            # 计算confidence score
            self.logger.info("    计算confidence scores ...")
            conf_results = self._calculate_confidence(
                test_gene_ids, neighbor_indices, neighbor_sims
            )

            # 合并预测结果和confidence结果
            fold_result = pd.DataFrame({
                'gene_id': test_gene_ids,
                'fold': fold,
                'true_label': pred_df['true_label'].values[:len(test_idx)],
                'pred_label': pred_df['pred_label'].values[:len(test_idx)],
                'prob_0': pred_df['prob_0'].values[:len(test_idx)],
                'prob_1': pred_df['prob_1'].values[:len(test_idx)],
                'prob_2': pred_df['prob_2'].values[:len(test_idx)],
                'confidence_score': conf_results['confidence_score'],
                'consistency': conf_results['consistency'],
                'excess_similarity': conf_results['excess_similarity'],
                'predicted_cazy': conf_results['predicted_cazy'],
                'predicted_ec': conf_results['predicted_ec'],
                'n_neighbors_used': conf_results['n_neighbors_used'],
            })

            all_results.append(fold_result)
            self.logger.info(f"    Fold {fold}完成: {len(fold_result):,} samples")

        # 合并所有fold
        self.validation_df = pd.concat(all_results, ignore_index=True)
        self.logger.info(f"\n  总计验证样本: {len(self.validation_df):,}")

        # 保存
        output_path = f"{self.config.output_dir}/validation_with_confidence.csv"
        self.validation_df.to_csv(output_path, index=False)
        self.logger.info(f"  保存到: {output_path}")

        return self.validation_df

    def _search_neighbors(self, query_features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """FAISS检索邻居"""
        if self.faiss_available:
            import faiss
            faiss.normalize_L2(query_features)
            distances, indices = self.faiss_index.search(
                query_features, self.config.faiss_top_k
            )
            return indices, distances
        else:
            # CPU fallback
            from sklearn.metrics.pairwise import cosine_similarity
            sims = cosine_similarity(query_features, self.pos_features)
            indices = np.argsort(-sims, axis=1)[:, :self.config.faiss_top_k]
            distances = np.take_along_axis(sims, indices, axis=1)
            return indices, distances

    def _calculate_confidence(self,
                              query_gene_ids: np.ndarray,
                              neighbor_indices: np.ndarray,
                              neighbor_sims: np.ndarray) -> Dict:
        """
        计算confidence score（与prediction_pipeline一致）
        """
        n = len(query_gene_ids)

        confidence_scores = np.zeros(n)
        consistencies = np.zeros(n)
        excess_similarities = np.zeros(n)
        predicted_cazy_list = []
        predicted_ec_list = []
        n_neighbors_used = np.zeros(n, dtype=int)

        for i in tqdm(range(n), desc="Confidence calc", leave=False):
            # 收集有效邻居
            valid_neighbors = []

            for row_idx, sim in zip(neighbor_indices[i], neighbor_sims[i]):
                row_idx = int(row_idx)
                if row_idx >= len(self.pos_gene_ids):
                    continue

                ref_gene_id = self.pos_gene_ids[row_idx]
                meta_idx = self.pos_geneid_to_metaidx.get(ref_gene_id)
                if meta_idx is None:
                    continue

                row = self.pos_metadata.iloc[meta_idx]

                # 提取CAZy
                cazy_fams = []
                if pd.notna(row.get('CAZy_Families')):
                    val = str(row['CAZy_Families'])
                    if val not in ('NA', 'nan', '-', ''):
                        cazy_fams = [f.strip() for f in val.split('|') if f.strip()]

                # 提取EC
                ec_nums = []
                if pd.notna(row.get('EC_Numbers')):
                    val = str(row['EC_Numbers'])
                    if val not in ('NA', 'nan', '-', ''):
                        ec_nums = [e.strip() for e in val.split('|') if e.strip()]

                if len(cazy_fams) > 0 or len(ec_nums) > 0:
                    valid_neighbors.append({
                        'similarity': float(sim),
                        'cazy': cazy_fams,
                        'ec': ec_nums,
                    })

                if len(valid_neighbors) >= self.config.top_neighbors_for_consensus:
                    break

            n_neighbors_used[i] = len(valid_neighbors)

            if len(valid_neighbors) < 3:
                confidence_scores[i] = 0.0
                predicted_cazy_list.append('NA')
                predicted_ec_list.append('NA')
                continue

            # 共识投票
            top_k = min(self.config.top_neighbors_for_consensus, len(valid_neighbors))
            cazy_counter = Counter()
            for n in valid_neighbors[:top_k]:
                for fam in n['cazy']:
                    cazy_counter[fam] += 1

            # 一致性
            if cazy_counter:
                consistency = max(cazy_counter.values()) / top_k
            else:
                consistency = 0.0

            # 超额相似度
            avg_sim = np.mean([n['similarity'] for n in valid_neighbors[:top_k]])
            excess_sim = max(0, avg_sim - self.config.similarity_baseline) * self.config.excess_similarity_scale
            excess_sim = min(excess_sim, 1.0)

            # 置信度
            w_con = self.config.consistency_weight
            w_excess = self.config.excess_similarity_weight
            confidence = consistency * w_con + excess_sim * w_excess
            confidence = min(confidence, 1.0)

            confidence_scores[i] = confidence
            consistencies[i] = consistency
            excess_similarities[i] = excess_sim

            # 预测CAZy和EC
            consensus_cazy = [fam for fam, count in cazy_counter.items()
                              if count / top_k >= self.config.min_consensus_ratio]
            predicted_cazy_list.append('|'.join(consensus_cazy) if consensus_cazy else 'NA')

            # 最佳EC
            ec_counter = Counter()
            for n in valid_neighbors[:top_k]:
                for ec in n['ec']:
                    ec_counter[ec] += 1

            if ec_counter:
                best_ec = sorted(ec_counter.items(),
                                key=lambda x: (x[0].count('.'), x[1]),
                                reverse=True)[0][0]
                predicted_ec_list.append(best_ec)
            else:
                predicted_ec_list.append('NA')

        return {
            'confidence_score': confidence_scores,
            'consistency': consistencies,
            'excess_similarity': excess_similarities,
            'predicted_cazy': predicted_cazy_list,
            'predicted_ec': predicted_ec_list,
            'n_neighbors_used': n_neighbors_used,
        }

    # ---------- 分析1: Precision-Recall分析 ----------

    def analyze_precision_recall(self):
        """
        精度-召回分析
        对GNN预测为positive (class 1)的样本，评估confidence score与真实precision的关系
        """
        self.logger.info("\n[Step 5] Precision-Recall分析")

        df = self.validation_df

        # 筛选GNN预测为positive的样本
        pos_mask = (df['pred_label'] == 1)
        pos_df = df[pos_mask].copy()

        self.logger.info(f"  GNN预测为Positive的样本: {len(pos_df):,}")

        # 在这些样本中，true_label==1的是true positive，true_label==0的是false positive
        # true_label==2 (hard)不参与计算（未知）
        eval_mask = pos_df['true_label'].isin([0, 1])
        eval_df = pos_df[eval_mask].copy()

        self.logger.info(f"  可评估的样本 (true_label=0或1): {len(eval_df):,}")
        self.logger.info(f"    True Positive (true_label=1): {(eval_df['true_label']==1).sum():,}")
        self.logger.info(f"    False Positive (true_label=0): {(eval_df['true_label']==0).sum():,}")

        if len(eval_df) == 0 or (eval_df['true_label'] == 1).sum() == 0:
            self.logger.error("  可评估样本不足，无法计算PR曲线")
            return {}

        y_true = (eval_df['true_label'] == 1).astype(int).values
        y_scores = eval_df['confidence_score'].values

        # 计算PR曲线
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        ap = average_precision_score(y_true, y_scores)

        self.logger.info(f"  Average Precision (AP): {ap:.4f}")

        # 计算F1
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        best_f1_idx = np.argmax(f1_scores)
        best_f1 = f1_scores[best_f1_idx]
        best_threshold = thresholds[best_f1_idx] if best_f1_idx < len(thresholds) else 1.0

        self.logger.info(f"  最优F1: {best_f1:.4f} @ threshold={best_threshold:.4f}")
        self.logger.info(f"    Precision={precision[best_f1_idx]:.4f}, Recall={recall[best_f1_idx]:.4f}")

        # 保存结果
        self.pr_results = {
            'precision': precision,
            'recall': recall,
            'thresholds': thresholds,
            'f1_scores': f1_scores,
            'average_precision': ap,
            'best_f1': best_f1,
            'best_threshold': best_threshold,
            'n_evaluable': len(eval_df),
            'n_true_positive': int((eval_df['true_label']==1).sum()),
            'n_false_positive': int((eval_df['true_label']==0).sum()),
        }

        return self.pr_results

    # ---------- 分析2: Tier阈值验证 ----------

    def validate_tier_thresholds(self):
        """
        验证当前Tier阈值下的实际precision
        """
        self.logger.info("\n[Step 6] Tier阈值验证")

        df = self.validation_df
        pos_mask = (df['pred_label'] == 1)
        eval_mask = pos_mask & df['true_label'].isin([0, 1])
        eval_df = df[eval_mask].copy()

        if len(eval_df) == 0:
            self.logger.error("  可评估样本不足")
            return {}

        t1 = self.config.tier1_threshold
        t2 = self.config.tier2_threshold
        t3 = self.config.tier3_threshold

        tier_masks = {
            'Tier_1': eval_df['confidence_score'] >= t1,
            'Tier_2': (eval_df['confidence_score'] >= t2) & (eval_df['confidence_score'] < t1),
            'Tier_3': (eval_df['confidence_score'] >= t3) & (eval_df['confidence_score'] < t2),
            'Tier_Low': eval_df['confidence_score'] < t3,
        }

        tier_stats = {}

        self.logger.info("  当前Tier阈值下的实际性能:")
        self.logger.info("  " + "-" * 70)
        self.logger.info(f"  {'Tier':<12} {'Threshold':<20} {'N':<10} {'Precision':<12} {'Recall':<12} {'F1':<10}")
        self.logger.info("  " + "-" * 70)

        for tier_name, mask in tier_masks.items():
            tier_df = eval_df[mask]
            n = len(tier_df)

            if n == 0:
                self.logger.info(f"  {tier_name:<12} {'N/A':<20} {0:<10} {'N/A':<12} {'N/A':<12} {'N/A':<10}")
                tier_stats[tier_name] = {'n': 0, 'precision': None, 'recall': None, 'f1': None}
                continue

            y_true = (tier_df['true_label'] == 1).astype(int).values
            y_pred = np.ones(len(y_true))  # 预测为positive

            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)

            threshold_str = {
                'Tier_1': f'≥ {t1:.2f}',
                'Tier_2': f'{t2:.2f} - {t1:.2f}',
                'Tier_3': f'{t3:.2f} - {t2:.2f}',
                'Tier_Low': f'< {t3:.2f}',
            }[tier_name]

            self.logger.info(f"  {tier_name:<12} {threshold_str:<20} {n:<10} {prec:<12.4f} {rec:<12.4f} {f1:<10.4f}")

            tier_stats[tier_name] = {
                'n': int(n),
                'precision': float(prec),
                'recall': float(rec),
                'f1': float(f1),
                'threshold': threshold_str,
            }

        self.logger.info("  " + "-" * 70)

        self.tier_stats = tier_stats
        return tier_stats

    # ---------- 分析3: 权重消融实验 ----------

    def weight_ablation(self):
        """
        权重消融实验：测试不同Wc/We组合的效果
        """
        self.logger.info("\n[Step 7] 权重消融实验")

        df = self.validation_df
        pos_mask = (df['pred_label'] == 1)
        eval_mask = pos_mask & df['true_label'].isin([0, 1])
        eval_df = df[eval_mask].copy()

        if len(eval_df) == 0:
            self.logger.error("  可评估样本不足")
            return {}

        y_true = (eval_df['true_label'] == 1).astype(int).values
        consistency = eval_df['consistency'].values
        excess_sim = eval_df['excess_similarity'].values

        results = []

        self.logger.info(f"  测试权重组合 (约束: Wc + We ≤ 1.0):")
        self.logger.info("  " + "-" * 50)
        self.logger.info(f"  {'Wc':<6} {'We':<6} {'Avg Precision':<15} {'Best F1':<12} {'Best Thresh':<12}")
        self.logger.info("  " + "-" * 50)

        for wc in self.config.weight_grid_wc:
            for we in self.config.weight_grid_we:
                if wc + we > 1.0:
                    continue

                # 重新计算confidence score
                conf = consistency * wc + excess_sim * we
                conf = np.clip(conf, 0, 1)

                # 计算PR
                precision, recall, thresholds = precision_recall_curve(y_true, conf)
                ap = average_precision_score(y_true, conf)

                f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
                best_idx = np.argmax(f1_scores)
                best_f1 = f1_scores[best_idx]
                best_thresh = thresholds[best_idx] if best_idx < len(thresholds) else 1.0

                results.append({
                    'wc': wc,
                    'we': we,
                    'ap': ap,
                    'best_f1': best_f1,
                    'best_threshold': best_thresh,
                })

                marker = " <-- 当前" if (wc == self.config.consistency_weight and
                                         we == self.config.excess_similarity_weight) else ""
                self.logger.info(f"  {wc:<6.1f} {we:<6.1f} {ap:<15.4f} {best_f1:<12.4f} {best_thresh:<12.4f}{marker}")

        self.logger.info("  " + "-" * 50)

        # 找到最优组合
        best = max(results, key=lambda x: x['best_f1'])
        self.logger.info(f"\n  最优组合: Wc={best['wc']}, We={best['we']} "
                        f"(F1={best['best_f1']:.4f}, AP={best['ap']:.4f})")

        self.weight_results = results
        return results

    # ---------- 分析4: 阈值敏感性分析 ----------

    def threshold_sensitivity(self):
        """
        测试不同Tier阈值组合的效果
        """
        self.logger.info("\n[Step 8] 阈值敏感性分析")

        df = self.validation_df
        pos_mask = (df['pred_label'] == 1)
        eval_mask = pos_mask & df['true_label'].isin([0, 1])
        eval_df = df[eval_mask].copy()

        y_true = (eval_df['true_label'] == 1).astype(int).values
        conf = eval_df['confidence_score'].values

        # 测试几组阈值
        threshold_configs = [
            (0.70, 0.50, 0.30, "严格"),
            (0.60, 0.40, 0.25, "当前"),
            (0.55, 0.35, 0.20, "宽松"),
            (0.50, 0.30, 0.15, "更宽松"),
        ]

        self.logger.info("  不同阈值配置下的Tier 1+2性能:")
        self.logger.info("  " + "-" * 60)
        self.logger.info(f"  {'配置':<10} {'Tier1':<8} {'Tier2':<8} {'Tier3':<8} {'Tier1+2 N':<12} {'Tier1+2 Prec':<12} {'Tier1+2 Rec':<12}")
        self.logger.info("  " + "-" * 60)

        for t1, t2, t3, name in threshold_configs:
            # Tier 1 + Tier 2
            tier12_mask = conf >= t2
            tier12_y_true = y_true[tier12_mask]

            if len(tier12_y_true) == 0:
                continue

            n = len(tier12_y_true)
            prec = precision_score(tier12_y_true, np.ones(n), zero_division=0)
            rec = recall_score(tier12_y_true, np.ones(n), zero_division=0)

            marker = " <-- 当前" if name == "当前" else ""
            self.logger.info(f"  {name:<10} {t1:<8.2f} {t2:<8.2f} {t3:<8.2f} {n:<12} {prec:<12.4f} {rec:<12.4f}{marker}")

        self.logger.info("  " + "-" * 60)

    # ---------- 可视化 ----------

    def generate_plots(self):
        """生成所有可视化图表"""
        self.logger.info("\n[Step 9] 生成可视化图表")

        self._plot_precision_recall_curve()
        self._plot_confidence_distribution()
        self._plot_tier_validation()
        self._plot_weight_ablation()
        self._plot_confidence_calibration()

    def _plot_precision_recall_curve(self):
        """PR曲线"""
        if not hasattr(self, 'pr_results'):
            return

        pr = self.pr_results

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # 左图：PR曲线
        ax1.plot(pr['recall'], pr['precision'], 'b-', linewidth=2,
                label=f"AP = {pr['average_precision']:.3f}")
        ax1.fill_between(pr['recall'], pr['precision'], alpha=0.3)

        # 标记当前Tier阈值
        t1, t2, t3 = self.config.tier1_threshold, self.config.tier2_threshold, self.config.tier3_threshold

        for thresh, color, name in [(t1, 'green', 'Tier 1'),
                                     (t2, 'orange', 'Tier 2'),
                                     (t3, 'blue', 'Tier 3')]:
            idx = np.argmin(np.abs(pr['thresholds'] - thresh)) if len(pr['thresholds']) > 0 else 0
            if idx < len(pr['precision']):
                ax1.plot(pr['recall'][idx], pr['precision'][idx], 'o',
                        color=color, markersize=10, label=f"{name} (θ={thresh:.2f})")

        ax1.set_xlabel('Recall', fontsize=12)
        ax1.set_ylabel('Precision', fontsize=12)
        ax1.set_title('(a) Precision-Recall Curve', fontsize=13, fontweight='bold')
        ax1.legend(loc='lower left')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([0, 1])
        ax1.set_ylim([0, 1.05])

        # 右图：F1 vs Threshold
        ax2.plot(pr['thresholds'], pr['f1_scores'][:-1], 'r-', linewidth=2)
        ax2.axvline(t1, color='green', linestyle='--', label=f'Tier 1 (θ={t1:.2f})')
        ax2.axvline(t2, color='orange', linestyle='--', label=f'Tier 2 (θ={t2:.2f})')
        ax2.axvline(t3, color='blue', linestyle='--', label=f'Tier 3 (θ={t3:.2f})')

        best_idx = np.argmax(pr['f1_scores'])
        ax2.plot(pr['best_threshold'], pr['f1_scores'][best_idx], 'r*',
                markersize=15, label=f"Best F1={pr['best_f1']:.3f}")

        ax2.set_xlabel('Confidence Threshold', fontsize=12)
        ax2.set_ylabel('F1 Score', fontsize=12)
        ax2.set_title('(b) F1 Score vs Threshold', fontsize=13, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        path = f"{self.config.output_dir}/precision_recall_analysis.png"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"  保存: {path}")

    def _plot_confidence_distribution(self):
        """置信度分布"""
        df = self.validation_df
        pos_mask = (df['pred_label'] == 1)
        eval_df = df[pos_mask & df['true_label'].isin([0, 1])]

        fig, ax = plt.subplots(figsize=(10, 6))

        # True positive的置信度分布
        tp_conf = eval_df[eval_df['true_label'] == 1]['confidence_score'].values
        fp_conf = eval_df[eval_df['true_label'] == 0]['confidence_score'].values

        ax.hist(tp_conf, bins=50, alpha=0.6, color='green', label=f'True Positive (n={len(tp_conf)})')
        ax.hist(fp_conf, bins=50, alpha=0.6, color='red', label=f'False Positive (n={len(fp_conf)})')

        # Tier阈值线
        t1, t2, t3 = self.config.tier1_threshold, self.config.tier2_threshold, self.config.tier3_threshold
        ax.axvline(t1, color='green', linestyle='--', linewidth=2, label=f'Tier 1 (θ={t1:.2f})')
        ax.axvline(t2, color='orange', linestyle='--', linewidth=2, label=f'Tier 2 (θ={t2:.2f})')
        ax.axvline(t3, color='blue', linestyle='--', linewidth=2, label=f'Tier 3 (θ={t3:.2f})')

        ax.set_xlabel('Confidence Score', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Confidence Score Distribution by True Label', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = f"{self.config.output_dir}/confidence_distribution.png"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"  保存: {path}")

    def _plot_tier_validation(self):
        """Tier验证图"""
        if not hasattr(self, 'tier_stats'):
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        tiers = ['Tier_1', 'Tier_2', 'Tier_3', 'Tier_Low']
        precisions = [self.tier_stats.get(t, {}).get('precision', 0) or 0 for t in tiers]
        recalls = [self.tier_stats.get(t, {}).get('recall', 0) or 0 for t in tiers]
        counts = [self.tier_stats.get(t, {}).get('n', 0) for t in tiers]

        x = np.arange(len(tiers))
        width = 0.35

        bars1 = ax.bar(x - width/2, precisions, width, label='Precision', color='steelblue')
        bars2 = ax.bar(x + width/2, recalls, width, label='Recall', color='coral')

        # 在bar上标注count
        for i, (bar, count) in enumerate(zip(bars1, counts)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'n={count:,}',
                   ha='center', va='bottom', fontsize=9)

        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Tier-wise Precision and Recall (Validation Set)', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(tiers)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1.05])

        plt.tight_layout()
        path = f"{self.config.output_dir}/tier_threshold_validation.png"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"  保存: {path}")

    def _plot_weight_ablation(self):
        """权重消融图"""
        if not hasattr(self, 'weight_results'):
            return

        results = self.weight_results

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # 左图：AP热图
        wc_vals = sorted(set(r['wc'] for r in results))
        we_vals = sorted(set(r['we'] for r in results))

        ap_matrix = np.zeros((len(wc_vals), len(we_vals)))
        for r in results:
            i = wc_vals.index(r['wc'])
            j = we_vals.index(r['we'])
            ap_matrix[i, j] = r['ap']

        im = ax1.imshow(ap_matrix, cmap='YlOrRd', aspect='auto')
        ax1.set_xticks(range(len(we_vals)))
        ax1.set_yticks(range(len(wc_vals)))
        ax1.set_xticklabels([f'{w:.1f}' for w in we_vals])
        ax1.set_yticklabels([f'{w:.1f}' for w in wc_vals])
        ax1.set_xlabel('We (Excess Similarity Weight)', fontsize=12)
        ax1.set_ylabel('Wc (Consistency Weight)', fontsize=12)
        ax1.set_title('(a) Average Precision', fontsize=13, fontweight='bold')

        # 标注数值
        for i in range(len(wc_vals)):
            for j in range(len(we_vals)):
                text = ax1.text(j, i, f'{ap_matrix[i, j]:.3f}',
                               ha="center", va="center", color="black", fontsize=9)

        plt.colorbar(im, ax=ax1)

        # 右图：Best F1热图
        f1_matrix = np.zeros((len(wc_vals), len(we_vals)))
        for r in results:
            i = wc_vals.index(r['wc'])
            j = we_vals.index(r['we'])
            f1_matrix[i, j] = r['best_f1']

        im2 = ax2.imshow(f1_matrix, cmap='YlGnBu', aspect='auto')
        ax2.set_xticks(range(len(we_vals)))
        ax2.set_yticks(range(len(wc_vals)))
        ax2.set_xticklabels([f'{w:.1f}' for w in we_vals])
        ax2.set_yticklabels([f'{w:.1f}' for w in wc_vals])
        ax2.set_xlabel('We (Excess Similarity Weight)', fontsize=12)
        ax2.set_ylabel('Wc (Consistency Weight)', fontsize=12)
        ax2.set_title('(b) Best F1 Score', fontsize=13, fontweight='bold')

        for i in range(len(wc_vals)):
            for j in range(len(we_vals)):
                text = ax2.text(j, i, f'{f1_matrix[i, j]:.3f}',
                               ha="center", va="center", color="black", fontsize=9)

        plt.colorbar(im2, ax=ax2)

        plt.tight_layout()
        path = f"{self.config.output_dir}/weight_ablation.png"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"  保存: {path}")

    def _plot_confidence_calibration(self):
        """置信度校准图（confidence bin vs actual precision）"""
        df = self.validation_df
        pos_mask = (df['pred_label'] == 1)
        eval_df = df[pos_mask & df['true_label'].isin([0, 1])]

        if len(eval_df) == 0:
            return

        # 分10个bin
        n_bins = 10
        eval_df = eval_df.copy()
        eval_df['conf_bin'] = pd.qcut(eval_df['confidence_score'], q=n_bins, duplicates='drop')

        bin_stats = eval_df.groupby('conf_bin').agg({
            'true_label': ['count', lambda x: (x == 1).sum() / len(x)],
            'confidence_score': 'mean'
        }).reset_index()

        bin_stats.columns = ['bin', 'count', 'actual_precision', 'mean_confidence']

        fig, ax = plt.subplots(figsize=(10, 6))

        x = range(len(bin_stats))
        ax.plot(x, bin_stats['actual_precision'], 'o-', color='blue', linewidth=2,
               label='Actual Precision', markersize=8)
        ax.plot(x, bin_stats['mean_confidence'], 's--', color='red', linewidth=2,
               label='Mean Confidence', markersize=8)

        # 理想校准线
        ax.plot(x, bin_stats['mean_confidence'], 'k:', alpha=0.5, label='Perfect Calibration')

        ax.set_xlabel('Confidence Decile Bin', fontsize=12)
        ax.set_ylabel('Precision / Confidence', fontsize=12)
        ax.set_title('Confidence Calibration: Predicted vs Actual Precision', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'Bin {i+1}' for i in x], rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])

        # 在点上标注count
        for i, (row) in enumerate(bin_stats.itertuples()):
            ax.annotate(f'n={row.count}', (i, row.actual_precision),
                       textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8)

        plt.tight_layout()
        path = f"{self.config.output_dir}/confidence_calibration.png"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"  保存: {path}")

    # ---------- 报告生成 ----------

    def generate_report(self):
        """生成校准报告"""
        self.logger.info("\n[Step 10] 生成报告")

        report = {
            'timestamp': datetime.now().isoformat(),
            'configuration': {
                'consistency_weight': self.config.consistency_weight,
                'excess_similarity_weight': self.config.excess_similarity_weight,
                'similarity_baseline': self.config.similarity_baseline,
                'excess_similarity_scale': self.config.excess_similarity_scale,
                'tier1_threshold': self.config.tier1_threshold,
                'tier2_threshold': self.config.tier2_threshold,
                'tier3_threshold': self.config.tier3_threshold,
            },
            'validation_summary': {
                'total_samples': len(self.validation_df),
                'n_folds': self.config.n_folds,
            },
        }

        if hasattr(self, 'pr_results'):
            report['precision_recall_analysis'] = {
                'average_precision': float(self.pr_results['average_precision']),
                'best_f1': float(self.pr_results['best_f1']),
                'best_threshold': float(self.pr_results['best_threshold']),
                'n_evaluable': self.pr_results['n_evaluable'],
                'n_true_positive': self.pr_results['n_true_positive'],
                'n_false_positive': self.pr_results['n_false_positive'],
            }

        if hasattr(self, 'tier_stats'):
            report['tier_threshold_validation'] = self.tier_stats

        if hasattr(self, 'weight_results'):
            report['weight_ablation'] = [
                {
                    'wc': r['wc'],
                    'we': r['we'],
                    'average_precision': float(r['ap']),
                    'best_f1': float(r['best_f1']),
                    'best_threshold': float(r['best_threshold']),
                }
                for r in self.weight_results
            ]
            best = max(self.weight_results, key=lambda x: x['best_f1'])
            report['optimal_weights'] = {
                'wc': best['wc'],
                'we': best['we'],
                'best_f1': float(best['best_f1']),
                'average_precision': float(best['ap']),
            }

        # 保存JSON
        json_path = f"{self.config.output_dir}/calibration_report.json"
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2)
        self.logger.info(f"  JSON报告: {json_path}")

        # 生成Markdown报告
        md = self._generate_markdown_report()
        md_path = f"{self.config.output_dir}/calibration_report.md"
        with open(md_path, 'w') as f:
            f.write(md)
        self.logger.info(f"  Markdown报告: {md_path}")

    def _generate_markdown_report(self) -> str:
        """生成Markdown格式报告（用于论文）"""

        pr = getattr(self, 'pr_results', {})
        tier = getattr(self, 'tier_stats', {})

        md = f"""# CAAC Confidence Calibration Report

## 1. 实验设计

基于5-Fold交叉验证的验证集（n={len(self.validation_df):,}），对CAAC框架的置信度评分系统和Tier阈值进行经验验证。

## 2. 验证数据

| 指标 | 数值 |
|------|------|
| 总验证样本 | {len(self.validation_df):,} |
| 5-Fold覆盖 | 每折~172,000样本 |
| GNN预测Positive样本 | {len(self.validation_df[self.validation_df['pred_label']==1]):,} |
| 可评估样本（有ground truth） | {pr.get('n_evaluable', 'N/A'):,} |
| True Positive | {pr.get('n_true_positive', 'N/A'):,} |
| False Positive | {pr.get('n_false_positive', 'N/A'):,} |

## 3. 精度-召回分析

| 指标 | 数值 |
|------|------|
| Average Precision (AP) | {pr.get('average_precision', 'N/A'):.4f} |
| 最优F1 | {pr.get('best_f1', 'N/A'):.4f} |
| 最优阈值 | {pr.get('best_threshold', 'N/A'):.4f} |

## 4. Tier阈值验证结果

| Tier | 阈值范围 | 样本数 | 实际精度 | 实际召回 | 实际F1 |
|------|---------|--------|---------|---------|--------|
"""

        for tier_name in ['Tier_1', 'Tier_2', 'Tier_3', 'Tier_Low']:
            if tier_name in tier:
                t = tier[tier_name]
                md += f"| {tier_name} | {t.get('threshold', 'N/A')} | {t.get('n', 0):,} | "
                md += f"{t.get('precision', 'N/A') if t.get('precision') is not None else 'N/A'} | "
                md += f"{t.get('recall', 'N/A') if t.get('recall') is not None else 'N/A'} | "
                md += f"{t.get('f1', 'N/A') if t.get('f1') is not None else 'N/A'} |\n"
            else:
                md += f"| {tier_name} | N/A | 0 | N/A | N/A | N/A |\n"

        md += f"""
## 5. 权重消融实验

测试了以下权重组合（约束: Wc + We ≤ 1.0）：

| Wc | We | Average Precision | Best F1 | Best Threshold |
|----|-----|------------------|---------|---------------|
"""

        if hasattr(self, 'weight_results'):
            for r in sorted(self.weight_results, key=lambda x: (x['wc'], x['we'])):
                marker = " **(当前)**" if (r['wc'] == self.config.consistency_weight and
                                          r['we'] == self.config.excess_similarity_weight) else ""
                md += f"| {r['wc']:.1f} | {r['we']:.1f} | {r['ap']:.4f} | {r['best_f1']:.4f} | {r['best_threshold']:.4f}{marker} |\n"

        if hasattr(self, 'weight_results'):
            best = max(self.weight_results, key=lambda x: x['best_f1'])
            md += f"""
**最优权重组合**: Wc={best['wc']}, We={best['we']} (F1={best['best_f1']:.4f}, AP={best['ap']:.4f})
"""

        md += f"""
## 6. 结论与建议

### 6.1 当前参数验证结果

- **Tier 1 (≥{self.config.tier1_threshold})**: 实际精度 = {tier.get('Tier_1', {}).get('precision', 'N/A')}
- **Tier 2 ({self.config.tier2_threshold}-{self.config.tier1_threshold})**: 实际精度 = {tier.get('Tier_2', {}).get('precision', 'N/A')}
- **Tier 3 ({self.config.tier3_threshold}-{self.config.tier2_threshold})**: 实际精度 = {tier.get('Tier_3', {}).get('precision', 'N/A')}

### 6.2 参数优化建议

"""

        if hasattr(self, 'weight_results'):
            best = max(self.weight_results, key=lambda x: x['best_f1'])
            if best['wc'] != self.config.consistency_weight or best['we'] != self.config.excess_similarity_weight:
                md += f"""
**建议调整权重**: 当前 Wc={self.config.consistency_weight}, We={self.config.excess_similarity_weight} 并非最优。
验证数据显示 Wc={best['wc']}, We={best['we']} 可达到更高的F1分数（{best['best_f1']:.4f} vs 当前最优F1）。
"""
            else:
                md += """
**当前权重已接近最优**: Wc=0.8, We=0.2 在验证集上表现良好。
"""

        md += f"""
### 6.3 下游使用建议

1. **Tier 1预测**: 可直接整合到KEGG注释，预期假阳性率可控
2. **Tier 2预测**: 建议整合但标记为"computational prediction"
3. **Tier 3预测**: 仅用于探索性分析
4. **Tier Low**: 排除

## 7. 局限性

1. 验证基于5-Fold CV的test set，与最终预测数据分布可能略有差异
2. 权重消融实验的网格粒度有限（步长0.1）
3. Hard序列（true_label=2）不参与精度计算，因其真实功能未知

---
*Generated by CAAC Calibration Module*
*Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
"""
        return md

    # ---------- 主流程 ----------

    def run(self):
        """执行完整校准流程"""

        # Step 4: 计算Confidence Scores
        self.compute_confidence_scores()

        # Step 5: PR分析
        self.analyze_precision_recall()

        # Step 6: Tier阈值验证
        self.validate_tier_thresholds()

        # Step 7: 权重消融
        self.weight_ablation()

        # Step 8: 阈值敏感性
        self.threshold_sensitivity()

        # Step 9: 可视化
        self.generate_plots()

        # Step 10: 报告
        self.generate_report()

        self.logger.info("\n" + "=" * 80)
        self.logger.info("校准完成!")
        self.logger.info(f"输出目录: {self.config.output_dir}")
        self.logger.info("=" * 80)


# ==================== 命令行入口 ====================

def main():
    parser = argparse.ArgumentParser(description='CAAC置信度校准与Tier阈值验证')
    parser.add_argument('--output-dir',
                       default="/home/zjw/zjwdata/3_deep_learning/training_data/curated_v2/calibration_results",
                       help='输出目录')
    parser.add_argument('--device', default='cuda:0', help='计算设备')
    parser.add_argument('--batch-size', type=int, default=512, help='批大小')

    args = parser.parse_args()

    config = Config(
        output_dir=args.output_dir,
        device=args.device,
        batch_size=args.batch_size,
    )

    calibrator = CAACCalibrator(config)
    calibrator.run()


if __name__ == "__main__":
    main()