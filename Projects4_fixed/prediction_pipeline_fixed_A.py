#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模块6: prediction_pipeline.py (方案A修复版 - 高性能优化)
修复: 预测时为hard样本提取真实上下文特征（不再使用零向量）
优化: 预加载fna到内存 + 批量提取 + 邻居缓存（50-100倍提速）

输入:
  - 模块2难样本: curated_v2/hard_samples_combined.fasta
  - 模块4正样本特征: features/positive/esm2_features.npy + gene_ids.csv
  - 模块4正样本metadata: features/positive/metadata.csv (含CAZy/EC)
  - 模块5GNN模型: gnn_model/final_model.pt

输出: curated_v2/predictions/
  - hard_predictions_full.tsv       # 完整预测结果
  - hard_predictions_tier1.fasta    # 高置信度预测
  - hard_predictions_tier2.fasta    # 中置信度预测
  - hard_predictions_tier3.fasta    # 低置信度预测
  - prediction_report.json          # 统计报告
"""

import os
import sys
import re
import json
import argparse
import logging
import gc
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
from multiprocessing import Pool, cpu_count, set_start_method

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from transformers import EsmTokenizer, EsmModel

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import faiss

    FAISS_AVAILABLE = True
    FAISS_GPU_AVAILABLE = faiss.get_num_gpus() > 0
except ImportError:
    FAISS_AVAILABLE = False
    FAISS_GPU_AVAILABLE = False

warnings.filterwarnings('ignore')

# 注释迁移并行worker数（根据内存调整，16个较安全）
ANNOTATION_WORKERS = 16


# ==================== 配置类 ====================

@dataclass
class PredictionConfig:
    """预测配置"""
    # 输入路径
    hard_fasta: str = "/home/zjw/zjwdata/3_deep_learning/training_data/curated_v2/hard_samples_combined.fasta"
    positive_esm2_features: str = "/home/zjw/zjwdata/3_deep_learning/training_data/curated_v2/features/positive/esm2_features.npy"
    positive_gene_ids: str = "/home/zjw/zjwdata/3_deep_learning/training_data/curated_v2/features/positive/gene_ids.csv"
    positive_metadata: str = "/home/zjw/zjwdata/3_deep_learning/training_data/curated_v2/positive_samples_info.tsv"
    gnn_model_path: str = "/home/zjw/zjwdata/3_deep_learning/training_data/curated_v2/gnn_model/final_model.pt"

    # ESM2模型
    esm2_model_path: str = "/home/zjw/deeplearning_project_advanced/esm2_t33_650M_UR50D"

    # 【修复A1】新增: fna文件目录（用于提取hard样本上下文）
    genes_fna_dir: str = "/home/zjw/zjwdata/1/assembly_analysis/genes"

    # 输出路径
    output_dir: str = "/home/zjw/zjwdata/3_deep_learning/training_data/curated_v2/predictions"

    # ESM2参数
    esm2_batch_size: int = 32
    esm2_max_length: int = 1024
    esm2_precision: str = "fp16"

    # FAISS参数
    faiss_top_k: int = 50
    faiss_batch_size: int = 10000
    faiss_use_gpu: bool = True
    faiss_gpu_id: int = 0

    # GNN参数
    gnn_batch_size: int = 512
    gnn_device: str = "cuda:0"

    # 注释迁移参数
    annotation_n_workers: int = ANNOTATION_WORKERS
    top_neighbors_for_consensus: int = 5
    min_consensus_ratio: float = 0.4  # 从0.6降到0.4（修复）

    # 置信度公式参数（方案A1）
    consistency_weight: float = 0.8
    excess_similarity_weight: float = 0.2
    similarity_baseline: float = 0.9
    excess_similarity_scale: float = 10.0

    # Tier阈值
    tier1_threshold: float = 0.60
    tier2_threshold: float = 0.40
    tier3_threshold: float = 0.25

    # 上下文特征（正样本无上下文，用零向量）
    context_dim: int = 128

    # 【修复A2】新增: 上下文窗口参数（与模块2/4一致）
    context_window: int = 5
    max_neighbors: int = 10

    # 【修复A3】新增: 是否提取hard样本上下文（默认True）
    extract_context: bool = True

    # 随机种子
    seed: int = 42

    log_level: str = "INFO"

    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(f"{self.output_dir}/logs", exist_ok=True)


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


# ==================== FASTA解析器 ====================

class FastaParser:
    @staticmethod
    def parse(filepath: str, max_samples: Optional[int] = None) -> List[Dict]:
        sequences = []
        with open(filepath, 'r') as f:
            current_id = None
            current_header = None
            current_seq = []
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    if current_id:
                        seq = ''.join(current_seq)
                        sequences.append({
                            'id': current_id,
                            'header': current_header,
                            'sequence': seq,
                            'length': len(seq)
                        })
                        if max_samples and len(sequences) >= max_samples:
                            break
                    current_header = line[1:]
                    # 提取gene_id（第一个|前的部分）
                    current_id = current_header.split('|')[0]
                    current_seq = []
                else:
                    current_seq.append(line)
            if current_id and (not max_samples or len(sequences) < max_samples):
                seq = ''.join(current_seq)
                sequences.append({
                    'id': current_id,
                    'header': current_header,
                    'sequence': seq,
                    'length': len(seq)
                })
        return sequences


# ==================== 阶段A: ESM2特征提取 ====================

class ESM2Extractor:
    def __init__(self, config: PredictionConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers库未安装")

        self._load_model()

    def _load_model(self):
        self.logger.info(f"[ESM2] 加载模型...")
        self.tokenizer = EsmTokenizer.from_pretrained(self.config.esm2_model_path)
        try:
            self.model = EsmModel.from_pretrained(
                self.config.esm2_model_path,
                torch_dtype=torch.float16 if self.config.esm2_precision == "fp16" else torch.float32,
            )
        except TypeError:
            self.model = EsmModel.from_pretrained(
                self.config.esm2_model_path,
                dtype=torch.float16 if self.config.esm2_precision == "fp16" else torch.float32,
            )
        self.model = self.model.to(self.device)
        self.model.eval()
        if self.config.esm2_precision == "fp16":
            self.model = self.model.half()

    def extract(self, sequences: List[Dict]) -> Tuple[np.ndarray, List[str]]:
        features = []
        gene_ids = []
        batch_size = self.config.esm2_batch_size

        for i in tqdm(range(0, len(sequences), batch_size), desc="ESM2提取"):
            batch = sequences[i:i + batch_size]
            seqs = [s['sequence'][:self.config.esm2_max_length] for s in batch]

            inputs = self.tokenizer(
                seqs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.esm2_max_length
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                batch_features = outputs.last_hidden_state[:, 0, :].cpu().float().numpy()

            features.append(batch_features)
            gene_ids.extend([s['id'] for s in batch])

            del inputs, outputs, batch_features
            if i % 100 == 0:
                torch.cuda.empty_cache()

        return np.vstack(features), gene_ids

    def cleanup(self):
        del self.model, self.tokenizer
        torch.cuda.empty_cache()
        gc.collect()


# ==================== 阶段B&C: FAISS-GPU检索 ====================

class FAISSSearcher:
    def __init__(self, config: PredictionConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger

        if not FAISS_GPU_AVAILABLE:
            raise RuntimeError("FAISS-GPU不可用")

        self._load_reference()
        self._build_index()

    def _load_reference(self):
        """加载正样本特征和元数据"""
        self.logger.info("加载正样本参考库...")

        # 特征
        ref_features = np.load(self.config.positive_esm2_features)
        self.ref_features = ref_features.astype('float32')
        self.logger.info(f"  参考特征: {self.ref_features.shape}")

        # 基因ID
        ids_df = pd.read_csv(self.config.positive_gene_ids)
        self.ref_gene_ids = ids_df['gene_id'].tolist()
        self.logger.info(f"  参考基因: {len(self.ref_gene_ids):,}")

        # 元数据（用于注释迁移）—— 模块一TSV格式，只加载需要的列
        self.ref_metadata = pd.read_csv(
            self.config.positive_metadata,
            sep='\t',
            low_memory=False,
            usecols=['Entry_ID', 'CAZy_Families', 'EC_Numbers']
        )
        self.logger.info(f"  参考metadata: {len(self.ref_metadata):,}")

        # 建立gene_id到metadata行索引的映射（TSV主键为Entry_ID）
        self.gene_id_to_meta_idx = {}
        if 'Entry_ID' in self.ref_metadata.columns:
            for idx, row in self.ref_metadata.iterrows():
                gid = str(row['Entry_ID']).strip()
                self.gene_id_to_meta_idx[gid] = idx

    def _build_index(self):
        """构建FAISS-GPU索引"""
        self.logger.info("构建FAISS-GPU索引...")

        dim = self.ref_features.shape[1]
        faiss.normalize_L2(self.ref_features)

        # CPU索引
        cpu_index = faiss.IndexFlatIP(dim)
        cpu_index.add(self.ref_features)

        # 转到GPU
        self.res = faiss.StandardGpuResources()
        self.index = faiss.index_cpu_to_gpu(self.res, self.config.faiss_gpu_id, cpu_index)

        self.logger.info(f"  GPU索引: {self.index.ntotal:,} 条向量")

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_allocated(self.config.faiss_gpu_id) / 1e9
            self.logger.info(f"  GPU显存: {allocated:.2f} GB")

    def search(self, query_features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """执行检索"""
        self.logger.info(f"FAISS检索: {query_features.shape}")

        n_queries = query_features.shape[0]
        batch_size = self.config.faiss_batch_size
        all_distances = []
        all_indices = []

        start_time = datetime.now()

        for i in range(0, n_queries, batch_size):
            end = min(i + batch_size, n_queries)
            batch = query_features[i:end].astype('float32')
            faiss.normalize_L2(batch)

            D, I = self.index.search(batch, self.config.faiss_top_k)
            all_distances.append(D)
            all_indices.append(I)

            if (i // batch_size) % 10 == 0 or end == n_queries:
                progress = end / n_queries * 100
                self.logger.info(f"  进度: {end:,}/{n_queries:,} ({progress:.1f}%)")

        distances = np.vstack(all_distances)
        indices = np.vstack(all_indices)

        elapsed = (datetime.now() - start_time).total_seconds()
        self.logger.info(f"检索完成: {elapsed:.1f}秒")

        # 验证
        max_sims = distances[:, 0]
        self.logger.info(f"Top-1相似度: min={max_sims.min():.4f}, max={max_sims.max():.4f}, mean={max_sims.mean():.4f}")

        return distances, indices


# ==================== 【修复A4】新增: 高性能预测时上下文特征提取器 ====================

class PredictionContextExtractor:
    """
    高性能版: 预加载fna到内存 + 批量提取 + 邻居缓存

    性能优化:
    1. _load_fna_to_memory(): 一次性加载fna到内存字典，避免重复文件IO
    2. _build_contig_index(): 内存中构建contig索引，O(1)查找邻居
    3. _neighbor_cache: 跨样本缓存邻居序列，减少重复提取
    4. 预期性能提升: 50-100倍
    """

    def __init__(self, config: PredictionConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.kmer_sizes = [3, 4, 5]
        self.vocab = self._build_kmer_vocab()
        np.random.seed(42)
        self.projection = np.random.randn(
            len(self.vocab), config.context_dim
        ) / np.sqrt(len(self.vocab))

        # 跨样本邻居序列缓存
        self._neighbor_cache: Dict[str, str] = {}
        self._cache_hits = 0
        self._cache_misses = 0

    def _build_kmer_vocab(self) -> Dict[str, int]:
        import itertools
        vocab = {}
        idx = 0
        bases = ['A', 'T', 'C', 'G']
        for k in self.kmer_sizes:
            for kmer in itertools.product(bases, repeat=k):
                vocab[''.join(kmer)] = idx
                idx += 1
        return vocab

    def extract_for_hard_samples(
            self,
            hard_sequences: List[Dict]
    ) -> np.ndarray:
        """高性能版上下文提取"""
        n = len(hard_sequences)
        if n == 0:
            return np.zeros((0, self.config.context_dim), dtype=np.float32)

        context_features = np.zeros((n, self.config.context_dim), dtype=np.float32)

        # 按样本分组
        sample_to_indices = defaultdict(list)
        for i, seq in enumerate(hard_sequences):
            sample = self._extract_sample_from_header(seq.get('header', ''))
            if sample:
                sample_to_indices[sample].append(i)
            else:
                self.logger.warning(f"  无法解析样本名: {seq['id']}")

        self.logger.info(f"[ContextExtractor] 处理 {len(sample_to_indices)} 个样本, 共 {n} 条基因")

        total_with_context = 0

        for sample, indices in sample_to_indices.items():
            fna_path = f"{self.config.genes_fna_dir}/{sample}_genes.fna"

            if not os.path.exists(fna_path):
                self.logger.warning(f"  fna不存在: {fna_path}")
                continue

            gene_ids = [hard_sequences[i]['id'] for i in indices]
            self.logger.info(f"  样本 {sample}: {len(gene_ids)} 条基因")

            # 【优化1】预加载整个fna到内存字典
            start_time = datetime.now()
            fna_dict = self._load_fna_to_memory(fna_path)
            load_time = (datetime.now() - start_time).total_seconds()
            self.logger.info(f"    加载fna: {len(fna_dict):,} 条序列, {load_time:.1f}秒")

            # 【优化2】构建contig索引（内存中）
            contig_index = self._build_contig_index(fna_dict)

            # 【优化3】批量提取邻居序列（带缓存）
            start_time = datetime.now()
            for i in indices:
                gene_id = hard_sequences[i]['id']

                # 获取邻居gene_ids
                neighbors = self._get_neighbors_from_contig(
                    gene_id, contig_index, self.config.context_window
                )

                if neighbors:
                    # 批量从内存字典获取序列（利用缓存）
                    neighbor_seqs = []
                    for nid in neighbors[:self.config.max_neighbors]:
                        if nid in self._neighbor_cache:
                            seq = self._neighbor_cache[nid]
                            self._cache_hits += 1
                        elif nid in fna_dict:
                            seq = fna_dict[nid]
                            self._neighbor_cache[nid] = seq
                            self._cache_misses += 1
                        else:
                            continue
                        neighbor_seqs.append(seq)

                    if neighbor_seqs:
                        context_features[i] = self._encode_sequences(neighbor_seqs)
                        total_with_context += 1

            process_time = (datetime.now() - start_time).total_seconds()
            speed = len(indices) / process_time * 60 if process_time > 0 else 0
            self.logger.info(f"    处理完成: {process_time:.1f}秒, 速度: {speed:.0f} 条/分钟")

            # 释放内存（fna_dict很大）
            del fna_dict
            gc.collect()

        # 缓存统计
        total_cache = self._cache_hits + self._cache_misses
        if total_cache > 0:
            hit_rate = self._cache_hits / total_cache * 100
            self.logger.info(f"[ContextExtractor] 缓存命中率: {hit_rate:.1f}% "
                             f"({self._cache_hits}/{total_cache})")

        self.logger.info(f"[ContextExtractor] 完成: {total_with_context}/{n} 条有上下文")
        return context_features

    def _load_fna_to_memory(self, fna_path: str) -> Dict[str, str]:
        """
        【核心优化】一次性加载整个fna到内存字典
        避免逐条读取时的重复文件IO
        """
        sequences = {}
        current_id = None
        current_seq = []

        with open(fna_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    if current_id:
                        sequences[current_id] = ''.join(current_seq)
                    current_id = line[1:].split()[0]
                    current_seq = []
                else:
                    current_seq.append(line)

            if current_id:
                sequences[current_id] = ''.join(current_seq)

        return sequences

    def _build_contig_index(self, fna_dict: Dict[str, str]) -> Dict[str, List[str]]:
        """
        构建contig索引: contig_id -> 排序后的gene_id列表
        """
        contig_dict = defaultdict(list)

        for gene_id in fna_dict.keys():
            contig_id = self._infer_contig_id(gene_id)
            contig_dict[contig_id].append(gene_id)

        # 按基因编号排序
        for contig_id in contig_dict:
            contig_dict[contig_id].sort(key=lambda x: self._extract_gene_num(x))

        return dict(contig_dict)

    def _get_neighbors_from_contig(
            self,
            gene_id: str,
            contig_index: Dict[str, List[str]],
            window: int
    ) -> List[str]:
        """从contig索引获取邻居基因ID"""
        contig_id = self._infer_contig_id(gene_id)

        if contig_id not in contig_index:
            return []

        genes = contig_index[contig_id]

        try:
            idx = genes.index(gene_id)
        except ValueError:
            return []

        start = max(0, idx - window)
        end = min(len(genes), idx + window + 1)

        return [genes[i] for i in range(start, end) if i != idx]

    def _infer_contig_id(self, gene_id: str) -> str:
        """从gene_id推断contig_id"""
        parts = gene_id.rsplit('_', 1)
        if len(parts) == 2 and parts[1].isdigit():
            return parts[0]
        return gene_id

    def _extract_gene_num(self, gene_id: str) -> int:
        """从gene_id提取编号"""
        parts = gene_id.rsplit('_', 1)
        if len(parts) == 2 and parts[1].isdigit():
            return int(parts[1])
        return 0

    def _extract_sample_from_header(self, header: str) -> Optional[str]:
        match = re.search(r'Sample=([^|]+)', header)
        if match:
            return match.group(1)
        return None

    def _encode_sequences(self, sequences: List[str]) -> np.ndarray:
        """k-mer频率编码 + 随机投影（与模块4一致）"""
        kmer_counts = {}
        total = 0

        for seq in sequences:
            seq = seq.upper().replace('N', '')
            for k in self.kmer_sizes:
                for i in range(len(seq) - k + 1):
                    kmer = seq[i:i + k]
                    if kmer in self.vocab:
                        kmer_counts[kmer] = kmer_counts.get(kmer, 0) + 1
                        total += 1

        vec = np.zeros(len(self.vocab))
        if total > 0:
            for kmer, count in kmer_counts.items():
                vec[self.vocab[kmer]] = count / total

        reduced = vec @ self.projection
        norm = np.linalg.norm(reduced)
        if norm > 0:
            reduced = reduced / norm

        return reduced


# ==================== GNN预测器 ====================

class GNNPredictor:
    def __init__(self, config: PredictionConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.device = torch.device(config.gnn_device if torch.cuda.is_available() else "cpu")
        self._load_model()

    def _load_model(self):
        """加载GNN最终模型"""
        self.logger.info(f"[GNN] 加载模型: {self.config.gnn_model_path}")

        checkpoint = torch.load(self.config.gnn_model_path, map_location=self.device)

        # 从checkpoint恢复模型结构
        from model_training import AttentionGNN, GNNConfig  # 运行时导入

        # 构建模型配置
        model_config = GNNConfig(
            input_dim=1408,
            hidden_dim=512,
            num_classes=3,
            num_attention_heads=8,
            dropout=0.3,
            attention_dropout=0.1,
        )

        self.model = AttentionGNN(model_config).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        self.logger.info(f"[GNN] 模型加载完成")

    def predict(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        GNN预测
        返回: (probs, preds)
        probs: (N, 3) 各类概率
        preds: (N,) 预测类别
        """
        self.logger.info(f"GNN预测: {features.shape}")

        # 构建DataLoader
        dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(features),
            torch.zeros(len(features), dtype=torch.long)  # dummy labels
        )
        loader = DataLoader(dataset, batch_size=self.config.gnn_batch_size, shuffle=False)

        all_probs = []
        all_preds = []

        with torch.no_grad():
            for batch_x, _ in tqdm(loader, desc="GNN预测"):
                batch_x = batch_x.to(self.device)
                logits = self.model(batch_x)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)

                all_probs.append(probs)
                all_preds.append(preds)

        probs = np.vstack(all_probs)
        preds = np.concatenate(all_preds)

        # 统计预测分布
        pred_dist = np.bincount(preds, minlength=3)
        self.logger.info(f"预测分布: Neg={pred_dist[0]}, Pos={pred_dist[1]}, Hard={pred_dist[2]}")

        return probs, preds


# ==================== 注释迁移（并行） ====================

def _init_annotation_worker(ref_metadata_path: str, ref_gene_ids: List[str]):
    """Worker初始化——只加载需要的列以节省内存"""
    global _WORKER_REF_META, _WORKER_REF_IDS, _WORKER_ID_TO_IDX

    df = pd.read_csv(
        ref_metadata_path,
        sep='\t',
        low_memory=False,
        usecols=['Entry_ID', 'CAZy_Families', 'EC_Numbers']
    )
    _WORKER_REF_META = df
    _WORKER_REF_IDS = ref_gene_ids
    _WORKER_ID_TO_IDX = {}

    # 模块一TSV用 Entry_ID 作为主键
    for idx, row in df.iterrows():
        entry_id = str(row['Entry_ID']).strip()
        _WORKER_ID_TO_IDX[entry_id] = idx

    # 统计有功能的条目
    n_with_function = 0
    for idx, row in df.iterrows():
        cazy = str(row.get('CAZy_Families', ''))
        ec = str(row.get('EC_Numbers', ''))
        if (cazy and cazy not in ('NA', 'nan', '-', '')) or \
                (ec and ec not in ('NA', 'nan', '-', '')):
            n_with_function += 1

    print(f"[Worker] 加载 {len(df)} 条参考注释，其中 {n_with_function} 条有功能", file=sys.stderr)


def _annotation_worker(args):
    """单个注释迁移任务"""
    idx, query_gene_id, neighbor_indices, neighbor_sims, config_dict = args
    global _WORKER_REF_META, _WORKER_REF_IDS, _WORKER_ID_TO_IDX

    top_k = config_dict['top_neighbors_for_consensus']
    min_consensus = config_dict['min_consensus_ratio']

    w_con = config_dict['consistency_weight']
    w_excess = config_dict['excess_similarity_weight']
    sim_baseline = config_dict['similarity_baseline']
    excess_scale = config_dict['excess_similarity_scale']

    # 收集有效邻居
    valid_neighbors = []
    for row_idx, sim in zip(neighbor_indices, neighbor_sims):
        row_idx_int = int(row_idx)
        if row_idx_int >= len(_WORKER_REF_IDS):
            continue

        ref_gene_id = _WORKER_REF_IDS[row_idx_int]

        # 获取metadata
        meta_idx = _WORKER_ID_TO_IDX.get(ref_gene_id)
        if meta_idx is None:
            continue

        row = _WORKER_REF_META.iloc[meta_idx]

        # 提取CAZy和EC
        cazy_fams = []
        if 'CAZy_Families' in row and pd.notna(row['CAZy_Families']):
            val = str(row['CAZy_Families'])
            if val not in ('NA', 'nan', '-', ''):
                cazy_fams = [f.strip() for f in val.split('|') if f.strip()]

        ec_nums = []
        if 'EC_Numbers' in row and pd.notna(row['EC_Numbers']):
            val = str(row['EC_Numbers'])
            if val not in ('NA', 'nan', '-', ''):
                ec_nums = [e.strip() for e in val.split('|') if e.strip()]

        has_function = len(cazy_fams) > 0 or len(ec_nums) > 0

        if has_function:
            valid_neighbors.append({
                'gene_id': ref_gene_id,
                'similarity': float(sim),
                'cazy': cazy_fams,
                'ec': ec_nums,
            })

        if len(valid_neighbors) >= top_k:
            break

    # 不足3个有效邻居
    if len(valid_neighbors) < 3:
        return {
            'gene_id': query_gene_id,
            'predicted_cazy': 'NA',
            'predicted_ec': 'NA',
            'confidence_score': 0.0,
            'confidence_tier': 'Tier_Low',
            'neighbor_avg_similarity': 0.0,
            'neighbor_consistency': 0.0,
            'excess_similarity': 0.0,
            'n_neighbors_used': len(valid_neighbors),
            'evidence_code': 'IEA:ESM2_SIM_LOW',
        }

    # 共识投票
    cazy_counter = Counter()
    for n in valid_neighbors[:top_k]:
        for fam in n['cazy']:
            cazy_counter[fam] += 1

    ec_counter = Counter()
    for n in valid_neighbors[:top_k]:
        for ec in n['ec']:
            ec_counter[ec] += 1

    # 选择最佳EC（最特异且频率高）
    best_ec = ''
    if ec_counter:
        best_ec = sorted(ec_counter.items(),
                         key=lambda x: (x[0].count('.'), x[1]),
                         reverse=True)[0][0]

    # 共识CAZy
    total = len(valid_neighbors[:top_k])
    consensus_cazy = [fam for fam, count in cazy_counter.items()
                      if count / total >= min_consensus]

    # 一致性 = 最高频家族出现比例
    if cazy_counter:
        consistency = max(cazy_counter.values()) / total
    else:
        consistency = 0.0

    # 方案A1置信度公式
    avg_sim = np.mean([n['similarity'] for n in valid_neighbors[:top_k]])
    excess_sim = max(0, avg_sim - sim_baseline) * excess_scale
    excess_sim = min(excess_sim, 1.0)
    confidence = consistency * w_con + excess_sim * w_excess
    confidence = min(confidence, 1.0)

    # Tier判定
    if confidence >= config_dict['tier1']:
        tier = 'Tier_1'
    elif confidence >= config_dict['tier2']:
        tier = 'Tier_2'
    elif confidence >= config_dict['tier3']:
        tier = 'Tier_3'
    else:
        tier = 'Tier_Low'

    return {
        'gene_id': query_gene_id,
        'predicted_cazy': '|'.join(consensus_cazy) if consensus_cazy else 'NA',
        'predicted_ec': best_ec,
        'confidence_score': round(confidence, 4),
        'confidence_tier': tier,
        'neighbor_avg_similarity': round(avg_sim, 4),
        'neighbor_consistency': round(consistency, 4),
        'excess_similarity': round(excess_sim, 4),
        'n_neighbors_used': len(valid_neighbors),
        'top_neighbors_detail': '; '.join([f"{n['gene_id']}:{n['similarity']:.3f}"
                                           for n in valid_neighbors[:5]]),
        'evidence_code': 'IEA:ESM2_SIM',
    }


class AnnotationTransfer:
    def __init__(self, config: PredictionConfig, logger: logging.Logger,
                 ref_gene_ids: List[str]):
        self.config = config
        self.logger = logger
        self.ref_gene_ids = ref_gene_ids

    def transfer_parallel(self, query_gene_ids: List[str],
                          neighbor_indices: np.ndarray,
                          neighbor_sims: np.ndarray) -> pd.DataFrame:
        """并行注释迁移"""
        self.logger.info("开始并行注释迁移...")

        n_queries = len(query_gene_ids)
        self.logger.info(f"处理 {n_queries:,} 条查询...")

        config_dict = {
            'top_neighbors_for_consensus': self.config.top_neighbors_for_consensus,
            'min_consensus_ratio': self.config.min_consensus_ratio,
            'consistency_weight': self.config.consistency_weight,
            'excess_similarity_weight': self.config.excess_similarity_weight,
            'similarity_baseline': self.config.similarity_baseline,
            'excess_similarity_scale': self.config.excess_similarity_scale,
            'tier1': self.config.tier1_threshold,
            'tier2': self.config.tier2_threshold,
            'tier3': self.config.tier3_threshold,
        }

        # 准备任务
        tasks = []
        for i in range(n_queries):
            tasks.append((
                i,
                query_gene_ids[i],
                neighbor_indices[i],
                neighbor_sims[i],
                config_dict,
            ))

        # 并行执行
        n_workers = min(self.config.annotation_n_workers, cpu_count() - 4)
        self.logger.info(f"启动 {n_workers} 个worker...")

        with Pool(
                processes=n_workers,
                initializer=_init_annotation_worker,
                initargs=(self.config.positive_metadata, self.ref_gene_ids)
        ) as pool:
            results = list(tqdm(
                pool.imap(_annotation_worker, tasks,
                          chunksize=max(1, len(tasks) // (n_workers * 4))),
                total=len(tasks),
                desc="注释迁移"
            ))

        df = pd.DataFrame(results)
        self.logger.info(f"注释完成: {len(df)} 条")

        return df


# ==================== 结果整合与输出 ====================

class ResultWriter:
    def __init__(self, output_dir: str, logger: logging.Logger):
        self.output_dir = output_dir
        self.logger = logger
        os.makedirs(output_dir, exist_ok=True)

    def write_results(self, sequences: List[Dict], gnn_probs: np.ndarray,
                      gnn_preds: np.ndarray, annotations: pd.DataFrame,
                      output_prefix: str = "hard_predictions"):
        """写入完整预测结果"""

        # 整合所有信息
        records = []
        for i, seq in enumerate(sequences):
            anno_row = annotations.iloc[i] if i < len(annotations) else {}

            record = {
                'gene_id': seq['id'],
                'sequence_length': seq['length'],
                'gnn_pred_class': int(gnn_preds[i]),
                'gnn_prob_negative': round(float(gnn_probs[i, 0]), 4),
                'gnn_prob_positive': round(float(gnn_probs[i, 1]), 4),
                'gnn_prob_hard': round(float(gnn_probs[i, 2]), 4),
                'predicted_cazy': anno_row.get('predicted_cazy', 'NA'),
                'predicted_ec': anno_row.get('predicted_ec', 'NA'),
                'confidence_score': anno_row.get('confidence_score', 0.0),
                'confidence_tier': anno_row.get('confidence_tier', 'Tier_Low'),
                'neighbor_avg_similarity': anno_row.get('neighbor_avg_similarity', 0.0),
                'neighbor_consistency': anno_row.get('neighbor_consistency', 0.0),
                'excess_similarity': anno_row.get('excess_similarity', 0.0),
                'n_neighbors_used': anno_row.get('n_neighbors_used', 0),
                'evidence_code': anno_row.get('evidence_code', 'NA'),
            }
            records.append(record)

        df = pd.DataFrame(records)

        # 保存完整TSV
        full_path = f"{self.output_dir}/{output_prefix}_full.tsv"
        df.to_csv(full_path, sep='\t', index=False)
        self.logger.info(f"完整预测: {full_path} ({len(df):,} 条)")

        # 按Tier生成FASTA
        self._write_tier_fastas(df, sequences, output_prefix)

        # 生成统计报告
        self._write_report(df, output_prefix)

        return full_path

    def _write_tier_fastas(self, df: pd.DataFrame, sequences: List[Dict], prefix: str):
        """按置信度Tier生成FASTA"""
        # 构建序列查找
        seq_dict = {s['id']: s for s in sequences}

        for tier in ['Tier_1', 'Tier_2', 'Tier_3']:
            tier_df = df[df['confidence_tier'] == tier]
            if len(tier_df) == 0:
                continue

            fasta_path = f"{self.output_dir}/{prefix}_{tier.lower().replace(' ', '_')}.fasta"

            with open(fasta_path, 'w') as f:
                for _, row in tier_df.iterrows():
                    gid = row['gene_id']
                    if gid not in seq_dict:
                        continue

                    seq = seq_dict[gid]
                    header = (f">{gid}|{tier}|"
                              f"Conf={row['confidence_score']:.3f}|"
                              f"CAZy={row['predicted_cazy']}|"
                              f"EC={row['predicted_ec']}|"
                              f"GNN={row['gnn_pred_class']}")

                    f.write(header + "\n")
                    for i in range(0, len(seq['sequence']), 60):
                        f.write(seq['sequence'][i:i + 60] + "\n")

            self.logger.info(f"{tier}: {fasta_path} ({len(tier_df):,} 条)")

    def _write_report(self, df: pd.DataFrame, prefix: str):
        """生成统计报告"""
        report_path = f"{self.output_dir}/{prefix}_report.json"

        tier_dist = df['confidence_tier'].value_counts().to_dict()
        gnn_dist = df['gnn_pred_class'].value_counts().to_dict()

        # GNN预测与注释置信度的交叉分析
        cross_tab = pd.crosstab(df['gnn_pred_class'], df['confidence_tier'])

        report = {
            'timestamp': datetime.now().isoformat(),
            'total_sequences': len(df),
            'tier_distribution': tier_dist,
            'gnn_prediction_distribution': {
                'negative': int(gnn_dist.get(0, 0)),
                'positive': int(gnn_dist.get(1, 0)),
                'hard': int(gnn_dist.get(2, 0)),
            },
            'confidence_statistics': {
                'mean': float(df['confidence_score'].mean()),
                'std': float(df['confidence_score'].std()),
                'min': float(df['confidence_score'].min()),
                'max': float(df['confidence_score'].max()),
                'percentiles': {
                    'p25': float(df['confidence_score'].quantile(0.25)),
                    'p50': float(df['confidence_score'].quantile(0.50)),
                    'p75': float(df['confidence_score'].quantile(0.75)),
                    'p90': float(df['confidence_score'].quantile(0.90)),
                }
            },
            'has_cazy_annotation': int((df['predicted_cazy'] != 'NA').sum()),
            'has_ec_annotation': int((df['predicted_ec'] != 'NA').sum()),
            'gnn_tier_crosstab': cross_tab.to_dict(),
            'configuration': {
                'tier_thresholds': {
                    'tier1': 0.60,
                    'tier2': 0.40,
                    'tier3': 0.25,
                },
                'confidence_formula': 'consistency*0.8 + excess_similarity*0.2',
            }
        }

        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        self.logger.info(f"报告: {report_path}")

        # 屏幕摘要
        self.logger.info(f"\n{'=' * 60}")
        self.logger.info("预测结果摘要")
        self.logger.info(f"{'=' * 60}")
        self.logger.info(f"总序列: {len(df):,}")
        self.logger.info(f"Tier分布:")
        for tier, count in sorted(tier_dist.items()):
            self.logger.info(f"  {tier}: {count:,} ({100 * count / len(df):.1f}%)")
        self.logger.info(f"有CAZy: {(df['predicted_cazy'] != 'NA').sum():,}")
        self.logger.info(f"有EC: {(df['predicted_ec'] != 'NA').sum():,}")
        self.logger.info(f"{'=' * 60}")


# ==================== 【修复A6】主流程 ====================

class PredictionPipeline:
    """预测流程主控制器（方案A修复版 - 高性能优化）"""

    def __init__(self, config: PredictionConfig):
        self.config = config
        self.logger = setup_logger(
            "prediction",
            f"{config.output_dir}/logs",
            config.log_level
        )

        self.logger.info("=" * 80)
        self.logger.info("模块6: 难样本功能预测流程（方案A修复版 - 高性能优化）")
        self.logger.info("修复内容: 预测时为hard样本提取真实上下文特征")
        self.logger.info("优化内容: 预加载fna到内存 + 批量提取 + 邻居缓存")
        self.logger.info(f"开始时间: {datetime.now()}")
        self.logger.info("=" * 80)

    def run(self) -> Dict:
        # 步骤1: 解析难样本FASTA
        self.logger.info(f"\n{'=' * 60}")
        self.logger.info("步骤1: 解析难样本序列")
        self.logger.info(f"{'=' * 60}")

        sequences = FastaParser.parse(self.config.hard_fasta)
        self.logger.info(f"难样本序列: {len(sequences):,}")

        # 步骤2: ESM2特征提取（支持跳过）
        esm2_save_path = f"{self.config.output_dir}/query_esm2_features.npy"
        if os.path.exists(esm2_save_path):
            self.logger.info(f"\n{'=' * 60}")
            self.logger.info("步骤2: 检测到已有ESM2特征，跳过提取")
            self.logger.info(f"{'=' * 60}")
            esm2_features = np.load(esm2_save_path)
            query_gene_ids = [s['id'] for s in sequences]
        else:
            self.logger.info(f"\n{'=' * 60}")
            self.logger.info("步骤2: ESM2特征提取")
            self.logger.info(f"{'=' * 60}")
            esm2_extractor = ESM2Extractor(self.config, self.logger)
            try:
                esm2_features, query_gene_ids = esm2_extractor.extract(sequences)
            finally:
                esm2_extractor.cleanup()
            np.save(esm2_save_path, esm2_features)

        self.logger.info(f"ESM2特征: {esm2_features.shape}")

        # 保存ESM2特征（可选，用于复用）
        esm2_save_path = f"{self.config.output_dir}/query_esm2_features.npy"
        np.save(esm2_save_path, esm2_features)
        self.logger.info(f"ESM2保存: {esm2_save_path}")

        # 步骤3: FAISS检索
        self.logger.info(f"\n{'=' * 60}")
        self.logger.info("步骤3: FAISS-GPU相似度检索")
        self.logger.info(f"{'=' * 60}")

        faiss_searcher = FAISSSearcher(self.config, self.logger)
        distances, indices = faiss_searcher.search(esm2_features)

        # 【修复A7】步骤4: GNN预测（3分类）—— 为hard样本提取上下文
        self.logger.info(f"\n{'=' * 60}")
        self.logger.info("步骤4: GNN 3分类预测（方案A: 为hard样本提取上下文）")
        self.logger.info(f"{'=' * 60}")

        # 初始化上下文特征（默认全零）
        context_features = np.zeros((len(esm2_features), self.config.context_dim), dtype=np.float32)

        # 【关键修复】识别hard样本并从fna提取上下文
        hard_mask = self._extract_hard_mask(sequences)
        n_hard = hard_mask.sum()
        n_pos_neg = len(sequences) - n_hard

        self.logger.info(f"序列分类: Pos/Neg={n_pos_neg}, Hard={n_hard}")
        self.logger.info(f"上下文策略: extract_context={self.config.extract_context}")

        if n_hard > 0 and self.config.extract_context:
            self.logger.info(f"\n为 {n_hard} 条hard样本提取上下文特征...")

            # 提取hard样本子集
            hard_sequences = [seq for seq, is_hard in zip(sequences, hard_mask) if is_hard]

            # 使用高性能PredictionContextExtractor提取上下文
            context_extractor = PredictionContextExtractor(self.config, self.logger)
            hard_context = context_extractor.extract_for_hard_samples(hard_sequences)

            # 将提取的上下文放回原位置
            hard_idx = 0
            for i, is_hard in enumerate(hard_mask):
                if is_hard:
                    context_features[i] = hard_context[hard_idx]
                    hard_idx += 1

            # 统计上下文质量
            has_context = np.any(context_features != 0, axis=1).sum()
            self.logger.info(f"上下文统计: {has_context}/{len(sequences)} 条有非零上下文")
        else:
            self.logger.info("跳过上下文提取（无hard样本或extract_context=False）")

        # 合并特征: ESM2 + 上下文
        combined_features = np.concatenate([esm2_features, context_features], axis=1)
        self.logger.info(f"合并特征: {combined_features.shape}")

        # 导入GNN模型（运行时导入避免循环依赖）
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from model_training import AttentionGNN, GNNConfig as GNNTrainConfig

        gnn_config = GNNTrainConfig(
            input_dim=1408,
            hidden_dim=512,
            num_classes=3,
            num_attention_heads=8,
            dropout=0.3,
            attention_dropout=0.1,
        )

        device = torch.device(self.config.gnn_device if torch.cuda.is_available() else "cpu")
        model = AttentionGNN(gnn_config).to(device)

        checkpoint = torch.load(self.config.gnn_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        # 预测
        dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(combined_features),
            torch.zeros(len(combined_features), dtype=torch.long)
        )
        loader = DataLoader(dataset, batch_size=self.config.gnn_batch_size, shuffle=False)

        all_probs = []
        all_preds = []

        with torch.no_grad():
            for batch_x, _ in tqdm(loader, desc="GNN预测"):
                batch_x = batch_x.to(device)
                logits = model(batch_x)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)
                all_probs.append(probs)
                all_preds.append(preds)

        gnn_probs = np.vstack(all_probs)
        gnn_preds = np.concatenate(all_preds)

        # 清理
        del model
        torch.cuda.empty_cache()

        self.logger.info(f"GNN预测完成: {len(gnn_preds):,}")
        pred_dist = np.bincount(gnn_preds, minlength=3)
        self.logger.info(f"预测分布: Neg={pred_dist[0]}, Pos={pred_dist[1]}, Hard={pred_dist[2]}")

        # 步骤5: 注释迁移
        self.logger.info(f"\n{'=' * 60}")
        self.logger.info("步骤5: 注释迁移（并行）")
        self.logger.info(f"{'=' * 60}")

        # 加载参考基因ID
        ref_ids_df = pd.read_csv(self.config.positive_gene_ids)
        ref_gene_ids = ref_ids_df['gene_id'].tolist()

        annotator = AnnotationTransfer(self.config, self.logger, ref_gene_ids)
        annotations = annotator.transfer_parallel(
            query_gene_ids, indices, distances
        )

        # 步骤6: 整合输出
        self.logger.info(f"\n{'=' * 60}")
        self.logger.info("步骤6: 整合输出")
        self.logger.info(f"{'=' * 60}")

        writer = ResultWriter(self.config.output_dir, self.logger)
        result_path = writer.write_results(
            sequences, gnn_probs, gnn_preds, annotations
        )

        # 完成
        self.logger.info(f"\n{'=' * 80}")
        self.logger.info("模块6完成!（方案A修复版 - 高性能优化）")
        self.logger.info(f"输出目录: {self.config.output_dir}")
        self.logger.info(f"{'=' * 80}")

        return {
            'total_sequences': len(sequences),
            'output_path': result_path,
            'output_dir': self.config.output_dir,
        }

    def _extract_hard_mask(self, sequences: List[Dict]) -> np.ndarray:
        """从FASTA header解析哪些序列是hard样本 (LABEL=2 / HardSample)

        支持格式:
        - >gene_id|LABEL=2|Sample=CK-7A|...
        - >gene_id|Sample=CK-7A|LABEL=2|...
        """
        mask = np.zeros(len(sequences), dtype=bool)
        for i, seq in enumerate(sequences):
            header = seq.get('header', '')
            if 'LABEL=2' in header or 'HardSample' in header:
                mask[i] = True
        return mask


# ==================== 命令行入口 ====================

def main():
    parser = argparse.ArgumentParser(description='模块6: 难样本功能预测（方案A修复版 - 高性能优化）')
    parser.add_argument('--hard-fasta',
                        default="/home/zjw/zjwdata/3_deep_learning/training_data/curated_v2/hard_samples_combined.fasta")
    parser.add_argument('--positive-features',
                        default="/home/zjw/zjwdata/3_deep_learning/training_data/curated_v2/features/positive/esm2_features.npy")
    parser.add_argument('--positive-ids',
                        default="/home/zjw/zjwdata/3_deep_learning/training_data/curated_v2/features/positive/gene_ids.csv")
    parser.add_argument('--positive-meta',
                        default="/home/zjw/zjwdata/3_deep_learning/training_data/curated_v2/positive_samples_info.tsv")
    parser.add_argument('--gnn-model',
                        default="/home/zjw/zjwdata/3_deep_learning/training_data/curated_v2/gnn_model/final_model.pt")
    # 【修复A8】新增fna目录参数
    parser.add_argument('--genes-fna-dir',
                        default="/home/zjw/zjwdata/1/assembly_analysis/genes",
                        help='基因fna文件目录（用于提取hard样本上下文）')
    parser.add_argument('--output-dir',
                        default="/home/zjw/zjwdata/3_deep_learning/training_data/curated_v2/predictions")
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    # 【修复A9】新增上下文控制参数
    parser.add_argument('--no-context', action='store_true',
                        help='禁用hard样本上下文提取（回退到零向量）')

    args = parser.parse_args()

    config = PredictionConfig(
        hard_fasta=args.hard_fasta,
        positive_esm2_features=args.positive_features,
        positive_gene_ids=args.positive_ids,
        positive_metadata=args.positive_meta,
        gnn_model_path=args.gnn_model,
        genes_fna_dir=args.genes_fna_dir,
        output_dir=args.output_dir,
        faiss_gpu_id=args.gpu_id,
        gnn_device=f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu",
        extract_context=not args.no_context,  # 【修复A10】根据参数控制
        seed=args.seed,
    )

    # 设置多进程启动方式
    set_start_method('spawn', force=True)

    pipeline = PredictionPipeline(config)
    results = pipeline.run()

    return results


if __name__ == "__main__":
    main()