#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模块4: feature_extraction.py (修复版)
修复: 上下文特征预建索引 + 跳过已完成 + 批量提取
"""

import os
import sys
import re
import json
import argparse
import logging
import gc
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

try:
    from transformers import EsmTokenizer, EsmModel

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


# ==================== 配置类 ====================

@dataclass
class FeatureConfig:
    """特征提取配置"""
    clustered_dir: str = "/home/zjw/zjwdata/3_deep_learning/training_data/curated_v2/clustered"
    genes_fna_dir: str = "/home/zjw/zjwdata/1/assembly_analysis/genes"
    output_dir: str = "/home/zjw/zjwdata/3_deep_learning/training_data/curated_v2/features"
    esm2_model_path: str = "/home/zjw/deeplearning_project_advanced/esm2_t33_650M_UR50D"

    esm2_batch_size: int = 16
    esm2_max_length: int = 1024
    esm2_precision: str = "fp16"

    context_window: int = 5
    max_neighbors: int = 10
    context_dim: int = 128
    kmer_sizes: Tuple[int, ...] = (3, 4, 5)

    gpu_ids: Tuple[int, ...] = (0, 1)
    categories: Tuple[str, ...] = ('positive', 'negative', 'hard_strict', 'hard_expanded')

    log_level: str = "INFO"

    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)
        for cat in self.categories:
            os.makedirs(f"{self.output_dir}/{cat}", exist_ok=True)


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
            current_seq = []
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    if current_id:
                        seq = ''.join(current_seq)
                        sequences.append({'id': current_id, 'sequence': seq, 'length': len(seq)})
                        if max_samples and len(sequences) >= max_samples:
                            break
                    header = line[1:]
                    current_id = header.split()[0].split('|')[0]
                    current_seq = []
                else:
                    current_seq.append(line)
            if current_id and (not max_samples or len(sequences) < max_samples):
                seq = ''.join(current_seq)
                sequences.append({'id': current_id, 'sequence': seq, 'length': len(seq)})
        return sequences

    @staticmethod
    def parse_ids_only(filepath: str, max_ids: Optional[int] = None) -> List[str]:
        """只解析FASTA文件中的序列ID，不读取序列内容"""
        ids = []
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    header = line[1:]
                    gene_id = header.split()[0].split('|')[0]
                    ids.append(gene_id)
                    if max_ids and len(ids) >= max_ids:
                        break
        return ids


# ==================== ESM2特征提取器（双GPU版） ====================

class ESM2FeatureExtractor:
    def __init__(self, config: FeatureConfig, logger: logging.Logger, gpu_id: int):
        self.config = config
        self.logger = logger
        self.gpu_id = gpu_id
        self.device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers库未安装")
        self._load_model()

    def _load_model(self):
        self.logger.info(f"[GPU {self.gpu_id}] 加载ESM2模型...")
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
        total_params = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f"[GPU {self.gpu_id}] 模型加载完成: {total_params / 1e6:.0f}M参数")

    def extract(self, sequences: List[Dict]) -> np.ndarray:
        features = []
        batch_size = self.config.esm2_batch_size
        for i in tqdm(range(0, len(sequences), batch_size),
                      desc=f"ESM2 GPU{self.gpu_id}",
                      position=self.gpu_id):
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
            del inputs, outputs, batch_features
            if i % 100 == 0:
                torch.cuda.empty_cache()
        return np.vstack(features)

    def cleanup(self):
        del self.model, self.tokenizer
        torch.cuda.empty_cache()
        gc.collect()


# ==================== FNA索引构建器（核心修复） ====================

class FNAIndexBuilder:
    """
    预建fna索引：一次性扫描所有fna文件，构建基因位置索引
    索引格式: {gene_id: {'fna_path': str, 'contig_id': str, 'gene_num': int, 'file_offset': int}}
    """

    def __init__(self, genes_fna_dir: str, logger: logging.Logger):
        self.genes_fna_dir = genes_fna_dir
        self.logger = logger
        self.index: Dict[str, Dict] = {}
        self.contig_genes: Dict[str, List[str]] = {}  # contig_id -> [gene_id1, gene_id2, ...]
        self.gene_sequences: Dict[str, str] = {}  # 可选：缓存序列

    def build_index(self, target_gene_ids: Optional[Set[str]] = None) -> Dict[str, Dict]:
        """
        构建索引
        target_gene_ids: 如果只索引特定基因集合，加速构建
        """
        self.logger.info(f"构建fna索引: {self.genes_fna_dir}")

        fna_files = []
        if os.path.exists(self.genes_fna_dir):
            for fname in sorted(os.listdir(self.genes_fna_dir)):
                if fname.endswith('_genes.fna'):
                    fna_files.append(f"{self.genes_fna_dir}/{fname}")

        self.logger.info(f"找到 {len(fna_files)} 个fna文件")

        total_genes = 0
        for fna_path in tqdm(fna_files, desc="索引fna文件"):
            genes_in_file = self._index_single_fna(fna_path, target_gene_ids)
            total_genes += genes_in_file

        self.logger.info(f"索引完成: {len(self.index)} 条基因, {len(self.contig_genes)} 个contig")
        return self.index

    def _index_single_fna(self, fna_path: str, target_gene_ids: Optional[Set[str]] = None) -> int:
        """索引单个fna文件"""
        count = 0
        current_contig = None
        current_gene_num = 0

        with open(fna_path, 'r') as f:
            while True:
                line = f.readline()
                if not line:
                    break

                line = line.strip()
                if line.startswith('>'):
                    # 解析头部
                    header = line[1:]
                    gene_id = header.split()[0]

                    # 如果指定了目标集合，跳过不相关的
                    if target_gene_ids is not None and gene_id not in target_gene_ids:
                        # 跳过这个基因的序列
                        while True:
                            pos = f.tell()
                            next_line = f.readline()
                            if not next_line or next_line.startswith('>'):
                                if next_line:
                                    # 回退到头部行
                                    f.seek(pos)
                                break
                        continue

                    # 推断contig_id
                    contig_id = self._infer_contig_id(gene_id, header)

                    # 重置contig计数
                    if current_contig != contig_id:
                        current_contig = contig_id
                        current_gene_num = 0

                    # 记录索引
                    self.index[gene_id] = {
                        'fna_path': fna_path,
                        'contig_id': contig_id,
                        'gene_num': current_gene_num,
                    }

                    # 记录contig成员
                    if contig_id not in self.contig_genes:
                        self.contig_genes[contig_id] = []
                    self.contig_genes[contig_id].append(gene_id)

                    current_gene_num += 1
                    count += 1

        return count

    def _infer_contig_id(self, gene_id: str, header: str) -> str:
        """从基因ID推断contig"""
        parts = gene_id.rsplit('_', 1)
        if len(parts) == 2 and parts[1].isdigit():
            return parts[0]
        # 从header解析ID=1_1格式
        match = re.search(r'ID=(\d+)_\d+', header)
        if match:
            return f"contig_{match.group(1)}"
        return gene_id

    def get_neighbor_gene_ids(self, gene_id: str, window: int = 5) -> List[str]:
        """获取邻居基因ID列表"""
        if gene_id not in self.index:
            return []

        info = self.index[gene_id]
        contig_id = info['contig_id']
        gene_num = info['gene_num']

        if contig_id not in self.contig_genes:
            return []

        genes = self.contig_genes[contig_id]
        start_idx = max(0, gene_num - window)
        end_idx = min(len(genes), gene_num + window + 1)

        neighbors = []
        for i in range(start_idx, end_idx):
            if i != gene_num:
                neighbors.append(genes[i])

        return neighbors

    def get_gene_sequence(self, gene_id: str) -> Optional[str]:
        """从fna提取单个基因序列（按需读取）"""
        if gene_id not in self.index:
            return None

        info = self.index[gene_id]
        fna_path = info['fna_path']

        # 快速扫描fna找到目标基因
        with open(fna_path, 'r') as f:
            in_target = False
            seq_parts = []

            for line in f:
                line = line.strip()

                if line.startswith('>'):
                    current_id = line[1:].split()[0]

                    if in_target:
                        return ''.join(seq_parts)

                    if current_id == gene_id:
                        in_target = True
                        seq_parts = []

                elif in_target:
                    seq_parts.append(line)

            if in_target:
                return ''.join(seq_parts)

        return None


# ==================== 上下文特征提取器（修复版） ====================

class ContextFeatureExtractor:
    """修复版：基于预建索引的上下文特征提取"""

    def __init__(self, config: FeatureConfig, logger: logging.Logger, index_builder: FNAIndexBuilder):
        self.config = config
        self.logger = logger
        self.index_builder = index_builder

        # k-mer词汇表
        self.vocab = self._build_kmer_vocab()
        np.random.seed(42)
        self.projection = np.random.randn(len(self.vocab), config.context_dim) / np.sqrt(len(self.vocab))

    def _build_kmer_vocab(self) -> Dict[str, int]:
        import itertools
        vocab = {}
        idx = 0
        bases = ['A', 'T', 'C', 'G']
        for k in self.config.kmer_sizes:
            for kmer in itertools.product(bases, repeat=k):
                vocab[''.join(kmer)] = idx
                idx += 1
        return vocab

    def extract_batch(self, gene_ids: List[str]) -> np.ndarray:
        """批量提取上下文特征"""
        results = np.zeros((len(gene_ids), self.config.context_dim), dtype=np.float32)

        # 收集所有需要的邻居基因
        all_neighbor_ids = set()
        gene_to_neighbors = {}

        for gid in gene_ids:
            neighbors = self.index_builder.get_neighbor_gene_ids(gid, self.config.context_window)
            gene_to_neighbors[gid] = neighbors
            all_neighbor_ids.update(neighbors)

        self.logger.info(f"  需要提取 {len(all_neighbor_ids)} 个邻居序列...")

        # 批量提取邻居序列（按fna文件分组，减少磁盘IO）
        neighbor_sequences = self._batch_extract_sequences(all_neighbor_ids)

        # 编码每个基因的上下文
        for i, gid in enumerate(gene_ids):
            neighbors = gene_to_neighbors.get(gid, [])
            if not neighbors:
                continue

            seqs = [neighbor_sequences.get(nid, '') for nid in neighbors if nid in neighbor_sequences]
            if seqs:
                results[i] = self._encode_sequences(seqs)

        return results

    def _batch_extract_sequences(self, gene_ids: Set[str]) -> Dict[str, str]:
        """批量提取基因序列，按fna文件分组读取"""
        # 按fna文件分组
        fna_groups = {}
        for gid in gene_ids:
            if gid not in self.index_builder.index:
                continue
            fna_path = self.index_builder.index[gid]['fna_path']
            if fna_path not in fna_groups:
                fna_groups[fna_path] = set()
            fna_groups[fna_path].add(gid)

        # 逐个fna文件读取
        sequences = {}
        for fna_path, targets in tqdm(fna_groups.items(), desc="读取fna序列"):
            file_sequences = self._read_target_sequences_from_fna(fna_path, targets)
            sequences.update(file_sequences)

        return sequences

    def _read_target_sequences_from_fna(self, fna_path: str, target_ids: Set[str]) -> Dict[str, str]:
        """从单个fna读取目标序列"""
        sequences = {}

        with open(fna_path, 'r') as f:
            current_id = None
            current_seq = []

            for line in f:
                line = line.strip()

                if line.startswith('>'):
                    if current_id and current_id in target_ids:
                        sequences[current_id] = ''.join(current_seq)

                    header = line[1:]
                    current_id = header.split()[0]
                    current_seq = []

                    if current_id in target_ids:
                        # 继续读取
                        pass
                    else:
                        current_id = None

                elif current_id:
                    current_seq.append(line)

            # 最后一个
            if current_id and current_id in target_ids:
                sequences[current_id] = ''.join(current_seq)

        return sequences

    def _encode_sequences(self, sequences: List[str]) -> np.ndarray:
        """k-mer频率编码 + 随机投影"""
        kmer_counts = {}
        total = 0

        for seq in sequences:
            seq = seq.upper().replace('N', '')
            for k in self.config.kmer_sizes:
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


# ==================== 特征对齐器 ====================

class FeatureAligner:
    def __init__(self, config: FeatureConfig, logger: logging.Logger, index_builder: FNAIndexBuilder):
        self.config = config
        self.logger = logger
        self.context_extractor = ContextFeatureExtractor(config, logger, index_builder)

    def align_category(self, category: str, esm2_features: np.ndarray,
                       gene_ids: List[str]) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        self.logger.info(f"\n对齐类别: {category}")
        n = len(gene_ids)
        self.logger.info(f"  ESM2特征: {esm2_features.shape}")
        self.logger.info(f"  基因数: {n}")

        context_features = np.zeros((n, self.config.context_dim), dtype=np.float32)

        if category.startswith('hard_'):
            self.logger.info(f"  批量提取难样本上下文特征...")
            context_features = self.context_extractor.extract_batch(gene_ids)

        combined = np.concatenate([esm2_features, context_features], axis=1)

        label_map = {'positive': 1, 'negative': 0, 'hard_strict': 2, 'hard_expanded': 2}
        labels = np.full(n, label_map.get(category, -1), dtype=np.int64)

        metadata = pd.DataFrame({
            'gene_id': gene_ids,
            'category': category,
            'label': labels,
            'has_context': [not np.allclose(ctx, 0) for ctx in context_features],
        })

        self.logger.info(f"  合并特征: {combined.shape}")
        self.logger.info(f"  有上下文: {metadata['has_context'].sum()}/{n}")

        return combined, labels, metadata


# ==================== 主流程 ====================

class FeatureExtractionPipeline:
    def __init__(self, config: FeatureConfig):
        self.config = config
        self.logger = setup_logger(
            "feature_extraction",
            f"{config.output_dir}/logs",
            config.log_level
        )
        self.fasta_parser = FastaParser()
        self.fna_index_builder = None  # 延迟初始化
        self.aligner = None

    def run(self) -> Dict:
        self.logger.info("=" * 80)
        self.logger.info("模块4: ESM2 + 上下文特征提取 (修复版)")
        self.logger.info(f"开始时间: {datetime.now()}")
        self.logger.info(f"GPU: {self.config.gpu_ids}")
        self.logger.info("=" * 80)

        all_features = []
        all_labels = []
        all_metadata = []

        # 收集所有难样本基因ID（用于预建索引）
        hard_gene_ids = self._collect_hard_gene_ids()
        self.logger.info(f"难样本基因总数: {len(hard_gene_ids):,}")

        # 预建fna索引
        if hard_gene_ids:
            self.fna_index_builder = FNAIndexBuilder(self.config.genes_fna_dir, self.logger)
            self.fna_index_builder.build_index(target_gene_ids=hard_gene_ids)
            self.aligner = FeatureAligner(self.config, self.logger, self.fna_index_builder)

        for category in self.config.categories:
            # 检查是否已完成
            if self._is_category_done(category):
                self.logger.info(f"\n{'=' * 60}")
                self.logger.info(f"类别 {category} 已完成，跳过...")
                self.logger.info(f"{'=' * 60}")

                # 加载已有结果
                combined = np.load(f"{self.config.output_dir}/{category}/combined_features.npy")
                meta = pd.read_csv(f"{self.config.output_dir}/{category}/metadata.csv")
                labels = meta['label'].values

                all_features.append(combined)
                all_labels.append(labels)
                all_metadata.append(meta)
                continue

            result = self._process_category(category)
            all_features.append(result['features'])
            all_labels.append(result['labels'])
            all_metadata.append(result['metadata'])

        # 合并所有类别
        self.logger.info(f"\n{'=' * 60}")
        self.logger.info("合并所有类别...")

        final_features = np.vstack(all_features)
        final_labels = np.concatenate(all_labels)
        final_metadata = pd.concat(all_metadata, ignore_index=True)

        self._save_final(final_features, final_labels, final_metadata)

        self.logger.info(f"\n{'=' * 80}")
        self.logger.info("模块4完成!")
        self.logger.info(f"总样本: {len(final_features):,}")
        self.logger.info(f"特征维度: {final_features.shape[1]}")
        self.logger.info(f"类别分布: {dict(final_metadata['category'].value_counts())}")
        self.logger.info(f"{'=' * 80}")

        return {
            'feature_shape': final_features.shape,
            'total_samples': len(final_features),
            'output_dir': self.config.output_dir,
        }

    def _collect_hard_gene_ids(self) -> Set[str]:
        """收集所有难样本基因ID"""
        hard_ids = set()
        for cat in ['hard_strict', 'hard_expanded']:
            fasta_file = f"{self.config.clustered_dir}/{cat}_cdhit90.fasta"
            if os.path.exists(fasta_file):
                ids = self.fasta_parser.parse_ids_only(fasta_file)
                hard_ids.update(ids)
        return hard_ids

    def _is_category_done(self, category: str) -> bool:
        """检查类别是否已完成"""
        combined_path = f"{self.config.output_dir}/{category}/combined_features.npy"
        meta_path = f"{self.config.output_dir}/{category}/metadata.csv"
        return os.path.exists(combined_path) and os.path.exists(meta_path)

    def _process_category(self, category: str) -> Dict:
        self.logger.info(f"\n{'=' * 60}")
        self.logger.info(f"处理类别: {category}")
        self.logger.info(f"{'=' * 60}")

        fasta_file = f"{self.config.clustered_dir}/{category}_cdhit90.fasta"
        sequences = self.fasta_parser.parse(fasta_file)
        gene_ids = [s['id'] for s in sequences]

        self.logger.info(f"序列数: {len(sequences):,}")

        # ESM2特征提取
        gpu_id = 0 if category in ('positive', 'negative') else 1
        extractor = ESM2FeatureExtractor(self.config, self.logger, gpu_id)

        try:
            esm2_features = extractor.extract(sequences)
        finally:
            extractor.cleanup()

        # 保存ESM2特征
        esm2_path = f"{self.config.output_dir}/{category}/esm2_features.npy"
        np.save(esm2_path, esm2_features)
        self.logger.info(f"ESM2保存: {esm2_path}")

        ids_df = pd.DataFrame({'gene_id': gene_ids})
        ids_df.to_csv(f"{self.config.output_dir}/{category}/gene_ids.csv", index=False)

        # 对齐特征
        if self.aligner:
            combined, labels, metadata = self.aligner.align_category(category, esm2_features, gene_ids)
        else:
            # 正/负样本无上下文
            context = np.zeros((len(gene_ids), self.config.context_dim), dtype=np.float32)
            combined = np.concatenate([esm2_features, context], axis=1)
            label_map = {'positive': 1, 'negative': 0, 'hard_strict': 2, 'hard_expanded': 2}
            labels = np.full(len(gene_ids), label_map.get(category, -1), dtype=np.int64)
            metadata = pd.DataFrame({
                'gene_id': gene_ids,
                'category': category,
                'label': labels,
                'has_context': False,
            })

        # 保存
        combined_path = f"{self.config.output_dir}/{category}/combined_features.npy"
        np.save(combined_path, combined)
        metadata.to_csv(f"{self.config.output_dir}/{category}/metadata.csv", index=False)

        return {
            'category': category,
            'features': combined,
            'labels': labels,
            'metadata': metadata,
        }

    def _save_final(self, features: np.ndarray, labels: np.ndarray, metadata: pd.DataFrame):
        self.logger.info(f"\n保存最终合并结果...")
        np.save(f"{self.config.output_dir}/all_features.npy", features)
        np.save(f"{self.config.output_dir}/all_labels.npy", labels)
        metadata.to_csv(f"{self.config.output_dir}/all_metadata.csv", index=False)

        stats = {
            'timestamp': datetime.now().isoformat(),
            'feature_shape': features.shape,
            'total_samples': len(features),
            'feature_dim': features.shape[1],
            'esm2_dim': 1280,
            'context_dim': self.config.context_dim,
            'category_distribution': metadata['category'].value_counts().to_dict(),
            'label_distribution': metadata['label'].value_counts().to_dict(),
            'context_coverage': metadata['has_context'].mean(),
        }

        with open(f"{self.config.output_dir}/extraction_report.json", 'w') as f:
            json.dump(stats, f, indent=2)

        self.logger.info(f"报告: {self.config.output_dir}/extraction_report.json")


# ==================== 命令行入口 ====================

def main():
    parser = argparse.ArgumentParser(description='模块4: ESM2 + 上下文特征提取 (修复版)')
    parser.add_argument('--clustered-dir',
                        default="/home/zjw/zjwdata/3_deep_learning/training_data/curated_v2/clustered")
    parser.add_argument('--genes-fna-dir',
                        default="/home/zjw/zjwdata/1/assembly_analysis/genes")
    parser.add_argument('--output-dir',
                        default="/home/zjw/zjwdata/3_deep_learning/training_data/curated_v2/features")
    parser.add_argument('--gpu-ids', nargs='+', type=int, default=[0, 1])
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--max-length', type=int, default=1024)

    args = parser.parse_args()

    config = FeatureConfig(
        clustered_dir=args.clustered_dir,
        genes_fna_dir=args.genes_fna_dir,
        output_dir=args.output_dir,
        gpu_ids=tuple(args.gpu_ids),
        esm2_batch_size=args.batch_size,
        esm2_max_length=args.max_length,
    )

    pipeline = FeatureExtractionPipeline(config)
    results = pipeline.run()

    return results


if __name__ == "__main__":
    main()