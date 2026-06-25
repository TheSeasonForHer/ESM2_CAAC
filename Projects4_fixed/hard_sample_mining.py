#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模块2: hard_sample_mining.py（修复版 v2.0）
功能: 难样本挖掘 + 上下文特征索引构建

修改点:
1. 输出目录同步为 curated_v2
2. 添加模块1正样本参考（用于难样本注释质量对比）
3. 上下文特征维度统一（正/负样本无上下文时用零向量）
4. 难样本作为GNN第三类标签（0=负, 1=正, 2=难）
5. 【修复】全局截断改为 Round-robin 轮流分配，确保所有样本都有代表
"""

import os
import sys
import re
import json
import argparse
import logging
from datetime import datetime
from collections import defaultdict, Counter, deque
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd
from tqdm import tqdm


# ==================== 配置类 ====================

@dataclass
class HardSampleConfig:
    """难样本挖掘配置"""
    # 输入路径（与模块1衔接）
    base_dir: str = "/home/zjw/zjwdata"

    # 模块1输出路径（新增）
    module1_output_dir: str = "/home/zjw/zjwdata/3_deep_learning/training_data/curated_v2"
    positive_info_tsv: str = "/home/zjw/zjwdata/3_deep_learning/training_data/curated_v2/positive_samples_info.tsv"

    # 宏基因组数据路径
    gene_catalog_dir: str = "/home/zjw/zjwdata/1/gene_catalog_analysis/per_sample_non_redundant_genes"
    annotation_dir: str = "/home/zjw/zjwdata/2/gene_function/annotation_results/integrated"
    genes_fna_dir: str = "/home/zjw/zjwdata/1/assembly_analysis/genes"

    # 输出路径（同步为curated_v2）
    output_dir: str = "/home/zjw/zjwdata/3_deep_learning/training_data/curated_v2"

    # 样本列表
    samples: Tuple[str, ...] = (
        'CK-7A', 'CK-7B', 'CK-7C',
        'CK-90A', 'CK-90B', 'CK-90D',
        'FM-1', 'FM-2', 'FM-3',
        'M3-6023-7A', 'M3-6023-7B', 'M3-6023-7D',
        'M3-90-A', 'M3-90-B', 'M3-90C',
        'T-31-7A', 'T-31-7B', 'T-31-7C',
        'TR-31-90A', 'TR-31-90C', 'TR-31-90D'
    )

    # 序列质量控制
    min_seq_length: int = 100
    max_seq_length: int = 2000
    max_n_ratio: float = 0.05

    # 难样本提取参数
    max_per_sample: int = 100000
    max_total_strict: int = 500000
    max_total_expanded: int = 300000

    # 上下文参数
    context_window: int = 5
    max_neighbors: int = 10
    context_feature_dim: int = 128  # 与GNN输入维度一致

    # 低质量注释关键词
    low_quality_keywords: Tuple[str, ...] = (
        'DUF', 'UPF', 'protein of unknown function', 'uncharacterized',
        'unknown function', 'hypothetical', 'putative uncharacterized',
        'domain-containing', 'family protein', 'conserved protein',
        'predicted protein', 'probable protein', 'similar to',
    )

    # 高质量功能关键词（与模块1正样本关键词对齐）
    high_quality_keywords: Tuple[str, ...] = (
        'hydrolase', 'transferase', 'oxidoreductase', 'lyase', 'isomerase',
        'ligase', 'synthase', 'kinase', 'phosphatase', 'dehydrogenase',
        'carbohydrate', 'glycoside', 'cellulase', 'xylanase', 'amylase',
        'fermentation', 'glycolysis', 'metabolism',
        'GH', 'GT', 'CE', 'AA', 'CBM', 'PL',
        'transporter', 'permease', 'pump', 'channel',
        'binding protein', 'receptor', 'regulator', 'sensor',
    )

    log_level: str = "INFO"


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


# ==================== 模块1正样本参考加载器（新增） ====================

class PositiveReferenceLoader:
    """
    加载模块1的正样本信息，用于难样本注释质量评估
    提供: CAZy家族集合, EC号集合, 高质量关键词集合
    """

    def __init__(self, positive_tsv_path: str, logger: logging.Logger):
        self.logger = logger
        self.cazy_families: Set[str] = set()
        self.ec_numbers: Set[str] = set()
        self.keywords: Set[str] = set()
        self._load(positive_tsv_path)

    def _load(self, filepath: str):
        """加载正样本TSV"""
        if not os.path.exists(filepath):
            self.logger.warning(f"正样本参考文件不存在: {filepath}")
            return

        self.logger.info(f"加载正样本参考: {filepath}")
        df = pd.read_csv(filepath, sep='\t', low_memory=False)

        # 解析CAZy家族
        if 'CAZy_Families' in df.columns:
            for _, row in df.iterrows():
                val = str(row['CAZy_Families'])
                if val and val != 'NA' and val != 'nan':
                    families = [f.strip() for f in val.split('|')]
                    self.cazy_families.update(families)

        # 解析EC号
        if 'EC_Numbers' in df.columns:
            for _, row in df.iterrows():
                val = str(row['EC_Numbers'])
                if val and val != 'NA' and val != 'nan':
                    ecs = [e.strip() for e in val.split('|')]
                    self.ec_numbers.update(ecs)

        # 解析关键词
        if 'Keywords' in df.columns:
            for _, row in df.iterrows():
                val = str(row['Keywords'])
                if val and val != 'NA' and val != 'nan':
                    kws = [k.strip() for k in val.split('|')]
                    self.keywords.update(kws)

        self.logger.info(f"正样本参考加载完成:")
        self.logger.info(f"  CAZy家族: {len(self.cazy_families)}")
        self.logger.info(f"  EC号: {len(self.ec_numbers)}")
        self.logger.info(f"  关键词: {len(self.keywords)}")

    def is_known_cazy_family(self, family: str) -> bool:
        """检查是否为已知的CAZy家族"""
        base = family.split('_')[0]
        return base in self.cazy_families or family in self.cazy_families

    def is_known_ec(self, ec: str) -> bool:
        """检查是否为已知的EC号"""
        return ec in self.ec_numbers

    def get_quality_hint(self, cazy_families: List[str], ec_numbers: List[str]) -> str:
        """根据与正样本的匹配程度给出质量提示"""
        if any(self.is_known_cazy_family(f) for f in cazy_families):
            return "high_confidence_cazy"
        if any(self.is_known_ec(e) for e in ec_numbers):
            return "high_confidence_ec"
        return "unknown"


# ==================== 数据结构 ====================

@dataclass
class GeneInfo:
    """基因信息"""
    gene_id: str
    sample: str
    sequence: str
    length: int
    quality_score: int = 0  # 0=无注释, 1=低质量, 2=高质量
    annotations: Dict = None
    # 新增: 与正样本的匹配信息
    positive_match_hint: str = "unknown"  # unknown, high_confidence_cazy, high_confidence_ec

    def __post_init__(self):
        if self.annotations is None:
            self.annotations = {}

    def to_fasta_header(self) -> str:
        quality_map = {0: 'Strict', 1: 'Expanded', 2: 'Annotated'}
        header = (f">{self.gene_id}|Sample={self.sample}|"
                  f"Length={self.length}|Quality={quality_map.get(self.quality_score, 'Unknown')}")
        if self.positive_match_hint != "unknown":
            header += f"|MatchHint={self.positive_match_hint}"
        header += "|HardSample"
        return header

    def to_dict(self) -> Dict:
        return {
            'gene_id': self.gene_id,
            'sample': self.sample,
            'length': self.length,
            'quality_score': self.quality_score,
            'positive_match_hint': self.positive_match_hint,
            'has_cazy': self.annotations.get('has_cazy', False),
            'has_kegg': self.annotations.get('has_kegg', False),
            'has_ec': self.annotations.get('has_ec', False),
            'has_any_function': self.annotations.get('has_any_function', False),
        }


@dataclass
class GenePosition:
    """基因位置信息"""
    gene_id: str
    contig_id: str
    start: int
    end: int
    strand: str
    gene_num: int
    sample: str

    def to_dict(self) -> Dict:
        return {
            'gene_id': self.gene_id,
            'contig_id': self.contig_id,
            'start': self.start,
            'end': self.end,
            'strand': self.strand,
            'gene_num': self.gene_num,
            'sample': self.sample,
        }


# ==================== 注释质量评估器（修复版） ====================

class AnnotationQualityAssessor:
    """
    修复版：使用模块1的正样本作为参考，更精确评估注释质量
    """

    def __init__(self, config: HardSampleConfig, logger: logging.Logger,
                 positive_ref: Optional[PositiveReferenceLoader] = None):
        self.config = config
        self.logger = logger
        self.positive_ref = positive_ref

        self.low_quality_pattern = re.compile(
            '|'.join(re.escape(kw) for kw in config.low_quality_keywords),
            re.IGNORECASE
        )
        self.high_quality_pattern = re.compile(
            '|'.join(re.escape(kw) for kw in config.high_quality_keywords),
            re.IGNORECASE
        )

    def assess(self, annotation_row: pd.Series, columns: Dict[str, str]) -> Tuple[int, str]:
        """
        评估注释质量
        返回: (quality_score, match_hint)
        quality_score: 0=无注释, 1=低质量, 2=高质量
        match_hint: 与正样本的匹配程度
        """
        # 检查是否有任何注释
        has_any = False
        cazy_fams = []
        ec_nums = []

        # CAZy
        has_cazy = False
        if 'cazy' in columns and columns['cazy'] in annotation_row.index:
            val = str(annotation_row[columns['cazy']])
            has_cazy = val and val not in ('-', 'nan', 'NA', '')
            if has_cazy:
                cazy_fams = [f.strip() for f in val.split('|') if f.strip()]
            has_any |= has_cazy

        # KEGG
        has_kegg = False
        if 'kegg' in columns and columns['kegg'] in annotation_row.index:
            val = str(annotation_row[columns['kegg']])
            has_kegg = val and val not in ('-', 'nan', 'NA', '')
            has_any |= has_kegg

        # EC
        has_ec = False
        if 'ec' in columns and columns['ec'] in annotation_row.index:
            val = str(annotation_row[columns['ec']])
            has_ec = val and val not in ('-', 'nan', 'NA', '')
            if has_ec:
                ec_nums = [e.strip() for e in val.split('|') if e.strip()]
            has_any |= has_ec

        # 描述/InterPro
        description = ""
        if 'interpro_desc' in columns and columns['interpro_desc'] in annotation_row.index:
            description = str(annotation_row[columns['interpro_desc']]).lower()
            has_any |= bool(description and description not in ('-', 'nan', 'na', ''))

        # 计算与正样本的匹配提示
        match_hint = "unknown"
        if self.positive_ref:
            match_hint = self.positive_ref.get_quality_hint(cazy_fams, ec_nums)

        if not has_any:
            return 0, match_hint  # 严格难样本

        # 检查描述质量
        if description:
            if self.high_quality_pattern.search(description):
                return 2, match_hint  # 高质量注释

            if self.low_quality_pattern.search(description):
                return 1, match_hint  # 扩展难样本

        # 有CAZy/KEGG/EC但描述为空或模糊
        if has_cazy or has_kegg or has_ec:
            # 如果与正样本高匹配，视为高质量
            if match_hint in ("high_confidence_cazy", "high_confidence_ec"):
                return 2, match_hint
            return 2, match_hint  # 有功能注释，视为高质量

        return 1, match_hint  # 默认低质量


# ==================== 基因位置解析器（保持不变） ====================

class GenePositionParser:
    POS_PATTERN_1 = re.compile(r'#\s*(\d+)\s*#\s*(\d+)\s*#\s*([+-]?1?)')
    POS_PATTERN_2 = re.compile(r'start[=:](\d+)[;:]end[=:](\d+)', re.IGNORECASE)
    POS_PATTERN_3 = re.compile(r'(\d+)[\.\-]+(\d+)')

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def parse_fna_file(self, fna_path: str, sample: str) -> Tuple[Dict[str, GenePosition], Dict[str, str]]:
        positions = {}
        sequences = {}

        if not os.path.exists(fna_path):
            self.logger.warning(f"文件不存在: {fna_path}")
            return positions, sequences

        current_id = None
        current_header = None
        current_seq = []

        with open(fna_path, 'r') as f:
            for line in f:
                line = line.strip()

                if line.startswith('>'):
                    if current_id:
                        sequences[current_id] = ''.join(current_seq)

                    current_header = line[1:]
                    current_id = self._extract_gene_id(current_header)
                    pos_info = self._parse_position(current_header, current_id)

                    if pos_info:
                        positions[current_id] = GenePosition(
                            gene_id=current_id,
                            contig_id=pos_info['contig_id'],
                            start=pos_info['start'],
                            end=pos_info['end'],
                            strand=pos_info['strand'],
                            gene_num=pos_info['gene_num'],
                            sample=sample,
                        )

                    current_seq = []
                else:
                    current_seq.append(line)

            if current_id:
                sequences[current_id] = ''.join(current_seq)

        return positions, sequences

    def _extract_gene_id(self, header: str) -> str:
        gene_id = header.split()[0]
        gene_id = gene_id.lstrip('>')
        return gene_id

    def _parse_position(self, header: str, gene_id: str) -> Optional[Dict]:
        match = self.POS_PATTERN_1.search(header)
        if match:
            start = int(match.group(1))
            end = int(match.group(2))
            strand = match.group(3) if match.group(3) else '1'
            contig_id = self._infer_contig(gene_id)
            gene_num = self._infer_gene_num(gene_id)
            return {
                'contig_id': contig_id,
                'start': start,
                'end': end,
                'strand': strand,
                'gene_num': gene_num,
            }

        match = self.POS_PATTERN_2.search(header)
        if match:
            start = int(match.group(1))
            end = int(match.group(2))
            contig_id = self._infer_contig(gene_id)
            gene_num = self._infer_gene_num(gene_id)
            return {
                'contig_id': contig_id,
                'start': start,
                'end': end,
                'strand': '+',
                'gene_num': gene_num,
            }

        contig_id = self._infer_contig(gene_id)
        gene_num = self._infer_gene_num(gene_id)

        if gene_num > 0:
            return {
                'contig_id': contig_id,
                'start': gene_num * 1000,
                'end': gene_num * 1000 + 500,
                'strand': '+',
                'gene_num': gene_num,
            }

        return None

    def _infer_contig(self, gene_id: str) -> str:
        parts = gene_id.rsplit('_', 1)
        if len(parts) == 2 and parts[1].isdigit():
            return parts[0]
        for sep in ['|', '#']:
            if sep in gene_id:
                return gene_id.split(sep)[0]
        return gene_id

    def _infer_gene_num(self, gene_id: str) -> int:
        parts = gene_id.rsplit('_', 1)
        if len(parts) == 2 and parts[1].isdigit():
            return int(parts[1])
        return 0

    def build_contig_index(self, all_positions: Dict[str, GenePosition]) -> Dict[str, List[GenePosition]]:
        contig_dict = defaultdict(list)

        for pos in all_positions.values():
            contig_dict[pos.contig_id].append(pos)

        for contig_id in contig_dict:
            contig_dict[contig_id].sort(key=lambda x: x.start)
            for idx, pos in enumerate(contig_dict[contig_id]):
                pos.gene_num = idx

        return dict(contig_dict)


# ==================== 上下文特征提取器（按需计算） ====================

class ContextFeatureExtractor:
    """
    上下文特征提取器 - 按需计算版
    关键修复: 正/负样本无上下文时用零向量填充，确保维度一致
    """

    def __init__(self, config: HardSampleConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.position_parser = GenePositionParser(logger)

        # k-mer编码器
        self.kmer_sizes = [3, 4, 5]
        self.vocab = self._build_kmer_vocab()

    def _build_kmer_vocab(self) -> Dict[str, int]:
        import itertools
        vocab = {}
        idx = 0
        bases = ['A', 'T', 'C', 'G']

        for k in self.kmer_sizes:
            for kmer in itertools.product(bases, repeat=k):
                vocab[''.join(kmer)] = idx
                idx += 1

        self.logger.info(f"K-mer词汇表大小: {len(vocab)}")
        return vocab

    def extract_context_for_gene(
            self,
            gene_id: str,
            contig_index: Dict[str, List[GenePosition]],
            all_sequences: Dict[str, str],
            all_positions: Dict[str, GenePosition]
    ) -> Optional[np.ndarray]:
        """为单个基因提取上下文特征"""
        if gene_id not in all_positions:
            return None

        target_pos = all_positions[gene_id]
        contig_id = target_pos.contig_id

        if contig_id not in contig_index:
            return None

        contig_genes = contig_index[contig_id]

        target_idx = -1
        for idx, pos in enumerate(contig_genes):
            if pos.gene_id == gene_id:
                target_idx = idx
                break

        if target_idx < 0:
            return None

        window = self.config.context_window
        start_idx = max(0, target_idx - window)
        end_idx = min(len(contig_genes), target_idx + window + 1)

        neighbor_seqs = []
        for i in range(start_idx, end_idx):
            if i == target_idx:
                continue

            neighbor_id = contig_genes[i].gene_id
            if neighbor_id in all_sequences:
                neighbor_seqs.append(all_sequences[neighbor_id])

        if not neighbor_seqs:
            return np.zeros(self.config.context_feature_dim)

        features = self._encode_neighbor_sequences(neighbor_seqs)
        return features

    def _encode_neighbor_sequences(self, sequences: List[str]) -> np.ndarray:
        """编码邻居序列为特征向量（降维到128维）"""
        kmer_counts = defaultdict(int)
        total_kmers = 0

        for seq in sequences:
            seq = seq.upper().replace('N', '')
            for k in self.kmer_sizes:
                for i in range(len(seq) - k + 1):
                    kmer = seq[i:i + k]
                    if kmer in self.vocab:
                        kmer_counts[kmer] += 1
                        total_kmers += 1

        vec = np.zeros(len(self.vocab))
        if total_kmers > 0:
            for kmer, count in kmer_counts.items():
                vec[self.vocab[kmer]] = count / total_kmers

        # 随机投影到128维
        np.random.seed(42)
        projection = np.random.randn(len(self.vocab), self.config.context_feature_dim) / np.sqrt(len(self.vocab))
        reduced = vec @ projection

        norm = np.linalg.norm(reduced)
        if norm > 0:
            reduced = reduced / norm

        return reduced

    def extract_batch(
            self,
            gene_ids: List[str],
            sample: str
    ) -> Dict[str, np.ndarray]:
        """批量提取上下文特征（按需）"""
        fna_path = f"{self.config.genes_fna_dir}/{sample}_genes.fna"

        positions, sequences = self.position_parser.parse_fna_file(fna_path, sample)
        contig_index = self.position_parser.build_contig_index(positions)

        results = {}
        for gene_id in gene_ids:
            feat = self.extract_context_for_gene(gene_id, contig_index, sequences, positions)
            if feat is not None:
                results[gene_id] = feat

        return results


# ==================== 难样本挖掘器（修复版） ====================

class HardSampleMiner:
    """难样本挖掘器"""

    def __init__(self, config: HardSampleConfig, logger: logging.Logger,
                 positive_ref: Optional[PositiveReferenceLoader] = None):
        self.config = config
        self.logger = logger
        self.quality_assessor = AnnotationQualityAssessor(config, logger, positive_ref)

    def mine_for_sample(self, sample: str) -> Tuple[List[GeneInfo], List[GeneInfo], int, int]:
        """为单个样本挖掘难样本"""
        self.logger.info(f"\n处理样本: {sample}")

        gene_file = f"{self.config.gene_catalog_dir}/{sample}/{sample}_non_redundant_genes.faa"
        anno_file = f"{self.config.annotation_dir}/{sample}/{sample}.integrated_annotations.tsv"

        if not os.path.exists(gene_file):
            self.logger.warning(f"  找不到基因文件: {gene_file}")
            return [], [], 0, 0

        annotations = self._load_annotations(anno_file)

        strict_hard = []
        expanded_hard = []
        total_genes = 0
        annotated_genes = 0

        with open(gene_file, 'r') as f:
            current_id = None
            current_seq = []

            for line in f:
                line = line.strip()

                if line.startswith('>'):
                    if current_id:
                        total_genes += 1
                        gene_info = self._process_gene(
                            current_id, current_seq, sample, annotations
                        )

                        if gene_info:
                            if gene_info.quality_score == 0:
                                strict_hard.append(gene_info)
                            elif gene_info.quality_score == 1:
                                expanded_hard.append(gene_info)

                            if gene_info.annotations.get('has_any_function', False):
                                annotated_genes += 1

                    current_id = line[1:].split()[0]
                    current_seq = []
                else:
                    current_seq.append(line)

            if current_id:
                total_genes += 1
                gene_info = self._process_gene(
                    current_id, current_seq, sample, annotations
                )
                if gene_info:
                    if gene_info.quality_score == 0:
                        strict_hard.append(gene_info)
                    elif gene_info.quality_score == 1:
                        expanded_hard.append(gene_info)

                    if gene_info.annotations.get('has_any_function', False):
                        annotated_genes += 1

        max_per = self.config.max_per_sample
        if len(strict_hard) > max_per // 2:
            strict_hard = strict_hard[:max_per // 2]
        if len(expanded_hard) > max_per // 2:
            expanded_hard = expanded_hard[:max_per // 2]

        self.logger.info(f"  总基因: {total_genes:,} | 已注释: {annotated_genes:,}")
        self.logger.info(f"  严格难样本: {len(strict_hard):,} | 扩展难样本: {len(expanded_hard):,}")

        return strict_hard, expanded_hard, total_genes, annotated_genes

    def _load_annotations(self, anno_file: str) -> pd.DataFrame:
        if not os.path.exists(anno_file):
            return pd.DataFrame()

        try:
            df = pd.read_csv(anno_file, sep='\t', low_memory=False)
            return df
        except Exception as e:
            self.logger.error(f"  注释文件加载失败: {e}")
            return pd.DataFrame()

    def _process_gene(
            self,
            gene_id: str,
            seq_parts: List[str],
            sample: str,
            annotations: pd.DataFrame
    ) -> Optional[GeneInfo]:
        """处理单个基因"""
        sequence = ''.join(seq_parts)
        seq_len = len(sequence)

        if seq_len < self.config.min_seq_length or seq_len > self.config.max_seq_length:
            return None

        n_ratio = (sequence.upper().count('N') + sequence.upper().count('X')) / seq_len
        if n_ratio > self.config.max_n_ratio:
            return None

        anno_row = None
        if not annotations.empty:
            id_col = annotations.columns[0]
            matches = annotations[annotations[id_col] == gene_id]
            if len(matches) > 0:
                anno_row = matches.iloc[0]

        if anno_row is not None:
            columns = {}
            for col in annotations.columns:
                col_lower = col.lower()
                if 'cazy' in col_lower:
                    columns['cazy'] = col
                elif 'kegg' in col_lower or 'ko' in col_lower:
                    columns['kegg'] = col
                elif col_lower == 'ec' or 'ec_' in col_lower:
                    columns['ec'] = col
                elif 'interpro' in col_lower and 'desc' in col_lower:
                    columns['interpro_desc'] = col

            quality_score, match_hint = self.quality_assessor.assess(anno_row, columns)

            has_cazy = columns.get('cazy') and str(anno_row[columns['cazy']]) not in ('-', 'nan', 'NA', '')
            has_kegg = columns.get('kegg') and str(anno_row[columns['kegg']]) not in ('-', 'nan', 'NA', '')
            has_ec = columns.get('ec') and str(anno_row[columns['ec']]) not in ('-', 'nan', 'NA', '')

            gene_annotations = {
                'has_cazy': bool(has_cazy),
                'has_kegg': bool(has_kegg),
                'has_ec': bool(has_ec),
                'has_any_function': bool(has_cazy or has_kegg or has_ec),
            }
        else:
            quality_score = 0
            match_hint = "unknown"
            gene_annotations = {
                'has_cazy': False,
                'has_kegg': False,
                'has_ec': False,
                'has_any_function': False,
            }

        return GeneInfo(
            gene_id=gene_id,
            sample=sample,
            sequence=sequence,
            length=seq_len,
            quality_score=quality_score,
            annotations=gene_annotations,
            positive_match_hint=match_hint,
        )


# ==================== 数据写入器（修复版） ====================

class HardSampleWriter:
    """难样本数据写入器"""

    def __init__(self, output_dir: str, logger: logging.Logger):
        self.output_dir = output_dir
        self.logger = logger
        os.makedirs(output_dir, exist_ok=True)

    def write_hard_samples(
            self,
            strict_hard: List[GeneInfo],
            expanded_hard: List[GeneInfo],
            sample_stats: List[Dict]
    ):
        """写入难样本文件"""

        # 严格难样本 FASTA
        strict_path = f"{self.output_dir}/hard_samples_strict.fasta"
        with open(strict_path, 'w') as f:
            for gene in strict_hard:
                f.write(gene.to_fasta_header() + "\n")
                for i in range(0, len(gene.sequence), 60):
                    f.write(gene.sequence[i:i + 60] + "\n")
        self.logger.info(f"严格难样本: {len(strict_hard):,} -> {strict_path}")

        # 扩展难样本 FASTA
        expanded_path = f"{self.output_dir}/hard_samples_expanded.fasta"
        with open(expanded_path, 'w') as f:
            for gene in expanded_hard:
                f.write(gene.to_fasta_header() + "\n")
                for i in range(0, len(gene.sequence), 60):
                    f.write(gene.sequence[i:i + 60] + "\n")
        self.logger.info(f"扩展难样本: {len(expanded_hard):,} -> {expanded_path}")

        # 合并版 FASTA（LABEL=2标记，用于GNN第三类）
        combined_path = f"{self.output_dir}/hard_samples_combined.fasta"
        with open(combined_path, 'w') as f:
            for gene in strict_hard + expanded_hard:
                header = f">{gene.gene_id}|LABEL=2|Sample={gene.sample}|Length={gene.length}|HardSample"
                if gene.positive_match_hint != "unknown":
                    header += f"|MatchHint={gene.positive_match_hint}"
                f.write(header + "\n")
                for i in range(0, len(gene.sequence), 60):
                    f.write(gene.sequence[i:i + 60] + "\n")
        self.logger.info(f"合并难样本: {len(strict_hard) + len(expanded_hard):,} -> {combined_path}")

        # 元数据 TSV
        all_hard = strict_hard + expanded_hard
        metadata_df = pd.DataFrame([g.to_dict() for g in all_hard])
        metadata_path = f"{self.output_dir}/hard_samples_metadata.tsv"
        metadata_df.to_csv(metadata_path, sep='\t', index=False)
        self.logger.info(f"元数据: {metadata_path}")

        # 基因位置索引
        self._write_position_index(all_hard)

        # 上下文特征提取配置（新增: 包含维度信息，与模块1衔接）
        self._write_context_config()

        # 统计报告
        self._write_stats(strict_hard, expanded_hard, sample_stats)

    def _write_position_index(self, all_hard: List[GeneInfo]):
        """写入基因位置索引"""
        sample_genes = defaultdict(list)
        for gene in all_hard:
            sample_genes[gene.sample].append(gene.gene_id)

        index = {
            'description': 'Gene position index for on-demand context feature extraction',
            'format': 'fna files contain gene positions in headers',
            'context_feature_dim': 128,  # 与模块1正样本零向量维度一致
            'samples': {},
        }

        for sample, gene_ids in sample_genes.items():
            index['samples'][sample] = {
                'fna_path': f"/home/zjw/zjwdata/1/assembly_analysis/genes/{sample}_genes.fna",
                'gene_count': len(gene_ids),
                'gene_ids': gene_ids,
            }

        index_path = f"{self.output_dir}/context_index.json"
        with open(index_path, 'w') as f:
            json.dump(index, f, indent=2)

        self.logger.info(f"位置索引: {index_path}")

    def _write_context_config(self):
        """写入上下文特征提取配置"""
        config = {
            'description': 'Configuration for on-demand context feature extraction',
            'method': 'k-mer frequency encoding with random projection',
            'kmer_sizes': [3, 4, 5],
            'output_dim': 128,
            'context_window': 5,
            'max_neighbors': 10,
            'extraction_class': 'ContextFeatureExtractor',
            'usage': 'Load ContextFeatureExtractor and call extract_batch(gene_ids, sample)',
            'note': 'Positive/negative samples from Module 1 will use zero vectors for context features',
        }

        config_path = f"{self.output_dir}/context_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        self.logger.info(f"上下文配置: {config_path}")

    def _write_stats(
            self,
            strict_hard: List[GeneInfo],
            expanded_hard: List[GeneInfo],
            sample_stats: List[Dict]
    ):
        stats_path = f"{self.output_dir}/hard_samples_stats.txt"

        total_strict = len(strict_hard)
        total_expanded = len(expanded_hard)
        total_hard = total_strict + total_expanded

        length_dist = Counter()
        for gene in strict_hard + expanded_hard:
            if gene.length < 200:
                length_dist['100-200'] += 1
            elif gene.length < 500:
                length_dist['200-500'] += 1
            elif gene.length < 1000:
                length_dist['500-1000'] += 1
            else:
                length_dist['1000+'] += 1

        # 匹配提示分布
        match_hint_dist = Counter()
        for gene in strict_hard + expanded_hard:
            match_hint_dist[gene.positive_match_hint] += 1

        with open(stats_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("难样本挖掘统计报告\n")
            f.write(f"生成时间: {datetime.now()}\n")
            f.write("=" * 70 + "\n\n")

            f.write("【难样本定义】\n")
            f.write("严格难样本 (Quality=Strict):\n")
            f.write("  - 完全无任何注释\n")
            f.write("  - 真正的功能'暗物质'\n\n")
            f.write("扩展难样本 (Quality=Expanded):\n")
            f.write("  - 有低质量注释：DUF家族、未知功能蛋白\n")
            f.write("  - 无明确功能描述\n\n")

            f.write("【整体统计】\n")
            total_genes = sum(s['total_genes'] for s in sample_stats)
            total_annotated = sum(s['annotated'] for s in sample_stats)
            f.write(f"总基因数: {total_genes:,}\n")
            f.write(f"已注释基因: {total_annotated:,} ({100 * total_annotated / max(total_genes, 1):.1f}%)\n")
            f.write(f"严格难样本: {total_strict:,} ({100 * total_strict / max(total_genes, 1):.1f}%)\n")
            f.write(f"扩展难样本: {total_expanded:,} ({100 * total_expanded / max(total_genes, 1):.1f}%)\n")
            f.write(f"总难样本: {total_hard:,} ({100 * total_hard / max(total_genes, 1):.1f}%)\n\n")

            f.write("【正样本匹配提示分布】\n")
            for hint, count in match_hint_dist.most_common():
                f.write(f"  {hint}: {count:,}\n")
            f.write("\n")

            f.write("【各样本统计】\n")
            for stat in sample_stats:
                f.write(f"{stat['sample']:12s}: {stat['total_genes']:>6,} 基因 | ")
                f.write(f"注释 {stat['annotated']:>6,} | ")
                f.write(f"严格 {stat['strict']:>5,} | ")
                f.write(f"扩展 {stat['expanded']:>5,} | ")
                f.write(f"难样本率 {stat['hard_rate']:>5.1f}%\n")
            f.write("\n")

            f.write("【长度分布】\n")
            for range_name, count in sorted(length_dist.items()):
                f.write(f"  {range_name}: {count:,} ({100 * count / max(total_hard, 1):.1f}%)\n")
            f.write("\n")

            f.write("【GNN标签说明】\n")
            f.write("  难样本在GNN中标记为 LABEL=2\n")
            f.write("  与模块1正样本(LABEL=1)和负样本(LABEL=0)共同构成3分类任务\n")
            f.write("  上下文特征维度: 128维（正/负样本用零向量填充）\n\n")

            f.write("【输出文件】\n")
            f.write(f"严格难样本: {self.output_dir}/hard_samples_strict.fasta\n")
            f.write(f"扩展难样本: {self.output_dir}/hard_samples_expanded.fasta\n")
            f.write(f"合并难样本: {self.output_dir}/hard_samples_combined.fasta\n")
            f.write(f"元数据: {self.output_dir}/hard_samples_metadata.tsv\n")
            f.write(f"位置索引: {self.output_dir}/context_index.json\n")
            f.write(f"上下文配置: {self.output_dir}/context_config.json\n")

        self.logger.info(f"统计报告: {stats_path}")


# ==================== 主流程（修复版 v2.0） ====================

class HardSampleMiningPipeline:
    """难样本挖掘主流程"""

    def __init__(self, config: HardSampleConfig):
        self.config = config
        self.logger = setup_logger("hard_sample_mining", f"{config.output_dir}/logs", config.log_level)

        # 加载模块1正样本参考（新增）
        self.positive_ref = None
        if os.path.exists(config.positive_info_tsv):
            self.positive_ref = PositiveReferenceLoader(config.positive_info_tsv, self.logger)
        else:
            self.logger.warning(f"模块1正样本参考未找到: {config.positive_info_tsv}")
            self.logger.warning("难样本评估将不使用正样本参考")

        self.miner = HardSampleMiner(config, self.logger, self.positive_ref)
        self.writer = HardSampleWriter(config.output_dir, self.logger)

    def _truncate_round_robin(self, all_genes: List[GeneInfo], max_total: int) -> List[GeneInfo]:
        """
        【核心修复】按样本轮流截断，确保每个样本至少保留部分序列，避免后进先截
        策略: 从每个样本队列头部轮流取1条，直到达到上限或所有样本耗尽
        """
        if len(all_genes) <= max_total:
            return all_genes

        # 按样本分组为队列（保持原顺序）
        by_sample = defaultdict(deque)
        for g in all_genes:
            by_sample[g.sample].append(g)

        result = []
        samples = list(by_sample.keys())
        sample_idx = 0
        rounds = 0

        # 轮流从每个样本取1条
        while len(result) < max_total and any(by_sample[s] for s in samples):
            sample = samples[sample_idx % len(samples)]
            if by_sample[sample]:
                result.append(by_sample[sample].popleft())
            sample_idx += 1
            if sample_idx % len(samples) == 0:
                rounds += 1

        self.logger.info(f"  Round-robin 截断: {len(all_genes):,} -> {len(result):,} "
                        f"(保留 {len(samples)} 个样本, 最多 {rounds} 轮)")

        # 统计各样本保留数量
        sample_counts = defaultdict(int)
        for g in result:
            sample_counts[g.sample] += 1
        for s, c in sorted(sample_counts.items()):
            self.logger.info(f"    {s}: {c:,} 条")

        return result

    def run(self):
        self.logger.info("=" * 80)
        self.logger.info("模块2: 难样本挖掘与上下文索引构建（修复版 v2.0）")
        self.logger.info(f"开始时间: {datetime.now()}")
        self.logger.info("修复内容: 输出目录同步, 正样本参考, GNN 3分类标签, Round-robin全局截断")
        self.logger.info("=" * 80)

        all_strict = []
        all_expanded = []
        sample_stats = []

        for idx, sample in enumerate(self.config.samples, 1):
            self.logger.info(f"\n{'-' * 60}")
            self.logger.info(f"[{idx}/{len(self.config.samples)}] 处理 {sample}")
            self.logger.info(f"{'-' * 60}")

            strict, expanded, total, annotated = self.miner.mine_for_sample(sample)

            total_hard = len(strict) + len(expanded)
            hard_rate = 100 * total_hard / max(total, 1)

            sample_stats.append({
                'sample': sample,
                'total_genes': total,
                'annotated': annotated,
                'strict': len(strict),
                'expanded': len(expanded),
                'total_hard': total_hard,
                'hard_rate': hard_rate,
            })

            all_strict.extend(strict)
            all_expanded.extend(expanded)

        # 【修复】全局截断改为 Round-robin 轮流分配
        self.logger.info(f"\n{'=' * 60}")
        self.logger.info("全局截断（Round-robin 按比例分配）")
        self.logger.info(f"{'=' * 60}")

        all_strict = self._truncate_round_robin(all_strict, self.config.max_total_strict)
        all_expanded = self._truncate_round_robin(all_expanded, self.config.max_total_expanded)

        # 写入输出
        self.logger.info(f"\n{'=' * 60}")
        self.logger.info("写入输出文件")
        self.logger.info(f"{'=' * 60}")

        self.writer.write_hard_samples(all_strict, all_expanded, sample_stats)

        # 完成
        total_hard = len(all_strict) + len(all_expanded)
        self.logger.info(f"\n{'=' * 80}")
        self.logger.info("模块2完成!")
        self.logger.info(f"严格难样本: {len(all_strict):,}")
        self.logger.info(f"扩展难样本: {len(all_expanded):,}")
        self.logger.info(f"总计: {total_hard:,}")
        self.logger.info(f"输出目录: {self.config.output_dir}")
        self.logger.info("=" * 80)

        return {
            'strict_count': len(all_strict),
            'expanded_count': len(all_expanded),
            'total_hard': total_hard,
            'output_dir': self.config.output_dir,
        }


# ==================== 命令行入口 ====================

def main():
    parser = argparse.ArgumentParser(description='模块2: 难样本挖掘（修复版 v2.0）')
    parser.add_argument('--base-dir', default="/home/zjw/zjwdata")
    parser.add_argument('--module1-output',
                        default="/home/zjw/zjwdata/3_deep_learning/training_data/curated_v2",
                        help='模块1输出目录（用于加载正样本参考）')
    parser.add_argument('--gene-catalog-dir',
                        default="/home/zjw/zjwdata/1/gene_catalog_analysis/per_sample_non_redundant_genes")
    parser.add_argument('--annotation-dir',
                        default="/home/zjw/zjwdata/2/gene_function/annotation_results/integrated")
    parser.add_argument('--genes-fna-dir',
                        default="/home/zjw/zjwdata/1/assembly_analysis/genes")
    parser.add_argument('--output-dir',
                        default="/home/zjw/zjwdata/3_deep_learning/training_data/curated_v2",
                        help='输出目录（应与模块1一致）')
    parser.add_argument('--max-strict', type=int, default=500000)
    parser.add_argument('--max-expanded', type=int, default=300000)

    args = parser.parse_args()

    config = HardSampleConfig(
        base_dir=args.base_dir,
        module1_output_dir=args.module1_output,
        positive_info_tsv=f"{args.module1_output}/positive_samples_info.tsv",
        gene_catalog_dir=args.gene_catalog_dir,
        annotation_dir=args.annotation_dir,
        genes_fna_dir=args.genes_fna_dir,
        output_dir=args.output_dir,
        max_total_strict=args.max_strict,
        max_total_expanded=args.max_expanded,
    )

    pipeline = HardSampleMiningPipeline(config)
    results = pipeline.run()

    return results


if __name__ == "__main__":
    main()