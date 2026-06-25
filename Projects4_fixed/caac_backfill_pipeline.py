#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CAAC回补分析管道 (Strategy B)
功能: 将CAAC高置信度预测回补到KEGG注释中，重新计算通路丰度与覆盖度
作者: AI Assistant
日期: 2026-04-25

流程:
1. 从ko00001.tsv构建EC→KO映射表
2. 筛选CAAC高置信度预测 (Tier_1 + Tier_2, confidence >= 0.40)
3. 逐样本提取原始KEGG注释 → 生成回补KEGG注释 → 合并
4. 调用pathway_abundance.py重新计算通路丰度
5. 对比回补前后通路覆盖度，生成论文统计报告
"""

import os
import sys
import re
import json
import argparse
import logging
import subprocess
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, field


# ==================== 配置 ====================

@dataclass
class BackfillConfig:
    """回补分析配置"""

    # CAAC预测结果
    caac_predictions: str = "/home/zjw/zjwdata/3_deep_learning/training_data/curated_v2/predictions/hard_predictions_full.tsv"

    # KEGG数据库
    kegg_db: str = "/mnt/databases/kegg/2025/ko00001.tsv"

    # 输入路径模板
    integrated_dir: str = "/mnt/zjwdata/2/gene_function/annotation_results/integrated"
    abundance_dir: str = "/mnt/zjwdata/2/gene_function/annotation_results/abundance"

    # 输出根目录
    output_dir: str = "/home/zjw/zjwdata/3_deep_learning/training_data/caac_backfill_results"

    # pathway_abundance.py 路径 (修正)
    pathway_script: str = "/home/zjw/Projects2/gene_function/pathway_abundance.py"

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

    # 筛选阈值
    min_confidence: float = 0.40  # Tier_2及以上
    min_tier: str = "Tier_2"  # 包含Tier_1和Tier_2
    allowed_tiers: Set[str] = None

    # EC映射策略
    ec_match_mode: str = "exact_then_prefix"  # exact:精确匹配, prefix:前缀匹配, exact_then_prefix:先精确后前缀

    def __post_init__(self):
        if self.allowed_tiers is None:
            self.allowed_tiers = {"Tier_1", "Tier_2"}
        os.makedirs(self.output_dir, exist_ok=True)
        for subdir in ['logs', 'per_sample', 'merged_kegg', 'pathway_results', 'summary']:
            os.makedirs(f"{self.output_dir}/{subdir}", exist_ok=True)


# ==================== 日志系统 ====================

def setup_logger(name: str, log_dir: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if logger.handlers:
        logger.handlers.clear()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fh = logging.FileHandler(f"{log_dir}/{name}_{timestamp}.log", encoding='utf-8')
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ==================== EC→KO映射构建器 ====================

class EC2KOMapper:
    """从KEGG ko00001.tsv构建EC→KO映射表"""

    EC_PATTERN = re.compile(r'\[EC:([\d\.\-]+)\]')

    def __init__(self, kegg_db_path: str, logger: logging.Logger):
        self.kegg_db_path = kegg_db_path
        self.logger = logger
        self.ec_to_kos: Dict[str, List[str]] = defaultdict(list)
        self.ko_to_ecs: Dict[str, List[str]] = defaultdict(list)
        self._build_mapping()

    def _build_mapping(self):
        """解析KEGG数据库，构建EC→KO映射"""
        self.logger.info(f"解析KEGG数据库: {self.kegg_db_path}")

        if not os.path.exists(self.kegg_db_path):
            raise FileNotFoundError(f"KEGG数据库不存在: {self.kegg_db_path}")

        line_count = 0
        ec_hits = 0

        with open(self.kegg_db_path, 'r', encoding='utf-8') as f:
            header = f.readline().strip()
            self.logger.info(f"数据库表头: {header}")

            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split('\t')
                if len(parts) < 6:
                    continue

                ko_id = parts[4].strip()
                gene_symbol = parts[5].strip()

                if not ko_id.startswith('K'):
                    continue

                # 提取EC号
                ec_matches = self.EC_PATTERN.findall(gene_symbol)

                for ec in ec_matches:
                    ec_clean = ec.strip()
                    if ec_clean:
                        self.ec_to_kos[ec_clean].append(ko_id)
                        self.ko_to_ecs[ko_id].append(ec_clean)
                        ec_hits += 1

                line_count += 1

        # 去重
        for ec in self.ec_to_kos:
            self.ec_to_kos[ec] = list(dict.fromkeys(self.ec_to_kos[ec]))

        self.logger.info(f"解析完成: {line_count} 行, {len(self.ec_to_kos)} 个唯一EC, {ec_hits} 个EC-KO关联")
        self.logger.info(f"平均每个EC对应 {np.mean([len(v) for v in self.ec_to_kos.values()]):.1f} 个KO")

    def get_ko(self, ec: str, mode: str = "exact_then_prefix") -> Optional[str]:
        """
        根据EC号获取KO号

        mode:
            exact: 精确匹配
            prefix: 前缀匹配 (如 2.7.1.1 匹配 2.7.1.-)
            exact_then_prefix: 先精确，失败后尝试前缀
        """
        if not ec or ec in ('NA', 'nan', '-', ''):
            return None

        ec = ec.strip()

        # 策略1: 精确匹配
        if mode in ("exact", "exact_then_prefix"):
            if ec in self.ec_to_kos:
                kos = self.ec_to_kos[ec]
                return kos[0]  # 返回第一个KO

        # 策略2: 前缀匹配
        if mode in ("prefix", "exact_then_prefix"):
            # 尝试逐级缩短: 1.2.3.4 -> 1.2.3.- -> 1.2.-.- -> 1.-.-.-
            parts = ec.split('.')
            for i in range(len(parts), 0, -1):
                prefix = '.'.join(parts[:i - 1] + ['-']) if i > 1 else parts[0]
                # 更合理的做法：逐级替换最后一位为-
                for j in range(len(parts) - 1, 0, -1):
                    test_parts = parts.copy()
                    test_parts[j] = '-'
                    test_ec = '.'.join(test_parts)
                    if test_ec in self.ec_to_kos:
                        kos = self.ec_to_kos[test_ec]
                        self.logger.debug(f"前缀匹配: {ec} -> {test_ec} -> {kos[0]}")
                        return kos[0]

        return None

    def get_all_kos(self, ec: str) -> List[str]:
        """获取EC对应的所有KO（用于统计）"""
        if ec in self.ec_to_kos:
            return self.ec_to_kos[ec]
        return []

    def save_mapping(self, output_path: str):
        """保存映射表为JSON（缓存）"""
        mapping_dict = {k: v for k, v in self.ec_to_kos.items()}
        with open(output_path, 'w') as f:
            json.dump(mapping_dict, f, indent=2)
        self.logger.info(f"EC→KO映射表已缓存: {output_path}")


# ==================== CAAC预测加载器 ====================

class CAACPredictionLoader:
    """加载并筛选CAAC预测结果"""

    def __init__(self, predictions_file: str, logger: logging.Logger):
        self.predictions_file = predictions_file
        self.logger = logger
        self.predictions: pd.DataFrame = None
        self._load()

    def _load(self):
        """加载预测文件"""
        self.logger.info(f"加载CAAC预测结果: {self.predictions_file}")

        if not os.path.exists(self.predictions_file):
            raise FileNotFoundError(f"预测文件不存在: {self.predictions_file}")

        self.predictions = pd.read_csv(self.predictions_file, sep='\t', low_memory=False)

        self.logger.info(f"预测总数: {len(self.predictions):,}")
        self.logger.info(f"列名: {list(self.predictions.columns)}")

        # 统计Tier分布
        tier_dist = self.predictions['confidence_tier'].value_counts()
        self.logger.info(f"Tier分布:\n{tier_dist}")

        # 统计predicted_ec分布
        has_ec = self.predictions['predicted_ec'].notna() & (self.predictions['predicted_ec'] != 'NA')
        self.logger.info(f"有EC预测: {has_ec.sum():,} / {len(self.predictions):,}")

    def filter_high_confidence(self, min_confidence: float = 0.40,
                               allowed_tiers: Set[str] = None) -> pd.DataFrame:
        """筛选高置信度预测"""
        if allowed_tiers is None:
            allowed_tiers = {"Tier_1", "Tier_2"}

        # 条件1: Tier在允许列表中
        mask_tier = self.predictions['confidence_tier'].isin(allowed_tiers)

        # 条件2: 置信度 >= 阈值
        mask_conf = self.predictions['confidence_score'] >= min_confidence

        # 条件3: 有EC预测（用于KEGG映射）
        mask_ec = self.predictions['predicted_ec'].notna() & \
                  (self.predictions['predicted_ec'] != 'NA') & \
                  (self.predictions['predicted_ec'] != '-')

        # 条件4: GNN预测为Positive(1)或Hard(2)（排除Negative(0)）
        mask_gnn = self.predictions['gnn_pred_class'].isin([1, 2])

        combined_mask = mask_tier & mask_conf & mask_ec & mask_gnn

        filtered = self.predictions[combined_mask].copy()

        self.logger.info(f"高置信度筛选: {len(filtered):,} / {len(self.predictions):,}")
        self.logger.info(f"  Tier条件: {mask_tier.sum():,}")
        self.logger.info(f"  置信度条件: {mask_conf.sum():,}")
        self.logger.info(f"  EC条件: {mask_ec.sum():,}")
        self.logger.info(f"  GNN类别条件: {mask_gnn.sum():,}")

        return filtered


# ==================== 样本整合注释处理器 ====================

class SampleAnnotationProcessor:
    """处理单个样本的整合注释文件"""

    def __init__(self, sample: str, config: BackfillConfig, logger: logging.Logger):
        self.sample = sample
        self.config = config
        self.logger = logger

        self.integrated_file = f"{config.integrated_dir}/{sample}/{sample}.integrated_annotations.tsv"
        self.abundance_file = f"{config.abundance_dir}/{sample}/{sample}.gene_abundance.tsv"

        self.integrated_df: pd.DataFrame = None
        self.original_kegg_df: pd.DataFrame = None
        self.backfill_df: pd.DataFrame = None
        self.merged_kegg_df: pd.DataFrame = None

    def load_integrated(self) -> bool:
        """加载整合注释文件"""
        if not os.path.exists(self.integrated_file):
            self.logger.warning(f"整合注释文件不存在: {self.integrated_file}")
            return False

        self.logger.info(f"[{self.sample}] 加载整合注释...")
        self.integrated_df = pd.read_csv(self.integrated_file, sep='\t', low_memory=False)

        # 统计
        total_genes = len(self.integrated_df)
        annotated = (self.integrated_df['Any_Annotation'] == 'Yes').sum()
        unannotated = total_genes - annotated

        self.logger.info(f"[{self.sample}] 总基因: {total_genes:,}, 已注释: {annotated:,}, 未注释: {unannotated:,}")

        return True

    def extract_original_kegg(self) -> pd.DataFrame:
        """从整合注释中提取原始KEGG注释（GeneID + KO）"""
        if self.integrated_df is None:
            raise ValueError("请先调用load_integrated()")

        # 筛选有KEGG注释的基因
        has_ko = self.integrated_df['KEGG_KO'].notna() & \
                 (self.integrated_df['KEGG_KO'] != '-') & \
                 (self.integrated_df['KEGG_KO'] != 'NA')

        original = self.integrated_df[has_ko][['GeneID', 'KEGG_KO']].copy()
        original.columns = ['GeneID', 'KO']

        # 去重：同一个GeneID可能有多个KO（用分号分隔）
        expanded_rows = []
        for _, row in original.iterrows():
            gene_id = row['GeneID']
            kos = str(row['KO']).split(';')
            for ko in kos:
                ko = ko.strip()
                if ko and ko.startswith('K'):
                    expanded_rows.append({'GeneID': gene_id, 'KO': ko})

        self.original_kegg_df = pd.DataFrame(expanded_rows)
        self.logger.info(f"[{self.sample}] 原始KEGG注释: {len(self.original_kegg_df):,} 条 (GeneID-KO对)")

        return self.original_kegg_df

    def generate_backfill(self, caac_filtered: pd.DataFrame, ec_mapper: EC2KOMapper) -> pd.DataFrame:
        """
        生成回补KEGG注释

        逻辑:
        1. 从整合注释中获取未注释基因的GeneID列表
        2. 与CAAC预测匹配
        3. EC→KO映射
        4. 生成回补文件
        """
        if self.integrated_df is None:
            raise ValueError("请先调用load_integrated()")

        self.logger.info(f"[{self.sample}] 生成回补注释...")

        # 获取未注释基因
        unannotated_mask = (self.integrated_df['Any_Annotation'] != 'Yes') | \
                           (self.integrated_df['KEGG_KO'].isna()) | \
                           (self.integrated_df['KEGG_KO'] == '-') | \
                           (self.integrated_df['KEGG_KO'] == 'NA')

        unannotated_genes = set(self.integrated_df[unannotated_mask]['GeneID'].tolist())
        self.logger.info(f"[{self.sample}] 未注释基因: {len(unannotated_genes):,}")

        # 与CAAC预测匹配
        caac_in_sample = caac_filtered[caac_filtered['gene_id'].isin(unannotated_genes)].copy()
        self.logger.info(f"[{self.sample}] CAAC预测匹配的未注释基因: {len(caac_in_sample):,}")

        if len(caac_in_sample) == 0:
            self.logger.warning(f"[{self.sample}] 无匹配的回补基因")
            self.backfill_df = pd.DataFrame(
                columns=['GeneID', 'KO', 'predicted_ec', 'confidence_score', 'confidence_tier'])
            return self.backfill_df

        # EC→KO映射
        backfill_records = []
        ec_stats = Counter()

        for _, row in caac_in_sample.iterrows():
            gene_id = row['gene_id']
            ec = str(row['predicted_ec']).strip()

            ko = ec_mapper.get_ko(ec, mode=self.config.ec_match_mode)

            if ko:
                backfill_records.append({
                    'GeneID': gene_id,
                    'KO': ko,
                    'predicted_ec': ec,
                    'confidence_score': row['confidence_score'],
                    'confidence_tier': row['confidence_tier'],
                    'gnn_pred_class': row['gnn_pred_class']
                })
                ec_stats['mapped'] += 1
            else:
                ec_stats['unmapped'] += 1

        self.backfill_df = pd.DataFrame(backfill_records)

        self.logger.info(f"[{self.sample}] EC→KO映射统计: {ec_stats}")
        self.logger.info(f"[{self.sample}] 成功回补: {len(self.backfill_df):,} 条")

        if len(self.backfill_df) > 0:
            tier_dist = self.backfill_df['confidence_tier'].value_counts()
            self.logger.info(f"[{self.sample}] 回补Tier分布:\n{tier_dist}")

        return self.backfill_df

    def merge_kegg(self) -> pd.DataFrame:
        """合并原始KEGG + 回补KEGG"""
        if self.original_kegg_df is None or self.backfill_df is None:
            raise ValueError("请先调用extract_original_kegg()和generate_backfill()")

        # 合并：回补优先（如果同一个GeneID在原始和回补中都有，保留回补的）
        # 但通常未注释基因不会在原始中出现，所以直接拼接即可

        combined = pd.concat([self.original_kegg_df, self.backfill_df[['GeneID', 'KO']]],
                             ignore_index=True)

        # 去重
        combined = combined.drop_duplicates(subset=['GeneID', 'KO'])

        self.merged_kegg_df = combined
        self.logger.info(f"[{self.sample}] 合并后KEGG注释: {len(self.merged_kegg_df):,} 条")
        self.logger.info(f"  原始: {len(self.original_kegg_df):,}, 回补: {len(self.backfill_df):,}")

        return self.merged_kegg_df

    def save_merged_kegg(self, output_path: str):
        """保存合并后的KEGG注释文件"""
        if self.merged_kegg_df is None:
            raise ValueError("请先调用merge_kegg()")

        self.merged_kegg_df.to_csv(output_path, sep='\t', index=False)
        self.logger.info(f"[{self.sample}] 合并KEGG注释已保存: {output_path}")


# ==================== 通路丰度计算调用器 ====================

class PathwayAbundanceRunner:
    """调用pathway_abundance.py计算通路丰度"""

    def __init__(self, config: BackfillConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger

    def run(self, sample: str, kegg_file: str, output_subdir: str) -> bool:
        """
        调用pathway_abundance.py

        output_subdir: 'original' 或 'backfilled'
        """
        abundance_file = f"{self.config.abundance_dir}/{sample}/{sample}.gene_abundance.tsv"

        if not os.path.exists(abundance_file):
            self.logger.error(f"[{sample}] 基因丰度文件不存在: {abundance_file}")
            return False

        output_dir = f"{self.config.output_dir}/pathway_results/{sample}/{output_subdir}"
        os.makedirs(output_dir, exist_ok=True)

        # 构建命令列表，每个参数独立
        cmd = [
            'python', self.config.pathway_script,
            '--gene_abundance', abundance_file,
            '--kegg_annotations', kegg_file,
            '--pathway_db', self.config.kegg_db,
            '-o', output_dir,
            '-p', sample,
            '--method', 'sum',
            '--normalization', 'none',
            '--abundance_column', 'TPM'
        ]

        # 日志中打印可读命令
        self.logger.info(f"[{sample}] 计算通路丰度 ({output_subdir})...")
        self.logger.info(f"  命令: {' '.join(cmd)}")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

            if result.returncode != 0:
                self.logger.error(f"[{sample}] 通路丰度计算失败:\n{result.stderr}")
                return False

            self.logger.info(f"[{sample}] 通路丰度计算完成: {output_dir}")
            return True

        except subprocess.TimeoutExpired:
            self.logger.error(f"[{sample}] 通路丰度计算超时")
            return False
        except Exception as e:
            self.logger.error(f"[{sample}] 通路丰度计算异常: {e}")
            return False


# ==================== 回补对比分析器 ====================

class BackfillComparator:
    """对比回补前后的通路覆盖度"""

    def __init__(self, config: BackfillConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.results: List[Dict] = []

    def compare_sample(self, sample: str) -> Optional[Dict]:
        """对比单个样本的回补效果"""
        original_coverage_file = f"{self.config.output_dir}/pathway_results/{sample}/original/{sample}.pathway_coverage.tsv"
        backfilled_coverage_file = f"{self.config.output_dir}/pathway_results/{sample}/backfilled/{sample}.pathway_coverage.tsv"

        if not os.path.exists(original_coverage_file) or not os.path.exists(backfilled_coverage_file):
            self.logger.warning(f"[{sample}] 覆盖度文件缺失，跳过对比")
            return None

        # 读取
        orig_df = pd.read_csv(original_coverage_file, sep='\t')
        back_df = pd.read_csv(backfilled_coverage_file, sep='\t')

        # 合并对比
        merged = orig_df.merge(back_df, on=['Pathway_ID', 'Pathway_Name'],
                               suffixes=('_orig', '_back'), how='outer')

        # 计算提升
        merged['Coverage_increase'] = merged['Coverage_Percentage_back'].fillna(0) - \
                                      merged['Coverage_Percentage_orig'].fillna(0)
        merged['KO_increase'] = merged['Annotated_KOs_back'].fillna(0) - \
                                merged['Annotated_KOs_orig'].fillna(0)

        # 统计
        n_pathways = len(merged)
        n_improved = (merged['Coverage_increase'] > 0).sum()
        avg_increase = merged['Coverage_increase'].mean()
        max_increase = merged['Coverage_increase'].max()

        # 找出提升最大的通路
        top_improved = merged.nlargest(10, 'Coverage_increase')[['Pathway_ID', 'Pathway_Name',
                                                                 'Coverage_Percentage_orig',
                                                                 'Coverage_Percentage_back',
                                                                 'Coverage_increase']]

        result = {
            'sample': sample,
            'total_pathways': n_pathways,
            'improved_pathways': int(n_improved),
            'avg_coverage_increase': float(avg_increase),
            'max_coverage_increase': float(max_increase),
            'top_improved': top_improved.to_dict('records')
        }

        self.results.append(result)

        self.logger.info(f"[{sample}] 回补效果:")
        self.logger.info(f"  总通路: {n_pathways}")
        self.logger.info(f"  覆盖度提升的通路: {n_improved}")
        self.logger.info(f"  平均覆盖度提升: {avg_increase:.2f}%")
        self.logger.info(f"  最大覆盖度提升: {max_increase:.2f}%")

        # 保存详细对比
        output_file = f"{self.config.output_dir}/summary/{sample}_comparison.tsv"
        merged.to_csv(output_file, sep='\t', index=False)

        return result

    def generate_summary_report(self):
        """生成汇总报告"""
        if not self.results:
            self.logger.warning("无对比结果可汇总")
            return

        report_file = f"{self.config.output_dir}/summary/backfill_summary_report.json"

        with open(report_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'configuration': {
                    'min_confidence': self.config.min_confidence,
                    'allowed_tiers': list(self.config.allowed_tiers),
                    'ec_match_mode': self.config.ec_match_mode
                },
                'sample_results': self.results,
                'overall': {
                    'n_samples': len(self.results),
                    'avg_improved_pathways': np.mean([r['improved_pathways'] for r in self.results]),
                    'avg_coverage_increase': np.mean([r['avg_coverage_increase'] for r in self.results]),
                    'max_coverage_increase': max([r['max_coverage_increase'] for r in self.results])
                }
            }, f, indent=2)

        self.logger.info(f"汇总报告已保存: {report_file}")

        # 生成CSV汇总表
        summary_df = pd.DataFrame([
            {
                'Sample': r['sample'],
                'Total_Pathways': r['total_pathways'],
                'Improved_Pathways': r['improved_pathways'],
                'Avg_Coverage_Increase_%': f"{r['avg_coverage_increase']:.2f}",
                'Max_Coverage_Increase_%': f"{r['max_coverage_increase']:.2f}"
            }
            for r in self.results
        ])

        csv_file = f"{self.config.output_dir}/summary/backfill_summary_table.csv"
        summary_df.to_csv(csv_file, index=False)
        self.logger.info(f"汇总表格已保存: {csv_file}")
        self.logger.info(f"\n汇总结果:\n{summary_df.to_string(index=False)}")


# ==================== 主流程 ====================

class CAACBackfillPipeline:
    """CAAC回补分析主控制器"""

    def __init__(self, config: BackfillConfig):
        self.config = config
        self.logger = setup_logger("caac_backfill", f"{config.output_dir}/logs")

        self.logger.info("=" * 80)
        self.logger.info("CAAC回补分析管道启动")
        self.logger.info(f"输出目录: {config.output_dir}")
        self.logger.info("=" * 80)

        # 初始化组件
        self.ec_mapper = EC2KOMapper(config.kegg_db, self.logger)
        self.caac_loader = CAACPredictionLoader(config.caac_predictions, self.logger)
        self.pathway_runner = PathwayAbundanceRunner(config, self.logger)
        self.comparator = BackfillComparator(config, self.logger)

    def run(self):
        """执行完整回补流程"""

        # Step 1: 筛选高置信度CAAC预测
        self.logger.info("\n" + "=" * 60)
        self.logger.info("Step 1: 筛选高置信度CAAC预测")
        self.logger.info("=" * 60)

        caac_filtered = self.caac_loader.filter_high_confidence(
            min_confidence=self.config.min_confidence,
            allowed_tiers=self.config.allowed_tiers
        )

        # 保存筛选后的预测
        filtered_file = f"{self.config.output_dir}/summary/caac_filtered_predictions.tsv"
        caac_filtered.to_csv(filtered_file, sep='\t', index=False)
        self.logger.info(f"筛选后的预测已保存: {filtered_file}")

        # Step 2: 逐样本处理
        self.logger.info("\n" + "=" * 60)
        self.logger.info("Step 2: 逐样本回补分析")
        self.logger.info("=" * 60)

        for idx, sample in enumerate(self.config.samples, 1):
            self.logger.info(f"\n{'#' * 60}")
            self.logger.info(f"# [{idx}/{len(self.config.samples)}] 处理样本: {sample}")
            self.logger.info(f"{'#' * 60}")

            # 检查是否已处理
            backfilled_coverage = f"{self.config.output_dir}/pathway_results/{sample}/backfilled/{sample}.pathway_coverage.tsv"
            if os.path.exists(backfilled_coverage):
                self.logger.info(f"[{sample}] 检测到已完成的回补结果，跳过...")
                self.comparator.compare_sample(sample)
                continue

            # 2.1 加载样本整合注释
            processor = SampleAnnotationProcessor(sample, self.config, self.logger)
            if not processor.load_integrated():
                continue

            # 2.2 提取原始KEGG
            processor.extract_original_kegg()

            # 2.3 生成回补KEGG
            processor.generate_backfill(caac_filtered, self.ec_mapper)

            # 2.4 合并
            processor.merge_kegg()

            # 2.5 保存原始KEGG注释
            original_kegg_file = f"{self.config.output_dir}/merged_kegg/{sample}_original_kegg.tsv"
            processor.original_kegg_df.to_csv(original_kegg_file, sep='\t', index=False)
            self.logger.info(f"[{sample}] 原始KEGG注释已保存: {original_kegg_file}")

            # 2.6 保存合并KEGG注释
            merged_kegg_file = f"{self.config.output_dir}/merged_kegg/{sample}_merged_kegg.tsv"
            processor.save_merged_kegg(merged_kegg_file)

            # 2.7 保存回补详情
            if processor.backfill_df is not None and len(processor.backfill_df) > 0:
                backfill_detail_file = f"{self.config.output_dir}/per_sample/{sample}_backfill_detail.tsv"
                processor.backfill_df.to_csv(backfill_detail_file, sep='\t', index=False)

            # 2.8 计算原始通路丰度
            self.pathway_runner.run(sample, original_kegg_file, 'original')

            # 2.9 计算回补后通路丰度
            self.pathway_runner.run(sample, merged_kegg_file, 'backfilled')

            # 2.10 对比
            self.comparator.compare_sample(sample)

        # Step 3: 汇总
        self.logger.info("\n" + "=" * 60)
        self.logger.info("Step 3: 生成汇总报告")
        self.logger.info("=" * 60)

        self.comparator.generate_summary_report()

        self.logger.info("\n" + "=" * 80)
        self.logger.info("CAAC回补分析完成!")
        self.logger.info(f"结果目录: {self.config.output_dir}")
        self.logger.info("=" * 80)


# ==================== 命令行入口 ====================

def main():
    parser = argparse.ArgumentParser(description='CAAC回补分析管道')
    parser.add_argument('--min-confidence', type=float, default=0.40,
                        help='最小置信度阈值 (默认: 0.40)')
    parser.add_argument('--ec-match-mode', default='exact_then_prefix',
                        choices=['exact', 'prefix', 'exact_then_prefix'],
                        help='EC→KO匹配模式')
    parser.add_argument('--samples', nargs='+', default=None,
                        help='指定处理的样本（默认处理全部21个）')

    args = parser.parse_args()

    config = BackfillConfig()
    config.min_confidence = args.min_confidence
    config.ec_match_mode = args.ec_match_mode
    # 确保目录存在（__post_init__已在实例化时执行，但修改字段后不会自动再次执行，手动调用安全）
    config.__post_init__()

    if args.samples:
        config.samples = tuple(args.samples)

    pipeline = CAACBackfillPipeline(config)
    pipeline.run()

    print(f"\n{'=' * 60}")
    print("回补分析完成!")
    print(f"结果目录: {config.output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()