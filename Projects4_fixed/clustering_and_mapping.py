#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模块3: clustering_and_mapping.py
功能: CD-HIT聚类去冗余 + ID映射构建

输入: 模块1/2输出 (curated_v2/)
  - positive_samples.fasta
  - negative_samples.fasta
  - hard_samples_strict.fasta
  - hard_samples_expanded.fasta

输出: curated_v2/clustered/
  - {category}_cdhit90.fasta      # 代表序列
  - {category}_cdhit90.fasta.clstr # 聚类文件
  - {category}_id_mapping.csv      # member_id -> representative_id
  - clustering_report.json         # 完整统计
"""

import os
import sys
import re
import json
import argparse
import logging
import subprocess
import shutil
import random
from datetime import datetime
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, asdict

import pandas as pd
import numpy as np
from tqdm import tqdm


# ==================== 配置类 ====================

@dataclass
class ClusterConfig:
    """聚类配置"""
    # 输入目录（模块1/2输出）
    input_dir: str = "/home/zjw/zjwdata/3_deep_learning/training_data/curated_v2"

    # 输出目录
    output_dir: str = "/home/zjw/zjwdata/3_deep_learning/training_data/curated_v2/clustered"

    # 输入文件（与模块1/2输出命名一致）
    input_files: Tuple[Tuple[str, str, int], ...] = (
        # (category_name, filename, max_samples)
        ('positive', 'positive_samples.fasta', 1000000),
        ('negative', 'negative_samples.fasta', 200000),
        ('hard_strict', 'hard_samples_strict.fasta', 500000),
        ('hard_expanded', 'hard_samples_expanded.fasta', 300000),
    )

    # CD-HIT参数
    cdhit_path: str = "/usr/local/bin/cd-hit"
    similarity_threshold: float = 0.90  # -c: 序列相似度阈值
    alignment_coverage: float = 0.90  # -aS: 对齐覆盖度（短序列相对于长序列）
    alignment_coverage_long: float = 0.90  # -aL: 对齐覆盖度（长序列相对于短序列）
    memory_limit_mb: int = 120000  # -M: 内存限制(MB)，125GB服务器留余量
    threads: int = 0  # -T: 0=使用所有CPU线程(96核)
    word_size: int = 5  # -n: 字长（90%相似度用5）
    description_length: int = 50  # -d: 描述长度
    sort_by_length: int = 1  # -sc: 1=按长度降序（保留最长代表序列）

    # 难样本特殊参数（稍宽松以保留多样性）
    hard_similarity_threshold: float = 0.90
    hard_alignment_coverage: float = 0.85

    # 采样随机种子
    random_seed: int = 42

    # 日志
    log_level: str = "INFO"

    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)


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


# ==================== 序列采样器 ====================

class SequenceSampler:
    """FASTA序列采样器（内存高效版）"""

    def __init__(self, seed: int = 42):
        random.seed(seed)

    def count_sequences(self, filepath: str) -> int:
        """统计FASTA序列数（使用grep快速计数）"""
        try:
            result = subprocess.run(
                ['grep', '-c', '^>', filepath],
                capture_output=True, text=True, check=True
            )
            return int(result.stdout.strip())
        except subprocess.CalledProcessError:
            # 文件为空或不存在
            return 0

    def sample_fasta(self, input_file: str, output_file: str, max_samples: int) -> int:
        """
        随机采样FASTA序列
        策略: 先收集所有序列起始位置，随机选择，流式写入
        """
        total = self.count_sequences(input_file)

        if total <= max_samples:
            # 未超过上限，直接复制
            shutil.copy2(input_file, output_file)
            return total

        # 超过上限，需要采样
        # 第一步: 收集所有序列在文件中的起始位置
        seq_positions = []
        with open(input_file, 'r') as f:
            for line_num, line in enumerate(f):
                if line.startswith('>'):
                    seq_positions.append(line_num)

        # 随机选择
        selected_positions = set(random.sample(seq_positions, max_samples))

        # 流式写入采样结果
        with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
            in_selected = False
            for line_num, line in enumerate(f_in):
                if line_num in selected_positions:
                    in_selected = True
                    f_out.write(line)
                elif line.startswith('>'):
                    in_selected = False
                elif in_selected:
                    f_out.write(line)

        return max_samples


# ==================== CD-HIT执行器 ====================

class CDHitRunner:
    """CD-HIT聚类执行器"""

    def __init__(self, config: ClusterConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger

        # 验证cd-hit可用
        self._validate_cdhit()

    def _validate_cdhit(self):
        """验证CD-HIT安装"""
        if not os.path.exists(self.config.cdhit_path):
            raise RuntimeError(f"CD-HIT未找到: {self.config.cdhit_path}")

        try:
            result = subprocess.run(
                [self.config.cdhit_path, '-h'],
                capture_output=True, text=True
            )
            version_match = re.search(r'CD-HIT version (\S+)', result.stderr)
            if version_match:
                self.logger.info(f"CD-HIT版本: {version_match.group(1)}")
            else:
                self.logger.warning("无法解析CD-HIT版本")
        except Exception as e:
            self.logger.warning(f"CD-HIT版本检查失败: {e}")

    def run_clustering(
            self,
            category: str,
            input_file: str,
            output_file: str,
            log_file: str,
            is_hard: bool = False
    ) -> Tuple[int, int, float]:
        """
        执行CD-HIT聚类

        返回: (input_count, output_count, reduction_rate)
        """
        # 根据类别选择参数
        if is_hard:
            sim = self.config.hard_similarity_threshold
            cov = self.config.hard_alignment_coverage
        else:
            sim = self.config.similarity_threshold
            cov = self.config.alignment_coverage

        cmd = [
            self.config.cdhit_path,
            '-i', input_file,
            '-o', output_file,
            '-c', str(sim),
            '-aS', str(cov),
            '-aL', str(self.config.alignment_coverage_long),
            '-g', '1',  # 最优模式（慢但准确）
            '-T', str(self.config.threads),
            '-M', str(self.config.memory_limit_mb),
            '-n', str(self.config.word_size),
            '-d', str(self.config.description_length),
            '-sc', str(self.config.sort_by_length),
        ]

        self.logger.info(f"\n{'=' * 60}")
        self.logger.info(f"运行CD-HIT: {category}")
        self.logger.info(f"  输入: {input_file}")
        self.logger.info(f"  输出: {output_file}")
        self.logger.info(f"  参数: similarity={sim * 100}%, coverage={cov * 100}%")
        self.logger.info(f"  命令: {' '.join(cmd)}")
        self.logger.info(f"{'=' * 60}")

        # 执行
        start_time = datetime.now()

        with open(log_file, 'w') as log_f:
            result = subprocess.run(cmd, stdout=log_f, stderr=subprocess.STDOUT)

        elapsed = (datetime.now() - start_time).total_seconds()

        if result.returncode != 0:
            self.logger.error(f"✗ CD-HIT失败，查看日志: {log_file}")
            raise RuntimeError(f"CD-HIT失败: {category}")

        # 统计结果
        input_count = self._count_fasta_sequences(input_file)
        output_count = self._count_fasta_sequences(output_file)
        reduction_rate = 100 * (input_count - output_count) / max(input_count, 1)

        self.logger.info(f"✓ CD-HIT完成 ({elapsed / 60:.1f}分钟)")
        self.logger.info(f"  输入: {input_count:,} 条")
        self.logger.info(f"  输出: {output_count:,} 条")
        self.logger.info(f"  去冗余: {input_count - output_count:,} 条 ({reduction_rate:.1f}%)")

        return input_count, output_count, reduction_rate

    def _count_fasta_sequences(self, filepath: str) -> int:
        """统计FASTA序列数"""
        if not os.path.exists(filepath):
            return 0

        try:
            result = subprocess.run(
                ['grep', '-c', '^>', filepath],
                capture_output=True, text=True, check=True
            )
            return int(result.stdout.strip())
        except:
            # 备用方法
            count = 0
            with open(filepath, 'r') as f:
                for line in f:
                    if line.startswith('>'):
                        count += 1
            return count


# ==================== 聚类解析器 ====================

class ClusterParser:
    """解析CD-HIT .clstr文件，构建ID映射"""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def parse_clstr_file(self, clstr_file: str) -> Dict[str, str]:
        """
        解析.clstr文件

        返回: dict[member_id] = representative_id
        """
        if not os.path.exists(clstr_file):
            raise FileNotFoundError(f"聚类文件不存在: {clstr_file}")

        clusters = {}
        current_cluster_id = None
        current_representative = None

        with open(clstr_file, 'r') as f:
            for line in f:
                line = line.strip()

                if not line:
                    continue

                if line.startswith('>Cluster'):
                    # 新聚类开始
                    current_cluster_id = line.split()[1]
                    current_representative = None

                elif current_cluster_id is not None:
                    # 解析成员行
                    # 格式: 0	2799aa, >PF04998.6|RPOC2_CHLRE/275-3073... *
                    # 或:   1	2214aa, >PF06317.1|Q6Y625_9VIRU/1-2214... at 80%

                    # 提取序列ID（>和...之间的内容）
                    match = re.search(r'>([^.]+)\.\.\.', line)
                    if not match:
                        # 尝试其他格式
                        match = re.search(r'>(\S+)', line)

                    if match:
                        seq_id = match.group(1)

                        # 检查是否是代表序列（以*结尾）
                        if line.endswith('*'):
                            current_representative = seq_id
                            clusters[seq_id] = seq_id  # 代表序列映射到自己
                        elif current_representative:
                            clusters[seq_id] = current_representative

        return clusters

    def build_mapping_dataframe(self, mapping: Dict[str, str]) -> pd.DataFrame:
        """将映射转换为DataFrame"""
        records = []
        for member_id, rep_id in mapping.items():
            records.append({
                'member_id': member_id,
                'representative_id': rep_id,
                'is_representative': member_id == rep_id
            })

        return pd.DataFrame(records)

    def get_cluster_statistics(self, mapping: Dict[str, str]) -> Dict:
        """获取聚类统计信息"""
        # 统计代表序列
        representatives = set(mapping.values())

        # 聚类大小分布
        cluster_sizes = Counter()
        rep_to_members = defaultdict(list)

        for member, rep in mapping.items():
            rep_to_members[rep].append(member)

        for rep, members in rep_to_members.items():
            cluster_sizes[len(members)] += 1

        return {
            'total_members': len(mapping),
            'total_representatives': len(representatives),
            'singleton_clusters': cluster_sizes.get(1, 0),
            'cluster_size_distribution': dict(sorted(cluster_sizes.items())[:20]),  # Top 20
        }


# ==================== 数据写入器 ====================

class ClusterWriter:
    """聚类结果写入器"""

    def __init__(self, output_dir: str, logger: logging.Logger):
        self.output_dir = output_dir
        self.logger = logger
        os.makedirs(output_dir, exist_ok=True)

    def save_mapping_csv(self, mapping: Dict[str, str], category: str):
        """保存ID映射到CSV"""
        df = pd.DataFrame([
            {'member_id': k, 'representative_id': v}
            for k, v in mapping.items()
        ])

        output_file = f"{self.output_dir}/{category}_id_mapping.csv"
        df.to_csv(output_file, index=False)

        self.logger.info(f"ID映射: {output_file}")
        self.logger.info(f"  总条目: {len(df):,}")
        self.logger.info(f"  代表序列: {df['representative_id'].nunique():,}")

        return output_file

    def copy_clustered_fasta(self, source_file: str, category: str):
        """复制聚类后的FASTA到输出目录（标准化命名）"""
        dest_file = f"{self.output_dir}/{category}_cdhit90.fasta"
        shutil.copy2(source_file, dest_file)
        self.logger.info(f"聚类FASTA: {dest_file}")
        return dest_file

    def save_report(self, results: Dict):
        """保存完整报告"""
        report_file = f"{self.output_dir}/clustering_report.json"

        with open(report_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        self.logger.info(f"完整报告: {report_file}")

    def save_summary_txt(self, results: Dict):
        """保存文本摘要"""
        summary_file = f"{self.output_dir}/clustering_summary.txt"

        with open(summary_file, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("CD-HIT聚类去冗余统计报告\n")
            f.write(f"生成时间: {datetime.now()}\n")
            f.write("=" * 70 + "\n\n")

            f.write("【全局配置】\n")
            config = results['configuration']
            f.write(f"CD-HIT路径: {config['cdhit_path']}\n")
            f.write(f"相似度阈值: {config['similarity_threshold'] * 100:.0f}%\n")
            f.write(f"对齐覆盖度: {config['alignment_coverage'] * 100:.0f}%\n")
            f.write(f"内存限制: {config['memory_limit_mb']:,} MB\n")
            f.write(f"线程数: {config['threads']} (0=所有CPU)\n\n")

            f.write("【各类别处理结果】\n")
            total_before = 0
            total_after = 0

            for cat_name, cat_result in results['categories'].items():
                f.write(f"\n{cat_name}:\n")
                f.write(f"  原始序列: {cat_result['input_count']:,}\n")
                f.write(f"  采样后: {cat_result['sampled_count']:,}\n")
                f.write(f"  去冗余后: {cat_result['output_count']:,}\n")
                f.write(f"  去冗余率: {cat_result['reduction_rate']:.1f}%\n")
                f.write(f"  代表序列: {cat_result['representative_count']:,}\n")
                f.write(f"  聚类数: {cat_result['cluster_count']:,}\n")

                total_before += cat_result['sampled_count']
                total_after += cat_result['output_count']

            f.write(f"\n{'=' * 70}\n")
            f.write(f"总计采样后: {total_before:,}\n")
            f.write(f"总计去冗余后: {total_after:,}\n")
            f.write(f"总体去冗余率: {100 * (total_before - total_after) / max(total_before, 1):.1f}%\n")
            f.write(f"{'=' * 70}\n\n")

            f.write("【输出文件】\n")
            for cat_name in results['categories'].keys():
                f.write(f"  {cat_name}:\n")
                f.write(f"    FASTA: {self.output_dir}/{cat_name}_cdhit90.fasta\n")
                f.write(f"    聚类: {self.output_dir}/{cat_name}_cdhit90.fasta.clstr\n")
                f.write(f"    映射: {self.output_dir}/{cat_name}_id_mapping.csv\n")

            f.write("\n【下一步】\n")
            f.write("1. ESM2特征提取 (模块4)\n")
            f.write("2. 上下文特征按需计算 (模块4)\n")
            f.write("3. GNN模型训练 (模块5)\n")

        self.logger.info(f"文本摘要: {summary_file}")


# ==================== 主流程 ====================

class ClusteringPipeline:
    """CD-HIT聚类与ID映射构建主流程"""

    def __init__(self, config: ClusterConfig):
        self.config = config
        self.logger = setup_logger(
            "clustering",
            f"{config.output_dir}/logs",
            config.log_level
        )

        self.sampler = SequenceSampler(seed=config.random_seed)
        self.cdhit_runner = CDHitRunner(config, self.logger)
        self.cluster_parser = ClusterParser(self.logger)
        self.writer = ClusterWriter(config.output_dir, self.logger)

    def run(self) -> Dict:
        """执行完整聚类流程"""
        self.logger.info("=" * 80)
        self.logger.info("模块3: CD-HIT聚类去冗余与ID映射构建")
        self.logger.info(f"开始时间: {datetime.now()}")
        self.logger.info(f"硬件: {self.config.threads}线程, {self.config.memory_limit_mb}MB内存")
        self.logger.info("=" * 80)

        # 验证输入文件
        self._validate_inputs()

        # 处理每个类别
        all_results = {
            'configuration': asdict(self.config),
            'categories': {},
            'start_time': datetime.now().isoformat(),
        }

        for category, filename, max_samples in self.config.input_files:
            self.logger.info(f"\n{'=' * 60}")
            self.logger.info(f"处理类别: {category}")
            self.logger.info(f"{'=' * 60}")

            result = self._process_category(category, filename, max_samples)
            all_results['categories'][category] = result

        # 保存报告
        all_results['end_time'] = datetime.now().isoformat()
        self.writer.save_report(all_results)
        self.writer.save_summary_txt(all_results)

        # 完成
        self.logger.info(f"\n{'=' * 80}")
        self.logger.info("模块3完成!")
        self.logger.info(f"输出目录: {self.config.output_dir}")
        self.logger.info(f"{'=' * 80}")

        return all_results

    def _validate_inputs(self):
        """验证输入文件存在"""
        for category, filename, _ in self.config.input_files:
            filepath = f"{self.config.input_dir}/{filename}"
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"输入文件不存在: {filepath}")

            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            self.logger.info(f"✓ {category}: {filepath} ({size_mb:.1f} MB)")

    def _process_category(
            self,
            category: str,
            filename: str,
            max_samples: int
    ) -> Dict:
        """处理单个类别"""
        input_file = f"{self.config.input_dir}/{filename}"
        sampled_file = f"{self.config.output_dir}/{category}_sampled.fasta"
        clustered_file = f"{self.config.output_dir}/{category}_cdhit90.fasta"
        log_file = f"{self.config.output_dir}/{category}_cdhit90.log"
        clstr_file = f"{clustered_file}.clstr"

        # 步骤1: 采样（如需要）
        self.logger.info(f"\n步骤1: 采样检查...")
        original_count = self.sampler.count_sequences(input_file)
        sampled_count = self.sampler.sample_fasta(input_file, sampled_file, max_samples)

        self.logger.info(f"  原始序列: {original_count:,}")
        self.logger.info(f"  采样后: {sampled_count:,}")
        if sampled_count < original_count:
            self.logger.info(f"  采样比例: {sampled_count / original_count * 100:.1f}%")

        # 步骤2: CD-HIT聚类
        self.logger.info(f"\n步骤2: CD-HIT聚类...")
        is_hard = category.startswith('hard_')

        try:
            input_count, output_count, reduction_rate = self.cdhit_runner.run_clustering(
                category=category,
                input_file=sampled_file,
                output_file=clustered_file,
                log_file=log_file,
                is_hard=is_hard
            )
        finally:
            # 清理采样文件（可选，保留以节省空间）
            if os.path.exists(sampled_file) and sampled_count == original_count:
                # 未采样，直接复制的情况，删除重复
                os.remove(sampled_file)
                self.logger.info(f"  清理采样文件（与原始相同）")

        # 步骤3: 解析聚类文件
        self.logger.info(f"\n步骤3: 解析聚类文件...")
        mapping = self.cluster_parser.parse_clstr_file(clstr_file)

        # 统计
        stats = self.cluster_parser.get_cluster_statistics(mapping)

        self.logger.info(f"  总成员: {stats['total_members']:,}")
        self.logger.info(f"  代表序列: {stats['total_representatives']:,}")
        self.logger.info(f"  单例聚类: {stats['singleton_clusters']:,}")

        # 步骤4: 保存映射
        self.logger.info(f"\n步骤4: 保存ID映射...")
        self.writer.save_mapping_csv(mapping, category)

        # 步骤5: 复制/确认FASTA
        # CD-HIT已直接输出到clustered_file，无需复制

        return {
            'category': category,
            'input_file': input_file,
            'original_count': original_count,
            'sampled_count': sampled_count,
            'input_count': input_count,
            'output_count': output_count,
            'reduction_rate': reduction_rate,
            'representative_count': stats['total_representatives'],
            'cluster_count': stats['total_representatives'],
            'singleton_clusters': stats['singleton_clusters'],
            'mapping_count': len(mapping),
            'output_fasta': clustered_file,
            'output_clstr': clstr_file,
        }


# ==================== 命令行入口 ====================

def main():
    parser = argparse.ArgumentParser(description='模块3: CD-HIT聚类去冗余与ID映射构建')
    parser.add_argument('--input-dir',
                        default="/home/zjw/zjwdata/3_deep_learning/training_data/curated_v2",
                        help='模块1/2输出目录')
    parser.add_argument('--output-dir',
                        default="/home/zjw/zjwdata/3_deep_learning/training_data/curated_v2/clustered",
                        help='聚类输出目录')
    parser.add_argument('--positive-max', type=int, default=1000000,
                        help='正样本最大数量')
    parser.add_argument('--negative-max', type=int, default=200000,
                        help='负样本最大数量')
    parser.add_argument('--hard-strict-max', type=int, default=500000,
                        help='严格难样本最大数量')
    parser.add_argument('--hard-expanded-max', type=int, default=300000,
                        help='扩展难样本最大数量')
    parser.add_argument('--similarity', type=float, default=0.90,
                        help='CD-HIT相似度阈值')
    parser.add_argument('--coverage', type=float, default=0.90,
                        help='CD-HIT对齐覆盖度')
    parser.add_argument('--threads', type=int, default=0,
                        help='CD-HIT线程数 (0=所有CPU)')
    parser.add_argument('--memory', type=int, default=120000,
                        help='CD-HIT内存限制(MB)')
    parser.add_argument('--seed', type=int, default=42,
                        help='采样随机种子')

    args = parser.parse_args()

    # 构建输入文件配置
    input_files = (
        ('positive', 'positive_samples.fasta', args.positive_max),
        ('negative', 'negative_samples.fasta', args.negative_max),
        ('hard_strict', 'hard_samples_strict.fasta', args.hard_strict_max),
        ('hard_expanded', 'hard_samples_expanded.fasta', args.hard_expanded_max),
    )

    config = ClusterConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        input_files=input_files,
        similarity_threshold=args.similarity,
        alignment_coverage=args.coverage,
        threads=args.threads,
        memory_limit_mb=args.memory,
        random_seed=args.seed,
    )

    pipeline = ClusteringPipeline(config)
    results = pipeline.run()

    return results


if __name__ == "__main__":
    main()