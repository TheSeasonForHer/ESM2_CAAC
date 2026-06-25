#!/usr/bin/env python3
"""
build_non_redundant_genes.py - 增量版
支持增量处理新样本，跳过已有结果，路径适配 /mnt/zjwdata/1/
"""

import os
import subprocess
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import logging
from collections import defaultdict
import matplotlib as mpl
import shutil

# 设置中文字体
def setup_chinese_font():
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False

        available_fonts = set([f.name for f in mpl.font_manager.fontManager.ttflist])
        chinese_fonts = ['SimHei', 'Microsoft YaHei', 'STSong', 'STKaiti', 'STHeiti']

        for font in chinese_fonts:
            if font in available_fonts:
                plt.rcParams['font.sans-serif'] = [font] + plt.rcParams['font.sans-serif']
                print(f"✅ 使用中文字体: {font}")
                break
        else:
            print("⚠️ 未找到中文字体，使用英文标签")

    except Exception as e:
        print(f"字体设置失败: {e}，使用英文标签")


setup_chinese_font()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class NonRedundantGeneBuilder:
    def __init__(self, assembly_dir, output_dir):
        self.assembly_dir = Path(assembly_dir)
        self.output_dir = Path(output_dir)

        self.global_nr_dir = self.output_dir / "global_non_redundant_genes"
        self.per_sample_nr_dir = self.output_dir / "per_sample_non_redundant_genes"

        self.genes_dir = self.assembly_dir / "genes"
        self.proteins_dir = self.assembly_dir / "proteins"

        self.global_nr_dir.mkdir(parents=True, exist_ok=True)
        self.per_sample_nr_dir.mkdir(parents=True, exist_ok=True)

        # 缓存已有结果，避免重复处理
        self.existing_per_sample = set()
        self.existing_global = False

    def find_gene_files(self):
        """查找所有基因预测文件"""
        logger.info("正在查找基因预测文件...")

        if not self.genes_dir.exists():
            logger.error(f"基因目录不存在: {self.genes_dir}")
            return {}

        if not self.proteins_dir.exists():
            logger.error(f"蛋白质目录不存在: {self.proteins_dir}")
            return {}

        faa_files = list(self.proteins_dir.glob("*.faa"))
        fna_files = list(self.genes_dir.glob("*.fna"))

        logger.info(f"蛋白质目录: {len(faa_files)} 个文件")
        logger.info(f"基因目录: {len(fna_files)} 个文件")

        sample_files = {}

        for faa_file in faa_files:
            sample_name = faa_file.stem.replace('_proteins', '')
            fna_file = self.genes_dir / f"{sample_name}_genes.fna"

            if fna_file.exists():
                sample_files[sample_name] = {
                    'faa': faa_file,
                    'fna': fna_file
                }

        return sample_files

    def check_existing_results(self, sample_files):
        """检查哪些样本已有 per_sample 结果"""
        existing = set()
        to_process = {}

        for sample_name, files in sample_files.items():
            # 检查 per_sample 输出是否完整
            sample_output_dir = self.per_sample_nr_dir / sample_name
            nr_faa = sample_output_dir / f"{sample_name}_non_redundant_genes.faa"
            nr_fna = sample_output_dir / f"{sample_name}_non_redundant_genes.fna"
            report = sample_output_dir / f"{sample_name}_non_redundant_genes_report.txt"

            if nr_faa.exists() and nr_fna.exists() and report.exists() and nr_faa.stat().st_size > 1000:
                existing.add(sample_name)
                self.existing_per_sample.add(sample_name)
            else:
                # 检查输入文件有效性
                faa_size = files['faa'].stat().st_size
                fna_size = files['fna'].stat().st_size
                if faa_size > 1000 and fna_size > 1000:
                    to_process[sample_name] = files
                else:
                    logger.warning(f"❌ {sample_name}: 输入文件过小，跳过")

        logger.info(f"已有 per_sample 结果: {len(existing)} 个样本")
        logger.info(f"需要处理: {len(to_process)} 个样本")

        return existing, to_process

    def build_per_sample_gene_catalogs(self, sample_files):
        """为每个样本单独构建非冗余基因目录 - 增量版"""
        logger.info("正在为每个样本构建非冗余基因目录...")

        existing, to_process = self.check_existing_results(sample_files)
        sample_stats = []

        # 加载已有统计
        stats_csv = self.per_sample_nr_dir / "per_sample_statistics.csv"
        if stats_csv.exists():
            existing_stats = pd.read_csv(stats_csv)
            for _, row in existing_stats.iterrows():
                sample_stats.append(row.to_dict())
            logger.info(f"已加载 {len(existing_stats)} 个已有统计记录")

        for sample_name, files in to_process.items():
            logger.info(f"处理样本: {sample_name}")

            sample_output_dir = self.per_sample_nr_dir / sample_name
            sample_output_dir.mkdir(parents=True, exist_ok=True)

            try:
                # 清理可能的残留（如果之前跑一半失败了）
                for f in sample_output_dir.glob("*"):
                    if f.is_file():
                        f.unlink()

                nr_protein_file, cluster_file = self.run_cd_hit_single_sample(
                    files['faa'], sample_output_dir, sample_name
                )

                nr_gene_count = self.count_genes_in_file(nr_protein_file)
                original_gene_count = self.count_genes_in_file(files['faa'])

                nr_nucleotide_file = self.extract_nucleotide_sequences_single_sample(
                    nr_protein_file, files['fna'], sample_output_dir, sample_name
                )

                clustering_stats = {
                    'original_genes': original_gene_count,
                    'non_redundant_genes': nr_gene_count,
                    'total_clusters': nr_gene_count,
                    'singleton_clusters': nr_gene_count,
                    'multi_gene_clusters': 0,
                    'reduction_ratio': (
                        (original_gene_count - nr_gene_count) / original_gene_count * 100
                        if original_gene_count > 0 else 0
                    ),
                    'avg_cluster_size': 1.0,
                    'max_cluster_size': 1,
                    'cluster_data': []
                }

                gene_characteristics = self.analyze_gene_characteristics_single_sample(
                    nr_protein_file, nr_nucleotide_file, sample_name
                )

                self.generate_single_sample_report(
                    sample_name, clustering_stats, gene_characteristics, sample_output_dir
                )

                stats = {
                    'sample': sample_name,
                    'original_genes': original_gene_count,
                    'non_redundant_genes': nr_gene_count,
                    'reduction_ratio': clustering_stats['reduction_ratio'],
                    'avg_protein_length': gene_characteristics['avg_protein_length'],
                    'avg_gc_content': gene_characteristics['avg_gc_content']
                }
                sample_stats.append(stats)

                logger.info(
                    f"✅ {sample_name}: {original_gene_count} → {nr_gene_count} "
                    f"(减少 {clustering_stats['reduction_ratio']:.1f}%)"
                )

            except Exception as e:
                logger.error(f"❌ {sample_name} 处理失败: {e}")
                import traceback
                logger.error(traceback.format_exc())
                continue

        # 保存更新后的统计
        if sample_stats:
            stats_df = pd.DataFrame(sample_stats)
            stats_df.to_csv(self.per_sample_nr_dir / "per_sample_statistics.csv",
                            index=False, encoding='utf-8-sig')
            logger.info(f"统计信息已保存: {len(sample_stats)} 个样本")

        return sample_stats

    def run_cd_hit_single_sample(self, input_faa, output_dir, sample_name):
        """为单个样本运行CD-HIT"""
        logger.info(f"样本 {sample_name}: 运行CD-HIT...")

        output_faa = output_dir / f"{sample_name}_non_redundant_genes.faa"
        cluster_file = output_dir / f"{sample_name}_gene_clusters.clstr"

        cmd = [
            "cd-hit",
            "-i", str(input_faa),
            "-o", str(output_faa),
            "-c", "0.95",
            "-aS", "0.9",
            "-g", "1",
            "-T", "8",
            "-M", "16000",
            "-d", "0"
        ]

        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)

            with open(cluster_file, 'w', encoding='utf-8') as f:
                f.write(result.stdout)

            return output_faa, cluster_file

        except subprocess.CalledProcessError as e:
            logger.error(f"CD-HIT失败: {e}")
            raise

    def extract_nucleotide_sequences_single_sample(self, nr_protein_file, nucleotide_file, output_dir, sample_name):
        """提取非冗余核酸序列"""
        output_fna = output_dir / f"{sample_name}_non_redundant_genes.fna"

        nr_gene_ids = set()
        with open(nr_protein_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith('>'):
                    gene_id = line[1:].split()[0].strip()
                    nr_gene_ids.add(gene_id)

        gene_to_sequence = {}
        with open(nucleotide_file, 'r', encoding='utf-8') as f:
            current_id = None
            current_seq = []
            for line in f:
                if line.startswith('>'):
                    if current_id and current_seq:
                        gene_to_sequence[current_id] = ''.join(current_seq)
                    current_id = line[1:].split()[0].strip()
                    current_seq = []
                else:
                    current_seq.append(line.strip())
            if current_id and current_seq:
                gene_to_sequence[current_id] = ''.join(current_seq)

        found = 0
        with open(output_fna, 'w', encoding='utf-8') as f:
            for gene_id in nr_gene_ids:
                if gene_id in gene_to_sequence:
                    f.write(f">{gene_id}\n")
                    f.write(f"{gene_to_sequence[gene_id]}\n")
                    found += 1

        logger.info(f"{sample_name}: 提取 {found} 个核酸序列")
        return output_fna

    def count_genes_in_file(self, fasta_file):
        count = 0
        with open(fasta_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith('>'):
                    count += 1
        return count

    def analyze_gene_characteristics_single_sample(self, nr_protein_file, nr_nucleotide_file, sample_name):
        """分析单个样本基因特征"""
        protein_lengths = []
        with open(nr_protein_file, 'r', encoding='utf-8') as f:
            current_seq = ""
            for line in f:
                if line.startswith('>'):
                    if current_seq:
                        protein_lengths.append(len(current_seq))
                    current_seq = ""
                else:
                    current_seq += line.strip()
            if current_seq:
                protein_lengths.append(len(current_seq))

        gc_contents = []
        with open(nr_nucleotide_file, 'r', encoding='utf-8') as f:
            current_seq = ""
            for line in f:
                if line.startswith('>'):
                    if current_seq:
                        seq = current_seq.upper()
                        gc = (seq.count('G') + seq.count('C')) / len(seq) if len(seq) > 0 else 0
                        gc_contents.append(gc)
                    current_seq = ""
                else:
                    current_seq += line.strip()
            if current_seq:
                seq = current_seq.upper()
                gc = (seq.count('G') + seq.count('C')) / len(seq) if len(seq) > 0 else 0
                gc_contents.append(gc)

        return {
            'protein_lengths': protein_lengths,
            'gc_contents': gc_contents,
            'avg_protein_length': np.nanmean(protein_lengths) if protein_lengths else 0,
            'avg_gc_content': np.nanmean(gc_contents) if gc_contents else 0,
            'min_protein_length': np.nanmin(protein_lengths) if protein_lengths else 0,
            'max_protein_length': np.nanmax(protein_lengths) if protein_lengths else 0,
        }

    def generate_single_sample_report(self, sample_name, clustering_stats, gene_characteristics, output_dir):
        """生成单个样本报告"""
        report_file = output_dir / f"{sample_name}_non_redundant_genes_report.txt"

        report = [
            f"=== Sample {sample_name} Non-redundant Gene Catalog Report ===",
            "",
            "1. Gene Statistics:",
            f"   Original Genes: {clustering_stats['original_genes']:,}",
            f"   Non-redundant Genes: {clustering_stats['non_redundant_genes']:,}",
            f"   Data Redundancy Reduction: {clustering_stats['reduction_ratio']:.1f}%",
            "",
            "2. Gene Characteristics:",
            f"   Average Protein Length: {gene_characteristics['avg_protein_length']:.1f} amino acids",
            f"   Protein Length Range: {gene_characteristics['min_protein_length']} - {gene_characteristics['max_protein_length']} amino acids",
            f"   Average GC Content: {gene_characteristics['avg_gc_content']:.3f}",
            "",
            "=== Report Generation Completed ==="
        ]

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))

    def merge_gene_files(self, sample_files):
        """合并所有样本基因文件 - 增量版：只合并新样本"""
        logger.info("合并基因文件用于全局分析...")

        merged_faa = self.output_dir / "all_genes_combined.faa"
        merged_fna = self.output_dir / "all_genes_combined.fna"
        sample_info_file = self.output_dir / "sample_gene_statistics.csv"

        # 如果已有合并文件且所有样本都已处理，直接返回
        if merged_faa.exists() and merged_fna.exists():
            existing_samples = set()
            if sample_info_file.exists():
                existing_df = pd.read_csv(sample_info_file)
                existing_samples = set(existing_df['sample'].tolist())

            new_samples = set(sample_files.keys()) - existing_samples
            if not new_samples:
                logger.info("所有样本已合并，跳过")
                stats_df = pd.read_csv(sample_info_file)
                return merged_faa, merged_fna, stats_df
            else:
                logger.info(f"发现 {len(new_samples)} 个新样本需要追加合并")

        # 全新合并或重建
        sample_stats = []
        total_genes = 0

        with open(merged_faa, 'w', encoding='utf-8') as out_faa, \
             open(merged_fna, 'w', encoding='utf-8') as out_fna:

            for sample_name, files in sorted(sample_files.items()):
                gene_count = 0

                with open(files['faa'], 'r', encoding='utf-8') as in_faa:
                    for line in in_faa:
                        if line.startswith('>'):
                            original_id = line[1:].split()[0].strip()
                            out_faa.write(f">{sample_name}_{original_id}\n")
                            gene_count += 1
                            total_genes += 1
                        else:
                            out_faa.write(line)

                with open(files['fna'], 'r', encoding='utf-8') as in_fna:
                    for line in in_fna:
                        if line.startswith('>'):
                            original_id = line[1:].split()[0].strip()
                            out_fna.write(f">{sample_name}_{original_id}\n")
                        else:
                            out_fna.write(line)

                sample_stats.append({
                    'sample': sample_name,
                    'gene_count': gene_count,
                    'protein_file': str(files['faa']),
                    'nucleotide_file': str(files['fna'])
                })

                logger.info(f"样本 {sample_name}: {gene_count} 个基因")

        stats_df = pd.DataFrame(sample_stats)
        stats_df.to_csv(sample_info_file, index=False, encoding='utf-8-sig')

        logger.info(f"合并完成: 总共 {total_genes} 个基因")
        return merged_faa, merged_fna, stats_df

    def run_cd_hit_global(self, input_faa):
        """全局CD-HIT - 检查是否已有结果"""
        output_faa = self.global_nr_dir / "global_non_redundant_genes.faa"
        cluster_file = self.global_nr_dir / "global_gene_clusters.clstr"

        if output_faa.exists() and output_faa.stat().st_size > 1000:
            logger.info("全局CD-HIT结果已存在，跳过")
            self.existing_global = True
            return output_faa, cluster_file

        logger.info("运行全局CD-HIT...")

        cmd = [
            "cd-hit",
            "-i", str(input_faa),
            "-o", str(output_faa),
            "-c", "0.95",
            "-aS", "0.9",
            "-g", "1",
            "-T", "8",
            "-M", "16000",
            "-d", "0"
        ]

        result = subprocess.run(cmd, check=True, capture_output=True, text=True)

        with open(cluster_file, 'w', encoding='utf-8') as f:
            f.write(result.stdout)

        return output_faa, cluster_file

    def extract_nucleotide_sequences_global(self, nr_protein_file, merged_nucleotide_file):
        """提取全局非冗余核酸序列"""
        output_fna = self.global_nr_dir / "global_non_redundant_genes.fna"

        if output_fna.exists() and output_fna.stat().st_size > 1000:
            logger.info("全局核酸序列已存在，跳过")
            return output_fna

        logger.info("提取全局非冗余核酸序列...")

        nr_gene_ids = set()
        with open(nr_protein_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith('>'):
                    nr_gene_ids.add(line[1:].split()[0].strip())

        gene_to_sequence = {}
        with open(merged_nucleotide_file, 'r', encoding='utf-8') as f:
            current_id = None
            current_seq = []
            for line in f:
                if line.startswith('>'):
                    if current_id and current_seq:
                        gene_to_sequence[current_id] = ''.join(current_seq)
                    current_id = line[1:].split()[0].strip()
                    current_seq = []
                else:
                    current_seq.append(line.strip())
            if current_id and current_seq:
                gene_to_sequence[current_id] = ''.join(current_seq)

        found = 0
        with open(output_fna, 'w', encoding='utf-8') as f:
            for gene_id in nr_gene_ids:
                if gene_id in gene_to_sequence:
                    f.write(f">{gene_id}\n")
                    f.write(f"{gene_to_sequence[gene_id]}\n")
                    found += 1

        logger.info(f"提取 {found} 个全局核酸序列")
        return output_fna

    def analyze_gene_characteristics_global(self, nr_protein_file, nr_nucleotide_file):
        """分析全局基因特征"""
        protein_lengths = []
        with open(nr_protein_file, 'r', encoding='utf-8') as f:
            current_seq = ""
            for line in f:
                if line.startswith('>'):
                    if current_seq:
                        protein_lengths.append(len(current_seq))
                    current_seq = ""
                else:
                    current_seq += line.strip()
            if current_seq:
                protein_lengths.append(len(current_seq))

        gc_contents = []
        with open(nr_nucleotide_file, 'r', encoding='utf-8') as f:
            current_seq = ""
            for line in f:
                if line.startswith('>'):
                    if current_seq:
                        seq = current_seq.upper()
                        gc = (seq.count('G') + seq.count('C')) / len(seq) if len(seq) > 0 else 0
                        gc_contents.append(gc)
                    current_seq = ""
                else:
                    current_seq += line.strip()
            if current_seq:
                seq = current_seq.upper()
                gc = (seq.count('G') + seq.count('C')) / len(seq) if len(seq) > 0 else 0
                gc_contents.append(gc)

        return {
            'protein_lengths': protein_lengths,
            'gc_contents': gc_contents,
            'avg_protein_length': np.nanmean(protein_lengths) if protein_lengths else 0,
            'avg_gc_content': np.nanmean(gc_contents) if gc_contents else 0,
            'min_protein_length': np.nanmin(protein_lengths) if protein_lengths else 0,
            'max_protein_length': np.nanmax(protein_lengths) if protein_lengths else 0,
        }

    def build_gene_catalog(self):
        """构建非冗余基因目录 - 增量版"""
        logger.info("开始构建非冗余基因目录...")

        sample_files = self.find_gene_files()
        if not sample_files:
            logger.error("未找到基因预测文件")
            return

        # per_sample 增量处理
        per_sample_stats = self.build_per_sample_gene_catalogs(sample_files)

        # 全局处理（如果样本有变化则重建）
        logger.info("构建全局非冗余基因目录...")

        merged_faa, merged_fna, sample_stats = self.merge_gene_files(sample_files)
        original_gene_count = sample_stats['gene_count'].sum()

        nr_protein_file, cluster_file = self.run_cd_hit_global(merged_faa)
        nr_gene_count = self.count_genes_in_file(nr_protein_file)

        nr_nucleotide_file = self.extract_nucleotide_sequences_global(nr_protein_file, merged_fna)

        gene_characteristics = self.analyze_gene_characteristics_global(nr_protein_file, nr_nucleotide_file)

        logger.info("\n=== 非冗余基因目录构建完成 ===")
        logger.info(f"Per Sample: {self.per_sample_nr_dir}")
        logger.info(f"Global: {self.global_nr_dir}")
        logger.info(f"全局统计: {original_gene_count:,} → {nr_gene_count:,} "
                    f"(减少 {(original_gene_count - nr_gene_count) / original_gene_count * 100:.1f}%)")

        # 生成全局报告
        self.generate_global_report(original_gene_count, nr_gene_count, gene_characteristics)

    def generate_global_report(self, original, nr, characteristics):
        """生成全局报告"""
        report_file = self.global_nr_dir / "global_non_redundant_genes_report.txt"

        report = [
            "=== Global Non-redundant Gene Catalog Report ===",
            "",
            "1. Statistics:",
            f"   Original Genes: {original:,}",
            f"   Non-redundant Genes: {nr:,}",
            f"   Reduction Ratio: {(original - nr) / original * 100:.1f}%",
            "",
            "2. Characteristics:",
            f"   Average Protein Length: {characteristics['avg_protein_length']:.1f} aa",
            f"   Average GC Content: {characteristics['avg_gc_content']:.3f}",
            "",
            "=== Report Generation Completed ==="
        ]

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))

        logger.info(f"全局报告已保存: {report_file}")


def main():
    assembly_dir = "/mnt/zjwdata/1/assembly_analysis"
    output_dir = "/mnt/zjwdata/1/gene_catalog_analysis"

    builder = NonRedundantGeneBuilder(assembly_dir, output_dir)
    builder.build_gene_catalog()


if __name__ == "__main__":
    main()