#!/usr/bin/env python3
"""
Kraken2 + Bracken物种注释脚本 - 多样本版本（原始数据版）
用于玉米饲料发酵微生物组成分析
直接使用原始fastq数据进行物种注释
修复了错误处理逻辑，以文件存在性判断成功与否
"""

import subprocess
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
import logging
import glob


class MultiSampleKraken2PipelineRaw:
    def __init__(self, raw_data_dir, base_output_dir, kraken_db, threads=16, confidence=0.2):
        """
        初始化多样本Kraken2管道（原始数据版）

        Args:
            raw_data_dir: 原始fastq数据的根目录
            base_output_dir: 输出结果的基目录
            kraken_db: Kraken2数据库路径
            threads: 线程数
            confidence: 置信度阈值
        """
        self.raw_data_dir = Path(raw_data_dir)
        self.base_output_dir = Path(base_output_dir)
        self.kraken_db = kraken_db
        self.threads = threads
        self.confidence = confidence

        # 创建输出目录
        self.base_output_dir.mkdir(parents=True, exist_ok=True)

        # 设置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.base_output_dir / 'multi_sample_kraken2_bracken_raw.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def find_fastq_samples(self):
        """
        查找所有的原始fastq样本数据
        每个样本目录下应包含_R1.fq.gz和_R2.fq.gz文件
        """
        self.logger.info("正在查找fastq样本数据...")

        samples = {}

        # 查找样本目录
        sample_dirs = [d for d in self.raw_data_dir.glob("*") if d.is_dir()]

        for sample_dir in sample_dirs:
            sample_name = sample_dir.name

            # 查找该样本的fastq文件
            r1_files = list(sample_dir.glob("*_R1.fq.gz"))
            r2_files = list(sample_dir.glob("*_R2.fq.gz"))

            if not r1_files or not r2_files:
                # 如果未找到标准的fastq文件，尝试其他常见格式
                r1_files = list(sample_dir.glob("*_1.fq.gz")) + list(sample_dir.glob("*_1.fastq.gz"))
                r2_files = list(sample_dir.glob("*_2.fq.gz")) + list(sample_dir.glob("*_2.fastq.gz"))

            # 确保找到匹配的R1和R2文件
            if r1_files and r2_files:
                # 取第一个匹配的文件（假设每个样本只有一个配对）
                r1_file = r1_files[0]
                r2_file = r2_files[0]

                # 验证文件大小（可选）
                if r1_file.stat().st_size > 0 and r2_file.stat().st_size > 0:
                    samples[sample_name] = {
                        'r1_file': r1_file,
                        'r2_file': r2_file,
                        'output_dir': self.base_output_dir / sample_name
                    }
                    self.logger.info(f"找到样本 {sample_name}: {r1_file.name} / {r2_file.name}")
                else:
                    self.logger.warning(f"样本 {sample_name} 的fastq文件大小异常，跳过")
            else:
                self.logger.warning(f"样本 {sample_name} 未找到完整的fastq配对文件")

        self.logger.info(f"总共找到 {len(samples)} 个有效样本")
        return samples

    def run_kraken2_paired_end(self, r1_file, r2_file, output_dir):
        """为双端fastq数据运行Kraken2进行物种分类"""
        output_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"开始处理样本: {r1_file.parent.name}")
        self.logger.info(f"R1文件: {r1_file.name}")
        self.logger.info(f"R2文件: {r2_file.name}")
        self.logger.info(f"Kraken2置信度阈值: {self.confidence}")

        kraken_output = output_dir / "kraken_output.txt"
        kraken_report = output_dir / "kraken_report.txt"

        # Kraken2双端命令
        cmd = [
            "kraken2",
            "--db", self.kraken_db,
            "--threads", str(self.threads),
            "--report", str(kraken_report),
            "--output", str(kraken_output),
            "--use-names",
            "--confidence", str(self.confidence),
            "--paired",  # 指定为双端数据
            "--gzip-compressed",  # 输入为gzip压缩格式
            str(r1_file),
            str(r2_file)
        ]

        self.logger.info(f"运行命令: {' '.join(cmd)}")

        try:
            # 运行命令但不检查返回码，因为Kraken2有时会返回非零退出码但实际成功
            result = subprocess.run(cmd, capture_output=True, text=True)

            # 记录Kraken2输出
            if result.stdout:
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        self.logger.info(f"Kraken2输出: {line.strip()}")

            if result.stderr:
                for line in result.stderr.strip().split('\n'):
                    if line.strip():
                        self.logger.warning(f"Kraken2警告: {line.strip()}")

            # 检查报告文件是否存在且不为空
            if kraken_report.exists() and kraken_report.stat().st_size > 0:
                self.logger.info(f"Kraken2运行完成: {r1_file.parent.name}")
                self.logger.info(f"报告文件已生成: {kraken_report}")
                return str(kraken_output), str(kraken_report)
            else:
                self.logger.error(f"Kraken2运行失败: 报告文件未生成或为空")
                if result.returncode != 0:
                    self.logger.error(f"Kraken2返回码: {result.returncode}")
                return None, None

        except FileNotFoundError:
            self.logger.error("Kraken2未安装或不在PATH中")
            return None, None
        except Exception as e:
            self.logger.error(f"运行Kraken2时发生异常: {e}")
            return None, None

    def run_bracken_single(self, kraken_report, output_dir, level='S'):
        """为单个样本运行Bracken进行物种丰度估计"""
        if kraken_report is None:
            return None

        self.logger.info(f"开始Bracken物种丰度估计 (水平: {level})...")

        bracken_output = output_dir / f"bracken_output_{level}.txt"
        bracken_report = output_dir / f"bracken_report_{level}.txt"

        cmd = [
            "bracken",
            "-d", self.kraken_db,
            "-i", str(kraken_report),
            "-o", str(bracken_output),
            "-r", "150",  # 读长，根据实际数据调整
            "-l", level,
            "-t", "10"
        ]

        self.logger.info(f"运行命令: {' '.join(cmd)}")

        try:
            # 运行Bracken，不检查返回码
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.stdout:
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        self.logger.info(f"Bracken输出: {line.strip()}")

            if result.stderr:
                for line in result.stderr.strip().split('\n'):
                    if line.strip():
                        self.logger.warning(f"Bracken警告: {line.strip()}")

            # 检查输出文件是否存在
            if bracken_output.exists() and bracken_output.stat().st_size > 0:
                self.logger.info("Bracken运行完成")
                return str(bracken_output)
            else:
                self.logger.error("Bracken运行失败: 输出文件未生成或为空")
                return None

        except FileNotFoundError:
            self.logger.error("Bracken未安装或不在PATH中")
            return None
        except Exception as e:
            self.logger.error(f"运行Bracken时发生异常: {e}")
            return None

    def parse_bracken_output_single(self, bracken_file, sample_name):
        """解析单个样本的Bracken输出文件"""
        if bracken_file is None or not Path(bracken_file).exists():
            self.logger.warning(f"Bracken输出文件不存在: {bracken_file}")
            return pd.DataFrame()

        self.logger.info(f"解析Bracken输出: {bracken_file}")

        try:
            df = pd.read_csv(bracken_file, sep='\t')

            if df.empty:
                self.logger.warning(f"Bracken输出为空: {sample_name}")
                return df

            # 计算相对丰度
            total_reads = df['new_est_reads'].sum()
            if total_reads > 0:
                df['relative_abundance'] = df['new_est_reads'] / total_reads * 100
            else:
                df['relative_abundance'] = 0
                self.logger.warning(f"样本 {sample_name} 总读段数为0")

            # 添加样本信息
            df['sample_name'] = sample_name

            # 过滤低丰度物种（可根据需要调整阈值）
            df_filtered = df[df['relative_abundance'] > 0.01].copy()

            self.logger.info(f"解析完成 {sample_name}: 共{len(df_filtered)}个物种 (丰度>0.01%)")
            return df_filtered

        except Exception as e:
            self.logger.error(f"解析Bracken输出失败 {sample_name}: {e}")
            return pd.DataFrame()

    def generate_species_composition_table_single(self, bracken_df, output_dir):
        """为单个样本生成物种组成表"""
        if bracken_df.empty:
            self.logger.warning("Bracken结果为空，无法生成物种组成表")
            return pd.DataFrame()

        # 选择关键列并排序
        composition_table = bracken_df[[
            'name', 'taxonomy_id', 'taxonomy_lvl', 'new_est_reads',
            'relative_abundance', 'fraction_total_reads', 'sample_name'
        ]].copy()

        composition_table = composition_table.sort_values(
            'relative_abundance', ascending=False
        )

        # 保存物种组成表
        output_file = output_dir / "species_composition_table.csv"
        composition_table.to_csv(output_file, index=False, encoding='utf-8-sig')
        self.logger.info(f"物种组成表已保存: {output_file}")

        return composition_table

    def visualize_species_composition_single(self, composition_table, output_dir, sample_name, top_n=20):
        """为单个样本可视化物种组成"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            plt.style.use('default')
            sns.set_palette("husl")

            # 取前top_n个物种
            if len(composition_table) < top_n:
                top_n = len(composition_table)
                self.logger.info(f"样本 {sample_name} 物种数少于{top_n}，显示所有{top_n}个物种")

            top_species = composition_table.head(top_n).copy()

            # 创建图表
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

            # 饼图
            if not top_species.empty and top_species['relative_abundance'].sum() > 0:
                wedges, texts, autotexts = ax1.pie(
                    top_species['relative_abundance'],
                    labels=top_species['name'],
                    autopct='%1.1f%%',
                    startangle=90
                )
                ax1.set_title(
                    f'{sample_name} - Top {top_n} Species Composition\n(Kraken2+Bracken, Confidence: {self.confidence})')

                # 改善饼图标签可读性
                for text in texts:
                    text.set_fontsize(8)
                for autotext in autotexts:
                    autotext.set_fontsize(8)
                    autotext.set_color('white')
                    autotext.set_weight('bold')
            else:
                ax1.text(0.5, 0.5, 'No species data', ha='center', va='center', fontsize=12)
                ax1.set_title(f'{sample_name} - No Species Data')

            # 条形图
            if not top_species.empty:
                y_pos = np.arange(len(top_species))
                bars = ax2.barh(y_pos, top_species['relative_abundance'])
                ax2.set_yticks(y_pos)
                ax2.set_yticklabels(top_species['name'], fontsize=8)
                ax2.set_xlabel('Relative Abundance (%)')
                ax2.set_title('Species Abundance Distribution')
                ax2.invert_yaxis()

                # 在条形上添加数值
                for i, bar in enumerate(bars):
                    width = bar.get_width()
                    ax2.text(width + 0.1, bar.get_y() + bar.get_height() / 2,
                             f'{width:.2f}%', ha='left', va='center', fontsize=7)
            else:
                ax2.text(0.5, 0.5, 'No species data', ha='center', va='center', fontsize=12)

            plt.tight_layout()
            plt.savefig(output_dir / 'species_composition_plot.png',
                        dpi=300, bbox_inches='tight')
            plt.close()

            self.logger.info(f"{sample_name} 物种组成可视化图已保存 (前{top_n}个物种)")

        except ImportError:
            self.logger.warning("matplotlib或seaborn未安装，跳过可视化")
        except Exception as e:
            self.logger.error(f"{sample_name} 可视化失败: {e}")

    def generate_analysis_report_single(self, composition_table, r1_file, r2_file, output_dir, sample_name):
        """为单个样本生成分析报告"""
        report_file = output_dir / "analysis_report.txt"

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"=== {sample_name} - Kraken2+Bracken物种注释分析报告 ===\n\n")
            f.write(f"输入文件1: {r1_file}\n")
            f.write(f"输入文件2: {r2_file}\n")
            f.write(f"分析时间: {pd.Timestamp.now()}\n")
            f.write(f"Kraken2数据库: {self.kraken_db}\n")
            f.write(f"Kraken2置信度阈值: {self.confidence}\n")
            f.write(f"数据类型: 双端原始fastq\n\n")

            if not composition_table.empty:
                f.write("物种组成统计:\n")
                f.write(f"- 总注释物种数: {len(composition_table)}\n")
                f.write(f"- 总估计读段数: {composition_table['new_est_reads'].sum():,}\n")
                f.write(f"- 前10个物种占总丰度: {composition_table.head(10)['relative_abundance'].sum():.2f}%\n")
                f.write(
                    f"- 最丰富的物种: {composition_table.iloc[0]['name']} ({composition_table.iloc[0]['relative_abundance']:.2f}%)\n\n")

                f.write("前20个最丰富的物种:\n")
                f.write("Rank\tSpecies\tRelative Abundance(%)\n")
                for i, row in composition_table.head(20).iterrows():
                    f.write(f"{i + 1}\t{row['name']}\t{row['relative_abundance']:.2f}\n")
            else:
                f.write("警告: 未检测到任何物种或注释结果为空\n\n")

            f.write("\n分析参数:\n")
            f.write(f"- 线程数: {self.threads}\n")
            f.write(f"- Kraken2置信度阈值: {self.confidence}\n")
            f.write(f"- Bracken读长: 150\n")
            f.write(f"- 最小丰度显示: 0.01%\n")
            f.write(f"- 数据类型: 双端测序 (paired-end)\n")

            f.write("\n输出文件:\n")
            for file_path in output_dir.glob("*"):
                if file_path.is_file():
                    f.write(f"- {file_path.name}\n")

        self.logger.info(f"{sample_name} 分析报告已保存: {report_file}")

    def check_existing_results(self, output_dir):
        """检查是否已存在处理结果，避免重复运行"""
        # 检查关键文件是否存在
        kraken_report = output_dir / "kraken_report.txt"
        bracken_output = output_dir / "bracken_output_S.txt"
        species_table = output_dir / "species_composition_table.csv"

        # 如果所有关键文件都存在且不为空，则认为已处理完成
        if (kraken_report.exists() and kraken_report.stat().st_size > 0 and
                bracken_output.exists() and bracken_output.stat().st_size > 0 and
                species_table.exists() and species_table.stat().st_size > 0):
            return True
        return False

    def process_single_sample(self, sample_name, sample_info):
        """处理单个样本"""
        self.logger.info(f"\n{'=' * 60}")
        self.logger.info(f"开始处理样本: {sample_name}")
        self.logger.info(f"{'=' * 60}")

        r1_file = sample_info['r1_file']
        r2_file = sample_info['r2_file']
        output_dir = sample_info['output_dir']

        # 检查是否已存在处理结果
        if self.check_existing_results(output_dir):
            self.logger.info(f"样本 {sample_name} 已处理完成，跳过")

            # 加载现有结果
            bracken_file = output_dir / "bracken_output_S.txt"
            bracken_df = self.parse_bracken_output_single(str(bracken_file), sample_name)

            if not bracken_df.empty:
                return bracken_df
            else:
                self.logger.warning(f"样本 {sample_name} 的现有结果为空，重新处理")

        # 1. 运行Kraken2（双端）
        kraken_output, kraken_report = self.run_kraken2_paired_end(r1_file, r2_file, output_dir)
        if kraken_report is None:
            self.logger.error(f"{sample_name} Kraken2处理失败，跳过后续步骤")
            return None

        # 2. 运行Bracken (种水平)
        bracken_file = self.run_bracken_single(kraken_report, output_dir, level='S')

        # 3. 解析Bracken结果
        bracken_df = self.parse_bracken_output_single(bracken_file, sample_name)

        # 4. 生成物种组成表
        composition_table = self.generate_species_composition_table_single(bracken_df, output_dir)

        # 5. 可视化
        if not composition_table.empty:
            self.visualize_species_composition_single(composition_table, output_dir, sample_name)

        # 6. 生成分析报告
        self.generate_analysis_report_single(composition_table, r1_file, r2_file, output_dir, sample_name)

        self.logger.info(f"{sample_name} 处理完成!")
        return composition_table

    def combine_all_results(self, all_results):
        """合并所有样本的结果到一个表格"""
        self.logger.info("合并所有样本结果...")

        combined_dfs = []
        for sample_name, df in all_results.items():
            if df is not None and not df.empty:
                combined_dfs.append(df)

        if combined_dfs:
            combined_df = pd.concat(combined_dfs, ignore_index=True)

            # 创建样本-物种矩阵（适合下游分析）
            pivot_table = pd.pivot_table(
                combined_df,
                values='relative_abundance',
                index='name',
                columns='sample_name',
                fill_value=0
            )

            # 保存合并结果
            combined_output = self.base_output_dir / "combined_species_abundance.csv"
            pivot_table.to_csv(combined_output, encoding='utf-8-sig')
            self.logger.info(f"合并结果已保存: {combined_output}")

            # 保存原始合并数据
            raw_combined = self.base_output_dir / "all_samples_raw_results.csv"
            combined_df.to_csv(raw_combined, index=False, encoding='utf-8-sig')

            return combined_df, pivot_table
        else:
            self.logger.warning("没有有效的结果可合并")
            return pd.DataFrame(), pd.DataFrame()

    def generate_summary_report(self, all_results):
        """生成所有样本的汇总报告"""
        self.logger.info("生成汇总报告...")

        summary_file = self.base_output_dir / "multi_sample_summary_report.txt"

        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("=== 多样本Kraken2+Bracken物种注释汇总报告 ===\n\n")
            f.write(f"分析时间: {pd.Timestamp.now()}\n")
            f.write(f"原始数据目录: {self.raw_data_dir}\n")
            f.write(f"总处理样本数: {len(all_results)}\n")
            f.write(f"Kraken2数据库: {self.kraken_db}\n")
            f.write(f"置信度阈值: {self.confidence}\n\n")

            f.write("各样本处理结果:\n")
            f.write("样本名称\t物种数量\t总读段数\t最丰富物种\t最丰富物种丰度(%)\n")

            successful_samples = 0
            for sample_name, result in all_results.items():
                if result is not None and not result.empty:
                    species_count = len(result)
                    total_reads = result['new_est_reads'].sum()
                    top_species = result.iloc[0]['name']
                    top_abundance = result.iloc[0]['relative_abundance']

                    f.write(f"{sample_name}\t{species_count}\t{total_reads:,}\t{top_species}\t{top_abundance:.2f}\n")
                    successful_samples += 1
                else:
                    f.write(f"{sample_name}\t处理失败\t-\t-\t-\n")

            f.write(f"\n总结:\n")
            f.write(f"- 成功处理样本数: {successful_samples}/{len(all_results)}\n")
            f.write(f"- 失败样本数: {len(all_results) - successful_samples}\n")

        self.logger.info(f"汇总报告已保存: {summary_file}")

    def run_multi_sample_pipeline(self):
        """运行多样本Kraken2+Bracken流程（原始数据版）"""
        self.logger.info("开始多样本Kraken2+Bracken物种注释流程（原始数据版）...")
        self.logger.info(f"原始数据目录: {self.raw_data_dir}")
        self.logger.info(f"输出基目录: {self.base_output_dir}")
        self.logger.info(f"数据库: {self.kraken_db}")
        self.logger.info(f"置信度阈值: {self.confidence}")

        # 1. 查找所有样本
        samples = self.find_fastq_samples()
        if not samples:
            self.logger.error("未找到任何fastq样本文件!")
            return {}

        # ===== 核心修复：运行前显式过滤已完成的样本 =====
        pending_samples = {}
        skipped_samples = []
        for sample_name, sample_info in samples.items():
            output_dir = sample_info['output_dir']
            if self.check_existing_results(output_dir):
                skipped_samples.append(sample_name)
            else:
                pending_samples[sample_name] = sample_info

        self.logger.info(f"【状态统计】原始样本总数: {len(samples)}")
        self.logger.info(f"【状态统计】已完成样本（将跳过）: {len(skipped_samples)}个 -> {skipped_samples}")
        self.logger.info(f"【状态统计】待处理样本（将运行）: {len(pending_samples)}个 -> {list(pending_samples.keys())}")

        if not pending_samples:
            self.logger.info("所有样本已处理完成，无需运行。正在加载已有结果生成汇总...")
            # 直接加载所有已有结果并生成汇总
            all_results = {}
            for sample_name, sample_info in samples.items():
                bracken_file = sample_info['output_dir'] / "bracken_output_S.txt"
                bracken_df = self.parse_bracken_output_single(str(bracken_file), sample_name)
                all_results[sample_name] = bracken_df
            combined_df, pivot_table = self.combine_all_results(all_results)
            self.generate_summary_report(all_results)
            self.logger.info("汇总报告已生成，流程结束。")
            return all_results, combined_df, pivot_table
        # ===== 修复结束 =====

        # 2. 分别处理每个样本（只遍历待处理样本）
        all_results = {}
        for sample_name, sample_info in pending_samples.items():
            result = self.process_single_sample(sample_name, sample_info)
            all_results[sample_name] = result

        # 3. 加载已跳过的样本结果（确保汇总包含全部21个）
        for sample_name, sample_info in samples.items():
            if sample_name in skipped_samples:
                bracken_file = sample_info['output_dir'] / "bracken_output_S.txt"
                bracken_df = self.parse_bracken_output_single(str(bracken_file), sample_name)
                all_results[sample_name] = bracken_df

        # 4. 合并所有结果
        combined_df, pivot_table = self.combine_all_results(all_results)

        # 5. 生成汇总报告
        self.generate_summary_report(all_results)

        self.logger.info("多样本Kraken2+Bracken物种注释流程完成!")
        return all_results, combined_df, pivot_table


def main():
    # 设置路径参数
    raw_data_dir = "/home/zjw/zjwdata/Raw-BYMB2024072902-ZXMB01-21-yumi"  # 原始fastq数据目录
    base_output_dir = "/mnt/zjwdata/2/raw/species_annotation_results_raw"  # 输出结果的基目录
    kraken_db = "/home/databases/kraken2_database/"  # Kraken2数据库路径

    print("开始多样本物种注释分析（原始数据版）...")
    print(f"原始数据目录: {raw_data_dir}")
    print(f"输出基目录: {base_output_dir}")
    print(f"数据库: {kraken_db}")
    print(f"置信度: 0.2")

    try:
        pipeline = MultiSampleKraken2PipelineRaw(
            raw_data_dir=raw_data_dir,
            base_output_dir=base_output_dir,
            kraken_db=kraken_db,
            threads=16,
            confidence=0.2
        )

        all_results, combined_df, pivot_table = pipeline.run_multi_sample_pipeline()

        print(f"\n分析完成! 结果保存在: {base_output_dir}")
        successful = len([r for r in all_results.values() if r is not None and not r.empty])
        print(f"成功处理 {successful}/{len(all_results)} 个样本")

        # 显示处理成功的样本
        successful_samples = [name for name, result in all_results.items()
                              if result is not None and not result.empty]
        if successful_samples:
            print(f"\n成功处理的样本: {', '.join(successful_samples)}")

    except Exception as e:
        print(f"分析失败: {e}")
        import traceback
        print(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()