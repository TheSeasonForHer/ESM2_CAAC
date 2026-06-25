#!/usr/bin/env python3
"""
玉米青贮20样本宏基因组数据质控分析 - 全缓存增量版
支持从已有 fastp JSON 自动重建缓存，修复 FastQC 文件名匹配问题
"""

import os
import sys
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
import json
import matplotlib
import time
from tqdm import tqdm
import concurrent.futures

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import matplotlib.font_manager as fm


class CornSilageQC:
    def __init__(self, raw_data_path, output_dir, max_workers=4):
        self.raw_data_path = Path(raw_data_path)
        self.output_dir = Path(output_dir)
        self.samples = []
        self.max_workers = max_workers

        # 缓存文件路径
        self.qc_cache_file = self.output_dir / "reports" / "qc_cache.json"
        self.qc_cache = self._load_qc_cache()

        # 设置中文字体
        self.setup_chinese_font()

        # 创建输出目录（不清理已有结果）
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / "fastqc_raw").mkdir(exist_ok=True)
        (self.output_dir / "fastp_reports").mkdir(exist_ok=True)
        (self.output_dir / "cleaned_data").mkdir(exist_ok=True)
        (self.output_dir / "reports").mkdir(exist_ok=True)

    def _load_qc_cache(self):
        """加载质控缓存：优先读缓存文件，没有则自动从已有 fastp json 重建"""
        cache = {}

        # 1. 尝试加载正式缓存文件
        if self.qc_cache_file.exists():
            try:
                with open(self.qc_cache_file, 'r') as f:
                    cache = json.load(f)
                print(f"📦 已加载缓存文件: {len(cache)} 个样本")
                return cache
            except Exception as e:
                print(f"⚠ 缓存文件损坏: {e}，尝试重建...")

        # 2. 自动重建：扫描已有 fastp json 报告
        fastp_dir = self.output_dir / "fastp_reports"
        if fastp_dir.exists():
            rebuilt = 0
            for json_file in fastp_dir.glob("*_fastp_report.json"):
                name = json_file.stem.replace("_fastp_report", "")
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                    b = data['summary']['before_filtering']
                    a = data['summary']['after_filtering']
                    cache[name] = {
                        'sample': name,
                        'before_qc_reads': b['total_reads'],
                        'after_qc_reads': a['total_reads'],
                        'before_qc_bases': b['total_bases'],
                        'after_qc_bases': a['total_bases'],
                        'q20_rate': a['q20_rate'],
                        'q30_rate': a['q30_rate'],
                        'gc_content': a['gc_content'],
                        'read1_mean_length': b['read1_mean_length'],
                        'read2_mean_length': b['read2_mean_length'],
                    }
                    rebuilt += 1
                except Exception:
                    continue

            if rebuilt:
                print(f"📦 从已有 fastp 报告重建缓存: {rebuilt} 个样本")
                try:
                    self.qc_cache_file.parent.mkdir(exist_ok=True)
                    with open(self.qc_cache_file, 'w') as f:
                        json.dump(cache, f, indent=2)
                    print(f"💾 重建缓存已保存至 {self.qc_cache_file}")
                except Exception as e:
                    print(f"⚠ 缓存保存失败: {e}")

        return cache

    def _save_qc_cache(self, qc_results):
        """保存质控缓存"""
        cache = {r['sample']: r for r in qc_results}
        try:
            with open(self.qc_cache_file, 'w') as f:
                json.dump(cache, f, indent=2)
            print(f"💾 缓存已保存: {len(cache)} 个样本")
        except Exception as e:
            print(f"⚠ 缓存保存失败: {e}")

    def setup_chinese_font(self):
        """设置中文字体"""
        try:
            font_paths = [
                '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf',
                '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',
                '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
                '/System/Library/Fonts/Arial.ttf',
                'C:/Windows/Fonts/simhei.ttf',
            ]
            chinese_font = None
            for font_path in font_paths:
                if Path(font_path).exists():
                    chinese_font = fm.FontProperties(fname=font_path)
                    break

            if chinese_font is None:
                plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
                plt.rcParams['axes.unicode_minus'] = False
                print("⚠ 未找到中文字体，将使用英文标签")
            else:
                plt.rcParams['font.family'] = chinese_font.get_name()
                plt.rcParams['axes.unicode_minus'] = False
                print(f"✓ 使用中文字体: {chinese_font.get_name()}")
        except Exception as e:
            print(f"⚠ 字体设置失败: {e}")
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
            plt.rcParams['axes.unicode_minus'] = False

    def use_english_labels(self):
        self.labels = {
            'title': 'Corn Silage 20 Samples Metagenomics Data Quality Report',
            'retention_rate': 'Read Retention Rate by Sample',
            'q30_dist': 'Q30 Score Distribution',
            'gc_dist': 'GC Content Distribution',
            'reads_comparison': 'Read Counts: Before vs After QC',
            'data_volume': 'Data Volume by Sample',
            'q20_vs_q30': 'Q20 vs Q30 Scores',
            'sample': 'Sample',
            'retention_pct': 'Retention Rate (%)',
            'q30_rate': 'Q30 Rate (%)',
            'gc_content': 'GC Content (%)',
            'before_qc': 'Before QC',
            'after_qc': 'After QC',
            'data_gb': 'Data Volume (GB)',
            'q20_rate': 'Q20 Rate (%)',
            'count': 'Count'
        }

    def discover_samples(self):
        """自动发现所有样本"""
        print("\n=== Discovering Sample Files ===")
        print(f"Data Directory: {self.raw_data_path}")

        sample_dirs = [d for d in self.raw_data_path.iterdir() if d.is_dir()]
        print(f"Found {len(sample_dirs)} sample directories")

        self.samples = []
        paired = 0
        for sample_dir in sorted(sample_dirs):
            r1_files = [f for f in sample_dir.glob("*_R1*.fq.gz") if not f.name.endswith('.md5')]
            r2_files = [f for f in sample_dir.glob("*_R2*.fq.gz") if not f.name.endswith('.md5')]
            if len(r1_files) == 1 and len(r2_files) == 1:
                self.samples.append({
                    'name': sample_dir.name,
                    'R1': r1_files[0],
                    'R2': r2_files[0],
                    'directory': sample_dir
                })
                paired += 1
            else:
                print(f"  ✗ {sample_dir.name}: R1={len(r1_files)}, R2={len(r2_files)}")

        cached = sum(1 for s in self.samples if s['name'] in self.qc_cache)
        print(f"\n=== 成功配对 {paired} 个样本 | 缓存命中 {cached} 个 ===")
        return self.samples

    def check_data_integrity(self):
        """检查数据完整性（仅检查新样本或缓存缺失的样本）"""
        print("\n=== Checking Data Integrity ===")
        total_gb = 0
        for s in self.samples:
            if s['name'] not in self.qc_cache:
                r1_size = s['R1'].stat().st_size / (1024**3)
                r2_size = s['R2'].stat().st_size / (1024**3)
                total_gb += r1_size + r2_size
                print(f"  {s['name']}: {r1_size+r2_size:.2f} GB (待处理)")
        if total_gb:
            print(f"待处理数据总量: {total_gb:.1f} GB")
        else:
            print("所有样本已有缓存，跳过完整性检查")

    def _fastqc_single(self, sample, fastqc_dir):
        """单个样本 FastQC"""
        cmd = [
            'fastqc', '-o', str(fastqc_dir), '-t', '4', '--extract',
            str(sample['R1']), str(sample['R2'])
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            if result.returncode == 0:
                return sample['name'], True, "success"
            else:
                log = fastqc_dir / f"{sample['name']}_fastqc_error.log"
                with open(log, 'w') as f:
                    f.write(f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}\n")
                return sample['name'], False, f"retcode_{result.returncode}"
        except subprocess.TimeoutExpired:
            return sample['name'], False, "timeout"
        except Exception as e:
            return sample['name'], False, str(e)

    def run_fastqc_analysis(self):
        """FastQC（仅跑缺失 zip 的样本）- 修复文件名匹配"""
        print("\n=== Running FastQC Quality Assessment ===")
        fastqc_dir = self.output_dir / "fastqc_raw"

        pending = []
        skipped = 0
        for sample in self.samples:
            # 修复：fastqc 输出会去掉 .fq.gz 后缀，直接用样本名+R1/R2 匹配
            z1 = fastqc_dir / f"{sample['name']}_R1_fastqc.zip"
            z2 = fastqc_dir / f"{sample['name']}_R2_fastqc.zip"
            if z1.exists() and z2.exists():
                skipped += 1
            else:
                pending.append(sample)

        if skipped:
            print(f"  ⏭ 跳过 {skipped} 个已有 FastQC 结果的样本")
        if not pending:
            print(f"  ✓ 全部 {len(self.samples)} 个样本 FastQC 就绪")
            return len(self.samples)

        print(f"  🚀 并行处理 {len(pending)} 个缺失样本")
        ok = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futures = {ex.submit(self._fastqc_single, s, fastqc_dir): s for s in pending}
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(pending), desc="FastQC"):
                name, success, msg = future.result()
                if success:
                    ok += 1
                else:
                    print(f"  ✗ {name}: {msg}")

        print(f"FastQC 就绪: {skipped + ok}/{len(self.samples)}")
        return skipped + ok

    def _fastp_single(self, sample):
        """单个样本 fastp"""
        r1_clean = self.output_dir / "cleaned_data" / f"{sample['name']}_R1_clean.fq.gz"
        r2_clean = self.output_dir / "cleaned_data" / f"{sample['name']}_R2_clean.fq.gz"
        html = self.output_dir / "fastp_reports" / f"{sample['name']}_fastp_report.html"
        json_report = self.output_dir / "fastp_reports" / f"{sample['name']}_fastp_report.json"

        cmd = [
            'fastp',
            '-i', str(sample['R1']), '-I', str(sample['R2']),
            '-o', str(r1_clean), '-O', str(r2_clean),
            '-h', str(html), '-j', str(json_report),
            '--detect_adapter_for_pe',
            '--qualified_quality_phred', '20',
            '--unqualified_percent_limit', '40',
            '--length_required', '50',
            '--correction',
            '--thread', '4',
            '--compression', '6'
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
            if result.returncode == 0 and json_report.exists():
                with open(json_report, 'r') as f:
                    data = json.load(f)
                b = data['summary']['before_filtering']
                a = data['summary']['after_filtering']
                return {
                    'sample': sample['name'],
                    'before_qc_reads': b['total_reads'],
                    'after_qc_reads': a['total_reads'],
                    'before_qc_bases': b['total_bases'],
                    'after_qc_bases': a['total_bases'],
                    'q20_rate': a['q20_rate'],
                    'q30_rate': a['q30_rate'],
                    'gc_content': a['gc_content'],
                    'read1_mean_length': b['read1_mean_length'],
                    'read2_mean_length': b['read2_mean_length'],
                    'status': 'success'
                }
            else:
                log = self.output_dir / "fastp_reports" / f"{sample['name']}_fastp_error.log"
                with open(log, 'w') as f:
                    f.write(f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}\n")
                return {'sample': sample['name'], 'status': 'failed', 'error': f'retcode_{result.returncode}'}
        except subprocess.TimeoutExpired:
            return {'sample': sample['name'], 'status': 'failed', 'error': 'timeout'}
        except Exception as e:
            return {'sample': sample['name'], 'status': 'failed', 'error': str(e)}

    def run_fastp_quality_control(self):
        """fastp（缓存命中样本彻底跳过，不读文件不执行命令）"""
        print("\n=== Running fastp Quality Control ===")

        qc_results = []
        to_run = []

        # 1. 直接从缓存加载已有样本（彻底跳过任何 I/O）
        for sample in self.samples:
            if sample['name'] in self.qc_cache:
                qc_results.append(self.qc_cache[sample['name']])
            else:
                to_run.append(sample)

        if qc_results:
            print(f"  ⏭ 缓存命中 {len(qc_results)} 个样本，彻底跳过")
        if not to_run:
            print(f"  ✓ 全部 {len(self.samples)} 个样本已有 fastp 结果")
            return qc_results

        # 2. 仅并行跑缺失样本
        print(f"  🚀 并行处理 {len(to_run)} 个新样本")
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futures = {ex.submit(self._fastp_single, s): s for s in to_run}
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(to_run), desc="fastp"):
                res = future.result()
                if res.get('status') == 'failed':
                    print(f"  ✗ {res['sample']}: {res.get('error')}")
                else:
                    res.pop('status', None)
                    qc_results.append(res)
                    ret = res['after_qc_reads'] / res['before_qc_reads'] * 100
                    print(f"  ✓ {res['sample']}: {res['before_qc_reads']:,} → {res['after_qc_reads']:,} ({ret:.1f}%)")

        # 3. 保存更新后的缓存
        self._save_qc_cache(qc_results)
        print(f"fastp 总计就绪: {len(qc_results)}/{len(self.samples)}")
        return qc_results

    def generate_multiqc_reports(self):
        """MultiQC（必须重新跑以包含全部样本，但可并行）"""
        print("\n=== Generating MultiQC Reports ===")

        def _run(input_dir, filename, title, comment):
            cmd = [
                'multiqc', str(input_dir),
                '-o', str(self.output_dir / "reports"),
                '--filename', filename,
                '--title', title,
                '--comment', comment,
                '--force'
            ]
            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True)
                return filename
            except Exception as e:
                print(f"✗ {filename}: {e}")
                return None

        tasks = []
        fastqc_dir = self.output_dir / "fastqc_raw"
        fastp_dir = self.output_dir / "fastp_reports"
        if any(fastqc_dir.glob("*")):
            tasks.append((fastqc_dir, 'multiqc_report_raw.html',
                         'Corn Silage 20 Samples - Raw Data Quality',
                         'Raw data quality assessment based on FastQC'))
        if any(fastp_dir.glob("*")):
            tasks.append((fastp_dir, 'multiqc_report_fastp.html',
                         'Corn Silage 20 Samples - FASTP QC Report',
                         'Quality control statistics based on fastp'))

        generated = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
            futures = [ex.submit(_run, d, f, t, c) for d, f, t, c in tasks]
            for future in concurrent.futures.as_completed(futures):
                res = future.result()
                if res:
                    generated.append(res)
                    print(f"✓ {res} generated")
        return generated

    def create_comprehensive_summary(self, qc_results):
        """生成综合质控总结"""
        print("\n=== Generating Comprehensive QC Summary ===")
        if not qc_results:
            print("⚠ No QC results to summarize")
            return None, None

        df = pd.DataFrame(qc_results)
        df['retention_rate'] = df['after_qc_reads'] / df['before_qc_reads'] * 100
        df['total_bases_gb'] = df['after_qc_bases'] / 1e9
        df['q30_percent'] = df['q30_rate'] * 100
        df['gc_percent'] = df['gc_content'] * 100

        stats = {
            'Total Samples': len(df),
            'Total Raw Reads': f"{df['before_qc_reads'].sum():,}",
            'Total Clean Reads': f"{df['after_qc_reads'].sum():,}",
            'Total Data Volume': f"{df['total_bases_gb'].sum():.1f} GB",
            'Average Retention Rate': f"{df['retention_rate'].mean():.1f}%",
            'Average Q30 Rate': f"{df['q30_percent'].mean():.1f}%",
            'Average GC Content': f"{df['gc_percent'].mean():.1f}%",
            'Average Read Length': f"{(df['read1_mean_length'].mean() + df['read2_mean_length'].mean()) / 2:.1f} bp"
        }

        df.to_csv(self.output_dir / "reports" / "qc_detailed_results.csv", index=False)
        df.to_excel(self.output_dir / "reports" / "qc_detailed_results.xlsx", index=False)

        with open(self.output_dir / "reports" / "qc_summary_report.txt", 'w') as f:
            f.write("Corn Silage 20 Samples Metagenomics Data QC Summary Report\n")
            f.write("=" * 70 + "\n\n")
            f.write("Project Information:\n")
            f.write(f"- Sample Type: Corn silage metagenomics\n")
            f.write(f"- Sample Count: {len(self.samples)} samples\n")
            f.write(f"- Data Directory: {self.raw_data_path}\n")
            f.write(f"- Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("Quality Statistics Summary:\n")
            for k, v in stats.items():
                f.write(f"- {k}: {v}\n")
            f.write(f"\nSample Details (Sorted by Retention Rate):\n")
            f.write("-" * 90 + "\n")
            for _, row in df.sort_values('retention_rate', ascending=False).iterrows():
                f.write(f"{row['sample']:15} | Raw: {row['before_qc_reads']:>10,} | "
                        f"Clean: {row['after_qc_reads']:>10,} | Retention: {row['retention_rate']:>5.1f}% | "
                        f"Q30: {row['q30_percent']:>5.1f}% | GC: {row['gc_percent']:>4.1f}%\n")

        print("✓ Comprehensive QC summary report generated")
        return df, stats

    def create_quality_visualizations(self, df):
        if df is None or len(df) == 0:
            print("⚠ No data for visualization")
            return

        print("\n=== Generating Quality Visualization Plots ===")
        self.use_english_labels()
        plt.style.use('default')
        sns.set_palette("husl")

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(self.labels['title'], fontsize=16, fontweight='bold')

        axes[0, 0].bar(range(len(df)), df['retention_rate'], color='skyblue', alpha=0.7)
        axes[0, 0].set_title(self.labels['retention_rate'], fontweight='bold')
        axes[0, 0].set_xlabel(self.labels['sample'])
        axes[0, 0].set_ylabel(self.labels['retention_pct'])
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].axhline(y=80, color='red', linestyle='--', alpha=0.5, label='80% baseline')
        axes[0, 0].legend()

        axes[0, 1].hist(df['q30_percent'], bins=10, color='lightgreen', alpha=0.7, edgecolor='black')
        axes[0, 1].set_title(self.labels['q30_dist'], fontweight='bold')
        axes[0, 1].set_xlabel(self.labels['q30_rate'])
        axes[0, 1].set_ylabel(self.labels['count'])
        axes[0, 1].axvline(x=85, color='red', linestyle='--', alpha=0.5, label='85% baseline')
        axes[0, 1].legend()

        axes[0, 2].hist(df['gc_percent'], bins=10, color='lightcoral', alpha=0.7, edgecolor='black')
        axes[0, 2].set_title(self.labels['gc_dist'], fontweight='bold')
        axes[0, 2].set_xlabel(self.labels['gc_content'])
        axes[0, 2].set_ylabel(self.labels['count'])

        axes[1, 0].scatter(df['before_qc_reads'], df['after_qc_reads'], alpha=0.6, color='purple')
        axes[1, 0].plot([df['before_qc_reads'].min(), df['before_qc_reads'].max()],
                        [df['before_qc_reads'].min(), df['before_qc_reads'].max()], 'r--', alpha=0.5)
        axes[1, 0].set_title(self.labels['reads_comparison'], fontweight='bold')
        axes[1, 0].set_xlabel(f'Reads {self.labels["before_qc"]}')
        axes[1, 0].set_ylabel(f'Reads {self.labels["after_qc"]}')

        axes[1, 1].bar(range(len(df)), df['total_bases_gb'], color='orange', alpha=0.7)
        axes[1, 1].set_title(self.labels['data_volume'], fontweight='bold')
        axes[1, 1].set_xlabel(self.labels['sample'])
        axes[1, 1].set_ylabel(self.labels['data_gb'])
        axes[1, 1].tick_params(axis='x', rotation=45)

        axes[1, 2].scatter(df['q20_rate'] * 100, df['q30_percent'], alpha=0.6, color='teal')
        axes[1, 2].set_title(self.labels['q20_vs_q30'], fontweight='bold')
        axes[1, 2].set_xlabel(self.labels['q20_rate'])
        axes[1, 2].set_ylabel(self.labels['q30_rate'])

        plt.tight_layout()
        plt.savefig(self.output_dir / "reports" / "quality_metrics_plots.png", dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / "reports" / "quality_metrics_plots.pdf", bbox_inches='tight')
        plt.close()
        print("✓ Quality visualization plots generated")

    def generate_final_report(self):
        print("\n=== Generating Final Analysis Report ===")
        report_file = self.output_dir / "reports" / "final_analysis_report.md"
        with open(report_file, 'w') as f:
            f.write("# Corn Silage 20 Samples Metagenomics Data Quality Control Report\n\n")
            f.write("## Project Overview\n\n")
            f.write(f"- **Sample Type**: Corn silage metagenomics\n")
            f.write(f"- **Sample Count**: {len(self.samples)} samples\n")
            f.write(f"- **Analysis Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"- **Data Directory**: {self.raw_data_path}\n")
            f.write(f"- **Output Directory**: {self.output_dir}\n\n")
            f.write("## Analysis Pipeline\n\n")
            f.write("1. **Sample Discovery**: Automatically identify and pair R1/R2 files\n")
            f.write("2. **Data Integrity Check**: Verify file sizes and pairing\n")
            f.write("3. **FastQC Analysis**: Raw data quality assessment\n")
            f.write("4. **FASTP Quality Control**: Read trimming and filtering\n")
            f.write("5. **MultiQC Reports**: Comprehensive quality reports\n")
            f.write("6. **Quality Visualization**: Statistical plots and charts\n\n")
            f.write("## Output Files\n\n")
            f.write("- `multiqc_report_raw.html`: Raw data quality report\n")
            f.write("- `multiqc_report_fastp.html`: QC statistics report\n")
            f.write("- `qc_summary_report.txt`: Text summary of QC results\n")
            f.write("- `quality_metrics_plots.png`: Quality metrics visualization\n")
            f.write("- `qc_detailed_results.csv`: Detailed QC results in CSV format\n")
            f.write("- `cleaned_data/`: Directory containing quality-filtered reads\n\n")
            f.write("## Next Steps\n\n")
            f.write("After quality control, the cleaned data can be used for:\n")
            f.write("- Metagenomic assembly\n")
            f.write("- Taxonomic profiling\n")
            f.write("- Functional annotation\n")
            f.write("- Metabolic network analysis\n")
        print("✓ Final analysis report generated")

    def run_complete_analysis(self):
        print("Starting Complete Analysis of 20 Corn Silage Metagenomics Samples")
        print("=" * 70)
        print(f"配置: 并行数={self.max_workers}, 输入={self.raw_data_path}, 输出={self.output_dir}")
        print("=" * 70)

        try:
            samples = self.discover_samples()
            if not samples:
                print("Error: No samples found!")
                return False

            self.check_data_integrity()
            self.run_fastqc_analysis()
            qc_results = self.run_fastp_quality_control()
            self.generate_multiqc_reports()

            if qc_results:
                df, stats = self.create_comprehensive_summary(qc_results)
                self.create_quality_visualizations(df)
            else:
                print("⚠ 没有可用的QC结果，跳过总结报告生成")

            self.generate_final_report()

            print("\n" + "=" * 70)
            print("ANALYSIS COMPLETED SUCCESSFULLY!")
            print(f"Results directory: {self.output_dir}")
            print("\nMain output files:")
            print(f"- Raw quality report: {self.output_dir}/reports/multiqc_report_raw.html")
            print(f"- QC report: {self.output_dir}/reports/multiqc_report_fastp.html")
            print(f"- QC summary: {self.output_dir}/reports/qc_summary_report.txt")
            print(f"- Visualization: {self.output_dir}/reports/quality_metrics_plots.png")
            print(f"- Cleaned data: {self.output_dir}/cleaned_data/")

            return True

        except Exception as e:
            print(f"\nERROR: Analysis failed - {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    raw_data_path = "/mnt/Raw-BYMB2024072902-ZXMB01-21-yumi"
    output_dir = "/mnt/zjwdata/1/corn_silage_qc_analysis"

    print("Corn Silage 20 Samples Metagenomics Data Quality Control")
    print("=" * 70)

    qc_analyzer = CornSilageQC(raw_data_path, output_dir, max_workers=4)
    success = qc_analyzer.run_complete_analysis()

    if success:
        print("\n🎉 Analysis completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Analysis failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()