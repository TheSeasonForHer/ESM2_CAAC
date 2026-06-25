#!/usr/bin/env python3
"""
quality_assessment.py - 增量版
自动扫描所有已完成样本，路径适配 /mnt/zjwdata/1/
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import json
import sys
from Bio import SeqIO
import warnings
import matplotlib.font_manager as fm

warnings.filterwarnings('ignore')


class AssemblyQualityReporter:
    def __init__(self, assembly_dir, output_dir):
        self.assembly_dir = Path(assembly_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        (self.output_dir / "plots").mkdir(exist_ok=True)
        (self.output_dir / "tables").mkdir(exist_ok=True)

        self.assembly_stats = []
        self.gene_stats = []

        self._setup_fonts()

    def _setup_fonts(self):
        try:
            chinese_fonts = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']
            available_fonts = [f.name for f in fm.fontManager.ttflist]

            for font_name in chinese_fonts:
                if any(font_name.lower() in f.lower() for f in available_fonts):
                    plt.rcParams['font.family'] = font_name
                    plt.rcParams['axes.unicode_minus'] = False
                    print(f"✅ 使用字体: {font_name}")
                    break
            else:
                print("⚠️ 未找到中文字体，使用英文标签")
                plt.rcParams['font.family'] = 'DejaVu Sans'

        except Exception as e:
            print(f"⚠️ 字体设置失败: {e}")
            plt.rcParams['font.family'] = 'DejaVu Sans'

    def discover_all_samples(self):
        """发现所有已完成样本"""
        print("🔍 扫描所有已完成的组装样本...")

        assemblies_dir = self.assembly_dir / "assemblies"
        if not assemblies_dir.exists():
            print(f"❌ 组装目录不存在: {assemblies_dir}")
            return []

        assembly_dirs = list(assemblies_dir.glob("*_megahit_k87_141"))
        print(f"找到 {len(assembly_dirs)} 个组装目录")

        all_samples = []
        for assembly_dir in sorted(assembly_dirs):
            sample_name = assembly_dir.name.replace("_megahit_k87_141", "")
            contig_file = assembly_dir / "final.contigs.fa"

            if contig_file.exists() and contig_file.stat().st_size > 1000:
                try:
                    with open(contig_file, 'r') as f:
                        if f.readline().strip().startswith('>'):
                            all_samples.append({
                                'sample_name': sample_name,
                                'contig_file': str(contig_file),
                                'assembly_dir': str(assembly_dir)
                            })
                            print(f"✅ {sample_name}")
                        else:
                            print(f"⚠️ 无效格式: {sample_name}")
                except Exception as e:
                    print(f"❌ 读取失败 {sample_name}: {e}")
            else:
                print(f"⚠️ 缺失或过小: {sample_name}")

        print(f"📊 总共 {len(all_samples)} 个有效样本")
        return all_samples

    def discover_gene_files(self):
        """发现所有基因预测文件"""
        print("🔍 扫描基因预测文件...")

        genes_dir = self.assembly_dir / "genes"
        if not genes_dir.exists():
            print("⚠️ 基因目录不存在")
            return []

        gene_files = list(genes_dir.glob("*_genes.fna"))
        gene_samples = []

        for gene_file in sorted(gene_files):
            sample_name = gene_file.name.replace("_genes.fna", "")
            if gene_file.stat().st_size > 100:
                gene_samples.append({
                    'sample_name': sample_name,
                    'gene_file': str(gene_file)
                })
                print(f"✅ {sample_name}")

        print(f"🧬 总共 {len(gene_samples)} 个基因文件")
        return gene_samples

    def collect_assembly_statistics(self):
        """收集组装统计"""
        print("📊 收集组装统计...")

        all_samples = self.discover_all_samples()
        if not all_samples:
            return False

        for sample_info in all_samples:
            stats = self.analyze_contig_file(sample_info['sample_name'], sample_info['contig_file'])
            if stats:
                self.assembly_stats.append(stats)

        print(f"✅ 成功收集 {len(self.assembly_stats)} 个样本")
        return len(self.assembly_stats) > 0

    def analyze_contig_file(self, sample_name, contig_file):
        """分析contig文件"""
        try:
            contig_lengths = []
            total_length = 0
            max_length = 0
            min_length = float('inf')

            print(f"  分析 {sample_name}...")

            with open(contig_file, 'r') as handle:
                for record in SeqIO.parse(handle, "fasta"):
                    length = len(record.seq)
                    contig_lengths.append(length)
                    total_length += length
                    max_length = max(max_length, length)
                    min_length = min(min_length, length)

            if not contig_lengths:
                return None

            contig_lengths.sort(reverse=True)
            n50 = n90 = 0
            cumulative = 0
            half = total_length * 0.5
            ninety = total_length * 0.9

            for length in contig_lengths:
                cumulative += length
                if cumulative >= half and n50 == 0:
                    n50 = length
                if cumulative >= ninety and n90 == 0:
                    n90 = length
                    break

            return {
                'sample_name': sample_name,
                'contig_file': contig_file,
                'total_contigs': len(contig_lengths),
                'total_length': total_length,
                'n50': n50,
                'n90': n90,
                'max_length': max_length,
                'min_length': min_length,
                'avg_length': total_length / len(contig_lengths),
                'gc_content': self.calculate_gc_content(contig_file)
            }

        except Exception as e:
            print(f"❌ {sample_name} 分析失败: {e}")
            return None

    def calculate_gc_content(self, contig_file):
        try:
            total_bases = 0
            gc_bases = 0

            with open(contig_file, 'r') as handle:
                for record in SeqIO.parse(handle, "fasta"):
                    seq = str(record.seq).upper()
                    total_bases += len(seq)
                    gc_bases += seq.count('G') + seq.count('C')

            return (gc_bases / total_bases * 100) if total_bases > 0 else 0

        except Exception as e:
            print(f"❌ GC计算失败: {e}")
            return 0

    def collect_gene_statistics(self):
        """收集基因统计"""
        print("🧬 收集基因统计...")

        gene_samples = self.discover_gene_files()
        if not gene_samples:
            return False

        for sample_info in gene_samples:
            stats = self.analyze_gene_file(sample_info['sample_name'], sample_info['gene_file'])
            if stats:
                self.gene_stats.append(stats)

        return len(self.gene_stats) > 0

    def analyze_gene_file(self, sample_name, gene_file):
        try:
            gene_lengths = []
            total_genes = 0

            print(f"  分析基因 {sample_name}...")

            with open(gene_file, 'r') as handle:
                for record in SeqIO.parse(handle, "fasta"):
                    gene_lengths.append(len(record.seq))
                    total_genes += 1

            if total_genes == 0:
                return None

            gene_lengths.sort(reverse=True)
            total_length = sum(gene_lengths)

            cumulative = 0
            half = total_length * 0.5
            gene_n50 = 0
            for length in gene_lengths:
                cumulative += length
                if cumulative >= half and gene_n50 == 0:
                    gene_n50 = length
                    break

            return {
                'sample_name': sample_name,
                'total_genes': total_genes,
                'total_gene_length': total_length,
                'gene_n50': gene_n50,
                'avg_gene_length': total_length / total_genes,
                'max_gene_length': max(gene_lengths),
                'min_gene_length': min(gene_lengths)
            }

        except Exception as e:
            print(f"❌ {sample_name} 基因分析失败: {e}")
            return None

    def create_visualizations(self):
        """创建可视化"""
        print("📈 生成可视化...")

        if not self.assembly_stats:
            return False

        df = pd.DataFrame(self.assembly_stats)

        self._create_assembly_overview_plot(df)
        self._create_contig_length_plot(df)
        self._create_gc_content_plot(df)
        self._create_sample_comparison_plot(df)

        if self.gene_stats:
            gene_df = pd.DataFrame(self.gene_stats)
            self._create_gene_statistics_plot(gene_df)

        print("✅ 可视化完成")
        return True

    def _create_assembly_overview_plot(self, df):
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(f'Corn Silage Metagenome Assembly Quality\nTotal Samples: {len(df)}',
                     fontsize=16, fontweight='bold')

        axes[0, 0].bar(df['sample_name'], df['total_contigs'], color='skyblue', alpha=0.7)
        axes[0, 0].set_title('Number of Contigs')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].tick_params(axis='x', rotation=45)

        axes[0, 1].bar(df['sample_name'], df['total_length'] / 1e6, color='lightgreen', alpha=0.7)
        axes[0, 1].set_title('Total Assembly Length')
        axes[0, 1].set_ylabel('Mbp')
        axes[0, 1].tick_params(axis='x', rotation=45)

        axes[0, 2].bar(df['sample_name'], df['n50'] / 1e3, color='salmon', alpha=0.7)
        axes[0, 2].set_title('N50')
        axes[0, 2].set_ylabel('Kbp')
        axes[0, 2].tick_params(axis='x', rotation=45)

        axes[1, 0].bar(df['sample_name'], df['gc_content'], color='gold', alpha=0.7)
        axes[1, 0].set_title('GC Content')
        axes[1, 0].set_ylabel('%')
        axes[1, 0].tick_params(axis='x', rotation=45)

        axes[1, 1].bar(df['sample_name'], df['avg_length'] / 1e3, color='orchid', alpha=0.7)
        axes[1, 1].set_title('Average Contig Length')
        axes[1, 1].set_ylabel('Kbp')
        axes[1, 1].tick_params(axis='x', rotation=45)

        axes[1, 2].bar(df['sample_name'], df['max_length'] / 1e3, color='lightcoral', alpha=0.7)
        axes[1, 2].set_title('Maximum Contig Length')
        axes[1, 2].set_ylabel('Kbp')
        axes[1, 2].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(self.output_dir / "plots" / "assembly_overview.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _create_contig_length_plot(self, df):
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        axes[0].scatter(df['n50'] / 1e3, df['n90'] / 1e3, s=100, alpha=0.7, color='blue')
        for _, row in df.iterrows():
            axes[0].annotate(row['sample_name'],
                             (row['n50'] / 1e3, row['n90'] / 1e3),
                             xytext=(5, 5), textcoords='offset points', fontsize=8)
        axes[0].set_xlabel('N50 (Kbp)')
        axes[0].set_ylabel('N90 (Kbp)')
        axes[0].set_title('N50 vs N90')
        axes[0].grid(True, alpha=0.3)

        length_data = [df['min_length'] / 1e3, df['avg_length'] / 1e3, df['max_length'] / 1e3]
        box = axes[1].boxplot(length_data, labels=['Min', 'Average', 'Max'], patch_artist=True)
        colors = ['lightblue', 'lightgreen', 'lightcoral']
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)
        axes[1].set_ylabel('Length (Kbp)')
        axes[1].set_title('Contig Length Distribution')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / "plots" / "contig_length_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _create_gc_content_plot(self, df):
        plt.figure(figsize=(10, 6))
        plt.hist(df['gc_content'], bins=15, alpha=0.7, color='teal', edgecolor='black')
        plt.axvline(df['gc_content'].mean(), color='red', linestyle='--',
                    label=f'Mean: {df["gc_content"].mean():.2f}%')
        plt.xlabel('GC Content (%)')
        plt.ylabel('Number of Samples')
        plt.title('GC Content Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / "plots" / "gc_content_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _create_sample_comparison_plot(self, df):
        metrics = ['total_contigs', 'total_length', 'n50', 'avg_length', 'gc_content']
        metric_names = ['Contig Count', 'Total Length', 'N50', 'Avg Length', 'GC Content']

        df_norm = df[metrics].copy()
        for col in metrics:
            if col != 'gc_content':
                df_norm[col] = (df_norm[col] - df_norm[col].min()) / (df_norm[col].max() - df_norm[col].min())

        plt.figure(figsize=(max(12, len(df) * 0.5), 8))
        sns.heatmap(df_norm.T,
                    xticklabels=df['sample_name'],
                    yticklabels=metric_names,
                    cmap='YlOrRd',
                    annot=df[metrics].T,
                    fmt='.0f',
                    cbar_kws={'label': 'Normalized Value'})
        plt.title('Sample Assembly Quality Comparison')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.output_dir / "plots" / "sample_comparison_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _create_gene_statistics_plot(self, gene_df):
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        axes[0, 0].bar(gene_df['sample_name'], gene_df['total_genes'], color='lightseagreen', alpha=0.7)
        axes[0, 0].set_title('Predicted Genes')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].tick_params(axis='x', rotation=45)

        axes[0, 1].bar(gene_df['sample_name'], gene_df['gene_n50'], color='mediumpurple', alpha=0.7)
        axes[0, 1].set_title('Gene N50')
        axes[0, 1].set_ylabel('bp')
        axes[0, 1].tick_params(axis='x', rotation=45)

        axes[1, 0].bar(gene_df['sample_name'], gene_df['avg_gene_length'], color='orange', alpha=0.7)
        axes[1, 0].set_title('Average Gene Length')
        axes[1, 0].set_ylabel('bp')
        axes[1, 0].tick_params(axis='x', rotation=45)

        axes[1, 1].bar(gene_df['sample_name'], gene_df['total_gene_length'] / 1e6, color='crimson', alpha=0.7)
        axes[1, 1].set_title('Total Gene Length')
        axes[1, 1].set_ylabel('Mbp')
        axes[1, 1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(self.output_dir / "plots" / "gene_statistics.png", dpi=300, bbox_inches='tight')
        plt.close()

    def create_excel_report(self):
        """创建Excel报告"""
        print("📋 生成Excel报告...")

        with pd.ExcelWriter(self.output_dir / "tables" / "assembly_quality_report.xlsx") as writer:

            if self.assembly_stats:
                assembly_df = pd.DataFrame(self.assembly_stats)
                assembly_df['total_length'] = assembly_df['total_length'].apply(lambda x: f"{x:,.0f}")
                assembly_df['n50'] = assembly_df['n50'].apply(lambda x: f"{x:,.0f}")
                assembly_df['n90'] = assembly_df['n90'].apply(lambda x: f"{x:,.0f}")
                assembly_df['max_length'] = assembly_df['max_length'].apply(lambda x: f"{x:,.0f}")
                assembly_df['avg_length'] = assembly_df['avg_length'].apply(lambda x: f"{x:,.0f}")
                assembly_df['gc_content'] = assembly_df['gc_content'].apply(lambda x: f"{x:.2f}%")
                assembly_df.to_excel(writer, sheet_name='Assembly Statistics', index=False)

            if self.gene_stats:
                gene_df = pd.DataFrame(self.gene_stats)
                gene_df['total_gene_length'] = gene_df['total_gene_length'].apply(lambda x: f"{x:,.0f}")
                gene_df['gene_n50'] = gene_df['gene_n50'].apply(lambda x: f"{x:,.0f}")
                gene_df['avg_gene_length'] = gene_df['avg_gene_length'].apply(lambda x: f"{x:.1f}")
                gene_df['max_gene_length'] = gene_df['max_gene_length'].apply(lambda x: f"{x:,.0f}")
                gene_df.to_excel(writer, sheet_name='Gene Statistics', index=False)

            summary_data = []
            if self.assembly_stats:
                assembly_df_num = pd.DataFrame(self.assembly_stats)
                summary_data.extend([
                    ['Total Samples', len(assembly_df_num)],
                    ['Average Contig Count', f"{assembly_df_num['total_contigs'].mean():.0f}"],
                    ['Average Total Length', f"{assembly_df_num['total_length'].mean():,.0f} bp"],
                    ['Average N50', f"{assembly_df_num['n50'].mean():,.0f} bp"],
                    ['Average GC Content', f"{assembly_df_num['gc_content'].mean():.2f}%"],
                    ['Max Contig Length', f"{assembly_df_num['max_length'].max():,.0f} bp"],
                ])

            if self.gene_stats:
                gene_df_num = pd.DataFrame(self.gene_stats)
                summary_data.extend([
                    ['Average Gene Count', f"{gene_df_num['total_genes'].mean():.0f}"],
                    ['Average Gene N50', f"{gene_df_num['gene_n50'].mean():,.0f} bp"],
                ])

            summary_df = pd.DataFrame(summary_data, columns=['Metric', 'Value'])
            summary_df.to_excel(writer, sheet_name='Quality Summary', index=False)

            if self.assembly_stats:
                detailed_info = []
                for stats in self.assembly_stats:
                    detailed_info.append({
                        'Sample': stats['sample_name'],
                        'Contigs': stats['total_contigs'],
                        'Total Length (bp)': stats['total_length'],
                        'N50 (bp)': stats['n50'],
                        'N90 (bp)': stats['n90'],
                        'Max Length (bp)': stats['max_length'],
                        'Avg Length (bp)': stats['avg_length'],
                        'GC Content (%)': stats['gc_content']
                    })
                pd.DataFrame(detailed_info).to_excel(writer, sheet_name='Sample Details', index=False)

        print("✅ Excel报告完成")
        return True

    def generate_html_report(self):
        """生成HTML报告"""
        print("🌐 生成HTML报告...")

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Corn Silage Assembly Quality Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background: #2c3e50; color: white; padding: 20px; border-radius: 10px; }}
                .section {{ margin: 30px 0; padding: 20px; border: 1px solid #ddd; border-radius: 8px; }}
                .plot {{ text-align: center; margin: 20px 0; }}
                .stats-table {{ width: 100%; border-collapse: collapse; }}
                .stats-table th, .stats-table td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                .stats-table th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🌽 Corn Silage Metagenome Assembly Quality Report</h1>
                <p>Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>Total Samples: {len(self.assembly_stats)}</p>
            </div>
        """

        if self.assembly_stats:
            html_content += """
            <div class="section">
                <h2>📊 Assembly Statistics</h2>
                <div class="plot"><img src="plots/assembly_overview.png" style="max-width: 100%;"></div>
                <div class="plot"><img src="plots/contig_length_distribution.png" style="max-width: 100%;"></div>
                <div class="plot"><img src="plots/gc_content_distribution.png" style="max-width: 100%;"></div>
                <div class="plot"><img src="plots/sample_comparison_heatmap.png" style="max-width: 100%;"></div>
            </div>
            """

        if self.gene_stats:
            html_content += """
            <div class="section">
                <h2>🧬 Gene Statistics</h2>
                <div class="plot"><img src="plots/gene_statistics.png" style="max-width: 100%;"></div>
            </div>
            """

        html_content += f"""
            <div class="section">
                <h2>📥 Downloads</h2>
                <ul>
                    <li><a href="tables/assembly_quality_report.xlsx">Excel Report</a></li>
                </ul>
            </div>
        </body>
        </html>
        """

        with open(self.output_dir / "assembly_quality_report.html", 'w', encoding='utf-8') as f:
            f.write(html_content)

        print("✅ HTML报告完成")
        return True

    def generate_report(self):
        """生成完整报告"""
        print("=" * 60)
        print("🌽 Corn Silage Metagenome Assembly Quality Report")
        print("=" * 60)

        if not self.collect_assembly_statistics():
            return False

        self.collect_gene_statistics()
        self.create_visualizations()
        self.create_excel_report()
        self.generate_html_report()

        print("=" * 60)
        print("🎉 报告生成完成!")
        print(f"📁 输出目录: {self.output_dir}")
        print("=" * 60)
        return True


def main():
    assembly_dir = "/mnt/zjwdata/1/assembly_analysis"
    output_dir = "/mnt/zjwdata/1/assembly_quality_report"

    if len(sys.argv) > 1:
        assembly_dir = sys.argv[1]
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]

    reporter = AssemblyQualityReporter(assembly_dir, output_dir)
    success = reporter.generate_report()

    if success:
        print("✅ 全部完成!")
        sys.exit(0)
    else:
        print("❌ 失败!")
        sys.exit(1)


if __name__ == "__main__":
    main()