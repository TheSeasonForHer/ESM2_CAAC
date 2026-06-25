import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import re
from matplotlib_venn import venn2
import warnings

warnings.filterwarnings('ignore')
from scipy.spatial.distance import pdist, squareform


class SpeciesComplementaryValidator:
    def __init__(self, metaphlan_dir, kraken2_dir, output_dir):
        self.metaphlan_dir = metaphlan_dir
        self.kraken2_dir = kraken2_dir
        self.output_dir = output_dir
        self.final_species_table = None

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'individual_reports'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'visualizations'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'conflict_analysis'), exist_ok=True)

    def parse_metaphlan_results(self, sample_id):
        """解析MetaPhlAn结果文件 - 针对实际格式优化"""
        sample_metaphlan_dir = os.path.join(self.metaphlan_dir, sample_id)

        if not os.path.exists(sample_metaphlan_dir):
            print(f"警告: 未找到样本 {sample_id} 的MetaPhlAn目录")
            return pd.DataFrame()

        # 优先读取标准输出文件
        for fname in ('species_profile.txt', 'metaphlan_profile.txt', 'profile.txt'):
            metaphlan_file = os.path.join(sample_metaphlan_dir, fname)
            if os.path.isfile(metaphlan_file):
                print(f"找到MetaPhlAn文件: {fname}")
                try:
                    # 使用pandas直接读取标准格式
                    df = pd.read_csv(metaphlan_file, sep='\t')

                    print(f"文件列名: {list(df.columns)}")
                    print(f"总行数: {len(df)}")

                    # 检查必需的列
                    required_cols = ['species_name', 'relative_abundance']
                    if not all(col in df.columns for col in required_cols):
                        print(f"文件缺少必需列，实际列: {list(df.columns)}")
                        return pd.DataFrame()

                    # 筛选有效的物种行（相对丰度 > 0）
                    valid_species = df[df['relative_abundance'] > 0].copy()

                    if valid_species.empty:
                        print(f"未找到有效物种（相对丰度 > 0）")
                        return pd.DataFrame()

                    # 标准化物种名称
                    valid_species['species_standardized'] = (
                        valid_species['species_name']
                        .str.replace('_', ' ')
                    )

                    result_df = valid_species[['species_standardized', 'relative_abundance']].copy()
                    result_df = result_df.rename(columns={'relative_abundance': 'metaphlan_abundance'})
                    result_df['metaphlan_reads'] = 0

                    print(f"成功解析 {len(result_df)} 个物种")
                    print(f"前5个物种: {list(result_df['species_standardized'].head())}")

                    return result_df

                except Exception as e:
                    print(f"解析MetaPhlAn文件失败 {metaphlan_file}: {e}")
                    return pd.DataFrame()

        print(f"未找到MetaPhlAn输出文件")
        return pd.DataFrame()

    def parse_kraken2_results(self, sample_id):
        """解析Kraken2结果文件，读取您的species_composition_table.csv"""
        sample_kraken2_dir = os.path.join(self.kraken2_dir, sample_id)

        if not os.path.exists(sample_kraken2_dir):
            print(f"警告: 未找到样本 {sample_id} 的Kraken2目录")
            return pd.DataFrame()

        # 查找物种组成表文件
        composition_file = os.path.join(sample_kraken2_dir, "species_composition_table.csv")

        if not os.path.exists(composition_file):
            print(f"警告: 未找到样本 {sample_id} 的species_composition_table.csv文件")
            print(f"在目录中查找其他文件...")
            # 尝试查找其他可能的文件
            for file in os.listdir(sample_kraken2_dir):
                file_path = os.path.join(sample_kraken2_dir, file)
                if 'species' in file.lower() and 'composition' in file.lower() and file.endswith('.csv'):
                    composition_file = file_path
                    print(f"找到替代文件: {file}")
                    break

            if not os.path.exists(composition_file):
                print(f"未找到物种组成表文件")
                return pd.DataFrame()

        try:
            print(f"读取Kraken2物种组成表: {composition_file}")
            df = pd.read_csv(composition_file, encoding='utf-8-sig')

            print(f"文件列名: {list(df.columns)}")
            print(f"总行数: {len(df)}")

            # 检查必需的列
            required_cols = ['name', 'relative_abundance']
            if not all(col in df.columns for col in required_cols):
                print(f"文件缺少必需列，实际列: {list(df.columns)}")
                print("尝试自动检测列名...")

                # 尝试找到包含物种名称的列
                name_col = None
                for col in df.columns:
                    if 'name' in col.lower() or 'species' in col.lower():
                        name_col = col
                        break

                # 尝试找到包含丰度的列
                abundance_col = None
                for col in df.columns:
                    if 'abundance' in col.lower() or 'relative' in col.lower():
                        abundance_col = col
                        break

                if name_col is None or abundance_col is None:
                    print(f"无法自动检测列名")
                    return pd.DataFrame()

                print(f"使用列: {name_col} 作为物种名称, {abundance_col} 作为相对丰度")

                # 创建新的DataFrame
                result_df = df[[name_col, abundance_col]].copy()
                result_df = result_df.rename(columns={
                    name_col: 'species',
                    abundance_col: 'relative_abundance'
                })
            else:
                # 使用标准列名
                result_df = df[['name', 'relative_abundance']].copy()
                result_df = result_df.rename(columns={'name': 'species'})

            # 过滤有效的物种行（相对丰度 > 0）
            valid_species = result_df[result_df['relative_abundance'] > 0].copy()

            if valid_species.empty:
                print(f"未找到有效物种（相对丰度 > 0）")
                return pd.DataFrame()

            # 标准化物种名称
            valid_species['species_standardized'] = (
                valid_species['species']
                .str.replace('_', ' ')
                .str.strip()
            )

            # 获取reads数（如果存在）
            if 'new_est_reads' in df.columns:
                reads_df = df[['name', 'new_est_reads']].copy()
                reads_df = reads_df.rename(columns={'name': 'species'})
                reads_df['species_standardized'] = (
                    reads_df['species']
                    .str.replace('_', ' ')
                    .str.strip()
                )

                # 合并reads数
                valid_species = pd.merge(
                    valid_species,
                    reads_df[['species_standardized', 'new_est_reads']],
                    on='species_standardized',
                    how='left'
                )
                valid_species['kraken2_reads'] = valid_species['new_est_reads'].fillna(0).astype(int)
            else:
                valid_species['kraken2_reads'] = 0

            # 重命名列以匹配输出格式
            final_df = valid_species[['species_standardized', 'relative_abundance', 'kraken2_reads']].copy()
            final_df = final_df.rename(columns={'relative_abundance': 'kraken2_abundance'})

            print(f"成功解析 {len(final_df)} 个物种")
            print(f"前5个物种: {list(final_df['species_standardized'].head())}")

            return final_df

        except Exception as e:
            print(f"解析Kraken2文件失败 {composition_file}: {e}")
            import traceback
            print(traceback.format_exc())
            return pd.DataFrame()

    def standardize_species_names(self, df, tool_name):
        """标准化物种名称以提高匹配率"""
        if df.empty:
            return df

        df_std = df.copy()

        def clean_species_name(name):
            if pd.isna(name):
                return ""
            # 移除括号内容
            name = re.sub(r'\([^)]*\)', '', name)
            # 移除方括号内容
            name = re.sub(r'\[.*?\]', '', name)
            # 标准化空格
            name = re.sub(r'\s+', ' ', name).strip()
            return name

        # 根据DataFrame的列结构选择正确的列名
        if 'species_standardized' in df_std.columns:
            # 已经是标准化后的名称
            df_std['species_standardized'] = df_std['species_standardized'].apply(clean_species_name)
        elif 'species' in df_std.columns:
            df_std['species_standardized'] = df_std['species'].apply(clean_species_name)
        else:
            print(f"警告: {tool_name} 数据框中未找到物种名称列")
            return pd.DataFrame()

        return df_std

    def complementary_validation(self, metaphlan_df, kraken2_df, sample_id):
        """执行互补验证分析"""
        # 标准化物种名称
        metaphlan_std = self.standardize_species_names(metaphlan_df, 'metaphlan')
        kraken2_std = self.standardize_species_names(kraken2_df, 'kraken2')

        # 获取物种集合
        metaphlan_species = set(metaphlan_std['species_standardized']) if not metaphlan_std.empty else set()
        kraken2_species = set(kraken2_std['species_standardized']) if not kraken2_std.empty else set()

        # 合并所有物种
        all_species = metaphlan_species.union(kraken2_species)

        results = []
        conflict_species = []  # 存储冲突物种信息

        validation_report = {
            'sample_id': sample_id,
            'total_species': len(all_species),
            'metaphlan_only': len(metaphlan_species - kraken2_species),
            'kraken2_only': len(kraken2_species - metaphlan_species),
            'common_species': len(metaphlan_species.intersection(kraken2_species)),
            'species_details': []
        }

        for species in all_species:
            # 查找在两个工具中的丰度
            metaphlan_abundance = 0
            kraken2_abundance = 0
            metaphlan_reads = 0
            kraken2_reads = 0
            confidence = 'low'

            # MetaPhlAn中的丰度
            if not metaphlan_std.empty:
                metaphlan_match = metaphlan_std[metaphlan_std['species_standardized'] == species]
                if not metaphlan_match.empty:
                    metaphlan_abundance = metaphlan_match['metaphlan_abundance'].iloc[0]
                    metaphlan_reads = metaphlan_match['metaphlan_reads'].iloc[0]

            # Kraken2中的丰度
            if not kraken2_std.empty:
                kraken2_match = kraken2_std[kraken2_std['species_standardized'] == species]
                if not kraken2_match.empty:
                    kraken2_abundance = kraken2_match['kraken2_abundance'].iloc[0]
                    kraken2_reads = kraken2_match['kraken2_reads'].iloc[0]

            # 确定检测来源
            detected_by = 'both' if metaphlan_abundance > 0 and kraken2_abundance > 0 else (
                'metaphlan_only' if metaphlan_abundance > 0 else 'kraken2_only'
            )

            # 确定置信度
            if metaphlan_abundance > 0 and kraken2_abundance > 0:
                # 两个工具都检测到，检查丰度一致性
                max_abundance = max(metaphlan_abundance, kraken2_abundance)
                if max_abundance > 0:
                    abundance_ratio = min(metaphlan_abundance, kraken2_abundance) / max_abundance
                    if abundance_ratio > 0.5:  # 丰度差异小于2倍
                        confidence = 'high'
                    else:
                        confidence = 'medium'
                        # 记录冲突物种
                        conflict_species.append({
                            'species_standardized': species,
                            'metaphlan_abundance': metaphlan_abundance,
                            'kraken2_abundance': kraken2_abundance,
                            'metaphlan_reads': metaphlan_reads,
                            'kraken2_reads': kraken2_reads,
                            'abundance_ratio': abundance_ratio
                        })
            elif metaphlan_abundance > 1.0 or kraken2_abundance > 1.0:  # 任一工具丰度>1%
                confidence = 'medium'
            else:
                confidence = 'low'

            # 最终丰度：优先使用一致的，否则使用检测到的
            if confidence == 'high':
                final_abundance = (metaphlan_abundance + kraken2_abundance) / 2
            elif metaphlan_abundance > 0:
                final_abundance = metaphlan_abundance
            else:
                final_abundance = kraken2_abundance

            # 关键物种判定
            is_key_species = (
                    confidence == 'high' and
                    final_abundance >= 1.0 and
                    detected_by == 'both'
            )

            species_detail = {
                'species_standardized': species,
                'metaphlan_abundance': metaphlan_abundance,
                'kraken2_abundance': kraken2_abundance,
                'metaphlan_reads': metaphlan_reads,
                'kraken2_reads': kraken2_reads,
                'final_abundance': final_abundance,
                'confidence': confidence,
                'detected_by': detected_by,
                'is_key_species': is_key_species
            }

            results.append(species_detail)
            validation_report['species_details'].append(species_detail)

        # 保存冲突物种列表
        if conflict_species:
            conflict_df = pd.DataFrame(conflict_species)
            conflict_file = os.path.join(self.output_dir, 'conflict_analysis', f'{sample_id}_conflict_species.tsv')
            conflict_df.to_csv(conflict_file, sep='\t', index=False)
            print(f"冲突物种列表已保存: {conflict_file}")

        return pd.DataFrame(results), validation_report

    def create_visualization(self, validation_report, sample_id):
        """为单个样本创建可视化"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'Sample {sample_id} - Species Annotation Complementary Validation', fontsize=16)

            # 1. Venn图 - 物种检测重叠
            ax1 = axes[0, 0]
            metaphlan_only = validation_report['metaphlan_only']
            kraken2_only = validation_report['kraken2_only']
            common = validation_report['common_species']

            venn = venn2(subsets=(metaphlan_only, kraken2_only, common),
                         set_labels=('MetaPhlAn', 'Kraken2'), ax=ax1)
            ax1.set_title('Species Detection Overlap')

            # 2. 置信度分布
            ax2 = axes[0, 1]
            confidence_data = [detail['confidence'] for detail in validation_report['species_details']]
            confidence_counts = pd.Series(confidence_data).value_counts()
            colors = {'high': 'green', 'medium': 'orange', 'low': 'red'}
            confidence_colors = [colors.get(conf, 'gray') for conf in confidence_counts.index]

            bars = ax2.bar(confidence_counts.index, confidence_counts.values, color=confidence_colors)
            ax2.set_title('Confidence Level Distribution')
            ax2.set_ylabel('Number of Species')

            # 在柱子上添加数值
            for bar, count in zip(bars, confidence_counts.values):
                ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                         str(count), ha='center', va='bottom')

            # 3. 关键物种分布
            ax3 = axes[1, 0]
            key_species_data = [detail['is_key_species'] for detail in validation_report['species_details']]
            key_counts = pd.Series(key_species_data).value_counts()

            if not key_counts.empty:
                labels = []
                sizes = []
                colors_pie = []
                if True in key_counts.index:
                    labels.append('Key Species')
                    sizes.append(key_counts[True])
                    colors_pie.append('red')
                if False in key_counts.index:
                    labels.append('Non-key Species')
                    sizes.append(key_counts[False])
                    colors_pie.append('lightgray')

                wedges, texts, autotexts = ax3.pie(sizes, labels=labels, autopct='%1.1f%%',
                                                   startangle=90, colors=colors_pie)
                ax3.set_title('Key Species Distribution')
            else:
                ax3.text(0.5, 0.5, 'No Key Species', ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title('Key Species Distribution')

            # 4. 丰度相关性散点图（仅共同检测的物种）
            ax4 = axes[1, 1]
            common_species = [detail for detail in validation_report['species_details']
                              if detail['detected_by'] == 'both']

            if common_species:
                metaphlan_abundances = [detail['metaphlan_abundance'] for detail in common_species]
                kraken2_abundances = [detail['kraken2_abundance'] for detail in common_species]

                ax4.scatter(metaphlan_abundances, kraken2_abundances, alpha=0.6)
                ax4.plot([0, max(metaphlan_abundances)], [0, max(metaphlan_abundances)], 'r--', alpha=0.8)
                ax4.set_xlabel('MetaPhlAn Abundance (%)')
                ax4.set_ylabel('Kraken2 Abundance (%)')
                ax4.set_title('Abundance Correlation (Common Species)')

                # 计算相关系数
                if len(common_species) > 1:
                    correlation = np.corrcoef(metaphlan_abundances, kraken2_abundances)[0, 1]
                    ax4.text(0.05, 0.95, f'R = {correlation:.3f}', transform=ax4.transAxes,
                             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'visualizations', f'{sample_id}_validation.png'),
                        dpi=300, bbox_inches='tight')
            plt.close()
            print(f"可视化图表已保存: {sample_id}_validation.png")
        except Exception as e:
            print(f"创建可视化失败: {e}")

    def generate_sample_report(self, validation_report, sample_id):
        """生成单个样本的详细报告"""
        try:
            report_file = os.path.join(self.output_dir, 'individual_reports', f'{sample_id}_report.txt')

            with open(report_file, 'w') as f:
                f.write(f"=== Complementary Validation Report for Sample {sample_id} ===\n\n")

                f.write("SUMMARY STATISTICS:\n")
                f.write(f"Total species detected: {validation_report['total_species']}\n")
                f.write(f"Common species (both tools): {validation_report['common_species']}\n")
                f.write(f"MetaPhlAn only species: {validation_report['metaphlan_only']}\n")
                f.write(f"Kraken2 only species: {validation_report['kraken2_only']}\n")
                if validation_report['total_species'] > 0:
                    agreement_rate = validation_report['common_species'] / validation_report['total_species'] * 100
                    f.write(f"Agreement rate: {agreement_rate:.1f}%\n\n")
                else:
                    f.write(f"Agreement rate: 0.0%\n\n")

                f.write("CONFIDENCE LEVEL BREAKDOWN:\n")
                confidence_counts = {}
                for detail in validation_report['species_details']:
                    conf = detail['confidence']
                    confidence_counts[conf] = confidence_counts.get(conf, 0) + 1

                for conf, count in confidence_counts.items():
                    percentage = count / validation_report['total_species'] * 100 if validation_report[
                                                                                         'total_species'] > 0 else 0
                    f.write(f"  {conf}: {count} species ({percentage:.1f}%)\n")
                f.write("\n")

                f.write("KEY SPECIES IDENTIFICATION:\n")
                key_species = [detail for detail in validation_report['species_details']
                               if detail['is_key_species']]
                f.write(f"Total key species identified: {len(key_species)}\n\n")

                f.write("TOP SPECIES (by final abundance):\n")
                sorted_species = sorted(validation_report['species_details'],
                                        key=lambda x: x['final_abundance'], reverse=True)[:15]

                f.write("Species\tFinal_Abundance(%)\tMetaPhlAn(%)\tKraken2(%)\tConfidence\tDetected_By\tKey_Species\n")
                for species in sorted_species:
                    f.write(f"{species['species_standardized']}\t{species['final_abundance']:.3f}\t")
                    f.write(f"{species['metaphlan_abundance']:.3f}\t{species['kraken2_abundance']:.3f}\t")
                    f.write(f"{species['confidence']}\t{species['detected_by']}\t{species['is_key_species']}\n")

            print(f"样本报告已保存: {sample_id}_report.txt")
        except Exception as e:
            print(f"生成样本报告失败: {e}")

    def export_micom_table(self, min_abundance=0.1, only_key=False):
        """生成 MICOM 可直接读取的丰度表 - 优化版本，归一化到1.0"""
        if self.final_species_table is None or self.final_species_table.empty:
            print("Warning: no species data available, MICOM table not generated.")
            return

        df = self.final_species_table.copy()

        # 应用筛选条件
        if only_key:
            df = df[df['is_key_species'] == True]
            print(f"筛选关键物种: {len(df)} 个物种")

        df = df[df['final_abundance'] >= min_abundance]

        if df.empty:
            print(
                f"Warning: no species passed the filter (min_abundance={min_abundance}, only_key={only_key}), MICOM table not generated.")
            return

        # 创建pivot表格
        micom_df = (df.pivot_table(index='sample_id',
                                   columns='species_standardized',
                                   values='final_abundance',
                                   fill_value=0)
                    .rename_axis(None, axis=1))

        # MICOM归一化：确保每行和为1.0
        row_sums = micom_df.sum(axis=1)
        # 只归一化非零行
        valid_rows = row_sums > 0
        micom_df.loc[valid_rows] = micom_df.loc[valid_rows].div(row_sums[valid_rows], axis=0)

        # 保存文件
        suffix = "_key_only" if only_key else "_all_species"
        out_path = os.path.join(self.output_dir, f'micom_abundance_table{suffix}.csv')
        micom_df.to_csv(out_path)

        # 验证归一化结果
        normalized_sums = micom_df.sum(axis=1)
        print(f"MICOM abundance table -> {out_path}")
        print(f"表形状: {micom_df.shape}")
        print(f"归一化验证 - 行和范围: [{normalized_sums.min():.3f}, {normalized_sums.max():.3f}]")

        return micom_df

    def plot_jaccard_heatmap(self):
        """绘制样本间Jaccard相似度热图"""
        if self.final_species_table is None or self.final_species_table.empty:
            print("Warning: no species data available for Jaccard heatmap.")
            return

        try:
            # 创建物种存在/缺失矩阵
            species_pivot = self.final_species_table.pivot_table(
                index='sample_id', columns='species_standardized',
                values='final_abundance', fill_value=0)

            # 二值化：只要有>0 就认为存在
            binary = (species_pivot > 0).astype(int)

            if len(binary) < 2:
                print("Warning: need at least 2 samples for Jaccard heatmap.")
                return

            # 计算Jaccard距离
            jaccard_dist = pdist(binary.values, metric='jaccard')
            jaccard_mat = squareform(jaccard_dist)
            jaccard_similarity = 1 - jaccard_mat  # 转换为相似度

            jaccard_df = pd.DataFrame(jaccard_similarity,
                                      index=binary.index,
                                      columns=binary.index)

            plt.figure(figsize=(10, 8))
            sns.heatmap(jaccard_df, annot=True, fmt=".3f", cmap="YlGnBu",
                        cbar_kws={'label': 'Jaccard Similarity'})
            plt.title("Jaccard Similarity Between Samples\n(Species Presence/Absence)")
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "jaccard_heatmap.png"), dpi=300)
            plt.close()

            print(f"Jaccard similarity heatmap saved")

            # 保存相似度矩阵
            jaccard_df.to_csv(os.path.join(self.output_dir, "jaccard_similarity_matrix.csv"))
        except Exception as e:
            print(f"绘制Jaccard热图失败: {e}")

    def generate_summary_report(self, summary_stats):
        """生成所有样本的汇总报告"""
        if not summary_stats:
            print("没有汇总数据可生成报告")
            return

        try:
            summary_df = pd.DataFrame(summary_stats)

            # 保存汇总统计
            summary_file = os.path.join(self.output_dir, 'validation_summary_statistics.csv')
            summary_df.to_csv(summary_file, index=False)

            # 创建汇总可视化
            plt.figure(figsize=(12, 8))

            # 样本间比较
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Cross-Sample Complementary Validation Summary', fontsize=16)

            # 1. 各样本检测物种数量
            ax1 = axes[0, 0]
            x_pos = np.arange(len(summary_df))
            width = 0.2

            ax1.bar(x_pos - width * 1.5, summary_df['common_species'], width, label='Common', color='blue')
            ax1.bar(x_pos - width * 0.5, summary_df['metaphlan_only'], width, label='MetaPhlAn Only', color='green')
            ax1.bar(x_pos + width * 0.5, summary_df['kraken2_only'], width, label='Kraken2 Only', color='orange')
            ax1.bar(x_pos + width * 1.5, summary_df['key_species'], width, label='Key Species', color='red')

            ax1.set_xlabel('Sample ID')
            ax1.set_ylabel('Number of Species')
            ax1.set_title('Species Detection by Tool')
            ax1.legend()
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(summary_df['sample_id'], rotation=45)

            # 2. 一致性率分布
            ax2 = axes[0, 1]
            ax2.hist(summary_df['agreement_rate'], bins=10, alpha=0.7, color='purple', edgecolor='black')
            ax2.set_xlabel('Agreement Rate (%)')
            ax2.set_ylabel('Number of Samples')
            ax2.set_title('Distribution of Agreement Rates')
            ax2.axvline(summary_df['agreement_rate'].mean(), color='red', linestyle='--',
                        label=f'Mean: {summary_df["agreement_rate"].mean():.1f}%')
            ax2.legend()

            # 3. 关键物种分布
            ax3 = axes[1, 0]
            ax3.hist(summary_df['key_species'], bins=10, alpha=0.7, color='crimson', edgecolor='black')
            ax3.set_xlabel('Key Species Count')
            ax3.set_ylabel('Number of Samples')
            ax3.set_title('Distribution of Key Species')
            ax3.axvline(summary_df['key_species'].mean(), color='red', linestyle='--',
                        label=f'Mean: {summary_df["key_species"].mean():.1f}')
            ax3.legend()

            # 4. 工具偏好散点图
            ax4 = axes[1, 1]
            ax4.scatter(summary_df['metaphlan_only'], summary_df['kraken2_only'], alpha=0.7)
            ax4.set_xlabel('MetaPhlAn Only Species')
            ax4.set_ylabel('Kraken2 Only Species')
            ax4.set_title('Tool Preference Across Samples')

            # 添加样本标签
            for i, row in summary_df.iterrows():
                ax4.annotate(row['sample_id'], (row['metaphlan_only'], row['kraken2_only']),
                             xytext=(5, 5), textcoords='offset points', fontsize=8)

            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'cross_sample_summary.png'),
                        dpi=300, bbox_inches='tight')
            plt.close()

            # 生成文本汇总报告
            report_file = os.path.join(self.output_dir, 'comprehensive_validation_report.txt')
            with open(report_file, 'w') as f:
                f.write("=== COMPREHENSIVE COMPLEMENTARY VALIDATION REPORT ===\n\n")

                f.write("OVERALL STATISTICS:\n")
                f.write(f"Total samples processed: {len(summary_df)}\n")
                f.write(f"Average species per sample: {summary_df['total_species'].mean():.1f}\n")
                f.write(f"Average agreement rate: {summary_df['agreement_rate'].mean():.1f}%\n")
                f.write(f"Average key species per sample: {summary_df['key_species'].mean():.1f}\n")
                f.write(f"Total MetaPhlAn-only species: {summary_df['metaphlan_only'].sum()}\n")
                f.write(f"Total Kraken2-only species: {summary_df['kraken2_only'].sum()}\n")
                f.write(f"Total common species: {summary_df['common_species'].sum()}\n")
                f.write(f"Total key species: {summary_df['key_species'].sum()}\n\n")

                f.write("SAMPLE-WISE SUMMARY:\n")
                f.write(
                    "Sample_ID\tTotal_Species\tCommon\tMetaPhlAn_Only\tKraken2_Only\tKey_Species\tAgreement_Rate(%)\n")
                for _, row in summary_df.iterrows():
                    f.write(f"{row['sample_id']}\t{row['total_species']}\t{row['common_species']}\t")
                    f.write(
                        f"{row['metaphlan_only']}\t{row['kraken2_only']}\t{row['key_species']}\t{row['agreement_rate']:.1f}\n")

            print("汇总报告和可视化已生成")
        except Exception as e:
            print(f"生成汇总报告失败: {e}")

    def process_all_samples(self):
        """处理所有样本"""
        # 获取样本列表
        metaphlan_samples = [d for d in os.listdir(self.metaphlan_dir)
                             if os.path.isdir(os.path.join(self.metaphlan_dir, d))]
        kraken2_samples = [d for d in os.listdir(self.kraken2_dir)
                           if os.path.isdir(os.path.join(self.kraken2_dir, d))]

        all_samples = sorted(list(set(metaphlan_samples + kraken2_samples)))
        print(f"找到 {len(all_samples)} 个样本进行处理")

        all_final_tables = []
        summary_stats = []

        for sample_id in all_samples:
            print(f"\n正在处理样本: {sample_id}")

            # 解析结果
            metaphlan_df = self.parse_metaphlan_results(sample_id)
            kraken2_df = self.parse_kraken2_results(sample_id)

            if metaphlan_df.empty and kraken2_df.empty:
                print(f"跳过样本 {sample_id}：两个工具都没有有效结果")
                continue

            print(f"MetaPhlAn检测到 {len(metaphlan_df)} 个物种")
            print(f"Kraken2检测到 {len(kraken2_df)} 个物种")

            # 互补验证
            final_table, validation_report = self.complementary_validation(
                metaphlan_df, kraken2_df, sample_id)

            if not final_table.empty:
                # 添加样本ID
                final_table['sample_id'] = sample_id
                all_final_tables.append(final_table)

                # 生成报告和可视化
                self.generate_sample_report(validation_report, sample_id)
                self.create_visualization(validation_report, sample_id)

                # 收集统计信息
                key_species_count = final_table['is_key_species'].sum()
                summary_stats.append({
                    'sample_id': sample_id,
                    'total_species': validation_report['total_species'],
                    'common_species': validation_report['common_species'],
                    'metaphlan_only': validation_report['metaphlan_only'],
                    'kraken2_only': validation_report['kraken2_only'],
                    'key_species': key_species_count,
                    'agreement_rate': validation_report['common_species'] / validation_report['total_species'] * 100 if
                    validation_report['total_species'] > 0 else 0
                })

                print(f"完成样本 {sample_id}: {validation_report['total_species']} 物种, {key_species_count} 关键物种")

        # 合并所有样本的最终结果
        if all_final_tables:
            self.final_species_table = pd.concat(all_final_tables, ignore_index=True)

            # 保存最终物种组成表
            final_output_file = os.path.join(self.output_dir, 'final_species_composition_table.csv')
            self.final_species_table.to_csv(final_output_file, index=False)

            # 生成汇总报告和输出MICOM表格
            self.generate_summary_report(summary_stats)
            self.export_micom_table(min_abundance=0.1, only_key=False)
            self.export_micom_table(min_abundance=0.1, only_key=True)
            self.plot_jaccard_heatmap()

            print(f"\n处理完成！最终结果保存在: {self.output_dir}")
            print(f"总共处理了 {len(summary_stats)} 个样本")
        else:
            print("没有找到任何有效数据")


# 使用示例
if __name__ == "__main__":
    # 设置路径 - 根据您的实际路径修改
    metaphlan_dir = "/mnt/zjwdata/2/raw/metaphlan_analysis_raw/per_sample_results/"
    kraken2_dir = "/mnt/zjwdata/2/raw/species_annotation_results_raw/"  # 修改为您的Kraken2输出目录
    output_dir = "/mnt/zjwdata/2/raw/complementary_validation_results/"

    # 创建验证器并运行
    validator = SpeciesComplementaryValidator(metaphlan_dir, kraken2_dir, output_dir)
    validator.process_all_samples()

    print("互补验证分析完成！")