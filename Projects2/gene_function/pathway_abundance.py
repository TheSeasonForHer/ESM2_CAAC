#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
代谢通路丰度量化脚本
基于基因注释和丰度数据计算代谢通路丰度
适配主控脚本调用方式
修复空数据导致的KeyError问题
更新：支持新版KEGG格式
更新：支持真实TPM值
更新：添加丰度列选择功能
修复：适配实际KEGG文件格式（6列）
简化版：使用简单KO映射
"""

import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
from collections import defaultdict

LOG = logging.getLogger(__name__)
__version__ = "1.5.1"
__author__ = ("Xingguo Zhang",)
__email__ = "invicoun@foxmail.com"


def setup_logging(log_file=None):
    """设置日志格式，支持输出到文件"""
    if log_file:
        # 创建日志目录
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        handlers = [
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    else:
        handlers = [logging.StreamHandler(sys.stdout)]

    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=handlers
    )

    if log_file:
        LOG.info(f"日志文件: {log_file}")


def load_gene_abundance(abundance_file, abundance_column='TPM'):
    """加载基因丰度数据，支持多种格式"""
    LOG.info(f"加载基因丰度数据: {abundance_file}")
    LOG.info(f"使用丰度列: {abundance_column}")

    try:
        # 支持多种格式：TSV, CSV
        if abundance_file.endswith('.tsv'):
            sep = '\t'
        else:
            sep = ','

        abundance_df = pd.read_csv(abundance_file, sep=sep)

        # 检查必要的列
        if 'GeneID' not in abundance_df.columns:
            # 尝试找到基因ID列
            gene_id_cols = [col for col in abundance_df.columns if 'gene' in col.lower() or 'id' in col.lower()]
            if gene_id_cols:
                abundance_df = abundance_df.rename(columns={gene_id_cols[0]: 'GeneID'})
                LOG.info(f"重命名列: {gene_id_cols[0]} -> GeneID")
            else:
                LOG.error("找不到GeneID列")
                return None

        # 设置索引
        abundance_df = abundance_df.set_index('GeneID')

        # 检查丰度列是否存在
        if abundance_column not in abundance_df.columns:
            # 尝试找到丰度列
            abundance_cols = [col for col in abundance_df.columns if
                              ('tpm' in col.lower() or 'abundance' in col.lower() or
                               'count' in col.lower() or 'reads' in col.lower())]
            if abundance_cols:
                abundance_column = abundance_cols[0]
                LOG.info(f"自动选择丰度列: {abundance_column}")
            else:
                LOG.error(f"找不到丰度列，可用列: {list(abundance_df.columns)}")
                return None

        # 提取丰度列作为DataFrame
        abundance_data = abundance_df[[abundance_column]].copy()
        abundance_data.columns = ['Abundance']

        LOG.info(f"基因丰度数据维度: {abundance_data.shape}")
        LOG.info(f"基因数量: {len(abundance_data)}")
        LOG.info(f"丰度统计: 最小值={abundance_data['Abundance'].min():.6f}, "
                 f"最大值={abundance_data['Abundance'].max():.2f}, "
                 f"平均值={abundance_data['Abundance'].mean():.4f}")

        return abundance_data
    except Exception as e:
        LOG.error(f"加载基因丰度数据失败: {e}")
        return None


def load_kegg_annotations(kegg_file):
    """加载KEGG注释数据"""
    LOG.info(f"加载KEGG注释: {kegg_file}")

    try:
        # 检查文件是否为空
        file_size = os.path.getsize(kegg_file)
        if file_size == 0:
            LOG.warning(f"KEGG注释文件为空: {kegg_file}")
            # 返回空的DataFrame
            return pd.DataFrame(columns=['GeneID', 'KO'])

        kegg_df = pd.read_csv(kegg_file, sep='\t')

        # 检查必要列
        required_cols = ['GeneID', 'KO']
        missing_cols = []
        for col in required_cols:
            if col not in kegg_df.columns:
                missing_cols.append(col)

        if missing_cols:
            # 尝试使用大小写不敏感的列名匹配
            actual_cols = list(kegg_df.columns)
            geneid_col = None
            ko_col = None

            for col in actual_cols:
                col_lower = col.lower()
                if 'gene' in col_lower and ('id' in col_lower or 'name' in col_lower):
                    geneid_col = col
                elif 'ko' in col_lower:
                    ko_col = col

            if geneid_col and ko_col:
                LOG.info(f"使用列映射: {geneid_col} -> GeneID, {ko_col} -> KO")
                kegg_df = kegg_df.rename(columns={geneid_col: 'GeneID', ko_col: 'KO'})
                LOG.info(f"KEGG注释基因数: {len(kegg_df)}")
                LOG.info(f"有KEGG注释的基因数: {kegg_df['KO'].notna().sum()}")
                return kegg_df
            else:
                LOG.error(f"KEGG注释文件缺少必要列: {missing_cols}")
                LOG.error(f"可用列: {list(kegg_df.columns)}")
                return None

        LOG.info(f"KEGG注释基因数: {len(kegg_df)}")
        LOG.info(f"有KEGG注释的基因数: {kegg_df['KO'].notna().sum()}")
        return kegg_df
    except Exception as e:
        LOG.error(f"加载KEGG注释失败: {e}")
        return None


def parse_simple_kegg_database(pathway_db_file):
    """简化版KEGG数据库解析，只提取KO信息"""
    LOG.info(f"简化版KEGG数据库解析: {pathway_db_file}")

    pathway_ko_mapping = defaultdict(list)
    pathway_info = {}

    try:
        with open(pathway_db_file, 'r') as f:
            # 跳过表头
            header_line = f.readline().strip()
            LOG.info(f"KEGG数据库表头: {header_line}")

            line_count = 0
            ko_count = 0
            pathway_ids = set()

            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split('\t')

                # 至少需要5列
                if len(parts) >= 5:
                    # 第3列是通路ID，第4列是通路描述，第5列是KO_id
                    pathway_id = parts[2]  # 通路ID
                    pathway_desc = parts[3]  # 通路描述
                    ko_id = parts[4]  # KO_id

                    # 清理通路ID：确保格式为 "ko00010"
                    if not pathway_id.startswith('ko'):
                        full_pathway_id = f"ko{pathway_id}"
                    else:
                        full_pathway_id = pathway_id

                    # 清理通路描述，移除[PATH:...]部分
                    if '[PATH:' in pathway_desc:
                        pathway_desc = pathway_desc.split('[PATH:')[0].strip()

                    # 存储通路信息
                    pathway_info[full_pathway_id] = pathway_desc

                    # 建立KO到通路的映射
                    if ko_id.startswith('K'):
                        pathway_ko_mapping[full_pathway_id].append(ko_id)
                        ko_count += 1
                        pathway_ids.add(full_pathway_id)

                    line_count += 1

                # 每读取10000行输出一次进度
                if line_count > 0 and line_count % 10000 == 0:
                    LOG.info(f"已处理 {line_count} 行，找到 {len(pathway_ids)} 个通路，{ko_count} 个KO映射")

        LOG.info(f"简化版KEGG数据库解析完成: {line_count} 行，{len(pathway_info)} 个通路，{ko_count} 个KO映射")
        return pathway_ko_mapping, pathway_info

    except Exception as e:
        LOG.error(f"简化版KEGG数据库解析失败: {e}")
        import traceback
        LOG.error(traceback.format_exc())
        return {}, {}


def parse_old_kegg_database(pathway_db_file):
    """解析旧版KEGG数据库格式"""
    LOG.info(f"解析旧版KEGG数据库格式: {pathway_db_file}")

    pathway_ko_mapping = defaultdict(list)
    pathway_info = {}

    try:
        with open(pathway_db_file, 'r') as f:
            current_pathway = ""
            line_count = 0
            ko_count = 0

            for line in f:
                line = line.strip()
                if not line:
                    continue

                line_count += 1

                if line.startswith('C') and '\t' in line:
                    # 通路行: C00010  Glycolysis / Gluconeogenesis
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        pathway_id = parts[0]
                        pathway_name = parts[1]
                        current_pathway = pathway_id
                        pathway_info[pathway_id] = pathway_name
                elif line.startswith('D') and '\t' in line and current_pathway:
                    # KO行: D      K00844  HK; hexokinase [EC:2.7.1.1]
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        ko_id = parts[1]
                        if ko_id.startswith('K'):
                            pathway_ko_mapping[current_pathway].append(ko_id)
                            ko_count += 1

        LOG.info(f"旧版KEGG数据库解析完成: {line_count} 行，{len(pathway_info)} 个通路，{ko_count} 个KO映射")
        return pathway_ko_mapping, pathway_info

    except Exception as e:
        LOG.error(f"解析旧版KEGG数据库失败: {e}")
        return {}, {}


def load_pathway_database(pathway_db_file):
    """加载通路数据库，支持新旧两种格式"""
    LOG.info(f"加载通路数据库: {pathway_db_file}")

    # 检查文件是否存在
    if not os.path.exists(pathway_db_file):
        LOG.error(f"通路数据库文件不存在: {pathway_db_file}")
        return None, None

    # 检查文件是否为空
    if os.path.getsize(pathway_db_file) == 0:
        LOG.warning(f"通路数据库文件为空: {pathway_db_file}")
        return {}, {}

    try:
        # 先读取第一行判断格式
        with open(pathway_db_file, 'r') as f:
            first_line = f.readline().strip()

        # 判断文件格式
        if first_line.startswith("Pathway_maps"):
            LOG.info("检测到新版KEGG格式（带表头）")
            pathway_ko_mapping, pathway_info = parse_simple_kegg_database(pathway_db_file)
        elif first_line.startswith("C") or first_line.startswith("D"):
            LOG.info("检测到旧版KEGG格式（无表头）")
            pathway_ko_mapping, pathway_info = parse_old_kegg_database(pathway_db_file)
        else:
            LOG.warning("无法识别KEGG格式，尝试简化版解析")
            pathway_ko_mapping, pathway_info = parse_simple_kegg_database(pathway_db_file)

            # 如果简化版解析失败，尝试旧版格式
            if not pathway_ko_mapping and not pathway_info:
                LOG.warning("简化版解析失败，尝试旧版格式")
                pathway_ko_mapping, pathway_info = parse_old_kegg_database(pathway_db_file)

        # 统计信息
        if pathway_ko_mapping:
            total_pathways = len(pathway_ko_mapping)
            total_kos = sum(len(kos) for kos in pathway_ko_mapping.values())
            LOG.info(f"加载通路数: {total_pathways}")
            LOG.info(f"加载通路信息数: {len(pathway_info)}")
            LOG.info(f"总KO映射数: {total_kos}")

            # 计算平均每个通路的KO数
            if total_pathways > 0:
                avg_ko = total_kos / total_pathways
                LOG.info(f"平均每个通路KO数: {avg_ko:.1f}")
        else:
            LOG.warning("通路数据库为空或解析失败")
            pathway_ko_mapping = defaultdict(list)
            pathway_info = {}

        return pathway_ko_mapping, pathway_info

    except Exception as e:
        LOG.error(f"加载通路数据库失败: {e}")
        import traceback
        LOG.error(traceback.format_exc())
        return None, None


def calculate_pathway_abundance(abundance_df, kegg_df, pathway_ko_mapping, pathway_info, method='sum',
                                abundance_column='Abundance'):
    """计算通路丰度，使用真实TPM值"""
    LOG.info(f"计算代谢通路丰度，方法: {method}")
    LOG.info(f"丰度列: {abundance_column}")

    # 检查输入数据
    if kegg_df.empty:
        LOG.warning("KEGG注释数据为空，无法计算通路丰度")
        # 返回空的DataFrame
        pathway_abundance = pd.DataFrame(columns=['Pathway_Name', 'Abundance', 'Gene_Count', 'Coverage'])
        return pathway_abundance

    if len(pathway_ko_mapping) == 0:
        LOG.warning("通路数据库为空，无法计算通路丰度")
        pathway_abundance = pd.DataFrame(columns=['Pathway_Name', 'Abundance', 'Gene_Count', 'Coverage'])
        return pathway_abundance

    # 创建基因-KO映射
    gene_ko_map = {}
    for _, row in kegg_df.iterrows():
        gene_id = row['GeneID']
        ko_id = row['KO']
        if pd.notna(ko_id) and gene_id in abundance_df.index:
            gene_ko_map[gene_id] = ko_id

    LOG.info(f"有KEGG注释且在丰度表中的基因数: {len(gene_ko_map)}")

    # 计算每个通路的丰度
    pathway_stats = []

    for pathway_id, ko_list in pathway_ko_mapping.items():
        pathway_genes = []
        pathway_abundance_values = []

        # 找到属于该通路的所有基因
        for gene_id, ko_id in gene_ko_map.items():
            if ko_id in ko_list:
                pathway_genes.append(gene_id)
                if gene_id in abundance_df.index:
                    pathway_abundance_values.append(abundance_df.loc[gene_id, 'Abundance'])

        if pathway_genes:
            # 计算通路丰度
            if method == 'sum':
                abundance = sum(pathway_abundance_values)
            elif method == 'mean':
                abundance = np.mean(pathway_abundance_values) if pathway_abundance_values else 0
            elif method == 'max':
                abundance = max(pathway_abundance_values) if pathway_abundance_values else 0
            else:
                abundance = sum(pathway_abundance_values)

            # 计算通路覆盖度（该通路中有注释的KO比例）
            total_kos = len(ko_list)
            annotated_kos = len(set(gene_ko_map.get(g, '') for g in pathway_genes) & set(ko_list))
            coverage = (annotated_kos / total_kos * 100) if total_kos > 0 else 0

            pathway_name = pathway_info.get(pathway_id, 'Unknown')

            pathway_stats.append({
                'Pathway_ID': pathway_id,
                'Pathway_Name': pathway_name,
                'Abundance': abundance,
                'Gene_Count': len(pathway_genes),
                'Coverage': coverage
            })

    # 转换为DataFrame
    if pathway_stats:
        pathway_abundance = pd.DataFrame(pathway_stats)
        pathway_abundance = pathway_abundance.sort_values('Abundance', ascending=False)
        LOG.info(f"计算完成: {len(pathway_stats)} 个通路有丰度数据")
    else:
        pathway_abundance = pd.DataFrame(columns=['Pathway_ID', 'Pathway_Name', 'Abundance', 'Gene_Count', 'Coverage'])
        LOG.warning("没有通路有丰度数据")

    return pathway_abundance


def calculate_pathway_coverage(kegg_df, pathway_ko_mapping, pathway_info):
    """计算通路覆盖度"""
    LOG.info("计算通路覆盖度")

    # 检查输入数据
    if kegg_df.empty:
        LOG.warning("KEGG注释数据为空，无法计算通路覆盖度")
        # 返回空的DataFrame
        coverage_df = pd.DataFrame(columns=['Pathway_ID', 'Pathway_Name', 'Total_KOs',
                                            'Annotated_KOs', 'Coverage_Percentage'])
        return coverage_df

    if len(pathway_ko_mapping) == 0:
        LOG.warning("通路数据库为空，无法计算通路覆盖度")
        coverage_df = pd.DataFrame(columns=['Pathway_ID', 'Pathway_Name', 'Total_KOs',
                                            'Annotated_KOs', 'Coverage_Percentage'])
        return coverage_df

    # 获取所有KO列表
    all_kos_in_data = set()
    for _, row in kegg_df.iterrows():
        ko_id = row['KO']
        if pd.notna(ko_id):
            all_kos_in_data.add(ko_id)

    LOG.info(f"数据中的唯一KO数量: {len(all_kos_in_data)}")

    coverage_stats = []
    for pathway_id, ko_list in pathway_ko_mapping.items():
        # 统计该通路中有注释的KO数量
        annotated_kos = set(ko_list) & all_kos_in_data
        total_kos = len(ko_list)
        coverage = len(annotated_kos) / total_kos if total_kos > 0 else 0

        coverage_stats.append({
            'Pathway_ID': pathway_id,
            'Pathway_Name': pathway_info.get(pathway_id, 'Unknown'),
            'Total_KOs': total_kos,
            'Annotated_KOs': len(annotated_kos),
            'Coverage_Percentage': coverage * 100
        })

    coverage_df = pd.DataFrame(coverage_stats)

    if not coverage_df.empty and 'Coverage_Percentage' in coverage_df.columns:
        coverage_df = coverage_df.sort_values('Coverage_Percentage', ascending=False)
        LOG.info(f"通路覆盖度统计完成: {len(coverage_df)} 个通路")

        # 统计覆盖度分布
        if len(coverage_df) > 0:
            coverage_ranges = [(0, 10), (10, 30), (30, 50), (50, 70), (70, 90), (90, 100)]
            for low, high in coverage_ranges:
                count = len(coverage_df[(coverage_df['Coverage_Percentage'] >= low) &
                                        (coverage_df['Coverage_Percentage'] < high)])
                LOG.info(f"  覆盖度 {low}-{high}%: {count} 个通路")
    else:
        LOG.warning("通路覆盖度数据为空或缺少'Coverage_Percentage'列")

    return coverage_df


def normalize_pathway_abundance(pathway_abundance, method='tss'):
    """标准化通路丰度"""
    LOG.info(f"标准化通路丰度，方法: {method}")

    # 检查数据是否为空
    if pathway_abundance.empty or 'Abundance' not in pathway_abundance.columns:
        LOG.warning("通路丰度数据为空或缺少'Abundance'列，跳过标准化")
        return pathway_abundance

    # 复制数据避免修改原数据
    normalized_df = pathway_abundance.copy()

    if method == 'tss':  # Total Sum Scaling
        total_abundance = normalized_df['Abundance'].sum()
        if total_abundance > 0:
            normalized_df['Abundance'] = normalized_df['Abundance'] / total_abundance * 1000000  # 转换为CPM
            LOG.info(f"TSS标准化完成: 总丰度={total_abundance:.2f}")
        else:
            LOG.warning("总丰度为0，无法进行TSS标准化")
    elif method == 'log':  # Log transformation
        normalized_df['Abundance'] = np.log1p(normalized_df['Abundance'])
        LOG.info("Log转换完成")
    elif method == 'clr':  # Centered Log Ratio
        # 避免零值问题
        abundance_positive = normalized_df['Abundance'] + 1e-10
        geometric_mean = np.exp(np.mean(np.log(abundance_positive)))
        normalized_df['Abundance'] = np.log(abundance_positive / geometric_mean)
        LOG.info("CLR标准化完成")
    elif method == 'zscore':  # Z-score标准化
        mean_val = normalized_df['Abundance'].mean()
        std_val = normalized_df['Abundance'].std()
        if std_val > 0:
            normalized_df['Abundance'] = (normalized_df['Abundance'] - mean_val) / std_val
            LOG.info(f"Z-score标准化完成: 均值={mean_val:.2f}, 标准差={std_val:.2f}")
        else:
            LOG.warning("标准差为0，无法进行Z-score标准化")
    else:  # 无标准化
        LOG.info("跳过标准化步骤")

    return normalized_df


def generate_abundance_report(pathway_abundance, output_prefix):
    """生成丰度报告"""
    try:
        if pathway_abundance.empty:
            LOG.warning("通路丰度数据为空，跳过报告生成")
            return None

        report_file = f"{output_prefix}.abundance_report.txt"

        with open(report_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("代谢通路丰度分析报告\n")
            f.write("=" * 80 + "\n\n")

            f.write("1. 总体统计:\n")
            f.write(f"   总通路数: {len(pathway_abundance)}\n")

            if 'Abundance' in pathway_abundance.columns:
                active_pathways = sum(pathway_abundance['Abundance'] > 0)
                f.write(f"   有丰度的通路数: {active_pathways}\n")
                f.write(f"   总丰度值: {pathway_abundance['Abundance'].sum():.2f}\n")
                f.write(f"   平均通路丰度: {pathway_abundance['Abundance'].mean():.4f}\n")
                f.write(f"   中位数通路丰度: {pathway_abundance['Abundance'].median():.4f}\n")
                f.write(f"   最大通路丰度: {pathway_abundance['Abundance'].max():.2f}\n")
                f.write(f"   最小通路丰度: {pathway_abundance['Abundance'].min():.6f}\n\n")

            if 'Gene_Count' in pathway_abundance.columns:
                total_genes = pathway_abundance['Gene_Count'].sum()
                f.write(f"   总基因数: {total_genes}\n")
                f.write(f"   平均每个通路基因数: {pathway_abundance['Gene_Count'].mean():.1f}\n\n")

            f.write("2. Top 20高丰度通路:\n")
            if 'Pathway_Name' in pathway_abundance.columns and 'Abundance' in pathway_abundance.columns:
                top_pathways = pathway_abundance.nlargest(20, 'Abundance')
                for idx, row in top_pathways.iterrows():
                    pathway_name = row['Pathway_Name'][:60] + "..." if len(row['Pathway_Name']) > 60 else row[
                        'Pathway_Name']
                    gene_count = row.get('Gene_Count', 'N/A')
                    coverage = row.get('Coverage', 'N/A')
                    coverage_str = f"{coverage:.1f}%" if isinstance(coverage, (int, float)) else str(coverage)
                    f.write(
                        f"   {pathway_name:<65} {row['Abundance']:>10.2f} (基因数: {gene_count}, 覆盖度: {coverage_str})\n")

            f.write("\n3. 丰度分布:\n")
            if 'Abundance' in pathway_abundance.columns:
                abundance_ranges = [(0, 0.1), (0.1, 1), (1, 10), (10, 100), (100, 1000), (1000, float('inf'))]
                range_labels = ["0-0.1", "0.1-1", "1-10", "10-100", "100-1000", ">1000"]

                for (low, high), label in zip(abundance_ranges, range_labels):
                    count = sum(1 for ab in pathway_abundance['Abundance'] if low <= ab < high)
                    percentage = (count / len(pathway_abundance)) * 100 if len(pathway_abundance) > 0 else 0
                    f.write(f"   丰度 {label}: {count:>3} 个通路 ({percentage:>5.1f}%)\n")

        LOG.info(f"丰度报告生成完成: {report_file}")
        return report_file

    except Exception as e:
        LOG.error(f"生成丰度报告失败: {e}")
        return None


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="代谢通路丰度量化脚本 - 适配主控脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("--gene_abundance", required=True,
                        help="基因丰度文件 (TSV/CSV格式)")
    parser.add_argument("--kegg_annotations", required=True,
                        help="KEGG注释文件")
    parser.add_argument("--pathway_db", default="/mnt/databases/kegg/2025/ko00001.tsv",
                        help="KEGG通路数据库文件 (默认: /mnt/databases/kegg/2025/ko00001.tsv)")
    parser.add_argument("-o", "--output", required=True,
                        help="输出目录")
    parser.add_argument("-p", "--prefix", required=True,
                        help="输出文件前缀")
    parser.add_argument("--method", choices=['sum', 'mean', 'max'], default='sum',
                        help="通路丰度计算方法 (默认: sum)")
    parser.add_argument("--normalization", choices=['tss', 'log', 'clr', 'zscore', 'none'], default='none',
                        help="标准化方法 (默认: none)")
    parser.add_argument("--abundance_column", default='TPM',
                        help="丰度列名 (默认: TPM)")
    parser.add_argument("--log", help="日志文件路径")

    args = parser.parse_args()

    # 设置日志
    setup_logging(args.log)

    LOG.info("开始代谢通路丰度量化流程")
    LOG.info(f"基因丰度文件: {args.gene_abundance}")
    LOG.info(f"KEGG注释文件: {args.kegg_annotations}")
    LOG.info(f"通路数据库: {args.pathway_db}")
    LOG.info(f"计算方法: {args.method}")
    LOG.info(f"标准化方法: {args.normalization}")
    LOG.info(f"丰度列名: {args.abundance_column}")

    # 检查输入文件
    for file_path in [args.gene_abundance, args.kegg_annotations, args.pathway_db]:
        if not os.path.exists(file_path):
            LOG.error(f"输入文件不存在: {file_path}")
            sys.exit(1)

    # 检查文件是否为空
    for file_path, file_name in [(args.gene_abundance, "基因丰度文件"),
                                 (args.kegg_annotations, "KEGG注释文件"),
                                 (args.pathway_db, "通路数据库")]:
        if os.path.getsize(file_path) == 0:
            LOG.warning(f"{file_name}为空: {file_path}")

    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)

    # 步骤1: 加载数据
    abundance_df = load_gene_abundance(args.gene_abundance, args.abundance_column)
    kegg_df = load_kegg_annotations(args.kegg_annotations)
    pathway_ko_mapping, pathway_info = load_pathway_database(args.pathway_db)

    if abundance_df is None:
        LOG.error("基因丰度数据加载失败")
        sys.exit(1)

    if kegg_df is None:
        LOG.error("KEGG注释数据加载失败")
        sys.exit(1)

    if pathway_ko_mapping is None or pathway_info is None:
        LOG.error("通路数据库加载失败")
        sys.exit(1)

    # 步骤2: 计算通路丰度
    pathway_abundance = calculate_pathway_abundance(
        abundance_df, kegg_df, pathway_ko_mapping, pathway_info, args.method, args.abundance_column
    )

    # 步骤3: 计算通路覆盖度
    pathway_coverage = calculate_pathway_coverage(kegg_df, pathway_ko_mapping, pathway_info)

    # 步骤4: 标准化
    if args.normalization != 'none' and not pathway_abundance.empty:
        pathway_abundance = normalize_pathway_abundance(
            pathway_abundance, args.normalization
        )
    else:
        LOG.info("跳过标准化步骤")

    # 步骤5: 保存结果
    output_prefix = os.path.join(args.output, args.prefix)

    # 保存通路丰度
    if not pathway_abundance.empty:
        pathway_abundance_file = f"{output_prefix}.pathway_abundance.tsv"
        pathway_abundance.to_csv(pathway_abundance_file, sep='\t', index=False)
        LOG.info(f"通路丰度保存到: {pathway_abundance_file}")

        # 生成丰度报告
        generate_abundance_report(pathway_abundance, output_prefix)
    else:
        LOG.warning("通路丰度数据为空，不保存文件")
        # 创建空文件
        pathway_abundance_file = f"{output_prefix}.pathway_abundance.tsv"
        with open(pathway_abundance_file, 'w') as f:
            f.write("Pathway_ID\tPathway_Name\tAbundance\tGene_Count\tCoverage\n")
        LOG.info(f"创建空的通路丰度文件: {pathway_abundance_file}")

    # 保存通路覆盖度
    if not pathway_coverage.empty:
        pathway_coverage_file = f"{output_prefix}.pathway_coverage.tsv"
        pathway_coverage.to_csv(pathway_coverage_file, sep='\t', index=False)
        LOG.info(f"通路覆盖度保存到: {pathway_coverage_file}")
    else:
        LOG.warning("通路覆盖度数据为空，不保存文件")
        # 创建空文件
        pathway_coverage_file = f"{output_prefix}.pathway_coverage.tsv"
        with open(pathway_coverage_file, 'w') as f:
            f.write("Pathway_ID\tPathway_Name\tTotal_KOs\tAnnotated_KOs\tCoverage_Percentage\n")
        LOG.info(f"创建空的通路覆盖度文件: {pathway_coverage_file}")

    # 生成摘要报告
    summary_file = f"{output_prefix}.pathway_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("代谢通路分析摘要报告\n")
        f.write("=" * 60 + "\n")
        f.write(f"分析时间: {pd.Timestamp.now()}\n")
        f.write(f"基因丰度文件: {args.gene_abundance}\n")
        f.write(f"KEGG注释文件: {args.kegg_annotations}\n")
        f.write(f"通路数据库: {args.pathway_db}\n")
        f.write(f"计算方法: {args.method}\n")
        f.write(f"标准化方法: {args.normalization}\n")
        f.write(f"丰度列名: {args.abundance_column}\n")
        f.write("\n")
        f.write(f"总基因数: {len(abundance_df) if abundance_df is not None else 0}\n")
        f.write(f"有KEGG注释的基因数: {len(kegg_df) if not kegg_df.empty else 0}\n")
        f.write(f"通路数量: {len(pathway_ko_mapping)}\n")
        f.write(f"计算出的通路丰度数量: {len(pathway_abundance)}\n")
        f.write(f"有覆盖度数据的通路数量: {len(pathway_coverage)}\n")

    LOG.info(f"摘要报告保存到: {summary_file}")
    LOG.info("代谢通路丰度量化完成!")


if __name__ == "__main__":
    main()