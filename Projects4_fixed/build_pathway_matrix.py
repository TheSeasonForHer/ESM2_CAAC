#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
构建样本×通路丰度矩阵 (回补前后对比版) - 修复版 v4
修复:
  1. 索引隐藏字符问题
  2. 通路分类函数
  3. 新增碳水化合物代谢专项统计
  4. 新增通路激活分析
  5. 修复Pathway_Name提取（从Pathway_ID解析）
"""

import os
import re
import pandas as pd
import numpy as np
from collections import defaultdict
from datetime import datetime

# ==================== 配置 ====================

BASE_DIR = "/home/zjw/zjwdata/3_deep_learning/training_data/caac_backfill_results"
OUTPUT_DIR = "/home/zjw/zjwdata/3_deep_learning/training_data/caac_backfill_results/network_analysis"
PATHWAY_RESULTS_DIR = f"{BASE_DIR}/pathway_results"

SAMPLES = [
    'CK-7A', 'CK-7B', 'CK-7C', 'CK-90A', 'CK-90B', 'CK-90D',
    'FM-1', 'FM-2', 'FM-3',
    'M3-6023-7A', 'M3-6023-7B', 'M3-6023-7D',
    'M3-90-A', 'M3-90-B', 'M3-90C',
    'T-31-7A', 'T-31-7B', 'T-31-7C',
    'TR-31-90A', 'TR-31-90C', 'TR-31-90D'
]

os.makedirs(OUTPUT_DIR, exist_ok=True)


def infer_group(sample_name: str) -> str:
    if sample_name.startswith('CK'):
        return 'CK'
    elif sample_name.startswith('FM'):
        return 'FM'
    elif sample_name.startswith('M3'):
        return 'M3'
    elif sample_name.startswith('T-'):
        return 'T'
    elif sample_name.startswith('TR'):
        return 'TR'
    else:
        return 'Unknown'


def extract_pure_id(full_id: str) -> str:
    """从完整通路ID中提取纯ko编号"""
    full_id = str(full_id).strip()
    match = re.match(r'(ko\d{5})', full_id)
    if match:
        return match.group(1)
    if ' ' in full_id:
        return full_id.split(' ')[0]
    return full_id


def extract_name_from_id(full_id: str) -> str:
    """从完整通路ID字符串中提取通路名称"""
    full_id = str(full_id).strip()
    # 匹配: "koXXXXX Name [PATH:...]" 或 "koXXXXX Name [BR:...]"
    m = re.match(r'^\S+\s+(.+?)\s*(?:\[PATH:|\[BR:|$)', full_id)
    if m:
        return m.group(1).strip()
    # 备选：如果没有 [PATH: 或 [BR:，取第一个空格后的所有内容
    if ' ' in full_id:
        return full_id.split(' ', 1)[1].strip()
    return full_id


def clean_index(idx):
    """清理索引中的隐藏字符"""
    if isinstance(idx, pd.Index):
        return idx.astype(str).str.strip().str.replace('\r', '').str.replace('\n', ' ')
    return str(idx).strip().replace('\r', '').replace('\n', ' ')


def read_pathway_abundance(sample: str, mode: str = 'backfilled') -> pd.DataFrame:
    filepath = f"{PATHWAY_RESULTS_DIR}/{sample}/{mode}/{sample}.pathway_abundance.tsv"
    if not os.path.exists(filepath):
        print(f"警告: 文件不存在 {filepath}")
        return pd.DataFrame(columns=['Pathway_ID', 'Pathway_Name', 'Abundance', 'Gene_Count', 'Coverage'])
    df = pd.read_csv(filepath, sep='\t')
    # 修复：如果Pathway_Name为空，从Pathway_ID提取
    if 'Pathway_Name' not in df.columns or df['Pathway_Name'].isna().all():
        df['Pathway_Name'] = df['Pathway_ID'].apply(extract_name_from_id)
    # 标准化Pathway_ID
    df['Pathway_ID'] = df['Pathway_ID'].astype(str).apply(lambda x: extract_pure_id(clean_index(pd.Index([x]))[0]))
    return df


def read_pathway_coverage(sample: str, mode: str = 'backfilled') -> pd.DataFrame:
    filepath = f"{PATHWAY_RESULTS_DIR}/{sample}/{mode}/{sample}.pathway_coverage.tsv"
    if not os.path.exists(filepath):
        print(f"警告: 文件不存在 {filepath}")
        return pd.DataFrame(columns=['Pathway_ID', 'Pathway_Name', 'Total_KOs', 'Annotated_KOs', 'Coverage_Percentage'])
    df = pd.read_csv(filepath, sep='\t')
    # 修复：如果Pathway_Name为空，从Pathway_ID提取
    if 'Pathway_Name' not in df.columns or df['Pathway_Name'].isna().all() or (df['Pathway_Name'] == '').all():
        df['Pathway_Name'] = df['Pathway_ID'].apply(extract_name_from_id)
    # 标准化Pathway_ID
    df['Pathway_ID'] = df['Pathway_ID'].astype(str).apply(lambda x: extract_pure_id(clean_index(pd.Index([x]))[0]))
    return df


def build_abundance_matrix(mode: str = 'backfilled') -> pd.DataFrame:
    print(f"\n构建通路丰度矩阵 ({mode})...")
    all_series = []
    all_pathway_names = {}

    for sample in SAMPLES:
        df = read_pathway_abundance(sample, mode)
        if len(df) == 0:
            print(f"  {sample}: 无数据")
            continue

        series = df.set_index('Pathway_ID')['Abundance']
        series.name = sample
        all_series.append(series)

        for _, row in df.iterrows():
            all_pathway_names[row['Pathway_ID']] = row['Pathway_Name']

        print(f"  {sample}: {len(df)} 个通路")

    if not all_series:
        raise ValueError("无有效数据")

    matrix = pd.concat(all_series, axis=1)
    matrix = matrix.fillna(0).astype(float)

    pathway_name_series = pd.Series(all_pathway_names, name='Pathway_Name')
    matrix.insert(0, 'Pathway_Name', pathway_name_series)

    print(f"矩阵维度: {matrix.shape}")
    print(f"非零值比例: {(matrix.iloc[:, 1:] > 0).sum().sum() / (matrix.shape[0] * matrix.shape[1]) * 100:.1f}%")

    return matrix


def build_coverage_matrix(mode: str = 'backfilled') -> pd.DataFrame:
    print(f"\n构建通路覆盖度矩阵 ({mode})...")
    all_series = []
    all_pathway_names = {}

    for sample in SAMPLES:
        df = read_pathway_coverage(sample, mode)
        if len(df) == 0:
            print(f"  {sample}: 无数据")
            continue

        series = df.set_index('Pathway_ID')['Coverage_Percentage']
        series.name = sample
        all_series.append(series)

        for _, row in df.iterrows():
            all_pathway_names[row['Pathway_ID']] = row['Pathway_Name']

        print(f"  {sample}: {len(df)} 个通路")

    if not all_series:
        raise ValueError("无有效数据")

    matrix = pd.concat(all_series, axis=1)
    matrix = matrix.fillna(0).astype(float)

    pathway_name_series = pd.Series(all_pathway_names, name='Pathway_Name')
    matrix.insert(0, 'Pathway_Name', pathway_name_series)

    print(f"矩阵维度: {matrix.shape}")

    return matrix


def build_coverage_increase_matrix(cov_orig: pd.DataFrame, cov_back: pd.DataFrame) -> pd.DataFrame:
    print("\n构建覆盖度提升矩阵...")

    cov_orig_pure = cov_orig.copy()
    cov_orig_pure.index = cov_orig_pure.index.map(extract_pure_id)

    cov_back_pure = cov_back.copy()
    cov_back_pure.index = cov_back_pure.index.map(extract_pure_id)

    cov_orig_pure = cov_orig_pure[~cov_orig_pure.index.duplicated(keep='first')]
    cov_back_pure = cov_back_pure[~cov_back_pure.index.duplicated(keep='first')]

    common_pathways = cov_orig_pure.index.intersection(cov_back_pure.index)
    print(f"共同通路: {len(common_pathways)}")

    sample_cols = [c for c in cov_orig_pure.columns if c != 'Pathway_Name']
    orig_vals = cov_orig_pure.loc[common_pathways, sample_cols].astype(float)
    back_vals = cov_back_pure.loc[common_pathways, sample_cols].astype(float)

    increase = back_vals - orig_vals

    names = cov_back_pure.loc[
        common_pathways, 'Pathway_Name'] if 'Pathway_Name' in cov_back_pure.columns else pd.Series(common_pathways,
                                                                                                   index=common_pathways)
    increase.insert(0, 'Pathway_Name', names)

    n_improved = (increase.iloc[:, 1:] > 0).sum().sum()
    n_total = increase.shape[0] * increase.shape[1]
    n_pathways_improved = (increase.iloc[:, 1:] > 0).any(axis=1).sum()
    n_unchanged = (increase.iloc[:, 1:] == 0).all(axis=1).sum()

    print(f"提升的样本-通路对: {n_improved} / {n_total}")
    print(f"至少一个样本有提升的通路数: {n_pathways_improved}")
    print(f"完全无变化的通路数: {n_unchanged}")
    print(f"平均提升: {increase.iloc[:, 1:].mean().mean():.3f}%")
    print(f"最大提升: {increase.iloc[:, 1:].max().max():.2f}%")

    positive_vals = increase.iloc[:, 1:].values[increase.iloc[:, 1:].values > 0]
    if len(positive_vals) > 0:
        print(f"有提升的样本-通路对的平均提升: {positive_vals.mean():.3f}%")
        print(f"有提升的样本-通路对的中位数提升: {np.median(positive_vals):.3f}%")

    return increase


def save_sample_metadata():
    metadata = pd.DataFrame({
        'Sample': SAMPLES,
        'Group': [infer_group(s) for s in SAMPLES],
        'Subgroup': [s.split('-')[1] if '-' in s else 'Unknown' for s in SAMPLES]
    })

    group_descriptions = {
        'CK': 'Control (no additive)',
        'FM': 'Fermentation microorganism',
        'M3': 'M3 additive treatment',
        'T': 'T treatment',
        'TR': 'TR combined treatment'
    }
    metadata['Group_Description'] = metadata['Group'].map(group_descriptions)

    output_file = f"{OUTPUT_DIR}/sample_metadata.csv"
    metadata.to_csv(output_file, index=False)
    print(f"\n样本元数据已保存: {output_file}")
    print(metadata.to_string(index=False))

    return metadata


# ==================== 基于名称的通路分类 ====================

CARB_KEYWORDS = [
    'glycolysis', 'citrate cycle', 'pyruvate', 'propanoate', 'butanoate',
    'starch', 'sucrose', 'galactose', 'fructose', 'mannose', 'amino sugar',
    'nucleotide sugar', 'pentose', 'glucuronate', 'ascorbate', 'aldarate',
    'inositol', 'sorbitol', 'glyoxylate', 'dicarboxylate', 'carbon fixation',
    'photosynthesis', 'glycan', 'glycosaminoglycan', 'chondroitin', 'keratan',
    'lipopolysaccharide', 'peptidoglycan', 'glycosphingolipid',
    'glycosylphosphatidylinositol', 'glycosyltransferase', 'gluconeogenesis',
    'pentose phosphate', 'fructose and mannose', 'ascorbate and aldarate',
    'starch and sucrose', 'galactose metabolism', 'amino sugar and nucleotide sugar',
    'glycosyl', 'carbohydrate', 'cellulose', 'xylan', 'pectin', 'chitin',
    'lignin', 'hemicellulose', 'cellobiose', 'maltose', 'lactose', 'trehalose'
]

ENERGY_KEYWORDS = [
    'oxidative phosphorylation', 'photosynthesis', 'photophosphorylation',
    'methane metabolism', 'nitrogen metabolism', 'sulfur metabolism',
    'selenium metabolism'
]

AA_KEYWORDS = [
    'alanine', 'aspartate', 'glutamate', 'glycine', 'serine', 'threonine',
    'cysteine', 'methionine', 'valine', 'leucine', 'isoleucine', 'lysine',
    'arginine', 'proline', 'histidine', 'phenylalanine', 'tyrosine',
    'tryptophan', 'phenylpropanoid', 'flavonoid', 'alkaloid', 'terpenoid',
    'polyketide', 'biosynthesis of antibiotics', 'amino acid', 'urea cycle',
    'purine', 'pyrimidine', 'nucleotide'
]

LIPID_KEYWORDS = [
    'fatty acid', 'glycerolipid', 'glycerophospholipid', 'sphingolipid',
    'arachidonic acid', 'linoleic acid', 'alpha-linolenic acid', 'steroid',
    'bile acid', 'fat digestion', 'lipid', 'phospholipid', 'cholesterol'
]


def infer_category_by_name(pathway_name: str) -> str:
    """基于通路名称关键词推断类别"""
    if pd.isna(pathway_name) or str(pathway_name).strip() == '':
        return 'Other'

    name = str(pathway_name).lower()

    if any(kw in name for kw in CARB_KEYWORDS):
        return 'Carbohydrate Metabolism'
    elif any(kw in name for kw in ENERGY_KEYWORDS):
        return 'Energy Metabolism'
    elif any(kw in name for kw in AA_KEYWORDS):
        return 'Amino Acid Metabolism'
    elif any(kw in name for kw in LIPID_KEYWORDS):
        return 'Lipid Metabolism'
    elif 'overview' in name or 'global' in name or 'map' in name or 'general' in name:
        return 'Global and Overview Maps'
    elif any(x in name for x in ['transporter', 'secretion', 'chemotaxis',
                                 'flagellar', 'biofilm', 'quorum', 'virulence',
                                 'resistance', 'toxin', 'antimicrobial']):
        return 'Environmental Information Processing'
    elif any(x in name for x in ['ribosome', 'transcription', 'replication',
                                 'repair', 'splice', 'rna degrad', 'rna poly',
                                 'translation', 'folding', 'sorting', 'degrad']):
        return 'Genetic Information Processing'
    elif any(x in name for x in ['cell cycle', 'cell division', 'apoptosis',
                                 'necroptosis', 'autophagy', 'ferroptosis',
                                 'oocyte', 'meiosis', 'mitosis']):
        return 'Cellular Processes'
    else:
        return 'Other'


def save_pathway_info(cov_matrix: pd.DataFrame):
    """保存通路信息（修复版：正确提取名称）"""
    # 提取纯通路ID
    pathway_ids = cov_matrix.index.map(extract_pure_id)

    # 提取通路名称：优先使用Pathway_Name列，否则从索引提取
    if 'Pathway_Name' in cov_matrix.columns:
        # 检查Pathway_Name列是否有效
        test_name = cov_matrix['Pathway_Name'].iloc[0] if len(cov_matrix) > 0 else ''
        if pd.notna(test_name) and str(test_name).strip() != '' and str(test_name).strip() != 'nan':
            pathway_names = cov_matrix['Pathway_Name'].values
        else:
            # 从索引提取
            pathway_names = cov_matrix.index.astype(str).map(extract_name_from_id)
    else:
        pathway_names = cov_matrix.index.astype(str).map(extract_name_from_id)

    info = pd.DataFrame({
        'Pathway_ID': pathway_ids,
        'Pathway_Name': pathway_names
    })

    # 去重
    info = info.drop_duplicates(subset='Pathway_ID', keep='first')
    info.set_index('Pathway_ID', inplace=True)

    # 基于名称分类
    info['Category'] = info['Pathway_Name'].apply(infer_category_by_name)

    output_file = f"{OUTPUT_DIR}/pathway_info.csv"
    info.to_csv(output_file)
    print(f"\n通路信息已保存: {output_file}")

    cat_counts = info['Category'].value_counts()
    print("\n通路类别分布:")
    print(cat_counts.to_string())

    return info


# ==================== CAAC提升专项分析 ====================

def analyze_carbohydrate_enhancement(cov_increase: pd.DataFrame, pathway_info: pd.DataFrame):
    """分析CAAC对碳水化合物代谢通路的专项提升"""
    print("\n" + "=" * 60)
    print("CAAC碳水化合物代谢通路专项分析")
    print("=" * 60)

    carb_pathways = pathway_info[pathway_info['Category'] == 'Carbohydrate Metabolism'].index
    print(f"碳水化合物代谢通路总数: {len(carb_pathways)}")

    if len(carb_pathways) == 0:
        print("警告: 未识别到碳水化合物代谢通路")
        return None

    carb_increase = cov_increase[cov_increase.index.isin(carb_pathways)].copy()
    sample_cols = [c for c in carb_increase.columns if c != 'Pathway_Name']

    n_carb_improved = (carb_increase[sample_cols] > 0).any(axis=1).sum()
    n_carb_total = len(carb_pathways)

    print(f"有提升的碳水化合物通路数: {n_carb_improved} / {n_carb_total}")
    print(f"碳水化合物通路提升率: {n_carb_improved / n_carb_total * 100:.1f}%")

    positive_mask = carb_increase[sample_cols] > 0
    positive_vals = carb_increase[sample_cols].values[positive_mask.values]
    if len(positive_vals) > 0:
        print(f"碳水化合物通路平均提升(有提升的): {positive_vals.mean():.3f}%")
        print(f"碳水化合物通路最大提升: {carb_increase[sample_cols].max().max():.2f}%")

    max_per_pathway = carb_increase[sample_cols].max(axis=1).sort_values(ascending=False)
    top10 = max_per_pathway.head(10)

    print(f"\nTop 10 提升最大的碳水化合物代谢通路:")
    results = []
    for pid, val in top10.items():
        name = pathway_info.loc[pid, 'Pathway_Name'] if pid in pathway_info.index else 'Unknown'
        print(f"  {pid}: {name} (最大提升 +{val:.2f}%)")
        results.append({
            'Pathway_ID': pid,
            'Pathway_Name': name,
            'Max_Increase': val,
            'Category': 'Carbohydrate Metabolism'
        })

    df = pd.DataFrame(results)
    output = f"{OUTPUT_DIR}/carbohydrate_enhancement_top10.csv"
    df.to_csv(output, index=False)
    print(f"\n已保存: {output}")

    return carb_increase


def analyze_pathway_activation(cov_orig: pd.DataFrame, cov_back: pd.DataFrame, pathway_info: pd.DataFrame):
    """分析通路激活（Coverage从0到>0）"""
    print("\n" + "=" * 60)
    print("通路激活分析 (Coverage: 0 → >0)")
    print("=" * 60)

    cov_orig_pure = cov_orig.copy()
    cov_orig_pure.index = cov_orig_pure.index.map(extract_pure_id)
    cov_back_pure = cov_back.copy()
    cov_back_pure.index = cov_back_pure.index.map(extract_pure_id)

    sample_cols = [c for c in cov_orig_pure.columns if c != 'Pathway_Name']

    activated_records = []
    for col in sample_cols:
        orig_zero = cov_orig_pure[col] == 0
        back_nonzero = cov_back_pure[col] > 0
        newly_activated = cov_orig_pure.index[orig_zero & back_nonzero]

        for pid in newly_activated:
            name = ''
            if pid in pathway_info.index:
                name = pathway_info.loc[pid, 'Pathway_Name']
            elif 'Pathway_Name' in cov_back_pure.columns:
                name = cov_back_pure.loc[pid, 'Pathway_Name']

            cat = pathway_info.loc[pid, 'Category'] if pid in pathway_info.index else 'Other'

            activated_records.append({
                'Sample': col,
                'Pathway_ID': pid,
                'Pathway_Name': name,
                'Category': cat,
                'Coverage_After': cov_back_pure.loc[pid, col]
            })

    df = pd.DataFrame(activated_records)
    print(f"新激活的样本-通路对总数: {len(df)}")

    if len(df) > 0:
        cat_counts = df['Category'].value_counts()
        print(f"\n按类别分布:")
        print(cat_counts.to_string())

        output = f"{OUTPUT_DIR}/pathway_activation.csv"
        df.to_csv(output, index=False)
        print(f"\n已保存: {output}")

    return df


def analyze_coverage_distribution(cov_orig: pd.DataFrame, cov_back: pd.DataFrame):
    """分析覆盖度分布变化"""
    print("\n" + "=" * 60)
    print("覆盖度分布变化分析")
    print("=" * 60)

    sample_cols = [c for c in cov_orig.columns if c != 'Pathway_Name']

    orig_flat = cov_orig[sample_cols].values.flatten()
    back_flat = cov_back[sample_cols].values.flatten()

    print(f"回补前覆盖度分布:")
    print(f"  均值: {orig_flat.mean():.2f}%")
    print(f"  中位数: {np.median(orig_flat):.2f}%")
    print(f"  标准差: {orig_flat.std():.2f}%")
    print(f"  0覆盖比例: {(orig_flat == 0).sum() / len(orig_flat) * 100:.1f}%")

    print(f"\n回补后覆盖度分布:")
    print(f"  均值: {back_flat.mean():.2f}%")
    print(f"  中位数: {np.median(back_flat):.2f}%")
    print(f"  标准差: {back_flat.std():.2f}%")
    print(f"  0覆盖比例: {(back_flat == 0).sum() / len(back_flat) * 100:.1f}%")

    diff_flat = back_flat - orig_flat
    positive_diff = diff_flat[diff_flat > 0]
    print(f"\n提升值分布 (仅正值):")
    print(f"  数量: {len(positive_diff)} / {len(diff_flat)}")
    print(f"  均值: {positive_diff.mean():.3f}%")
    print(f"  中位数: {np.median(positive_diff):.3f}%")
    print(f"  最大值: {positive_diff.max():.2f}%")


def main():
    print("=" * 80)
    print("样本×通路丰度矩阵构建 (修复版 v4)")
    print(f"输出目录: {OUTPUT_DIR}")
    print("=" * 80)

    abund_orig = build_abundance_matrix('original')
    abund_back = build_abundance_matrix('backfilled')

    cov_orig = build_coverage_matrix('original')
    cov_back = build_coverage_matrix('backfilled')

    cov_increase = build_coverage_increase_matrix(cov_orig, cov_back)

    metadata = save_sample_metadata()
    pathway_info = save_pathway_info(cov_back)

    analyze_coverage_distribution(cov_orig, cov_back)
    carb_increase = analyze_carbohydrate_enhancement(cov_increase, pathway_info)
    activation_df = analyze_pathway_activation(cov_orig, cov_back, pathway_info)

    print("\n" + "=" * 60)
    print("保存矩阵文件...")
    print("=" * 60)

    files_to_save = {
        'pathway_abundance_matrix_original.csv': abund_orig,
        'pathway_abundance_matrix_backfilled.csv': abund_back,
        'pathway_coverage_matrix_original.csv': cov_orig,
        'pathway_coverage_matrix_backfilled.csv': cov_back,
        'pathway_coverage_increase_matrix.csv': cov_increase,
    }

    for filename, matrix in files_to_save.items():
        filepath = f"{OUTPUT_DIR}/{filename}"
        matrix.to_csv(filepath)
        print(f"  {filename}: {matrix.shape}")

    print("\n" + "=" * 60)
    print("生成统计摘要...")
    print("=" * 60)

    sample_cols = [c for c in cov_back.columns if c != 'Pathway_Name']

    summary = {
        'timestamp': datetime.now().isoformat(),
        'n_samples': len(SAMPLES),
        'n_pathways': len(cov_back),
        'abundance_matrix_shape': list(abund_back.shape),
        'coverage_matrix_shape': list(cov_back.shape),
        'avg_coverage_before': float(cov_orig[sample_cols].mean().mean()),
        'avg_coverage_after': float(cov_back[sample_cols].mean().mean()),
        'avg_coverage_increase': float(cov_increase[sample_cols].mean().mean()),
        'max_coverage_increase': float(cov_increase[sample_cols].max().max()),
        'n_pathways_with_increase': int((cov_increase[sample_cols] > 0).any(axis=1).sum()),
        'n_pathways_unchanged': int((cov_increase[sample_cols] == 0).all(axis=1).sum()),
        'group_distribution': metadata['Group'].value_counts().to_dict()
    }

    if carb_increase is not None:
        carb_cols = [c for c in carb_increase.columns if c != 'Pathway_Name']
        summary['carb_pathways_total'] = len(carb_increase)
        summary['carb_pathways_improved'] = int((carb_increase[carb_cols] > 0).any(axis=1).sum())

    if activation_df is not None and len(activation_df) > 0:
        summary['n_activated_pathways'] = len(activation_df)
        summary['n_unique_activated_pathways'] = activation_df['Pathway_ID'].nunique()

    import json
    summary_file = f"{OUTPUT_DIR}/matrix_build_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n摘要已保存: {summary_file}")
    print(f"\n关键统计:")
    print(f"  平均覆盖度 (回补前): {summary['avg_coverage_before']:.2f}%")
    print(f"  平均覆盖度 (回补后): {summary['avg_coverage_after']:.2f}%")
    print(f"  平均提升: {summary['avg_coverage_increase']:.3f}%")
    print(f"  最大提升: {summary['max_coverage_increase']:.2f}%")
    print(f"  有提升的通路数: {summary['n_pathways_with_increase']}")
    print(f"  完全无变化的通路数: {summary['n_pathways_unchanged']}")
    if 'carb_pathways_improved' in summary:
        print(f"  有提升的碳水化合物通路数: {summary['carb_pathways_improved']}")
    if 'n_activated_pathways' in summary:
        print(f"  新激活的样本-通路对: {summary['n_activated_pathways']}")

    print("\n" + "=" * 80)
    print("矩阵构建完成!")
    print(f"所有结果保存在: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()