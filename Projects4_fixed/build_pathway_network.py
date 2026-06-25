#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通路级代谢网络构建与分析 (修复版 v2)
修改:
  1. 索引标准化，避免隐藏字符问题
  2. 相关性阈值0.8，FDR 0.005→
  3. 新增回补特异性边的通路类别分析
"""

import os
import json
import warnings
import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime
import re

try:
    import networkx as nx
except ImportError:
    raise ImportError("请安装 networkx: pip install networkx")

# ==================== 配置 ====================

BASE_DIR = "/home/zjw/zjwdata/3_deep_learning/training_data/caac_backfill_results/network_analysis"
OUTPUT_DIR = "/home/zjw/zjwdata/3_deep_learning/training_data/caac_backfill_results/network_analysis/network_results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 网络构建参数 - 调整后的更合理阈值
CORR_METHOD = 'spearman'
CORR_THRESHOLD = 0.70  # 从0.8降到0.7
PVALUE_THRESHOLD = 0.05
USE_FDR = True
FDR_ALPHA = 0.05  # 从0.005放宽到0.05
MIN_SAMPLES = 5


def extract_pure_id(full_id):
    """提取纯ko编号"""
    full_id = str(full_id).strip()
    match = re.match(r'(ko\d{5})', full_id)
    if match:
        return match.group(1)
    if ' ' in full_id:
        return full_id.split(' ')[0]
    return full_id


def benjamini_hochberg(pvalues, alpha=0.05):
    pvalues = np.asarray(pvalues)
    n = len(pvalues)
    if n == 0:
        return np.array([]), np.array([])
    sorted_indices = np.argsort(pvalues)
    sorted_p = pvalues[sorted_indices]
    bh_threshold = [alpha * (i + 1) / n for i in range(n)]
    valid = sorted_p <= bh_threshold
    if np.any(valid):
        last_valid = np.max(np.where(valid)[0])
    else:
        last_valid = -1
    rejected = np.zeros(n, dtype=bool)
    rejected[sorted_indices[:last_valid + 1]] = True
    adj_p = np.zeros(n)
    for i, idx in enumerate(sorted_indices):
        adj_p[idx] = sorted_p[i] * n / (i + 1)
    adj_p = np.minimum(adj_p, 1.0)
    for i in range(n - 2, -1, -1):
        adj_p[sorted_indices[i]] = min(adj_p[sorted_indices[i]], adj_p[sorted_indices[i + 1]])
    return rejected, adj_p


def load_matrices():
    print("加载矩阵数据...")

    files = {
        'abund_orig': 'pathway_abundance_matrix_original.csv',
        'abund_back': 'pathway_abundance_matrix_backfilled.csv',
        'cov_orig': 'pathway_coverage_matrix_original.csv',
        'cov_back': 'pathway_coverage_matrix_backfilled.csv',
        'cov_increase': 'pathway_coverage_increase_matrix.csv',
    }

    data = {}
    for key, filename in files.items():
        filepath = f"{BASE_DIR}/{filename}"
        df = pd.read_csv(filepath, index_col=0)
        # 标准化索引
        df.index = df.index.map(extract_pure_id)
        # 去重
        df = df[~df.index.duplicated(keep='first')]
        data[key] = df
        print(f"  {key}: {df.shape}")

    metadata = pd.read_csv(f"{BASE_DIR}/sample_metadata.csv")

    pathway_info = pd.read_csv(f"{BASE_DIR}/pathway_info.csv")
    if 'Pathway_ID' not in pathway_info.columns:
        pathway_info.columns = ['Pathway_ID', 'Pathway_Name', 'Category']

    # ===== 修复：先转换为字符串再使用.str accessor =====
    pathway_info['Pathway_Name'] = pathway_info['Pathway_Name'].astype(str).str.strip()
    pathway_info['Pathway_Name'] = pathway_info['Pathway_Name'].replace('nan', 'Unknown').replace('None',
                                                                                                  'Unknown').fillna(
        'Unknown')
    # ====================================================

    pathway_info.set_index('Pathway_ID', inplace=True)
    pathway_info.index = pathway_info.index.map(extract_pure_id)
    pathway_info = pathway_info[~pathway_info.index.duplicated(keep='first')]

    print(f"  metadata: {metadata.shape}")
    print(f"  pathway_info: {pathway_info.shape}")

    for key in ['abund_orig', 'abund_back']:
        df = data[key]
        if df.columns[0] == 'Pathway_Name':
            df = df.drop(columns=['Pathway_Name'])
            data[key] = df
        n_samples = df.shape[1]
        if n_samples != len(metadata):
            warnings.warn(f"{key} 样本数 {n_samples} 与元数据样本数 {len(metadata)} 不一致")

    return data, metadata, pathway_info


def extract_numeric_matrix(df: pd.DataFrame) -> pd.DataFrame:
    if 'Pathway_Name' in df.columns:
        df = df.drop(columns=['Pathway_Name'])
    for col in df.columns:
        if not np.issubdtype(df[col].dtype, np.number):
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def compute_correlation_network(abund_matrix: pd.DataFrame,
                                method: str = 'spearman',
                                corr_thresh: float = 0.70,
                                pval_thresh: float = 0.05,
                                use_fdr: bool = True,
                                fdr_alpha: float = 0.05) -> pd.DataFrame:
    print(f"\n计算相关性网络 (method={method}, threshold={corr_thresh}, FDR={use_fdr}, alpha={fdr_alpha})...")

    mat = abund_matrix.T
    valid_cols = (mat > 0).sum() >= MIN_SAMPLES
    mat_filtered = mat.loc[:, valid_cols]

    print(f"  有效通路: {mat_filtered.shape[1]} / {mat.shape[1]}")

    n_pathways = mat_filtered.shape[1]
    pathway_names = mat_filtered.columns.tolist()

    all_tests = []

    for i in range(n_pathways):
        for j in range(i + 1, n_pathways):
            p1 = pathway_names[i]
            p2 = pathway_names[j]

            x = mat_filtered.iloc[:, i].values
            y = mat_filtered.iloc[:, j].values

            mask = (x > 0) & (y > 0)
            if mask.sum() < MIN_SAMPLES:
                continue

            x_valid = x[mask]
            y_valid = y[mask]

            if method == 'spearman':
                corr, pval = stats.spearmanr(x_valid, y_valid)
            else:
                corr, pval = stats.pearsonr(x_valid, y_valid)

            all_tests.append((i, j, p1, p2, corr, pval, mask.sum()))

    if not all_tests:
        print("  警告: 无有效通路对")
        return pd.DataFrame()

    pvals = [t[5] for t in all_tests]
    if use_fdr:
        rejected, adj_pvals = benjamini_hochberg(pvals, alpha=fdr_alpha)
    else:
        rejected = [p <= pval_thresh for p in pvals]
        adj_pvals = pvals

    edges = []
    for (i, j, p1, p2, corr, pval, n_samp), rej, adj_p in zip(all_tests, rejected, adj_pvals):
        if rej and abs(corr) >= corr_thresh:
            edges.append({
                'source': p1,
                'target': p2,
                'correlation': corr,
                'pvalue': pval,
                'adj_pvalue': adj_p,
                'weight': abs(corr),
                'n_samples': n_samp
            })

    edges_df = pd.DataFrame(edges)

    if len(edges_df) > 0:
        edges_df = edges_df.sort_values('weight', ascending=False)
        print(f"  总测试对数: {len(all_tests)}")
        print(f"  显著边数: {len(edges_df)}")
        n_nodes = len(set(edges_df['source']) | set(edges_df['target']))
        density = 2 * len(edges_df) / (n_nodes * (n_nodes - 1)) if n_nodes > 1 else 0
        print(f"  节点数: {n_nodes}")
        print(f"  网络密度: {density:.4f}")
        print(f"  正相关边: {(edges_df['correlation'] > 0).sum()}")
        print(f"  负相关边: {(edges_df['correlation'] < 0).sum()}")
        if use_fdr:
            print(f"  FDR拒绝数: {sum(rejected)}")
    else:
        print("  警告: 无显著相关性边")

    return edges_df


def identify_backfill_specific_edges(edges_orig: pd.DataFrame,
                                     edges_back: pd.DataFrame,
                                     pathway_info: pd.DataFrame) -> pd.DataFrame:
    print("\n识别回补特异性边...")

    def edge_key(row):
        return tuple(sorted([row['source'], row['target']]))

    orig_edges = set(edges_orig.apply(edge_key, axis=1))
    back_edges = set(edges_back.apply(edge_key, axis=1))

    new_edges = back_edges - orig_edges

    new_edge_records = []
    for _, row in edges_back.iterrows():
        key = edge_key(row)
        if key in new_edges:
            record = row.to_dict()
            record['edge_type'] = 'new'
            s_cat = pathway_info.loc[row['source'], 'Category'] if row['source'] in pathway_info.index else 'Other'
            t_cat = pathway_info.loc[row['target'], 'Category'] if row['target'] in pathway_info.index else 'Other'
            record['source_category'] = s_cat
            record['target_category'] = t_cat
            record['cross_category'] = s_cat != t_cat
            new_edge_records.append(record)

    orig_corr = {(edge_key(row), row['correlation']) for _, row in edges_orig.iterrows()}
    orig_corr_dict = {k: v for k, v in orig_corr}

    enhanced_records = []
    for _, row in edges_back.iterrows():
        key = edge_key(row)
        if key in orig_edges and key not in new_edges:
            orig_corr_val = orig_corr_dict.get(key, 0)
            back_corr_val = row['correlation']
            if abs(back_corr_val) - abs(orig_corr_val) > 0.1:
                record = row.to_dict()
                record['edge_type'] = 'enhanced'
                record['correlation_increase'] = back_corr_val - orig_corr_val
                s_cat = pathway_info.loc[row['source'], 'Category'] if row['source'] in pathway_info.index else 'Other'
                t_cat = pathway_info.loc[row['target'], 'Category'] if row['target'] in pathway_info.index else 'Other'
                record['source_category'] = s_cat
                record['target_category'] = t_cat
                record['cross_category'] = s_cat != t_cat
                enhanced_records.append(record)

    all_specific = new_edge_records + enhanced_records

    if all_specific:
        df = pd.DataFrame(all_specific)
        print(f"  新出现边: {len(new_edge_records)}")
        print(f"  显著增强边: {len(enhanced_records)}")
        print(f"  回补特异性边总数: {len(df)}")

        cross_cat = df[df['cross_category'] == True]
        print(f"  跨类别边: {len(cross_cat)} ({len(cross_cat) / len(df) * 100:.1f}%)")

        if 'source_category' in df.columns:
            print(f"\n  回补特异性边的类别分布:")
            cat_pairs = df.apply(lambda r: tuple(sorted([r['source_category'], r['target_category']])), axis=1)
            print(cat_pairs.value_counts().head(10).to_string())
    else:
        print("  无回补特异性边")
        df = pd.DataFrame()

    return df


def map_pathway_info(pid, pathway_info):
    pid_str = str(pid)
    if pid_str in pathway_info.index:
        name = pathway_info.loc[pid_str, 'Pathway_Name']
        cat = pathway_info.loc[pid_str, 'Category']
    else:
        name, cat = 'Unknown', 'Other'
    if pd.isna(name) or name == '' or name == 'Unknown':
        name = pid_str
    if pd.isna(cat) or cat == '':
        cat = 'Other'
    return name, cat


def compute_node_topology(edges_df: pd.DataFrame, pathway_info: pd.DataFrame) -> tuple:
    print("\n计算节点拓扑属性...")

    if len(edges_df) == 0:
        print("  警告: 无边，跳过拓扑分析")
        return pd.DataFrame(), nx.Graph()

    G = nx.Graph()
    for _, row in edges_df.iterrows():
        G.add_edge(row['source'], row['target'], weight=row['weight'])

    nodes = list(G.nodes())
    print(f"  节点数: {len(nodes)}")

    degree = dict(G.degree())
    strength = {n: sum(d['weight'] for _, _, d in G.edges(n, data=True)) for n in nodes}
    betweenness = nx.betweenness_centrality(G, weight='weight', normalized=True)
    clustering = nx.clustering(G, weight='weight')

    records = []
    matched_count = 0
    for node in sorted(nodes):
        pname, cat = map_pathway_info(node, pathway_info)
        if pname != node and pname != 'Unknown':
            matched_count += 1
        records.append({
            'Pathway_ID': node,
            'Pathway_Name': pname,
            'Category': cat,
            'Degree': degree.get(node, 0),
            'Strength': round(strength.get(node, 0), 4),
            'Betweenness': round(betweenness.get(node, 0), 4),
            'Clustering_Coefficient': round(clustering.get(node, 0), 4)
        })
    print(f"  通路信息匹配成功: {matched_count}/{len(nodes)}")

    nodes_df = pd.DataFrame(records)
    nodes_df = nodes_df.sort_values('Degree', ascending=False)

    print(f"  Hub通路 (度≥5): {(nodes_df['Degree'] >= 5).sum()}")
    print(f"  平均度: {nodes_df['Degree'].mean():.2f}")

    return nodes_df, G


def detect_modules(G: nx.Graph, nodes_df: pd.DataFrame) -> pd.DataFrame:
    print("\n模块检测 (贪婪模块度算法)...")

    if G.number_of_edges() == 0:
        print("  跳过")
        nodes_df['Module'] = -1
        return nodes_df

    try:
        from networkx.algorithms.community import greedy_modularity_communities
        communities = list(greedy_modularity_communities(G, weight='weight'))
        module_map = {}
        for mod_id, comm in enumerate(communities):
            for node in comm:
                module_map[node] = mod_id
        nodes_df['Module'] = nodes_df['Pathway_ID'].map(module_map)
        nodes_df['Module'] = nodes_df['Module'].fillna(-1).astype(int)
        print(f"  检测到 {len(communities)} 个模块")
        module_sizes = nodes_df['Module'].value_counts()
        print(f"  模块大小分布 (前10):\n{module_sizes.head(10).to_string()}")
    except Exception as e:
        print(f"  模块检测失败: {e}")
        nodes_df['Module'] = -1

    return nodes_df


def generate_cytoscape_format(edges_df: pd.DataFrame, nodes_df: pd.DataFrame,
                              suffix: str):
    edge_file = f"{OUTPUT_DIR}/network_cytoscape_{suffix}_edges.csv"
    edges_df.to_csv(edge_file, index=False)
    node_file = f"{OUTPUT_DIR}/network_cytoscape_{suffix}_nodes.csv"
    nodes_df.to_csv(node_file, index=False)
    print(f"\nCytoscape格式已保存:")
    print(f"  边: {edge_file}")
    print(f"  节点: {node_file}")


def analyze_group_specific_patterns(abund_matrix: pd.DataFrame,
                                    metadata: pd.DataFrame,
                                    pathway_info: pd.DataFrame):
    print("\n分析组间特异性模式...")

    group_means = {}
    for group in metadata['Group'].unique():
        samples = metadata[metadata['Group'] == group]['Sample'].tolist()
        group_samples = [s for s in samples if s in abund_matrix.columns]
        if group_samples:
            group_means[group] = abund_matrix[group_samples].mean(axis=1)

    if not group_means:
        print("  警告: 无法按组聚合")
        return pd.DataFrame()

    group_df = pd.DataFrame(group_means)
    group_df['max_diff'] = group_df.max(axis=1) - group_df.min(axis=1)
    top_diff = group_df.nlargest(20, 'max_diff')
    top_diff = top_diff.reset_index()
    top_diff.rename(columns={'index': 'Pathway_ID'}, inplace=True)

    def get_name(pid):
        pname, _ = map_pathway_info(pid, pathway_info)
        return pname

    top_diff['Pathway_Name'] = top_diff['Pathway_ID'].apply(get_name)

    output_file = f"{OUTPUT_DIR}/group_specific_pathways.csv"
    top_diff.to_csv(output_file, index=False)
    print(f"  组间差异最大通路已保存: {output_file}")

    return top_diff


def main():
    print("=" * 80)
    print("通路级代谢网络构建与分析 (修复版 v2)")
    print(f"输出目录: {OUTPUT_DIR}")
    print("=" * 80)

    data, metadata, pathway_info = load_matrices()

    abund_orig = extract_numeric_matrix(data['abund_orig'])
    abund_back = extract_numeric_matrix(data['abund_back'])

    print("\n" + "=" * 60)
    print("构建回补前网络...")
    print("=" * 60)
    edges_orig = compute_correlation_network(abund_orig, CORR_METHOD, CORR_THRESHOLD,
                                             PVALUE_THRESHOLD, USE_FDR, FDR_ALPHA)

    print("\n" + "=" * 60)
    print("构建回补后网络...")
    print("=" * 60)
    edges_back = compute_correlation_network(abund_back, CORR_METHOD, CORR_THRESHOLD,
                                             PVALUE_THRESHOLD, USE_FDR, FDR_ALPHA)

    if len(edges_orig) > 0:
        edges_orig.to_csv(f"{OUTPUT_DIR}/network_edges_original.csv", index=False)
    if len(edges_back) > 0:
        edges_back.to_csv(f"{OUTPUT_DIR}/network_edges_backfilled.csv", index=False)

    specific_edges = identify_backfill_specific_edges(edges_orig, edges_back, pathway_info)
    if len(specific_edges) > 0:
        specific_edges.to_csv(f"{OUTPUT_DIR}/backfill_specific_edges.csv", index=False)

    print("\n" + "=" * 60)
    print("回补前网络拓扑分析...")
    print("=" * 60)
    nodes_orig, G_orig = compute_node_topology(edges_orig, pathway_info)
    if len(nodes_orig) > 0:
        nodes_orig = detect_modules(G_orig, nodes_orig)
        nodes_orig.to_csv(f"{OUTPUT_DIR}/network_nodes_original.csv", index=False)
        generate_cytoscape_format(edges_orig, nodes_orig, 'original')

    print("\n" + "=" * 60)
    print("回补后网络拓扑分析...")
    print("=" * 60)
    nodes_back, G_back = compute_node_topology(edges_back, pathway_info)
    if len(nodes_back) > 0:
        nodes_back = detect_modules(G_back, nodes_back)
        nodes_back.to_csv(f"{OUTPUT_DIR}/network_nodes_backfilled.csv", index=False)
        generate_cytoscape_format(edges_back, nodes_back, 'backfilled')

    if len(nodes_back) > 0:
        hub_threshold = nodes_back['Degree'].quantile(0.9)
        hubs = nodes_back[nodes_back['Degree'] >= hub_threshold].copy()
        hubs.to_csv(f"{OUTPUT_DIR}/hub_pathways.csv", index=False)
        print(f"\nHub通路 (度≥{hub_threshold:.0f}): {len(hubs)}")
        if len(hubs) > 0:
            display_cols = ['Pathway_ID', 'Pathway_Name', 'Degree', 'Category']
            print(hubs[display_cols].head(10).to_string(index=False))

    group_patterns = analyze_group_specific_patterns(abund_back, metadata, pathway_info)

    stats = {
        'timestamp': datetime.now().isoformat(),
        'parameters': {
            'correlation_method': CORR_METHOD,
            'correlation_threshold': CORR_THRESHOLD,
            'pvalue_threshold': PVALUE_THRESHOLD,
            'use_fdr': USE_FDR,
            'fdr_alpha': FDR_ALPHA,
            'min_samples': MIN_SAMPLES
        },
        'original_network': {
            'n_nodes': len(set(edges_orig['source']) | set(edges_orig['target'])) if len(edges_orig) > 0 else 0,
            'n_edges': len(edges_orig),
            'avg_degree': 2 * len(edges_orig) / len(nodes_orig) if len(edges_orig) > 0 and len(nodes_orig) > 0 else 0
        },
        'backfilled_network': {
            'n_nodes': len(set(edges_back['source']) | set(edges_back['target'])) if len(edges_back) > 0 else 0,
            'n_edges': len(edges_back),
            'n_specific_edges': len(specific_edges),
            'hub_pathways': hubs['Pathway_ID'].tolist() if len(nodes_back) > 0 else []
        }
    }

    with open(f"{OUTPUT_DIR}/network_topology_stats.json", 'w') as f:
        json.dump(stats, f, indent=2)

    print("\n" + "=" * 80)
    print("网络构建完成!")
    print(f"结果保存在: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()