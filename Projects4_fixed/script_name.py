#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CAAC回补验证与网络分析可视化脚本
读取真实数据绘制科研风格配图
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from scipy import stats

# ==================== 配置 ====================

# 数据路径（请根据实际路径修改）
BASE_DIR = "/home/zjw/zjwdata/3_deep_learning/training_data/caac_backfill_results/network_analysis"
NETWORK_DIR = os.path.join(BASE_DIR, "network_results")

# 输出路径
OUTPUT_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 学术配色
COLOR_ORIG = '#2E5C8A'  # 深蓝 - 回补前
COLOR_BACK = '#D9534F'  # 暖橙红 - 回补后/特异性边
COLOR_CARB = '#E8A838'  # 金黄 - 碳水化合物代谢
COLOR_NEUTRAL = '#666666'  # 中性灰

# 图片参数
plt.rcParams['font.family'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300


# ==================== 工具函数 ====================

def extract_pure_id(full_id):
    """提取纯ko编号"""
    full_id = str(full_id).strip()
    match = re.match(r'(ko\d{5})', full_id)
    if match:
        return match.group(1)
    if ' ' in full_id:
        return full_id.split(' ')[0]
    return full_id


def load_coverage_matrices():
    """加载回补前后的覆盖度矩阵"""
    print("加载覆盖度矩阵...")

    cov_orig = pd.read_csv(os.path.join(BASE_DIR, "pathway_coverage_matrix_original.csv"), index_col=0)
    cov_back = pd.read_csv(os.path.join(BASE_DIR, "pathway_coverage_matrix_backfilled.csv"), index_col=0)

    # 标准化索引
    cov_orig.index = cov_orig.index.map(extract_pure_id)
    cov_back.index = cov_back.index.map(extract_pure_id)

    # 去除Pathway_Name列（如果存在）
    if 'Pathway_Name' in cov_orig.columns:
        cov_orig = cov_orig.drop(columns=['Pathway_Name'])
    if 'Pathway_Name' in cov_back.columns:
        cov_back = cov_back.drop(columns=['Pathway_Name'])

    # 确保数值类型
    for col in cov_orig.columns:
        cov_orig[col] = pd.to_numeric(cov_orig[col], errors='coerce').fillna(0)
    for col in cov_back.columns:
        cov_back[col] = pd.to_numeric(cov_back[col], errors='coerce').fillna(0)

    print(f"  原始矩阵: {cov_orig.shape}")
    print(f"  回补矩阵: {cov_back.shape}")

    return cov_orig, cov_back


def load_pathway_info():
    """加载通路分类信息"""
    print("加载通路信息...")

    info = pd.read_csv(os.path.join(BASE_DIR, "pathway_info.csv"), index_col=0)
    info.index = info.index.map(extract_pure_id)

    # 清理Category列
    info['Category'] = info['Category'].astype(str).str.strip()
    info['Category'] = info['Category'].replace('nan', 'Other').replace('None', 'Other').fillna('Other')

    print(f"  通路信息: {info.shape}")
    print(f"  类别分布:\n{info['Category'].value_counts()}")

    return info


def load_network_edges():
    """加载网络边数据"""
    print("加载网络边数据...")

    edges_orig = pd.read_csv(os.path.join(NETWORK_DIR, "network_edges_original.csv"))
    edges_back = pd.read_csv(os.path.join(NETWORK_DIR, "network_edges_backfilled.csv"))
    specific_edges = pd.read_csv(os.path.join(NETWORK_DIR, "backfill_specific_edges.csv"))

    print(f"  原始边: {len(edges_orig)}")
    print(f"  回补边: {len(edges_back)}")
    print(f"  特异性边: {len(specific_edges)}")

    return edges_orig, edges_back, specific_edges


def load_hub_pathways():
    """加载Hub通路"""
    print("加载Hub通路...")

    hubs = pd.read_csv(os.path.join(NETWORK_DIR, "hub_pathways.csv"))
    print(f"  Hub通路: {len(hubs)}")

    return hubs


# ==================== 图1：覆盖度分布对比 ====================

def plot_coverage_comparison(cov_orig, cov_back, pathway_info):
    """绘制回补前后通路覆盖度分布对比图"""

    print("\n绘制图1: 覆盖度分布对比...")

    # 获取样本列
    sample_cols = [c for c in cov_orig.columns if c != 'Pathway_Name']

    # 展平为长格式数据
    orig_flat = cov_orig[sample_cols].values.flatten()
    back_flat = cov_back[sample_cols].values.flatten()

    # 计算整体统计
    mean_orig = np.mean(orig_flat)
    mean_back = np.mean(back_flat)

    # 识别碳水化合物代谢通路
    carb_pathways = pathway_info[pathway_info['Category'] == 'Carbohydrate Metabolism'].index.tolist()

    # 获取两个矩阵中都存在的碳水化合物通路（关键修复）
    common_carb = [p for p in carb_pathways if p in cov_orig.index and p in cov_back.index]
    print(f"  碳水化合物通路总数: {len(carb_pathways)}")
    print(f"  共同存在的碳水化合物通路: {len(common_carb)}")

    if len(common_carb) == 0:
        print("  警告: 未找到共同的碳水化合物代谢通路，跳过子图(b)")
        common_carb = carb_pathways  # 回退方案

    # 碳水化合物通路数据（按通路平均）- 使用共同索引
    carb_orig_avg = cov_orig.loc[common_carb, sample_cols].mean(axis=1)
    carb_back_avg = cov_back.loc[common_carb, sample_cols].mean(axis=1)

    # 确保两个Series索引一致
    carb_orig_avg, carb_back_avg = carb_orig_avg.align(carb_back_avg, join='inner')

    carb_increase = carb_back_avg - carb_orig_avg

    carb_df = pd.DataFrame({
        'Pathway_ID': carb_orig_avg.index,
        'Original': carb_orig_avg.values,
        'Backfilled': carb_back_avg.values,
        'Increase': carb_increase.values,
    })

    print(f"  碳水通路DataFrame: {carb_df.shape}")
    print(f"  有提升(>0.5%): {(carb_df['Increase'] > 0.5).sum()}")
    print(f"  无提升(≤0.5%): {(carb_df['Increase'] <= 0.5).sum()}")

    # 创建图形
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))

    # --- 左图：整体覆盖度分布（小提琴图+箱线图） ---
    ax1 = axes[0]

    positions = [1, 2]
    vp1 = ax1.violinplot([orig_flat, back_flat], positions=positions, widths=0.6,
                         showmeans=False, showmedians=False, showextrema=False)
    vp1['bodies'][0].set_facecolor(COLOR_ORIG)
    vp1['bodies'][0].set_alpha(0.6)
    vp1['bodies'][1].set_facecolor(COLOR_BACK)
    vp1['bodies'][1].set_alpha(0.6)

    bp = ax1.boxplot([orig_flat, back_flat], positions=positions, widths=0.15,
                     patch_artist=True, showfliers=False,
                     medianprops=dict(color='white', linewidth=1.5),
                     whiskerprops=dict(color=COLOR_NEUTRAL, linewidth=0.8),
                     capprops=dict(color=COLOR_NEUTRAL, linewidth=0.8))
    bp['boxes'][0].set_facecolor(COLOR_ORIG)
    bp['boxes'][0].set_alpha(0.9)
    bp['boxes'][1].set_facecolor(COLOR_BACK)
    bp['boxes'][1].set_alpha(0.9)

    # 均值虚线
    ax1.hlines(mean_orig, positions[0] - 0.1, positions[0] + 0.1,
               colors='white', linewidth=2, linestyles='--')
    ax1.hlines(mean_back, positions[1] - 0.1, positions[1] + 0.1,
               colors='white', linewidth=2, linestyles='--')

    ax1.set_xticks(positions)
    ax1.set_xticklabels(['Original\n(Before)', 'Backfilled\n(After)'], fontsize=11)
    ax1.set_ylabel('Pathway Coverage (%)', fontsize=12)
    ax1.set_ylim(-2, 65)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    # 均值标注
    ax1.annotate(f'Mean: {mean_orig:.2f}%', xy=(1, mean_orig),
                 xytext=(1.3, mean_orig + 8), fontsize=9, color=COLOR_ORIG,
                 arrowprops=dict(arrowstyle='->', color=COLOR_ORIG, lw=0.8))
    ax1.annotate(f'Mean: {mean_back:.2f}%', xy=(2, mean_back),
                 xytext=(2.3, mean_back + 8), fontsize=9, color=COLOR_BACK,
                 arrowprops=dict(arrowstyle='->', color=COLOR_BACK, lw=0.8))

    ax1.set_title('(a) Overall Pathway Coverage Distribution', fontsize=12,
                  fontweight='bold', pad=10, y=-0.18)

    # --- 右图：碳水化合物代谢通路散点对比 ---
    ax2 = axes[1]

    max_val = max(carb_df['Original'].max(), carb_df['Backfilled'].max()) + 5
    ax2.plot([0, max_val], [0, max_val], '--', color=COLOR_NEUTRAL,
             linewidth=0.8, alpha=0.5, label='No change')

    # 根据提升幅度分配颜色
    for _, row in carb_df.iterrows():
        if row['Increase'] > 0.5:
            color = COLOR_BACK  # 红色 - 有提升
            size = min(row['Increase'] * 12 + 25, 300)
        else:
            color = COLOR_ORIG  # 蓝色 - 无提升
            size = 45

        ax2.scatter(row['Original'], row['Backfilled'],
                    c=color, s=size, alpha=0.75,
                    edgecolors='white', linewidth=0.5, zorder=3)

    # 标注Top3提升通路
    top3 = carb_df.nlargest(3, 'Increase')
    for _, row in top3.iterrows():
        # 获取通路名称
        pid = row['Pathway_ID']
        pname = pathway_info.loc[pid, 'Pathway_Name'] if pid in pathway_info.index else pid

        # 截短名称
        short_name = pid
        if len(pname) > 30:
            short_name = pid
        else:
            short_name = f"{pid}\n{pname[:25]}"

        ax2.annotate(f"{short_name}\n+{row['Increase']:.1f}%",
                     xy=(row['Original'], row['Backfilled']),
                     xytext=(row['Original'] - 8, row['Backfilled'] + 6),
                     fontsize=7, ha='center',
                     arrowprops=dict(arrowstyle='->', color=COLOR_NEUTRAL, lw=0.6),
                     bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                               edgecolor='none', alpha=0.85))

    ax2.set_xlabel('Original Coverage (%)', fontsize=12)
    ax2.set_ylabel('Backfilled Coverage (%)', fontsize=12)
    ax2.set_xlim(-2, 55)
    ax2.set_ylim(-2, 60)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(alpha=0.3, linestyle='--')

    # 图例
    legend_elements = [
        mpatches.Patch(color=COLOR_ORIG, label='No / Minor increase (≤0.5%)'),
        mpatches.Patch(color=COLOR_BACK, label='Increased (>0.5%, CAAC backfilled)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_NEUTRAL,
                   markersize=8, label='Size ∝ increase magnitude')
    ]
    ax2.legend(handles=legend_elements, loc='upper left', fontsize=8.5,
               frameon=True, fancybox=False, edgecolor='gray')

    ax2.set_title('(b) Carbohydrate Metabolism Pathways', fontsize=12,
                  fontweight='bold', pad=10, y=-0.18)

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "fig3_coverage_comparison_real.png")
    plt.savefig(output_path, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.show()

    print(f"图1已保存: {output_path}")
    print(f"  碳水化合物通路: {len(carb_df)}条")
    print(f"  有提升(>0.5%): {(carb_df['Increase'] > 0.5).sum()}条")
    print(f"  无提升(≤0.5%): {(carb_df['Increase'] <= 0.5).sum()}条")


# ==================== 图2：网络拓扑对比 ====================

def plot_network_comparison(edges_orig, edges_back, specific_edges, hub_pathways, pathway_info):
    """绘制回补前后网络拓扑对比图"""

    print("\n绘制图2: 网络拓扑对比...")

    # 构建NetworkX图
    G_orig = nx.Graph()
    for _, row in edges_orig.iterrows():
        G_orig.add_edge(row['source'], row['target'], weight=row.get('weight', abs(row.get('correlation', 0))))

    G_back = nx.Graph()
    for _, row in edges_back.iterrows():
        G_back.add_edge(row['source'], row['target'], weight=row.get('weight', abs(row.get('correlation', 0))))

    # 获取回补特异性边
    orig_edge_set = set(tuple(sorted([row['source'], row['target']])) for _, row in edges_orig.iterrows())
    specific_edge_set = set()
    for _, row in specific_edges.iterrows():
        key = tuple(sorted([row['source'], row['target']]))
        specific_edge_set.add(key)

    # 计算度中心性
    degree_orig = dict(G_orig.degree())
    degree_back = dict(G_back.degree())

    # 确定Hub节点（使用回补后网络）
    hub_threshold = np.percentile(list(degree_back.values()), 90)
    hub_nodes = [n for n, d in degree_back.items() if d >= hub_threshold]

    # 碳水化合物代谢Hub节点
    carb_hub_nodes = []
    for n in hub_nodes:
        if n in pathway_info.index:
            if pathway_info.loc[n, 'Category'] == 'Carbohydrate Metabolism':
                carb_hub_nodes.append(n)

    print(f"  Hub节点: {len(hub_nodes)}")
    print(f"  碳水化合物Hub: {len(carb_hub_nodes)}")

    # 布局（使用spring layout）
    print("  计算布局...")
    pos_orig = nx.spring_layout(G_orig, k=0.25, iterations=60, seed=42)

    # 回补后网络复用布局
    pos_back = {}
    for n in G_back.nodes():
        if n in pos_orig:
            pos_back[n] = pos_orig[n]
        else:
            # 新节点放在随机位置
            pos_back[n] = np.random.randn(2) * 0.3

    # 创建图形
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))

    # 类别颜色映射
    cat_colors = {
        'Carbohydrate Metabolism': COLOR_CARB,
        'Genetic Information Processing': '#6B8E9F',
        'Amino Acid Metabolism': '#8FBC8F',
        'Environmental Information Processing': '#B0C4DE',
        'Lipid Metabolism': '#DDA0DD',
        'Cellular Processes': '#F0B8A0',
        'Energy Metabolism': '#FFD700',
        'Global and Overview Maps': '#D3D3D3',
        'Other': '#BBBBBB',
    }

    # --- 左图：回补前网络 ---
    ax1 = axes[0]

    # 采样绘制边（避免过于密集）
    edges_list = list(G_orig.edges())
    if len(edges_list) > 2500:
        sample_idx = np.random.choice(len(edges_list), size=2500, replace=False)
        edges_to_draw = [edges_list[i] for i in sample_idx]
    else:
        edges_to_draw = edges_list

    for u, v in edges_to_draw:
        if u in pos_orig and v in pos_orig:
            ax1.plot([pos_orig[u][0], pos_orig[v][0]],
                     [pos_orig[u][1], pos_orig[v][1]],
                     color='#CCCCCC', alpha=0.12, linewidth=0.25, zorder=1)

    # 绘制节点
    for n in G_orig.nodes():
        if n not in pos_orig:
            continue

        cat = pathway_info.loc[n, 'Category'] if n in pathway_info.index else 'Other'
        color = cat_colors.get(cat, '#BBBBBB')

        if n in hub_nodes:
            size = degree_orig.get(n, 1) * 1.8 + 35
            alpha = 0.85
            edge_w = 0.8
        else:
            size = degree_orig.get(n, 1) * 1.2 + 12
            alpha = 0.45
            edge_w = 0.3

        ax1.scatter(pos_orig[n][0], pos_orig[n][1], c=color, s=size,
                    alpha=alpha, edgecolors='white', linewidth=edge_w, zorder=2)

    density_orig = 2 * G_orig.number_of_edges() / (
                G_orig.number_of_nodes() * (G_orig.number_of_nodes() - 1)) if G_orig.number_of_nodes() > 1 else 0
    ax1.set_title(
        f'(a) Original Network\n{G_orig.number_of_nodes()} nodes, {G_orig.number_of_edges():,} edges, density={density_orig:.4f}',
        fontsize=11, fontweight='bold', pad=10, y=-0.15)
    ax1.axis('off')

    # --- 右图：回补后网络 ---
    ax2 = axes[1]

    # 绘制普通边
    edges_back_list = list(G_back.edges())
    for u, v in edges_back_list:
        if u not in pos_back or v not in pos_back:
            continue

        key = tuple(sorted([u, v]))
        is_specific = key in specific_edge_set

        if is_specific and np.random.random() < 0.5:
            # 回补特异性边（红色）
            ax2.plot([pos_back[u][0], pos_back[v][0]],
                     [pos_back[u][1], pos_back[v][1]],
                     color='#D9534F', alpha=0.3, linewidth=0.6, zorder=1)
        elif not is_specific and np.random.random() < 0.2:
            # 普通边（灰色）
            ax2.plot([pos_back[u][0], pos_back[v][0]],
                     [pos_back[u][1], pos_back[v][1]],
                     color='#CCCCCC', alpha=0.08, linewidth=0.2, zorder=1)

    # 绘制节点
    for n in G_back.nodes():
        if n not in pos_back:
            continue

        cat = pathway_info.loc[n, 'Category'] if n in pathway_info.index else 'Other'
        color = cat_colors.get(cat, '#BBBBBB')

        is_hub = n in hub_nodes
        is_carb_hub = n in carb_hub_nodes

        if is_hub:
            size = degree_back.get(n, 1) * 1.8 + 35
            alpha = 0.95 if is_carb_hub else 0.85
            edge_color = '#D9534F' if is_carb_hub else 'white'
            edge_w = 2.5 if is_carb_hub else 0.8
        else:
            size = degree_back.get(n, 1) * 1.2 + 12
            alpha = 0.45
            edge_color = 'white'
            edge_w = 0.3

        ax2.scatter(pos_back[n][0], pos_back[n][1], c=color, s=size,
                    alpha=alpha, edgecolors=edge_color, linewidth=edge_w, zorder=2)

        # 标注碳水化合物Hub
        if is_carb_hub:
            pname = pathway_info.loc[n, 'Pathway_Name'] if n in pathway_info.index else n
            short_name = str(pname)[:20] + '...' if len(str(pname)) > 20 else str(pname)
            ax2.annotate(f'Carb-Hub\n{short_name}',
                         xy=(pos_back[n][0], pos_back[n][1]),
                         xytext=(pos_back[n][0] + 0.12, pos_back[n][1] + 0.12),
                         fontsize=6.5, color='#D9534F', fontweight='bold',
                         bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                                   edgecolor='#D9534F', alpha=0.9, linewidth=0.8))

    density_back = 2 * G_back.number_of_edges() / (
                G_back.number_of_nodes() * (G_back.number_of_nodes() - 1)) if G_back.number_of_nodes() > 1 else 0
    ax2.set_title(
        f'(b) Backfilled Network\n{G_back.number_of_nodes()} nodes, {G_back.number_of_edges():,} edges, density={density_back:.4f}',
        fontsize=11, fontweight='bold', pad=10, y=-0.15)
    ax2.axis('off')

    # 图例
    legend_elements = [
        mpatches.Patch(color=COLOR_CARB, label='Carbohydrate Metabolism'),
        mpatches.Patch(color='#6B8E9F', label='Genetic Info. Processing'),
        mpatches.Patch(color='#8FBC8F', label='Amino Acid Metabolism'),
        mpatches.Patch(color='#BBBBBB', label='Other pathways'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#BBBBBB',
                   markersize=8, markeredgecolor='#D9534F', markeredgewidth=2,
                   label='Carbohydrate Hub'),
        plt.Line2D([0], [0], color='#D9534F', linewidth=1.5, alpha=0.5,
                   label=f'Backfill-specific edges ({len(specific_edges) / len(edges_back) * 100:.1f}%)'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=6, fontsize=9,
               frameon=True, fancybox=False, edgecolor='gray',
               bbox_to_anchor=(0.5, -0.01))

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    output_path = os.path.join(OUTPUT_DIR, "fig4_network_comparison_real.png")
    plt.savefig(output_path, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.show()

    print(f"图2已保存: {output_path}")


# ==================== 主函数 ====================

def main():
    print("=" * 60)
    print("CAAC回补验证与网络分析可视化")
    print("=" * 60)

    # 加载数据
    cov_orig, cov_back = load_coverage_matrices()
    pathway_info = load_pathway_info()
    edges_orig, edges_back, specific_edges = load_network_edges()
    hub_pathways = load_hub_pathways()

    # 绘制图1
    plot_coverage_comparison(cov_orig, cov_back, pathway_info)

    # 绘制图2
    plot_network_comparison(edges_orig, edges_back, specific_edges, hub_pathways, pathway_info)

    print("\n" + "=" * 60)
    print("所有图片已保存到:", OUTPUT_DIR)
    print("=" * 60)


if __name__ == "__main__":
    main()