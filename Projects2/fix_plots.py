#!/usr/bin/env python3
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

samples = ['TR-31-90D', 'M3-90C', 'M3-6023-7B', 'FM-3']
base = Path('/mnt/zjwdata/2/raw/species_annotation_results_raw')

for s in samples:
    csv = base / s / 'species_composition_table.csv'
    if not csv.exists():
        print(f"⚠️ {s}: 找不到物种组成表，跳过")
        continue

    df = pd.read_csv(csv)
    top = df.head(20)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # 饼图
    if not top.empty and top['relative_abundance'].sum() > 0:
        ax1.pie(top['relative_abundance'], labels=top['name'], autopct='%1.1f%%', startangle=90)
        ax1.set_title(f'{s} - Top 20 Species Composition\n(Kraken2+Bracken)')
        for text in ax1.texts:
            text.set_fontsize(8)
    else:
        ax1.text(0.5, 0.5, 'No data', ha='center', va='center')

    # 条形图
    if not top.empty:
        y = np.arange(len(top))
        bars = ax2.barh(y, top['relative_abundance'])
        ax2.set_yticks(y)
        ax2.set_yticklabels(top['name'], fontsize=8)
        ax2.set_xlabel('Relative Abundance (%)')
        ax2.set_title('Species Abundance Distribution')
        ax2.invert_yaxis()
        for i, bar in enumerate(bars):
            w = bar.get_width()
            ax2.text(w + 0.1, bar.get_y() + bar.get_height() / 2, f'{w:.2f}%',
                     ha='left', va='center', fontsize=7)

    plt.tight_layout()
    plt.savefig(base / s / 'species_composition_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ {s}: 图片已补生成")