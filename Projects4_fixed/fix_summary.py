#!/usr/bin/env python3
import os
import json
import pandas as pd
import numpy as np

output_dir = "/home/zjw/zjwdata/3_deep_learning/training_data/caac_backfill_results"
samples = [
    'CK-7A', 'CK-7B', 'CK-7C', 'CK-90A', 'CK-90B', 'CK-90D',
    'FM-1', 'FM-2', 'FM-3',
    'M3-6023-7A', 'M3-6023-7B', 'M3-6023-7D',
    'M3-90-A', 'M3-90-B', 'M3-90C',
    'T-31-7A', 'T-31-7B', 'T-31-7C',
    'TR-31-90A', 'TR-31-90C', 'TR-31-90D'
]

results = []

for sample in samples:
    comp_file = f"{output_dir}/summary/{sample}_comparison.tsv"
    if not os.path.exists(comp_file):
        print(f"警告: {sample} 对比文件不存在")
        continue

    df = pd.read_csv(comp_file, sep='\t')

    # 修复Pathway_Name: 优先使用_orig，如果不存在则用_back
    if 'Pathway_Name_orig' in df.columns:
        df['Pathway_Name'] = df['Pathway_Name_orig'].fillna(df.get('Pathway_Name_back', ''))
    elif 'Pathway_Name_back' in df.columns:
        df['Pathway_Name'] = df['Pathway_Name_back']

    n_pathways = len(df)
    n_improved = (df['Coverage_increase'] > 0).sum()
    avg_increase = df['Coverage_increase'].mean()
    max_increase = df['Coverage_increase'].max()

    # 计算关键代谢通路的提升（修复ID格式匹配问题）
    key_pathways = ['ko00010', 'ko00500', 'ko00520', 'ko00620', 'ko00650', 'ko00710']
    # 从完整的 Pathway_ID 中提取纯 ko 编号（例如 "ko00010 Glycolysis..." -> "ko00010"）
    key_mask = df['Pathway_ID'].str.extract(r'(ko\d+)', expand=False).isin(key_pathways)
    key_df = df[key_mask]
    key_avg = key_df['Coverage_increase'].mean() if len(key_df) > 0 else 0

    results.append({
        'Sample': sample,
        'Total_Pathways': n_pathways,
        'Improved_Pathways': int(n_improved),
        'Avg_Coverage_Increase_%': round(avg_increase, 2),
        'Max_Coverage_Increase_%': round(max_increase, 2),
        'Key_Metabolism_Avg_Increase_%': round(key_avg, 2)
    })

    # 保存修复后的对比文件（不包含临时添加的列）
    df.to_csv(comp_file, sep='\t', index=False)

# 保存汇总表
summary_df = pd.DataFrame(results)
csv_file = f"{output_dir}/summary/backfill_summary_table_FIXED.csv"
summary_df.to_csv(csv_file, index=False)

print(f"✅ 修复完成！汇总表已保存: {csv_file}")
print(f"\n{summary_df.to_string(index=False)}")

# 保存JSON报告
report = {
    'timestamp': pd.Timestamp.now().isoformat(),
    'n_samples': len(results),
    'overall': {
        'avg_improved_pathways': float(np.mean([r['Improved_Pathways'] for r in results])),
        'avg_coverage_increase': float(np.mean([r['Avg_Coverage_Increase_%'] for r in results])),
        'max_coverage_increase': float(max([r['Max_Coverage_Increase_%'] for r in results])),
        'avg_key_metabolism_increase': float(np.mean([r['Key_Metabolism_Avg_Increase_%'] for r in results]))
    },
    'sample_results': results
}

json_file = f"{output_dir}/summary/backfill_summary_report_FIXED.json"
with open(json_file, 'w') as f:
    json.dump(report, f, indent=2)

print(f"\n✅ JSON报告已保存: {json_file}")