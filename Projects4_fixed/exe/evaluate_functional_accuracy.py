#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评估功能注释准确性
输入: ground_truth.tsv + predictions.tsv
输出: accuracy_report.json
"""

import json
import sys
import pandas as pd
from collections import Counter


def evaluate(ground_truth_tsv, predictions_tsv, output_json):
    """评估功能注释准确性"""

    print(f"加载 ground truth: {ground_truth_tsv}")
    gt = pd.read_csv(ground_truth_tsv, sep='\t')

    print(f"加载 predictions: {predictions_tsv}")
    pred = pd.read_csv(predictions_tsv, sep='\t')

    # 合并
    merged = pd.merge(gt, pred, on='gene_id', how='inner')
    print(f"可评估序列: {len(merged):,}")

    # 1. CAZy家族准确率
    cazy_correct = 0
    cazy_top1_correct = 0

    for _, row in merged.iterrows():
        true_fams = set(str(row['true_cazy']).split('|'))
        pred_cazy = str(row.get('predicted_cazy', 'NA'))

        if pred_cazy != 'NA':
            pred_fams = set(pred_cazy.split('|'))
            # any match
            if true_fams & pred_fams:
                cazy_correct += 1
            # top-1
            pred_top1 = pred_cazy.split('|')[0]
            if any(pred_top1.startswith(tf.split('_')[0]) for tf in true_fams):
                cazy_top1_correct += 1

    # 2. EC准确率
    ec_exact = 0
    ec_level3 = 0
    valid_ec = 0

    for _, row in merged.iterrows():
        true_ec = str(row['true_ec'])
        pred_ec = str(row.get('predicted_ec', 'NA'))

        if true_ec != 'NA':
            valid_ec += 1
            if pred_ec != 'NA':
                true_ecs = true_ec.split('|')
                pred_ecs = pred_ec.split('|')

                # exact match
                if any(te in pred_ecs for te in true_ecs):
                    ec_exact += 1

                # level-3 match
                true_l3 = {'.'.join(te.split('.')[:3]) for te in true_ecs}
                pred_l3 = {'.'.join(pe.split('.')[:3]) for pe in pred_ecs}
                if true_l3 & pred_l3:
                    ec_level3 += 1

    # 3. 按Tier统计
    tier_stats = {}
    for tier in ['Tier_1', 'Tier_2', 'Tier_3', 'Tier_Low']:
        tier_df = merged[merged.get('confidence_tier', '') == tier]
        if len(tier_df) > 0:
            tier_correct = 0
            for _, row in tier_df.iterrows():
                true_fams = set(str(row['true_cazy']).split('|'))
                pred_cazy = str(row.get('predicted_cazy', 'NA'))
                if pred_cazy != 'NA' and (true_fams & set(pred_cazy.split('|'))):
                    tier_correct += 1
            tier_stats[tier] = {
                'count': len(tier_df),
                'cazy_accuracy': tier_correct / len(tier_df)
            }

    # 汇总
    results = {
        'total_evaluated': len(merged),
        'cazy_any_match': {
            'correct': cazy_correct,
            'total': len(merged),
            'accuracy': cazy_correct / len(merged) if len(merged) > 0 else 0
        },
        'cazy_top1': {
            'correct': cazy_top1_correct,
            'total': len(merged),
            'accuracy': cazy_top1_correct / len(merged) if len(merged) > 0 else 0
        },
        'ec_exact': {
            'correct': ec_exact,
            'total': valid_ec,
            'accuracy': ec_exact / valid_ec if valid_ec > 0 else 0
        },
        'ec_level3': {
            'correct': ec_level3,
            'total': valid_ec,
            'accuracy': ec_level3 / valid_ec if valid_ec > 0 else 0
        },
        'tier_stats': tier_stats,
    }

    # 保存
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)

    # 打印摘要
    print(f"\n{'=' * 60}")
    print("评估结果摘要")
    print(f"{'=' * 60}")
    print(f"总序列: {len(merged):,}")
    print(f"CAZy any-match: {cazy_correct}/{len(merged)} = {results['cazy_any_match']['accuracy']:.4f}")
    print(f"CAZy Top-1: {cazy_top1_correct}/{len(merged)} = {results['cazy_top1']['accuracy']:.4f}")
    print(f"EC exact: {ec_exact}/{valid_ec} = {results['ec_exact']['accuracy']:.4f}")
    print(f"EC level-3: {ec_level3}/{valid_ec} = {results['ec_level3']['accuracy']:.4f}")
    print(f"\nTier统计:")
    for tier, stats in tier_stats.items():
        print(f"  {tier}: {stats['count']}条, 准确率={stats['cazy_accuracy']:.4f}")
    print(f"{'=' * 60}")
    print(f"报告保存: {output_json}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("用法: python evaluate_functional_accuracy.py <ground_truth.tsv> <predictions.tsv> <output.json>")
        sys.exit(1)

    evaluate(sys.argv[1], sys.argv[2], sys.argv[3])