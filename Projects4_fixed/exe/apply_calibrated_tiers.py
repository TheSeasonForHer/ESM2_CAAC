#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
CAAC Calibrated Tier Application Module
将校准后的Tier阈值应用到已有的hard序列预测结果

功能:
1. 读取已有的hard_predictions_full.tsv
2. 加载校准报告中的优化阈值
3. 重新分配Tier（基于校准阈值）
4. 生成包含预期精度/FPR的新报告

输入: /home/zjw/zjwdata/3_deep_learning/training_data/curated_v2/predictions/hard_predictions_full.tsv
输出: /home/zjw/zjwdata/3_deep_learning/training_data/curated_v2/calibration_results/calibrated_predictions/
================================================================================
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd


# ==================== 配置 ====================

class ApplyConfig:
    """应用配置"""

    # 输入
    original_predictions_path: str = "/home/zjw/zjwdata/3_deep_learning/training_data/curated_v2/predictions/hard_predictions_full.tsv"
    calibration_report_path: str = "/home/zjw/zjwdata/3_deep_learning/training_data/curated_v2/calibration_results/calibration_report.json"

    # 输出（独立目录）
    output_dir: str = "/home/zjw/zjwdata/3_deep_learning/training_data/curated_v2/calibration_results/calibrated_predictions"

    # 如果校准报告不存在，使用的默认阈值
    default_tier1: float = 0.60
    default_tier2: float = 0.40
    default_tier3: float = 0.25


# ==================== 日志 ====================

def setup_logger(log_dir: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("apply_tiers")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        logger.handlers.clear()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fh = logging.FileHandler(f"{log_dir}/apply_tiers_{timestamp}.log")
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ==================== Tier应用器 ====================

class CalibratedTierApplier:
    """
    应用校准后的Tier阈值到已有预测结果
    """

    def __init__(self, config: ApplyConfig):
        self.config = config
        self.logger = setup_logger(f"{config.output_dir}/logs")

        self.logger.info("=" * 80)
        self.logger.info("CAAC校准Tier应用")
        self.logger.info("=" * 80)

        # 加载校准阈值
        self.tier_thresholds = self._load_calibration()

    def _load_calibration(self) -> Dict:
        """加载校准报告"""
        if os.path.exists(self.config.calibration_report_path):
            self.logger.info(f"加载校准报告: {self.config.calibration_report_path}")
            with open(self.config.calibration_report_path, 'r') as f:
                report = json.load(f)

            rec = report.get('recommendations', {})
            thresholds = {
                'tier1': rec.get('tier1_threshold', self.config.default_tier1),
                'tier2': rec.get('tier2_threshold', self.config.default_tier2),
                'tier3': rec.get('tier3_threshold', self.config.default_tier3),
            }

            # 加载预期精度
            results = report.get('results', {})
            tier_info = results.get('tier_thresholds', {})

            self.expected_precision = {
                'tier1': tier_info.get('tier1_expected', {}).get('precision'),
                'tier2': tier_info.get('tier2_expected', {}).get('precision'),
                'tier3': tier_info.get('tier3_expected', {}).get('precision'),
            }

            self.logger.info("✓ 使用校准后的阈值")

        else:
            self.logger.warning(f"校准报告不存在，使用默认阈值")
            thresholds = {
                'tier1': self.config.default_tier1,
                'tier2': self.config.default_tier2,
                'tier3': self.config.default_tier3,
            }
            self.expected_precision = {'tier1': None, 'tier2': None, 'tier3': None}

        self.logger.info(f"  Tier 1 threshold: {thresholds['tier1']:.4f}")
        self.logger.info(f"  Tier 2 threshold: {thresholds['tier2']:.4f}")
        self.logger.info(f"  Tier 3 threshold: {thresholds['tier3']:.4f}")

        return thresholds

    def apply(self):
        """应用校准阈值到已有预测结果"""

        # 加载已有预测
        self.logger.info(f"\n加载已有预测: {self.config.original_predictions_path}")
        df = pd.read_csv(self.config.original_predictions_path, sep='\t')
        self.logger.info(f"  共 {len(df):,} 条预测")

        # 重新分配Tier
        self.logger.info("\n重新分配Tier...")

        def assign_tier(confidence: float) -> str:
            if confidence >= self.tier_thresholds['tier1']:
                return 'Tier_1'
            elif confidence >= self.tier_thresholds['tier2']:
                return 'Tier_2'
            elif confidence >= self.tier_thresholds['tier3']:
                return 'Tier_3'
            else:
                return 'Tier_Low'

        df['calibrated_tier'] = df['confidence_score'].apply(assign_tier)

        # 添加预期精度
        def get_expected_precision(tier: str) -> Optional[float]:
            return {
                'Tier_1': self.expected_precision.get('tier1'),
                'Tier_2': self.expected_precision.get('tier2'),
                'Tier_3': self.expected_precision.get('tier3'),
                'Tier_Low': None,
            }.get(tier)

        def get_expected_fpr(tier: str) -> Optional[float]:
            prec = get_expected_precision(tier)
            return 1.0 - prec if prec is not None else None

        df['expected_precision'] = df['calibrated_tier'].apply(get_expected_precision)
        df['expected_fpr'] = df['calibrated_tier'].apply(get_expected_fpr)

        # 保存结果
        os.makedirs(self.config.output_dir, exist_ok=True)

        output_path = f"{self.config.output_dir}/hard_predictions_calibrated.tsv"
        df.to_csv(output_path, sep='\t', index=False)
        self.logger.info(f"\n校准后的预测: {output_path}")

        # 生成统计报告
        self._generate_report(df)

        return df

    def _generate_report(self, df: pd.DataFrame):
        """生成统计报告"""

        report = {
            'timestamp': datetime.now().isoformat(),
            'calibration_source': self.config.calibration_report_path,
            'tier_thresholds': self.tier_thresholds,
            'expected_precision': self.expected_precision,
            'tier_distribution': df['calibrated_tier'].value_counts().to_dict(),
            'statistics': {},
        }

        for tier in ['Tier_1', 'Tier_2', 'Tier_3', 'Tier_Low']:
            tier_df = df[df['calibrated_tier'] == tier]
            if len(tier_df) == 0:
                continue

            report['statistics'][tier] = {
                'count': int(len(tier_df)),
                'percentage': float(len(tier_df) / len(df) * 100),
                'mean_confidence': float(tier_df['confidence_score'].mean()),
                'has_cazy': int((tier_df['predicted_cazy'] != 'NA').sum()),
                'has_ec': int((tier_df['predicted_ec'] != 'NA').sum()),
            }

        # 保存JSON报告
        report_path = f"{self.config.output_dir}/calibrated_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        self.logger.info(f"统计报告: {report_path}")

        # 屏幕摘要
        self.logger.info(f"\n{'=' * 60}")
        self.logger.info("校准后的Tier分布")
        self.logger.info(f"{'=' * 60}")
        for tier, stats in sorted(report['statistics'].items()):
            prec_str = f" (ExpPrec: {self.expected_precision.get(tier.lower().replace('_', ''), 'N/A')})" if tier != 'Tier_Low' else ""
            self.logger.info(f"  {tier}: {stats['count']:,} ({stats['percentage']:.1f}%){prec_str}")
        self.logger.info(f"{'=' * 60}")


# ==================== 入口 ====================

def main():
    parser = argparse.ArgumentParser(description='应用校准后的Tier阈值')
    parser.add_argument('--predictions', default=ApplyConfig.original_predictions_path,
                        help='已有预测结果路径')
    parser.add_argument('--calibration-report', default=ApplyConfig.calibration_report_path,
                        help='校准报告路径')
    parser.add_argument('--output-dir', default=ApplyConfig.output_dir,
                        help='输出目录')

    args = parser.parse_args()

    config = ApplyConfig()
    config.original_predictions_path = args.predictions
    config.calibration_report_path = args.calibration_report
    config.output_dir = args.output_dir

    applier = CalibratedTierApplier(config)
    applier.apply()


if __name__ == "__main__":
    main()