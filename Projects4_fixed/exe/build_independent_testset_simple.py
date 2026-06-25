#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版独立测试集构建
策略: 从新版CAZyDB按家族分层抽样，确保与训练集家族分布不同
"""

import os
import re
import json
import random
import logging
from datetime import datetime
from collections import Counter

import pandas as pd

# ==================== 配置 ====================
NEW_CAZY_FASTA = "/mnt/databases/CAZyDB.07242025/CAZyDB.07242025.fa"
NEW_CAZY_ACTIVITIES = "/mnt/databases/CAZyDB.07242025/CAZyDB.08062022.fam-activities.txt"
TRAIN_POSITIVE = "/home/zjw/zjwdata/3_deep_learning/training_data/curated_v2/clustered/positive_cdhit90.fasta"

# 输出路径 - 修改为你想要的路径
OUTPUT_DIR = "/home/zjw/zjwdata/3_deep_learning/training_data/curated_v2/independent_validation"
# 或者用这个: OUTPUT_DIR = "/home/zjw/Projects4_fixed/independent_validation"

MAX_SAMPLES = 5000  # 减少样本量，确保质量
SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ==================== Step 1: 解析训练集家族分布 ====================
def parse_train_families(train_fasta):
    """解析训练集的CAZy家族分布"""
    families = Counter()
    with open(train_fasta, 'r') as f:
        for line in f:
            if line.startswith('>') and 'CAZy=' in line:
                # 格式: >id|...|CAZy=GH5|...
                match = re.search(r'CAZy=([^\s|]+)', line)
                if match:
                    fam = match.group(1).split('|')[0]  # 取第一个家族
                    base = re.match(r'^([A-Z]+)', fam).group(1)
                    families[base] += 1
    logger.info(f"训练集家族分布: {dict(families.most_common(10))}")
    return families


# ==================== Step 2: 解析EC映射 ====================
def parse_ec_mapping(activities_file):
    """解析家族-EC映射"""
    fam_to_ec = {}
    ec_pattern = re.compile(r'\(EC\s+([\d\.\-]+)\)')

    with open(activities_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('Copyright'):
                continue

            match = re.match(r'^(AA|CBM|CE|GH|GT|PL)(\d+)(\s+|$)', line)
            if not match:
                continue

            fam = match.group(1) + match.group(2)
            desc = line[match.end():].strip()
            ec_matches = ec_pattern.findall(desc)

            clean_ecs = []
            for ec in ec_matches:
                ec = ec.strip().rstrip('.').rstrip('-')
                if ec and ec != '-':
                    clean_ecs.append(ec)

            if clean_ecs:
                fam_to_ec[fam] = list(dict.fromkeys(clean_ecs))

    logger.info(f"EC映射: {len(fam_to_ec)} 个家族")
    return fam_to_ec


# ==================== Step 3: 解析新版CAZyDB并筛选 ====================
def parse_and_filter_new_cazy(fasta_path, fam_to_ec, train_families, max_samples=MAX_SAMPLES):
    """
    解析新版CAZyDB，筛选：
    1. 目标家族 GH/GT/CE
    2. 有EC映射
    3. 优先选择训练集中**较少**的家族（确保差异性）
    """

    # 先收集所有候选序列
    candidates = []

    with open(fasta_path, 'r') as f:
        current_id = None
        current_fams = []
        current_seq = []

        for line in f:
            line = line.strip()

            if line.startswith('>'):
                # 保存前一个
                if current_id and current_seq:
                    seq_str = ''.join(current_seq)
                    base_fams = []
                    has_ec = False
                    all_ecs = []

                    for fam in current_fams:
                        base_match = re.match(r'^([A-Z]+)', fam)
                        if base_match:
                            base = base_match.group(1)
                            base_fams.append(base)
                            # 检查EC
                            fam_base = fam.split('_')[0]
                            if fam_base in fam_to_ec:
                                has_ec = True
                                all_ecs.extend(fam_to_ec[fam_base])

                    # 筛选条件
                    if any(b in {'GH', 'GT', 'CE'} for b in base_fams) and has_ec:
                        # 计算"新颖性分数"：训练集中越少的家族分数越高
                        novelty = sum(1.0 / (train_families.get(b, 1) + 1) for b in base_fams)

                        candidates.append({
                            'id': current_id,
                            'families': current_fams,
                            'base_families': list(set(base_fams)),
                            'ec_list': list(dict.fromkeys(all_ecs)),
                            'sequence': seq_str,
                            'length': len(seq_str),
                            'novelty': novelty,
                        })

                # 解析新header
                parts = line[1:].split('|')
                current_id = parts[0] if parts else ""
                current_fams = parts[1:] if len(parts) > 1 else []
                current_seq = []
            else:
                current_seq.append(line)

        # 最后一个
        if current_id and current_seq:
            seq_str = ''.join(current_seq)
            base_fams = []
            has_ec = False
            all_ecs = []

            for fam in current_fams:
                base_match = re.match(r'^([A-Z]+)', fam)
                if base_match:
                    base = base_match.group(1)
                    base_fams.append(base)
                    fam_base = fam.split('_')[0]
                    if fam_base in fam_to_ec:
                        has_ec = True
                        all_ecs.extend(fam_to_ec[fam_base])

            if any(b in {'GH', 'GT', 'CE'} for b in base_fams) and has_ec:
                novelty = sum(1.0 / (train_families.get(b, 1) + 1) for b in base_fams)
                candidates.append({
                    'id': current_id,
                    'families': current_fams,
                    'base_families': list(set(base_fams)),
                    'ec_list': list(dict.fromkeys(all_ecs)),
                    'sequence': seq_str,
                    'length': len(seq_str),
                    'novelty': novelty,
                })

    logger.info(f"候选序列总数: {len(candidates):,}")

    # 按新颖性排序，优先选择训练集中少见的家族
    candidates.sort(key=lambda x: x['novelty'], reverse=True)

    # 分层抽样：确保各家族都有代表
    selected = []
    family_quota = {}  # 每个家族的配额

    # 先按家族分组
    by_family = {}
    for c in candidates:
        for bf in c['base_families']:
            if bf not in by_family:
                by_family[bf] = []
            by_family[bf].append(c)

    # 每个家族至少选一些
    for bf, seqs in by_family.items():
        quota = min(50, len(seqs))  # 每个家族最多50条
        family_quota[bf] = 0
        for s in seqs[:quota]:
            if s['id'] not in {x['id'] for x in selected}:
                selected.append(s)
                family_quota[bf] += 1

    # 如果还不够，从剩余候选中补充
    remaining = [c for c in candidates if c['id'] not in {x['id'] for x in selected}]
    random.seed(SEED)
    random.shuffle(remaining)

    while len(selected) < max_samples and remaining:
        selected.append(remaining.pop())

    # 最终截断
    selected = selected[:max_samples]

    logger.info(f"最终选择: {len(selected):,}")
    logger.info("家族分布:")
    final_fams = Counter()
    for s in selected:
        for bf in s['base_families']:
            final_fams[bf] += 1
    for fam, count in final_fams.most_common():
        logger.info(f"  {fam}: {count:,}")

    return selected


# ==================== Step 4: 构建模拟hard序列 ====================
def build_mock_hard(sequences, output_fasta, output_tsv):
    """构建模拟hard序列和ground truth"""

    with open(output_fasta, 'w') as f_fasta, open(output_tsv, 'w') as f_tsv:
        f_tsv.write("gene_id\ttrue_cazy\ttrue_ec\tsequence_length\n")

        for seq in sequences:
            fam_str = '|'.join(seq['families'])
            ec_str = '|'.join(seq['ec_list'])

            # FASTA: 模拟hard格式
            f_fasta.write(f">{seq['id']}|LABEL=2|Sample=IndependentTest|CAZy={fam_str}|EC={ec_str}\n")
            for i in range(0, len(seq['sequence']), 60):
                f_fasta.write(seq['sequence'][i:i + 60] + '\n')

            # TSV: ground truth
            f_tsv.write(f"{seq['id']}\t{fam_str}\t{ec_str}\t{seq['length']}\n")

    logger.info(f"模拟hard: {output_fasta} ({len(sequences):,}条)")
    logger.info(f"Ground truth: {output_tsv}")


# ==================== 主流程 ====================
def main():
    logger.info("=" * 70)
    logger.info("简化版独立测试集构建")
    logger.info("策略: 分层抽样，优先选择训练集中少见的家族")
    logger.info("=" * 70)

    # Step 1: 解析训练集家族
    logger.info("\n[Step 1] 解析训练集家族分布...")
    train_families = parse_train_families(TRAIN_POSITIVE)

    # Step 2: 解析EC映射
    logger.info("\n[Step 2] 解析EC映射...")
    fam_to_ec = parse_ec_mapping(NEW_CAZY_ACTIVITIES)

    # Step 3: 筛选新版CAZyDB
    logger.info("\n[Step 3] 筛选新版CAZyDB...")
    selected = parse_and_filter_new_cazy(NEW_CAZY_FASTA, fam_to_ec, train_families, MAX_SAMPLES)

    if len(selected) == 0:
        logger.error("没有选到任何序列！")
        return

    # Step 4: 构建模拟hard序列
    logger.info("\n[Step 4] 构建模拟hard序列...")
    mock_fasta = os.path.join(OUTPUT_DIR, "independent_test_mock_hard.fasta")
    gt_tsv = os.path.join(OUTPUT_DIR, "ground_truth.tsv")

    build_mock_hard(selected, mock_fasta, gt_tsv)

    # 保存配置
    config = {
        'timestamp': datetime.now().isoformat(),
        'new_cazy_db': NEW_CAZY_FASTA,
        'train_positive': TRAIN_POSITIVE,
        'strategy': 'stratified_sampling_by_novelty',
        'max_samples': MAX_SAMPLES,
        'final_count': len(selected),
        'mock_fasta': mock_fasta,
        'ground_truth': gt_tsv,
    }
    with open(os.path.join(OUTPUT_DIR, "config.json"), 'w') as f:
        json.dump(config, f, indent=2)

    logger.info(f"\n{'=' * 70}")
    logger.info("完成!")
    logger.info(f"{'=' * 70}")
    logger.info("\n下一步: 运行预测")
    logger.info(f"  python prediction_pipeline_fixed_A\\(3\\).py \\")
    logger.info(f"    --hard-fasta {mock_fasta} \\")
    logger.info(f"    --output-dir {OUTPUT_DIR}/predictions \\")
    logger.info(f"    --no-context")


if __name__ == "__main__":
    main()