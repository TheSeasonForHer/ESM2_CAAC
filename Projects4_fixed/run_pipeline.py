#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化重跑脚本 (跳过模块5 GNN训练)
已运行: 模块2 (hard_sample_mining) + 模块3 (clustering_and_mapping)
跳过: 模块5 (model_training) - 使用旧模型
"""

import os
import sys
import subprocess
import shutil
from datetime import datetime

# ==================== 配置 ====================

SCRIPT_DIR = "/home/zjw/Projects4_fixed"
DATA_DIR = "/home/zjw/zjwdata/3_deep_learning/training_data/curated_v2"
BACKFILL_DIR = "/home/zjw/zjwdata/3_deep_learning/training_data/caac_backfill_results"
OLD_MODEL_PATH = f"{DATA_DIR}/gnn_model/final_model.pt"

# 需要强制删除的缓存
CACHE_TO_REMOVE = [
    f"{DATA_DIR}/features/hard_strict/",
    f"{DATA_DIR}/features/hard_expanded/",
    f"{DATA_DIR}/predictions/query_esm2_features.npy",
]


# ==================== 日志工具 ====================

def log(msg):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}")
    sys.stdout.flush()


def run_cmd(cmd, desc):
    log(f"▶ {desc}")
    log(f"  命令: {cmd}")

    process = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, cwd=SCRIPT_DIR
    )

    for line in process.stdout:
        print(line, end='')
        sys.stdout.flush()

    process.wait()

    if process.returncode != 0:
        log(f"❌ {desc} 失败")
        return False
    else:
        log(f"✅ {desc} 完成")
        return True


# ==================== 各步骤 ====================

def step0_prepare():
    log("=" * 80)
    log("步骤0: 准备工作")
    log("=" * 80)

    # 删除缓存
    log("0.1 删除需要强制重跑的缓存...")
    for path in CACHE_TO_REMOVE:
        if os.path.exists(path):
            if os.path.isdir(path):
                shutil.rmtree(path)
                log(f"  删除目录: {path}")
            else:
                os.remove(path)
                log(f"  删除文件: {path}")
        else:
            log(f"  不存在(无需删除): {path}")

    # 检查旧模型
    log("0.2 检查旧模型...")
    if os.path.exists(OLD_MODEL_PATH):
        log(f"  ✅ 旧模型存在: {OLD_MODEL_PATH}")
    else:
        log(f"  ❌ 旧模型不存在: {OLD_MODEL_PATH}")
        return False

    log("")
    return True


def step3_feature_extraction():
    log("=" * 80)
    log("步骤3: 模块4 - ESM2 + 上下文特征提取")
    log("说明: positive/negative 会自动跳过，只处理 hard 样本")
    log("=" * 80)

    cmd = (
        f"python3 {SCRIPT_DIR}/feature_extraction.py "
        f"--clustered-dir {DATA_DIR}/clustered "
        f"--genes-fna-dir /home/zjw/zjwdata/1/assembly_analysis/genes "
        f"--output-dir {DATA_DIR}/features"
    )
    return run_cmd(cmd, "模块4 特征提取")


def step4_skip_training():
    log("=" * 80)
    log("步骤4: 模块5 - GNN训练 (跳过)")
    log(f"使用旧模型: {OLD_MODEL_PATH}")
    log("=" * 80)
    log("✅ 跳过训练，直接使用已有模型")
    log("")
    return True


def step5_prediction():
    log("=" * 80)
    log("步骤5: 模块6 - 难样本功能预测")
    log("=" * 80)

    cmd = (
        f"python3 {SCRIPT_DIR}/prediction_pipeline_fixed_A.py "
        f"--hard-fasta {DATA_DIR}/hard_samples_combined.fasta "
        f"--positive-features {DATA_DIR}/features/positive/esm2_features.npy "
        f"--positive-ids {DATA_DIR}/features/positive/gene_ids.csv "
        f"--positive-meta {DATA_DIR}/positive_samples_info.tsv "
        f"--gnn-model {OLD_MODEL_PATH} "
        f"--genes-fna-dir /home/zjw/zjwdata/1/assembly_analysis/genes "
        f"--output-dir {DATA_DIR}/predictions"
    )
    return run_cmd(cmd, "模块6 预测")


def step6_caac_backfill():
    log("=" * 80)
    log("步骤6: CAAC回补分析")
    log("说明: 旧17个样本会自动跳过，只处理新增4个样本")
    log("=" * 80)

    cmd = (
        f"python3 {SCRIPT_DIR}/caac_backfill_pipeline.py "
        f"--min-confidence 0.40 "
        f"--ec-match-mode exact_then_prefix"
    )
    return run_cmd(cmd, "CAAC回补分析")


def step7_build_matrix():
    log("=" * 80)
    log("步骤7: 通路矩阵构建")
    log("=" * 80)

    cmd = f"python3 {SCRIPT_DIR}/build_pathway_matrix.py"
    return run_cmd(cmd, "矩阵构建")


def step8_fix_summary():
    log("=" * 80)
    log("步骤8: 汇总表修复")
    log("=" * 80)

    cmd = f"python3 {SCRIPT_DIR}/fix_summary.py"
    return run_cmd(cmd, "汇总修复")


def step9_network():
    log("=" * 80)
    log("步骤9: 代谢网络分析")
    log("=" * 80)

    cmd = f"python3 {SCRIPT_DIR}/build_pathway_network.py"
    return run_cmd(cmd, "网络分析")


# ==================== 主流程 ====================

def main():
    log("")
    log("=" * 80)
    log("优化重跑流程 (跳过模块5 GNN训练)")
    log("已运行: 模块2 (难样本挖掘) + 模块3 (CD-HIT聚类)")
    log("跳过: 模块5 (GNN训练) - 使用旧模型")
    log("=" * 80)
    log("")

    start_time = datetime.now()

    steps = [
        (step0_prepare, "准备工作"),
        (step3_feature_extraction, "模块4 特征提取"),
        (step4_skip_training, "跳过模块5"),
        (step5_prediction, "模块6 预测"),
        (step6_caac_backfill, "CAAC回补"),
        (step7_build_matrix, "矩阵构建"),
        (step8_fix_summary, "汇总修复"),
        (step9_network, "网络分析"),
    ]

    for step_func, step_name in steps:
        if not step_func():
            log(f"❌ {step_name} 失败，终止执行")
            return 1

    elapsed = datetime.now() - start_time

    log("")
    log("=" * 80)
    log("✅ 全部完成！")
    log("=" * 80)
    log(f"总耗时: {elapsed}")
    log(f"使用模型: {OLD_MODEL_PATH}")
    log("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())