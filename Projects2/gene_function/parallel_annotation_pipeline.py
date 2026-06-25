#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
修正版主控脚本：并行运行四个注释流程
集成蛋白序列清理功能，解决InterProScan星号问题
输出目录修改为 /home/zjw/zjwdata/
修复路径不一致问题，改进蛋白序列清理逻辑
更新InterProScan数据库路径
优化JVM参数和命令执行方式
适配大内存服务器配置
修复InterProScan卡在99%的问题
添加代谢通路丰度分析功能
添加注释结果整合和整体评估功能
添加日志文件支持
修复文件扩展名和参数传递问题
修复日志日期格式问题
修复CAZy注释率100%问题 - 更新CAZy脚本调用参数
更新：使用Salmon计算真实基因丰度
更新：使用原始测序数据
更新：添加质量控制统计
更新：添加中文字体支持
"""

import os
import sys
import argparse
import logging
import subprocess
import glob
import json
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.font_manager as fm

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

LOG = logging.getLogger(__name__)
__version__ = "1.7.0"
__author__ = ("Jiewei Zhang",)
__email__ = "2694016293@qq.com"


def setup_chinese_font():
    """设置中文字体支持 - 修复版"""
    try:
        # 方法1: 使用默认字体，避免复杂的字体设置
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'Liberation Sans']
        plt.rcParams['axes.unicode_minus'] = False

        # 检查是否有中文字体，如果没有就使用英文标题
        try:
            # 尝试查找常见的中文字体
            fonts = fm.findSystemFonts()
            chinese_font_found = False

            for font_path in fonts:
                try:
                    font_prop = fm.FontProperties(fname=font_path)
                    font_name = font_prop.get_name()
                    # 检查是否包含中文字体名称
                    if any(chinese in font_name.lower() for chinese in
                           ['simhei', 'simsun', 'microsoft yahei', 'stheitisc', 'stsong']):
                        # 添加找到的中文字体
                        fm.fontManager.addfont(font_path)
                        font_name = fm.FontProperties(fname=font_path).get_name()
                        # 添加到字体列表的开头
                        current_fonts = plt.rcParams['font.sans-serif']
                        if font_name not in current_fonts:
                            plt.rcParams['font.sans-serif'] = [font_name] + current_fonts
                        chinese_font_found = True
                        LOG.info(f"找到中文字体: {font_name}")
                        break
                except:
                    continue

            if not chinese_font_found:
                LOG.warning("未找到中文字体，将使用英文标题")
        except Exception as e:
            LOG.warning(f"查找中文字体失败: {e}")

    except Exception as e:
        LOG.warning(f"设置字体失败: {e}")
        # 设置最安全的默认值
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False


def setup_logging(output_base):
    """设置日志格式，同时输出到文件和终端"""
    log_file = os.path.join(output_base, "annotation_pipeline.log")

    # 创建日志目录
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )

    LOG.info(f"日志文件: {log_file}")
    return log_file


def setup_environment():
    """设置环境变量"""
    # 设置InterProScan路径
    interproscan_path = "/root/tools/interproscan/interproscan-5.28-67.0"
    if interproscan_path not in os.environ.get('PATH', ''):
        os.environ['PATH'] = f"{interproscan_path}:{os.environ.get('PATH', '')}"
        LOG.info(f"设置InterProScan路径: {interproscan_path}")

    # 设置JVM内存参数
    os.environ['JAVA_OPTS'] = '-Xmx96g -Xms32g -XX:ParallelGCThreads=32'

    # 确保salmon在PATH中
    salmon_path = "/usr/bin"
    if salmon_path not in os.environ.get('PATH', ''):
        os.environ['PATH'] = f"{salmon_path}:{os.environ.get('PATH', '')}"
        LOG.info(f"添加salmon到PATH: {salmon_path}")


class StatusTracker:
    """状态跟踪器，用于记录和管理每个样本每个步骤的完成状态"""

    def __init__(self, output_base):
        self.status_file = os.path.join(output_base, "annotation_status.json")
        self.status = self.load_status()

    def load_status(self):
        """加载状态文件"""
        if os.path.exists(self.status_file):
            try:
                with open(self.status_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                LOG.warning(f"无法加载状态文件 {self.status_file}: {e}")
                return {}
        return {}

    def save_status(self):
        """保存状态文件"""
        try:
            with open(self.status_file, 'w') as f:
                json.dump(self.status, f, indent=2)
        except Exception as e:
            LOG.error(f"无法保存状态文件 {self.status_file}: {e}")

    def get_sample_status(self, sample):
        """获取样本的状态"""
        return self.status.get(sample, {})

    def update_step_status(self, sample, step, success, output_file=None):
        """更新步骤状态"""
        if sample not in self.status:
            self.status[sample] = {}

        step_status = {
            'success': success,
            'timestamp': datetime.now().isoformat()
        }

        if output_file and os.path.exists(output_file):
            step_status['output_file'] = output_file
            step_status['file_size'] = os.path.getsize(output_file)

        self.status[sample][step] = step_status
        self.save_status()

    def check_step_completed(self, sample, step, check_file=False):
        """检查步骤是否已完成"""
        if sample not in self.status or step not in self.status[sample]:
            return False

        step_status = self.status[sample][step]
        if not step_status.get('success', False):
            return False

        # 如果设置了检查文件，验证输出文件是否存在
        if check_file and 'output_file' in step_status:
            output_file = step_status['output_file']
            if not os.path.exists(output_file):
                LOG.warning(f"步骤 {step} 的输出文件不存在: {output_file}")
                return False

            # 检查文件大小是否一致
            if 'file_size' in step_status:
                actual_size = os.path.getsize(output_file)
                if actual_size != step_status['file_size']:
                    LOG.warning(f"步骤 {step} 的输出文件大小不一致: {actual_size} vs {step_status['file_size']}")
                    return False

        return True

    def get_incomplete_samples(self, samples, steps):
        """获取未完成的样本列表"""
        incomplete_samples = []
        for sample in samples:
            incomplete_steps = []
            for step in steps:
                if not self.check_step_completed(sample, step, check_file=True):
                    incomplete_steps.append(step)

            if incomplete_steps:
                incomplete_samples.append((sample, incomplete_steps))

        return incomplete_samples


def clean_protein_sequences(input_file, output_file):
    """清理蛋白序列文件，只移除序列末尾的星号（终止符）"""
    LOG.info(f"清理蛋白序列文件: {input_file} -> {output_file}")

    sequences_processed = 0
    sequences_cleaned = 0

    try:
        with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
            current_header = ""
            current_sequence = ""

            for line in fin:
                if line.startswith('>'):
                    # 处理前一个序列
                    if current_header and current_sequence:
                        # 只在序列末尾清理星号，而不是全部清理
                        if current_sequence.endswith('*\n'):
                            current_sequence = current_sequence[:-2] + '\n'
                            sequences_cleaned += 1
                        elif current_sequence.endswith('*'):
                            current_sequence = current_sequence[:-1]
                            sequences_cleaned += 1

                        fout.write(current_header)
                        fout.write(current_sequence)
                        sequences_processed += 1

                    # 开始新序列
                    current_header = line
                    current_sequence = ""
                else:
                    current_sequence += line

            # 处理最后一个序列
            if current_header and current_sequence:
                if current_sequence.endswith('*\n'):
                    current_sequence = current_sequence[:-2] + '\n'
                    sequences_cleaned += 1
                elif current_sequence.endswith('*'):
                    current_sequence = current_sequence[:-1]
                    sequences_cleaned += 1

                fout.write(current_header)
                fout.write(current_sequence)
                sequences_processed += 1

        LOG.info(f"处理完成: 共处理 {sequences_processed} 个序列，清理了 {sequences_cleaned} 个序列中的终止星号")
        return True

    except Exception as e:
        LOG.error(f"清理蛋白序列文件失败: {e}")
        return False


def run_command(cmd, description=""):
    """运行命令行工具"""
    LOG.info(f"运行: {description}")
    LOG.info(f"命令: {cmd}")

    try:
        # 实时打印，避免被 PIPE 吞掉
        result = subprocess.run(cmd, shell=True, check=True)
        LOG.info(f"{description} 完成")
        return True
    except subprocess.CalledProcessError as e:
        LOG.error(f"{description} 失败，返回码 {e.returncode}")
        return False


def validate_directories(output_base):
    """验证和创建必要的目录结构"""
    LOG.info("验证和创建输出目录结构...")

    required_dirs = {
        'annotation_results': os.path.join(output_base, "annotation_results"),
        'cleaned_proteins': os.path.join(output_base, "cleaned_proteins"),
        'interproscan_results': os.path.join(output_base, "annotation_results", "interproscan"),
        'kegg_results': os.path.join(output_base, "annotation_results", "kegg"),
        'cazy_results': os.path.join(output_base, "annotation_results", "cazy"),
        'abundance_results': os.path.join(output_base, "annotation_results", "abundance"),
        'pathway_results': os.path.join(output_base, "annotation_results", "pathway_abundance"),
        'integrated_results': os.path.join(output_base, "annotation_results", "integrated")
    }

    for dir_name, dir_path in required_dirs.items():
        os.makedirs(dir_path, exist_ok=True)
        LOG.info(f"✓ 确保目录存在: {dir_path}")

    return required_dirs


def check_database_paths():
    """检查数据库路径并返回正确的路径"""
    LOG.info("检查数据库路径...")

    database_paths = {}

    # 设置工具路径
    database_paths['interproscan_path'] = "/root/tools/interproscan/interproscan-5.28-67.0/interproscan.sh"

    # InterProScan数据库路径
    interproscan_base = "/root/tools/interproscan/interproscan-5.28-67.0"
    database_paths['pfam_db'] = os.path.join(interproscan_base, "data/pfam/31.0/pfam_a.hmm")
    database_paths['smart_db'] = os.path.join(interproscan_base, "data/smart/7.1/smart.HMMs")
    database_paths['tigrfam_db'] = os.path.join(interproscan_base, "data/tigrfam/15.0/TIGRFAMs_HMM.LIB")

    # 检查InterProScan数据库
    for db_name, db_path in [('Pfam', database_paths['pfam_db']),
                             ('SMART', database_paths['smart_db']),
                             ('TIGRFAM', database_paths['tigrfam_db'])]:
        if os.path.exists(db_path):
            LOG.info(f"✓ InterProScan {db_name}数据库: {db_path}")
        else:
            LOG.warning(f"⚠ InterProScan {db_name}数据库不存在: {db_path}")

    # KEGG数据库路径 - 使用实际路径
    kegg_base = "/mnt/databases/kegg/2025"
    if os.path.exists(kegg_base):
        database_paths['ko_list'] = os.path.join(kegg_base, "ko_list")
        database_paths['kegg_hierarchy'] = os.path.join(kegg_base, "ko00001.tsv")
        database_paths['profile_db'] = os.path.join(kegg_base, "profiles")

        # 检查文件是否存在
        for key, path in [('ko_list_old.tsv', database_paths['ko_list']),
                          ('kegg_hierarchy', database_paths['kegg_hierarchy']),
                          ('profile_db', database_paths['profile_db'])]:
            if os.path.exists(path):
                LOG.info(f"✓ KEGG {key}: {path}")
            else:
                LOG.warning(f"⚠ KEGG {key} 不存在: {path}")
    else:
        LOG.warning(f"⚠ KEGG基础目录不存在: {kegg_base}")

    # CAZy数据库路径 - 使用实际路径
    cazy_base = "/mnt/databases/cazy_db"
    if os.path.exists(cazy_base):
        database_paths['cazy_db'] = os.path.join(cazy_base, "cazy.dmnd")

        if os.path.exists(database_paths['cazy_db']):
            LOG.info(f"✓ CAZy数据库: {database_paths['cazy_db']}")
        else:
            LOG.warning(f"⚠ CAZy数据库不存在: {database_paths['cazy_db']}")
    else:
        LOG.warning(f"⚠ CAZy基础目录不存在: {cazy_base}")

    # 代谢通路数据库路径 - 使用实际路径
    database_paths['pathway_db'] = "/mnt/databases/kegg/2025/ko00001.tsv"
    if os.path.exists(database_paths['pathway_db']):
        LOG.info(f"✓ 代谢通路数据库: {database_paths['pathway_db']}")
    else:
        LOG.warning(f"⚠ 代谢通路数据库不存在: {database_paths['pathway_db']}")

    return database_paths


def get_sample_directories():
    """获取样本目录列表"""
    gene_dir = "/home/zjw/zjwdata/1/gene_catalog_analysis/per_sample_non_redundant_genes/"

    if not os.path.exists(gene_dir):
        LOG.error(f"基因目录不存在: {gene_dir}")
        return []

    samples = []
    for item in os.listdir(gene_dir):
        item_path = os.path.join(gene_dir, item)
        if os.path.isdir(item_path) and item not in ['per_sample_statistics.csv']:
            samples.append(item)

    LOG.info(f"找到 {len(samples)} 个样本: {samples}")
    return sorted(samples)


def find_protein_files(sample):
    """查找样本的蛋白文件"""
    gene_dir = "/home/zjw/zjwdata/1/gene_catalog_analysis/per_sample_non_redundant_genes/"
    sample_dir = os.path.join(gene_dir, sample)

    if not os.path.exists(sample_dir):
        LOG.error(f"样本目录不存在: {sample_dir}")
        return None

    # 查找可能的蛋白文件
    patterns = [
        f"{sample}_non_redundant_genes.faa",
        f"{sample}.faa",
        f"{sample}.fasta",
        f"{sample}.fa",
        "non_redundant_genes.faa",
        "genes.faa",
        "*.faa",
        "*.fasta"
    ]

    for pattern in patterns:
        files = glob.glob(os.path.join(sample_dir, pattern))
        if files:
            LOG.info(f"为样本 {sample} 找到蛋白文件: {files[0]}")
            return files[0]

    LOG.warning(f"在 {sample_dir} 中未找到蛋白文件")
    return None


def count_sequences(fasta_file):
    """统计FASTA文件中的序列数量"""
    if not fasta_file or not os.path.exists(fasta_file):
        return 0

    count = 0
    try:
        with open(fasta_file, 'r') as f:
            for line in f:
                if line.startswith('>'):
                    count += 1
        return count
    except Exception as e:
        LOG.warning(f"无法统计序列数量: {e}")
        return 0


def check_interproscan_databases(db_paths):
    """检查InterProScan数据库是否存在"""
    LOG.info("检查InterProScan数据库...")

    required_dbs = {
        'Pfam': db_paths.get('pfam_db'),
        'SMART': db_paths.get('smart_db'),
        'TIGRFAM': db_paths.get('tigrfam_db')
    }

    missing_dbs = []
    for db_name, db_path in required_dbs.items():
        if not db_path or not os.path.exists(db_path):
            missing_dbs.append(db_name)
            LOG.error(f"InterProScan {db_name}数据库不存在: {db_path}")
        else:
            LOG.info(f"InterProScan {db_name}数据库检查通过: {db_path}")

    if missing_dbs:
        LOG.error(f"缺少InterProScan数据库: {', '.join(missing_dbs)}")
        return False

    LOG.info("所有InterProScan数据库检查通过")
    return True


def setup_interproscan_environment_fixed():
    """修正InterProScan环境设置"""
    # 清理可能干扰的环境变量
    if 'JAVA_TOOL_OPTIONS' in os.environ:
        del os.environ['JAVA_TOOL_OPTIONS']
        LOG.info("已清理 JAVA_TOOL_OPTIONS 环境变量")

    # 设置InterProScan特定的环境变量
    os.environ['INTERPROSCAN_HOME'] = '/root/tools/interproscan/interproscan-5.28-67.0'

    # 设置JVM内存参数 - 增加到96GB
    os.environ['JAVA_OPTS'] = '-Xmx96g -Xms32g -XX:ParallelGCThreads=32'

    # 添加Java 8到PATH
    java_path = '/usr/lib/jvm/java-8-openjdk-amd64/jre/bin'
    if java_path not in os.environ.get('PATH', ''):
        os.environ['PATH'] = f"{java_path}:{os.environ.get('PATH', '')}"
        LOG.info(f"已添加Java 8到PATH: {java_path}")


def run_interproscan_annotation(sample, output_base, db_paths, status_tracker, force=False):
    """运行InterProScan注释 - 修正版"""
    LOG.info(f"为样本 {sample} 运行InterProScan注释")

    # 检查是否已经完成且不需要强制重运行
    if not force and status_tracker.check_step_completed(sample, 'interproscan', check_file=True):
        LOG.info(f"样本 {sample} 的InterProScan注释已完成，跳过")
        return True

    # 检查InterProScan数据库
    if not check_interproscan_databases(db_paths):
        LOG.error(f"InterProScan数据库检查失败，跳过样本 {sample}")
        status_tracker.update_step_status(sample, 'interproscan', False)
        return False

    protein_file = find_protein_files(sample)
    if not protein_file:
        LOG.error(f"样本 {sample} 未找到蛋白文件，跳过InterProScan注释")
        status_tracker.update_step_status(sample, 'interproscan', False)
        return False

    # 创建注释结果目录
    output_dir = os.path.join(output_base, "annotation_results", "interproscan", sample)
    os.makedirs(output_dir, exist_ok=True)

    # 创建清理蛋白序列目录
    cleaned_dir = os.path.join(output_base, "cleaned_proteins")
    os.makedirs(cleaned_dir, exist_ok=True)

    # 预处理：清理蛋白序列文件
    cleaned_protein_file = os.path.join(cleaned_dir, f"{sample}.cleaned.faa")
    if not clean_protein_sequences(protein_file, cleaned_protein_file):
        LOG.error(f"清理蛋白序列文件失败，使用原始文件")
        cleaned_protein_file = protein_file
    else:
        LOG.info(f"使用清理后的蛋白文件: {cleaned_protein_file}")

    # 设置修正版InterProScan环境
    setup_interproscan_environment_fixed()

    # 为每个样本创建单独的日志文件
    sample_log_file = os.path.join(output_dir, f"{sample}.interproscan.log")

    # 使用修正版的独立脚本运行InterProScan，取消超时
    cmd = f"""
    python /home/zjw/Projects2/gene_function/interproscan_standalone.py \
      -i {cleaned_protein_file} \
      -o {output_dir} \
      -p {sample} \
      -t 32 \
      --log {sample_log_file}
    """

    success = run_command(cmd, f"InterProScan注释 - {sample}")

    if success:
        output_file = os.path.join(output_dir, f"{sample}.simplified_annotations.tsv")
        if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
            LOG.info(f"样本 {sample} 的InterProScan注释完成")
            status_tracker.update_step_status(sample, 'interproscan', True, output_file)

            # 读取注释率报告
            rate_file = os.path.join(output_dir, f"{sample}.annotation_rate.txt")
            if os.path.exists(rate_file):
                with open(rate_file, 'r') as f:
                    for line in f:
                        if "注释率:" in line:
                            rate = line.split(":")[1].strip()
                            LOG.info(f"InterProScan注释率: {rate}")
        else:
            LOG.error(f"样本 {sample} 的InterProScan输出文件无效")
            status_tracker.update_step_status(sample, 'interproscan', False)
            success = False
    else:
        LOG.error(f"样本 {sample} 的InterProScan注释失败")
        status_tracker.update_step_status(sample, 'interproscan', False)

    return success


def run_kegg_annotation(sample, output_base, db_paths, status_tracker, force=False):
    """运行KEGG注释"""
    LOG.info(f"为样本 {sample} 运行KEGG注释")

    # 检查是否已经完成且不需要强制重运行
    if not force and status_tracker.check_step_completed(sample, 'kegg', check_file=True):
        LOG.info(f"样本 {sample} 的KEGG注释已完成，跳过")
        return True

    protein_file = find_protein_files(sample)
    if not protein_file:
        LOG.error(f"样本 {sample} 未找到蛋白文件，跳过KEGG注释")
        status_tracker.update_step_status(sample, 'kegg', False)
        return False

    # 创建注释结果目录
    output_dir = os.path.join(output_base, "annotation_results", "kegg", sample)
    os.makedirs(output_dir, exist_ok=True)

    # 创建清理蛋白序列目录
    cleaned_dir = os.path.join(output_base, "cleaned_proteins")
    os.makedirs(cleaned_dir, exist_ok=True)

    # 预处理：清理蛋白序列文件
    cleaned_protein_file = os.path.join(cleaned_dir, f"{sample}.cleaned.faa")
    if not clean_protein_sequences(protein_file, cleaned_protein_file):
        LOG.error(f"清理蛋白序列文件失败，使用原始文件")
        cleaned_protein_file = protein_file
    else:
        LOG.info(f"使用清理后的蛋白文件: {cleaned_protein_file}")

    # 检查KEGG数据库路径
    if not db_paths.get('profile_db') or not os.path.exists(db_paths['profile_db']):
        LOG.error("KEGG profile数据库不存在，跳过KEGG注释")
        status_tracker.update_step_status(sample, 'kegg', False)
        return False
    if not db_paths.get('ko_list') or not os.path.exists(db_paths['ko_list']):
        LOG.error("KO列表文件不存在，跳过KEGG注释")
        status_tracker.update_step_status(sample, 'kegg', False)
        return False

    # 统一使用 .kegg.out 作为输出文件名
    kegg_output_file = os.path.join(output_dir, f"{sample}.kegg.out")

    cmd = f"""
    exec_annotation -o {kegg_output_file} \
      --profile {db_paths['profile_db']} \
      --ko-list {db_paths['ko_list']} \
      --cpu 32 \
      --e-value 1e-05 \
      --format detail-tsv \
      {cleaned_protein_file}
    """

    success = run_command(cmd, f"KEGG注释 - {sample}")
    if success:
        LOG.info(f"样本 {sample} 的KEGG注释完成")

        # 处理结果 - 为每个样本创建单独的日志文件
        sample_log_file = os.path.join(output_dir, f"{sample}.kegg.log")

        process_cmd = f"""
        python /home/zjw/Projects2/gene_function/kegg_annotation.py \
          --parse_only \
          -i {cleaned_protein_file} \
          --output {output_dir} \
          --prefix {sample} \
          --ko_list {db_paths['ko_list']} \
          --kegg_hierarchy {db_paths.get('kegg_hierarchy', '')} \
          --log {sample_log_file}
        """
        success = run_command(process_cmd, f"处理KEGG结果 - {sample}")

        if success:
            output_file = os.path.join(output_dir, f"{sample}.kegg_annotations.tsv")
            if os.path.exists(output_file):
                file_size = os.path.getsize(output_file)
                if file_size > 0:
                    status_tracker.update_step_status(sample, 'kegg', True, output_file)

                    # 读取注释率报告
                    rate_file = os.path.join(output_dir, f"{sample}.annotation_rate.txt")
                    if os.path.exists(rate_file):
                        with open(rate_file, 'r') as f:
                            for line in f:
                                if "注释率:" in line:
                                    rate = line.split(":")[1].strip()
                                    LOG.info(f"KEGG注释率: {rate}")
                else:
                    # 输出文件为空，但可能确实是没注释到
                    LOG.warning(f"样本 {sample} 的KEGG注释结果为空")
                    status_tracker.update_step_status(sample, 'kegg', True, output_file)
            else:
                status_tracker.update_step_status(sample, 'kegg', False)
                success = False
        else:
            status_tracker.update_step_status(sample, 'kegg', False)
    else:
        LOG.error(f"样本 {sample} 的KEGG注释失败")
        status_tracker.update_step_status(sample, 'kegg', False)

    return success


def filter_blast_results(blast_file, output_file, evalue=1e-5, coverage=30):
    """过滤BLAST结果"""
    LOG.info(f"过滤BLAST结果: {blast_file} -> {output_file}")

    if not os.path.exists(blast_file):
        LOG.error(f"BLAST文件不存在: {blast_file}")
        return False

    file_size = os.path.getsize(blast_file)
    if file_size == 0:
        LOG.warning(f"BLAST文件为空: {blast_file}")
        # 创建空的过滤文件
        with open(output_file, 'w') as f:
            pass
        LOG.info(f"创建空的过滤文件: {output_file}")
        return True

    filtered_count = 0
    total_count = 0

    try:
        with open(blast_file, 'r') as infile, open(output_file, 'w') as outfile:
            for line in infile:
                total_count += 1
                parts = line.strip().split('\t')
                if len(parts) < 14:  # 需要包含stitle列
                    continue

                qseqid, sseqid, pident, length, mismatch, gapopen, qstart, qend, sstart, send, evalue_val, bitscore, qlen, slen = parts[
                                                                                                                                  :14]
                stitle = parts[14] if len(parts) > 14 else ""

                # 计算查询覆盖度
                try:
                    qcov = (int(qend) - int(qstart) + 1) / int(qlen) * 100
                except (ValueError, ZeroDivisionError):
                    continue

                # 过滤条件
                if float(evalue_val) <= evalue and qcov >= coverage:
                    outfile.write(line)
                    filtered_count += 1

        LOG.info(f"BLAST结果过滤完成: {filtered_count}/{total_count} 条记录通过过滤")
        if filtered_count == 0:
            LOG.warning("没有记录通过过滤条件")
        return True
    except Exception as e:
        LOG.error(f"过滤BLAST结果失败: {e}")
        return False


def run_cazy_annotation(sample, output_base, db_paths, status_tracker, force=False):
    """运行CAZy注释 - 使用修正版参数传递"""
    LOG.info(f"为样本 {sample} 运行CAZy注释")

    # 检查是否已经完成且不需要强制重运行
    if not force and status_tracker.check_step_completed(sample, 'cazy', check_file=True):
        LOG.info(f"样本 {sample} 的CAZy注释已完成，跳过")
        return True

    protein_file = find_protein_files(sample)
    if not protein_file:
        LOG.error(f"样本 {sample} 未找到蛋白文件，跳过CAZy注释")
        status_tracker.update_step_status(sample, 'cazy', False)
        return False

    # 创建注释结果目录
    output_dir = os.path.join(output_base, "annotation_results", "cazy", sample)
    os.makedirs(output_dir, exist_ok=True)

    # 创建清理蛋白序列目录
    cleaned_dir = os.path.join(output_base, "cleaned_proteins")
    os.makedirs(cleaned_dir, exist_ok=True)

    # 预处理：清理蛋白序列文件
    cleaned_protein_file = os.path.join(cleaned_dir, f"{sample}.cleaned.faa")
    if not clean_protein_sequences(protein_file, cleaned_protein_file):
        LOG.error(f"清理蛋白序列文件失败，使用原始文件")
        cleaned_protein_file = protein_file
    else:
        LOG.info(f"使用清理后的蛋白文件: {cleaned_protein_file}")

    # 检查CAZy数据库路径
    if not db_paths.get('cazy_db') or not os.path.exists(db_paths['cazy_db']):
        LOG.error("CAZy数据库不存在，跳过CAZy注释")
        status_tracker.update_step_status(sample, 'cazy', False)
        return False

    # 运行diamond
    raw_output = os.path.join(output_dir, f"{sample}.cazy.m6")
    filtered_output = os.path.join(output_dir, f"{sample}.cazy.filtered.m6")

    cmd = f"""
    diamond blastp \
      --query {cleaned_protein_file} \
      --db {db_paths['cazy_db']} \
      --outfmt 6 qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore qlen slen stitle \
      --max-target-seqs 5 \
      --evalue 1e-05 \
      --threads 32 \
      --out {raw_output}
    """

    success = run_command(cmd, f"CAZy注释 - {sample}")
    if success:
        LOG.info(f"样本 {sample} 的CAZy比对完成")

        # 运行过滤步骤
        if not os.path.exists(filtered_output):
            LOG.info(f"运行过滤步骤: {raw_output} -> {filtered_output}")
            if not filter_blast_results(raw_output, filtered_output, 1e-5, 30):
                LOG.error("结果过滤失败")
                status_tracker.update_step_status(sample, 'cazy', False)
                return False

        # 处理结果 - 为每个样本创建单独的日志文件
        sample_log_file = os.path.join(output_dir, f"{sample}.cazy.log")

        # 使用修正版参数调用CAZy脚本
        process_cmd = f"""
        python /home/zjw/Projects2/gene_function/cazy_annotation.py \
          --parse_only \
          -i {cleaned_protein_file} \
          -b {filtered_output} \
          --output {output_dir} \
          --prefix {sample} \
          --evalue 1e-05 \
          --coverage 30 \
          --log {sample_log_file}
        """
        success = run_command(process_cmd, f"处理CAZy结果 - {sample}")

        if success:
            output_file = os.path.join(output_dir, f"{sample}.cazy_annotations.tsv")
            if os.path.exists(output_file):
                file_size = os.path.getsize(output_file)
                if file_size > 0:
                    status_tracker.update_step_status(sample, 'cazy', True, output_file)

                    # 读取注释率报告
                    rate_file = os.path.join(output_dir, f"{sample}.annotation_rate.txt")
                    if os.path.exists(rate_file):
                        with open(rate_file, 'r') as f:
                            for line in f:
                                if "注释率:" in line:
                                    rate = line.split(":")[1].strip()
                                    LOG.info(f"CAZy注释率: {rate}")
                else:
                    # 输出文件为空，但可能确实是没注释到
                    LOG.warning(f"样本 {sample} 的CAZy注释结果为空")
                    status_tracker.update_step_status(sample, 'cazy', True, output_file)
            else:
                status_tracker.update_step_status(sample, 'cazy', False)
                success = False
        else:
            status_tracker.update_step_status(sample, 'cazy', False)
    else:
        LOG.error(f"样本 {sample} 的CAZy注释失败")
        status_tracker.update_step_status(sample, 'cazy', False)

    return success


def check_salmon_available():
    """检查salmon是否可用"""
    try:
        # 方法1：直接使用which命令
        result = subprocess.run(['which', 'salmon'], capture_output=True, text=True)
        if result.returncode == 0:
            salmon_path = result.stdout.strip()
            LOG.info(f"✓ salmon可用: {salmon_path}")

            # 获取salmon版本
            version_result = subprocess.run(['salmon', '--version'], capture_output=True, text=True)
            if version_result.returncode == 0:
                LOG.info(f"  salmon版本: {version_result.stdout.strip()}")

            return True
        else:
            LOG.warning("✗ salmon不可用")
            return False
    except Exception as e:
        LOG.error(f"检查salmon失败: {e}")
        return False


def calculate_gene_abundance(sample, output_base, db_paths, status_tracker, force=False):
    """计算基因丰度（使用salmon进行真实丰度计算）"""
    LOG.info(f"为样本 {sample} 计算基因丰度（使用salmon）")

    # 检查是否已经完成且不需要强制重运行
    if not force and status_tracker.check_step_completed(sample, 'abundance', check_file=True):
        LOG.info(f"样本 {sample} 的基因丰度计算已完成，跳过")
        return True

    # 检查salmon是否可用
    if not check_salmon_available():
        LOG.error("salmon工具未找到，请先安装salmon")
        LOG.info("安装命令: conda install -c bioconda salmon")
        status_tracker.update_step_status(sample, 'abundance', False)
        return False

    # 1. 查找文件
    # 基因文件（CDS序列）- 使用DNA序列文件
    gene_dir = "/home/zjw/zjwdata/1/gene_catalog_analysis/per_sample_non_redundant_genes/"
    gene_file = os.path.join(gene_dir, sample, f"{sample}_non_redundant_genes.fna")

    # 测序数据文件 - 使用原始数据
    raw_data_dir = "/mnt/zjwdata/1/corn_silage_qc_analysis/cleaned_data/"
    r1_file = os.path.join(raw_data_dir, f"{sample}_R1_clean.fq.gz")
    r2_file = os.path.join(raw_data_dir, f"{sample}_R2_clean.fq.gz")

    # 检查文件是否存在
    missing_files = []
    for file_path, file_name in [(gene_file, "基因序列文件"),
                                 (r1_file, "测序R1文件"),
                                 (r2_file, "测序R2文件")]:
        if not os.path.exists(file_path):
            missing_files.append(f"{file_name}: {file_path}")

    if missing_files:
        LOG.error(f"缺少必要文件，跳过丰度计算:")
        for msg in missing_files:
            LOG.error(f"  {msg}")
        status_tracker.update_step_status(sample, 'abundance', False)
        return False

    # 创建输出目录
    output_dir = os.path.join(output_base, "annotation_results", "abundance", sample)
    os.makedirs(output_dir, exist_ok=True)

    try:
        LOG.info(f"开始使用salmon计算样本 {sample} 的基因丰度")

        # 2. 创建salmon索引
        LOG.info(f"步骤1: 创建salmon索引")
        salmon_index_dir = os.path.join(output_dir, "salmon_index")

        # 检查是否已经有索引
        if not os.path.exists(salmon_index_dir) or not os.path.exists(
                os.path.join(salmon_index_dir, "complete_ref_lens.bin")):
            cmd = f"""
            salmon index -t {gene_file} \
              -i {salmon_index_dir} \
              -k 31 \
              -p 32
            """

            if not run_command(cmd, f"创建salmon索引 - {sample}"):
                LOG.error(f"创建salmon索引失败")
                return False
        else:
            LOG.info(f"salmon索引已存在，跳过创建")

        # 3. 运行salmon定量
        LOG.info(f"步骤2: 运行salmon定量")
        salmon_quant_dir = os.path.join(output_dir, "salmon_quant")

        cmd = f"""
        salmon quant -i {salmon_index_dir} \
          -l A \
          -1 {r1_file} \
          -2 {r2_file} \
          -p 32 \
          --validateMappings \
          --gcBias \
          --seqBias \
          -o {salmon_quant_dir}
        """

        if not run_command(cmd, f"salmon定量 - {sample}"):
            LOG.error(f"salmon定量失败")
            return False

        # 4. 读取salmon结果
        LOG.info(f"步骤3: 解析salmon结果")
        quant_file = os.path.join(salmon_quant_dir, "quant.sf")

        if not os.path.exists(quant_file):
            LOG.error(f"salmon输出文件不存在: {quant_file}")
            return False

        # 读取quant.sf文件
        gene_data = {}
        total_tpm = 0
        total_reads = 0

        with open(quant_file, 'r') as f:
            header = f.readline().strip().split('\t')
            # salmon输出格式: Name Length EffectiveLength TPM NumReads
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 5:
                    gene_id = parts[0]
                    length = float(parts[1])
                    effective_length = float(parts[2])
                    tpm = float(parts[3])
                    num_reads = float(parts[4])

                    gene_data[gene_id] = {
                        'TPM': tpm,
                        'NumReads': num_reads,
                        'Length': length,
                        'EffLength': effective_length
                    }

                    total_tpm += tpm
                    total_reads += num_reads

        LOG.info(f"解析完成: 检测到 {len(gene_data)} 个基因")
        LOG.info(f"总TPM: {total_tpm:.2f}")
        LOG.info(f"总比对reads数: {total_reads:.0f}")

        # 5. 生成丰度文件
        abundance_file = os.path.join(output_dir, f"{sample}.gene_abundance.tsv")

        with open(abundance_file, 'w') as f:
            f.write("GeneID\tTPM\tNumReads\tLength\tEffLength\n")

            for gene_id, data in gene_data.items():
                f.write(f"{gene_id}\t{data['TPM']:.6f}\t{data['NumReads']:.2f}\t")
                f.write(f"{data['Length']:.2f}\t{data['EffLength']:.2f}\n")

        # 6. 生成质量控制报告
        LOG.info(f"步骤4: 生成质量控制报告")
        qc_report = generate_qc_report(sample, salmon_quant_dir, gene_data, output_dir)

        # 7. 生成丰度分布图
        LOG.info(f"步骤5: 生成丰度分布图")
        generate_abundance_plots(sample, gene_data, output_dir)

        LOG.info(f"样本 {sample} 的基因丰度计算完成")
        LOG.info(f"  检测到基因数: {len(gene_data)}")
        LOG.info(f"  总TPM: {total_tpm:.2f}")
        LOG.info(f"  总比对reads: {total_reads:.0f}")

        status_tracker.update_step_status(sample, 'abundance', True, abundance_file)
        return True

    except Exception as e:
        LOG.error(f"使用salmon计算基因丰度失败: {e}")
        import traceback
        LOG.error(traceback.format_exc())
        status_tracker.update_step_status(sample, 'abundance', False)
        return False


def generate_qc_report(sample, salmon_quant_dir, gene_data, output_dir):
    """生成质量控制报告"""
    try:
        # 读取salmon的日志文件
        cmd_info_file = os.path.join(salmon_quant_dir, "cmd_info.json")
        lib_format_file = os.path.join(salmon_quant_dir, "lib_format_counts.json")

        qc_report_file = os.path.join(output_dir, f"{sample}.qc_report.txt")

        with open(qc_report_file, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write(f"样本 {sample} 质量控制报告\n")
            f.write("=" * 60 + "\n\n")

            f.write("1. 基因丰度统计:\n")
            f.write(f"   检测到的基因数: {len(gene_data)}\n")

            # 统计表达基因（TPM > 0）
            expressed_genes = sum(1 for data in gene_data.values() if data['TPM'] > 0)
            f.write(f"   表达基因数 (TPM > 0): {expressed_genes}\n")

            # 统计高表达基因（TPM > 10）
            high_expressed = sum(1 for data in gene_data.values() if data['TPM'] > 10)
            f.write(f"   高表达基因数 (TPM > 10): {high_expressed}\n")

            # 计算TPM分布
            tpm_values = [data['TPM'] for data in gene_data.values()]
            if tpm_values:
                f.write(f"   TPM中位数: {np.median(tpm_values):.4f}\n")
                f.write(f"   TPM平均值: {np.mean(tpm_values):.4f}\n")
                f.write(f"   TPM最大值: {np.max(tpm_values):.2f}\n")
                f.write(f"   TPM最小值: {np.min(tpm_values):.6f}\n")

            # 计算reads分布
            reads_values = [data['NumReads'] for data in gene_data.values()]
            if reads_values:
                f.write(f"   比对reads中位数: {np.median(reads_values):.2f}\n")
                f.write(f"   比对reads平均值: {np.mean(reads_values):.2f}\n")
                f.write(f"   比对reads最大值: {np.max(reads_values):.0f}\n")
                f.write(f"   总比对reads数: {sum(reads_values):.0f}\n")

            f.write("\n2. 表达水平分布:\n")

            # 按表达水平分类
            tpm_ranges = [(0, 0.1), (0.1, 1), (1, 10), (10, 100), (100, 1000), (1000, float('inf'))]
            range_labels = ["0-0.1", "0.1-1", "1-10", "10-100", "100-1000", ">1000"]

            for (low, high), label in zip(tpm_ranges, range_labels):
                count = sum(1 for data in gene_data.values() if low <= data['TPM'] < high)
                percentage = (count / len(gene_data)) * 100 if gene_data else 0
                f.write(f"   TPM {label}: {count} 个基因 ({percentage:.1f}%)\n")

            f.write("\n3. 序列特征:\n")

            # 基因长度统计
            lengths = [data['Length'] for data in gene_data.values()]
            if lengths:
                f.write(f"   基因长度中位数: {np.median(lengths):.0f} bp\n")
                f.write(f"   基因长度平均值: {np.mean(lengths):.0f} bp\n")
                f.write(f"   最短基因: {np.min(lengths):.0f} bp\n")
                f.write(f"   最长基因: {np.max(lengths):.0f} bp\n")

            # 有效长度统计
            eff_lengths = [data['EffLength'] for data in gene_data.values()]
            if eff_lengths:
                f.write(f"   有效长度中位数: {np.median(eff_lengths):.0f} bp\n")
                f.write(f"   有效长度平均值: {np.mean(eff_lengths):.0f} bp\n")

            # 读取salmon日志信息
            if os.path.exists(cmd_info_file):
                f.write("\n4. Salmon运行信息:\n")
                try:
                    with open(cmd_info_file, 'r') as cmd_f:
                        import json as json_module
                        cmd_info = json_module.load(cmd_f)
                        f.write(f"   运行命令: {cmd_info.get('cmd', 'N/A')}\n")
                except:
                    f.write("   无法读取cmd_info.json\n")

            if os.path.exists(lib_format_file):
                f.write("\n5. 文库格式统计:\n")
                try:
                    with open(lib_format_file, 'r') as lib_f:
                        import json as json_module
                        lib_info = json_module.load(lib_f)
                        f.write(f"   预期格式: {lib_info.get('expected_format', 'N/A')}\n")
                        f.write(f"   兼容对: {lib_info.get('compatible_fragments', 'N/A')}\n")
                except:
                    f.write("   无法读取lib_format_counts.json\n")

        LOG.info(f"质量控制报告生成完成: {qc_report_file}")
        return qc_report_file

    except Exception as e:
        LOG.error(f"生成质量控制报告失败: {e}")
        return None


def generate_abundance_plots(sample, gene_data, output_dir):
    """生成丰度分布图 - 修复版"""
    try:
        if not gene_data:
            LOG.warning("没有基因数据，跳过绘图")
            return

        # 设置字体 - 使用修复版
        setup_chinese_font()

        # 准备数据
        tpm_values = [data['TPM'] for data in gene_data.values() if data['TPM'] > 0]
        reads_values = [data['NumReads'] for data in gene_data.values() if data['NumReads'] > 0]

        if not tpm_values:
            LOG.warning("没有TPM数据，跳过绘图")
            return

        # 设置绘图风格
        plt.style.use('default')  # 使用默认风格，避免seaborn可能的问题

        # 创建图形
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. TPM分布直方图
        ax1 = axes[0, 0]
        log_tpm = np.log10([tpm + 1e-6 for tpm in tpm_values])
        ax1.hist(log_tpm, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel('log10(TPM+1e-6)')
        ax1.set_ylabel('Gene Count')
        ax1.set_title('TPM Distribution')
        ax1.grid(True, alpha=0.3)

        # 2. 累积分布图
        ax2 = axes[0, 1]
        sorted_tpm = np.sort(tpm_values)
        cum_frac = np.arange(1, len(sorted_tpm) + 1) / len(sorted_tpm)
        ax2.plot(sorted_tpm, cum_frac, 'b-', linewidth=2)
        ax2.set_xlabel('TPM')
        ax2.set_ylabel('Cumulative Fraction')
        ax2.set_title('TPM Cumulative Distribution')
        ax2.set_xscale('log')
        ax2.grid(True, alpha=0.3)

        # 3. 散点图：基因长度 vs TPM
        ax3 = axes[1, 0]
        lengths = [data['Length'] for data in gene_data.values() if data['TPM'] > 0]
        valid_tpm = [data['TPM'] for data in gene_data.values() if data['TPM'] > 0]

        if lengths and len(lengths) == len(valid_tpm):
            ax3.scatter(lengths, valid_tpm, alpha=0.5, s=10, color='green')
            ax3.set_xlabel('Gene Length (bp)')
            ax3.set_ylabel('TPM')
            ax3.set_title('Gene Length vs TPM')
            ax3.set_xscale('log')
            ax3.set_yscale('log')
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'No Data', ha='center', va='center')
            ax3.set_title('Gene Length vs TPM')

        # 4. 排名图：Top 20高表达基因
        ax4 = axes[1, 1]
        top_n = min(20, len(gene_data))
        sorted_genes = sorted(gene_data.items(), key=lambda x: x[1]['TPM'], reverse=True)[:top_n]

        if sorted_genes:
            gene_names = [gene[0] for gene in sorted_genes]
            tpm_values_top = [gene[1]['TPM'] for gene in sorted_genes]

            # 简化基因名显示
            short_names = []
            for name in gene_names:
                if len(name) > 20:
                    short_names.append(name[:17] + '...')
                else:
                    short_names.append(name)

            bars = ax4.barh(range(top_n), tpm_values_top, color='orange')
            ax4.set_yticks(range(top_n))
            ax4.set_yticklabels(short_names, fontsize=8)
            ax4.invert_yaxis()  # 从高到低显示
            ax4.set_xlabel('TPM')
            ax4.set_title(f'Top {top_n} Highly Expressed Genes')
            ax4.grid(True, alpha=0.3, axis='x')

            # 添加数值标签
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax4.text(width * 1.01, bar.get_y() + bar.get_height() / 2,
                         f'{width:.1f}', va='center', fontsize=8)
        else:
            ax4.text(0.5, 0.5, 'No Data', ha='center', va='center')
            ax4.set_title('Top 20 Highly Expressed Genes')

        # 使用英文标题避免字体问题
        plt.suptitle(f'Sample {sample} Gene Abundance Distribution',
                     fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # 保存图片
        plot_file = os.path.join(output_dir, f"{sample}.abundance_distribution.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close(fig)

        LOG.info(f"丰度分布图生成完成: {plot_file}")
        return plot_file

    except Exception as e:
        LOG.error(f"生成丰度分布图失败: {e}")
        import traceback
        LOG.error(traceback.format_exc())
        return None


def run_pathway_abundance_analysis(sample, output_base, db_paths, status_tracker, force=False):
    """运行代谢通路丰度分析 - 使用真实TPM值"""
    LOG.info(f"为样本 {sample} 运行代谢通路丰度分析（使用真实TPM值）")

    # 检查是否已经完成且不需要强制重运行
    if not force and status_tracker.check_step_completed(sample, 'pathway', check_file=True):
        LOG.info(f"样本 {sample} 的代谢通路丰度分析已完成，跳过")
        return True

    # 检查必要的输入文件
    kegg_annotation_file = os.path.join(output_base, "annotation_results", "kegg", sample,
                                        f"{sample}.kegg_annotations.tsv")
    abundance_file = os.path.join(output_base, "annotation_results", "abundance", sample,
                                  f"{sample}.gene_abundance.tsv")

    if not os.path.exists(kegg_annotation_file):
        LOG.error(f"KEGG注释文件不存在: {kegg_annotation_file}")
        status_tracker.update_step_status(sample, 'pathway', False)
        return False

    if not os.path.exists(abundance_file):
        LOG.error(f"基因丰度文件不存在: {abundance_file}")
        status_tracker.update_step_status(sample, 'pathway', False)
        return False

    # 创建代谢通路分析结果目录
    output_dir = os.path.join(output_base, "annotation_results", "pathway_abundance", sample)
    os.makedirs(output_dir, exist_ok=True)

    # 检查代谢通路数据库
    if not db_paths.get('pathway_db') or not os.path.exists(db_paths['pathway_db']):
        LOG.error("代谢通路数据库不存在，跳过代谢通路丰度分析")
        status_tracker.update_step_status(sample, 'pathway', False)
        return False

    # 为每个样本创建单独的日志文件
    sample_log_file = os.path.join(output_dir, f"{sample}.pathway.log")

    # 使用代谢通路丰度分析脚本 - 现在使用真实TPM值
    cmd = f"""
    python /home/zjw/Projects2/gene_function/pathway_abundance.py \
      --gene_abundance {abundance_file} \
      --kegg_annotations {kegg_annotation_file} \
      --pathway_db {db_paths['pathway_db']} \
      -o {output_dir} \
      -p {sample} \
      --method sum \
      --normalization none \
      --abundance_column TPM \
      --log {sample_log_file}
    """

    success = run_command(cmd, f"代谢通路丰度分析 - {sample}")

    if success:
        output_file = os.path.join(output_dir, f"{sample}.pathway_abundance.tsv")
        if os.path.exists(output_file):
            file_size = os.path.getsize(output_file)
            if file_size > 100:  # 空文件通常很小
                LOG.info(f"样本 {sample} 的代谢通路丰度分析完成")

                # 读取通路丰度结果并生成统计
                generate_pathway_stats(sample, output_dir, output_file)

                status_tracker.update_step_status(sample, 'pathway', True, output_file)
            else:
                # 输出文件很小，可能没有数据
                LOG.warning(f"样本 {sample} 的代谢通路丰度分析结果可能为空")
                status_tracker.update_step_status(sample, 'pathway', True, output_file)
        else:
            LOG.error(f"样本 {sample} 的代谢通路丰度分析输出文件无效")
            status_tracker.update_step_status(sample, 'pathway', False)
            success = False
    else:
        LOG.error(f"样本 {sample} 的代谢通路丰度分析失败")
        status_tracker.update_step_status(sample, 'pathway', False)

    return success


def generate_pathway_stats(sample, output_dir, pathway_file):
    """生成通路丰度统计 - 修复版"""
    try:
        if not os.path.exists(pathway_file):
            LOG.warning(f"通路丰度文件不存在: {pathway_file}")
            return

        # 读取通路丰度数据
        df = pd.read_csv(pathway_file, sep='\t')

        if df.empty:
            LOG.warning("通路丰度数据为空")
            return

        # 确保数据列是合适的类型
        if 'Abundance' in df.columns:
            df['Abundance'] = pd.to_numeric(df['Abundance'], errors='coerce').fillna(0)

        if 'Coverage' in df.columns:
            df['Coverage'] = pd.to_numeric(df['Coverage'], errors='coerce').fillna(0)

        if 'Pathway_Name' in df.columns:
            df['Pathway_Name'] = df['Pathway_Name'].astype(str)

        # 生成统计报告
        stats_file = os.path.join(output_dir, f"{sample}.pathway_stats.txt")

        with open(stats_file, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write(f"样本 {sample} 通路丰度统计\n")
            f.write("=" * 60 + "\n\n")

            f.write("1. 总体统计:\n")
            f.write(f"   总通路数: {len(df)}\n")

            # 统计有丰度的通路
            if 'Abundance' in df.columns:
                active_pathways = sum(df['Abundance'] > 0)
                f.write(f"   有丰度的通路数: {active_pathways}\n")
                f.write(f"   总丰度值: {df['Abundance'].sum():.2f}\n")
                f.write(f"   平均通路丰度: {df['Abundance'].mean():.4f}\n")
                f.write(f"   中位数通路丰度: {df['Abundance'].median():.4f}\n")
                f.write(f"   最大通路丰度: {df['Abundance'].max():.2f}\n")

            if 'Coverage' in df.columns:
                f.write(f"   平均通路覆盖度: {df['Coverage'].mean():.2f}%\n")
                f.write(f"   中位数通路覆盖度: {df['Coverage'].median():.2f}%\n")

            f.write("\n2. Top 20高丰度通路:\n")
            if 'Pathway_Name' in df.columns and 'Abundance' in df.columns:
                top_pathways = df.nlargest(20, 'Abundance')[['Pathway_Name', 'Abundance']]
                for idx, row in top_pathways.iterrows():
                    # 确保Pathway_Name是字符串
                    pathway_name = str(row['Pathway_Name']) if not pd.isna(row['Pathway_Name']) else "Unknown"
                    abundance_val = float(row['Abundance']) if not pd.isna(row['Abundance']) else 0

                    # 截断过长的通路名称
                    if len(pathway_name) > 50:
                        pathway_name = pathway_name[:47] + "..."

                    f.write(f"   {pathway_name:<50} {abundance_val:10.2f}\n")

        LOG.info(f"通路丰度统计生成完成: {stats_file}")

        # 生成通路丰度分布图
        generate_pathway_plot(sample, df, output_dir)

    except Exception as e:
        LOG.error(f"生成通路丰度统计失败: {e}")
        import traceback
        LOG.error(f"详细错误信息: {traceback.format_exc()}")


def generate_pathway_plot(sample, df, output_dir):
    """生成通路丰度分布图 - 修复版"""
    try:
        if df.empty or 'Abundance' not in df.columns:
            LOG.warning("数据为空或缺少Abundance列，跳过绘图")
            return

        # 设置字体
        setup_chinese_font()

        # 确保数据列是合适的类型
        df['Abundance'] = pd.to_numeric(df['Abundance'], errors='coerce').fillna(0)

        if 'Pathway_Name' in df.columns:
            df['Pathway_Name'] = df['Pathway_Name'].astype(str)

        plt.style.use('default')  # 使用默认风格
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # 1. 通路丰度分布直方图
        ax1 = axes[0]
        # 过滤掉零值
        non_zero = df[df['Abundance'] > 0]['Abundance']
        if len(non_zero) > 0:
            # 避免对数操作中的零或负值
            log_abundance = np.log10(non_zero + 1e-6)
            ax1.hist(log_abundance, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            ax1.set_xlabel('log10(Pathway Abundance+1e-6)')
            ax1.set_ylabel('Pathway Count')
            ax1.set_title('Pathway Abundance Distribution')
            ax1.grid(True, alpha=0.3)
        else:
            ax1.text(0.5, 0.5, 'No Non-Zero Abundance Data',
                     ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('Pathway Abundance Distribution')

        # 2. Top 15通路柱状图
        ax2 = axes[1]
        top_n = min(15, len(df))

        if top_n > 0:
            top_df = df.nlargest(top_n, 'Abundance')

            if not top_df.empty and 'Pathway_Name' in top_df.columns:
                pathway_names = []
                for idx, row in top_df.iterrows():
                    # 安全地获取通路名称
                    name = str(row['Pathway_Name']) if not pd.isna(row['Pathway_Name']) else f"Pathway_{idx}"
                    # 简化通路名称显示
                    if len(name) > 40:
                        name = name[:37] + '...'
                    pathway_names.append(name)

                # 确保Abundance是数值类型
                abundances = top_df['Abundance'].values

                bars = ax2.barh(range(top_n), abundances, color='lightcoral')
                ax2.set_yticks(range(top_n))
                ax2.set_yticklabels(pathway_names, fontsize=9)
                ax2.invert_yaxis()
                ax2.set_xlabel('Pathway Abundance')
                ax2.set_title(f'Top {top_n} High Abundance Pathways')
                ax2.grid(True, alpha=0.3, axis='x')
            else:
                ax2.text(0.5, 0.5, 'No Pathway Name Data',
                         ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title(f'Top {top_n} High Abundance Pathways')
        else:
            ax2.text(0.5, 0.5, 'No Data Available',
                     ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Top Pathways')

        plt.suptitle(f'Sample {sample} Pathway Abundance Distribution',
                     fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        plot_file = os.path.join(output_dir, f"{sample}.pathway_distribution.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close(fig)

        LOG.info(f"通路丰度分布图生成完成: {plot_file}")

    except Exception as e:
        LOG.error(f"生成通路丰度分布图失败: {e}")
        import traceback
        LOG.error(f"详细错误信息: {traceback.format_exc()}")
        return None


def integrate_annotations(sample, output_base, status_tracker, force=False):
    """整合三个数据库的注释结果，计算整体注释率"""
    LOG.info(f"为样本 {sample} 整合注释结果")

    # 检查是否已经完成且不需要强制重运行
    if not force and status_tracker.check_step_completed(sample, 'integrate', check_file=True):
        LOG.info(f"样本 {sample} 的注释整合已完成，跳过")
        return True

    # 获取三个注释文件路径
    interproscan_file = os.path.join(output_base, "annotation_results", "interproscan", sample,
                                     f"{sample}.simplified_annotations.tsv")
    kegg_file = os.path.join(output_base, "annotation_results", "kegg", sample, f"{sample}.kegg_annotations.tsv")
    cazy_file = os.path.join(output_base, "annotation_results", "cazy", sample, f"{sample}.cazy_annotations.tsv")

    # 获取基因丰度文件路径
    abundance_file = os.path.join(output_base, "annotation_results", "abundance", sample,
                                  f"{sample}.gene_abundance.tsv")

    # 检查文件是否存在
    files_exist = True
    missing_files = []
    for file_path, file_name in [(interproscan_file, "InterProScan注释文件"),
                                 (kegg_file, "KEGG注释文件"),
                                 (cazy_file, "CAZy注释文件")]:
        if not os.path.exists(file_path):
            files_exist = False
            missing_files.append(file_name)
            LOG.warning(f"{file_name}不存在: {file_path}")

    if not files_exist:
        LOG.warning(f"样本 {sample} 的注释文件不全，缺少: {', '.join(missing_files)}")
        status_tracker.update_step_status(sample, 'integrate', False)
        return False

    # 获取总基因数
    protein_file = find_protein_files(sample)
    total_genes = count_sequences(protein_file) if protein_file else 0

    try:
        # 读取三个注释文件
        # InterProScan注释
        interproscan_data = {}
        if os.path.exists(interproscan_file):
            try:
                df_ipr = pd.read_csv(interproscan_file, sep='\t')
                LOG.info(f"读取InterProScan注释: {len(df_ipr)} 行")
                for _, row in df_ipr.iterrows():
                    gene_id = row['ProteinID']
                    interproscan_data[gene_id] = {
                        'databases': row.get('Databases', ''),
                        'signatures': row.get('Signatures', ''),
                        'descriptions': row.get('Descriptions', ''),
                        'interpro_ids': row.get('InterPro_IDs', ''),
                        'go_terms': row.get('GO_Terms', '')
                    }
            except Exception as e:
                LOG.error(f"读取InterProScan注释文件失败: {e}")

        # KEGG注释
        kegg_data = {}
        if os.path.exists(kegg_file):
            try:
                df_kegg = pd.read_csv(kegg_file, sep='\t')
                LOG.info(f"读取KEGG注释: {len(df_kegg)} 行")
                for _, row in df_kegg.iterrows():
                    gene_id = row['GeneID']
                    kegg_data[gene_id] = {
                        'ko': row['KO'],
                        'ko_name': row['KO_Name'],
                        'pathway': row['Pathway'],
                        'kegg_score': row.get('Score', ''),
                        'kegg_evalue': row.get('Evalue', '')
                    }
            except Exception as e:
                LOG.error(f"读取KEGG注释文件失败: {e}")

        # CAZy注释
        cazy_data = {}
        if os.path.exists(cazy_file):
            try:
                df_cazy = pd.read_csv(cazy_file, sep='\t')
                LOG.info(f"读取CAZy注释: {len(df_cazy)} 行")
                for _, row in df_cazy.iterrows():
                    gene_id = row['GeneID']
                    cazy_data[gene_id] = {
                        'cazy_family': row['CAZy_Family'],
                        'cazy_description': row['Description'],
                        'cazy_identity': row.get('Identity', ''),
                        'cazy_evalue': row.get('Evalue', ''),
                        'cazy_coverage': row.get('Coverage', '')
                    }
            except Exception as e:
                LOG.error(f"读取CAZy注释文件失败: {e}")

        # 读取基因丰度数据
        abundance_data = {}
        if os.path.exists(abundance_file):
            try:
                df_abundance = pd.read_csv(abundance_file, sep='\t')
                LOG.info(f"读取基因丰度数据: {len(df_abundance)} 行")
                for _, row in df_abundance.iterrows():
                    gene_id = row['GeneID']
                    abundance_data[gene_id] = {
                        'tpm': row.get('TPM', 0),
                        'num_reads': row.get('NumReads', 0),
                        'length': row.get('Length', 0)
                    }
            except Exception as e:
                LOG.error(f"读取基因丰度文件失败: {e}")

        # 整合所有基因
        all_genes = set(list(interproscan_data.keys()) + list(kegg_data.keys()) + list(cazy_data.keys()))

        # 创建整合注释目录
        integrated_dir = os.path.join(output_base, "annotation_results", "integrated", sample)
        os.makedirs(integrated_dir, exist_ok=True)

        # 创建整合注释表
        integrated_file = os.path.join(integrated_dir, f"{sample}.integrated_annotations.tsv")

        with open(integrated_file, 'w') as f:
            # 写入表头
            f.write("GeneID\t")
            f.write("TPM\tNumReads\t")
            f.write(
                "InterProScan_Databases\tInterProScan_Signatures\tInterProScan_Descriptions\tInterProScan_InterProIDs\tInterProScan_GO_Terms\t")
            f.write("KEGG_KO\tKEGG_KO_Name\tKEGG_Pathway\tKEGG_Score\tKEGG_Evalue\t")
            f.write("CAZy_Family\tCAZy_Description\tCAZy_Identity\tCAZy_Evalue\tCAZy_Coverage\t")
            f.write("Any_Annotation\n")

            # 写入每个基因的注释
            for gene_id in sorted(all_genes):
                f.write(f"{gene_id}\t")

                # 基因丰度
                if gene_id in abundance_data:
                    ab = abundance_data[gene_id]
                    f.write(f"{ab['tpm']:.6f}\t{ab['num_reads']:.2f}\t")
                else:
                    f.write("0.000000\t0.00\t")

                # InterProScan注释
                if gene_id in interproscan_data:
                    ipr = interproscan_data[gene_id]
                    f.write(
                        f"{ipr['databases']}\t{ipr['signatures']}\t{ipr['descriptions']}\t{ipr['interpro_ids']}\t{ipr['go_terms']}\t")
                else:
                    f.write("-\t-\t-\t-\t-\t")

                # KEGG注释
                if gene_id in kegg_data:
                    kegg = kegg_data[gene_id]
                    f.write(
                        f"{kegg['ko']}\t{kegg['ko_name']}\t{kegg['pathway']}\t{kegg['kegg_score']}\t{kegg['kegg_evalue']}\t")
                else:
                    f.write("-\t-\t-\t-\t-\t")

                # CAZy注释
                if gene_id in cazy_data:
                    cazy = cazy_data[gene_id]
                    f.write(
                        f"{cazy['cazy_family']}\t{cazy['cazy_description']}\t{cazy['cazy_identity']}\t{cazy['cazy_evalue']}\t{cazy['cazy_coverage']}\t")
                else:
                    f.write("-\t-\t-\t-\t-\t")

                # 是否有任何注释
                has_annotation = gene_id in interproscan_data or gene_id in kegg_data or gene_id in cazy_data
                f.write(f"{'Yes' if has_annotation else 'No'}\n")

        # 计算注释率
        genes_with_any_annotation = len(all_genes)
        interproscan_genes = len(interproscan_data)
        kegg_genes = len(kegg_data)
        cazy_genes = len(cazy_data)

        # 生成注释率报告
        rate_file = os.path.join(integrated_dir, f"{sample}.annotation_rates.txt")

        with open(rate_file, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write(f"样本 {sample} 注释率统计报告\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"总基因数: {total_genes}\n")
            f.write(f"获得至少一个注释的基因数: {genes_with_any_annotation}\n")
            f.write(f"整体注释率: {genes_with_any_annotation / total_genes * 100:.2f}%\n\n")

            f.write("各数据库注释统计:\n")
            f.write(f"InterProScan注释基因数: {interproscan_genes} ({interproscan_genes / total_genes * 100:.2f}%)\n")
            f.write(f"KEGG注释基因数: {kegg_genes} ({kegg_genes / total_genes * 100:.2f}%)\n")
            f.write(f"CAZy注释基因数: {cazy_genes} ({cazy_genes / total_genes * 100:.2f}%)\n\n")

            # 计算表达基因的注释率
            expressed_genes = [g for g in abundance_data if abundance_data[g]['tpm'] > 0]
            expressed_annotated = [g for g in expressed_genes if g in all_genes]

            if expressed_genes:
                f.write("表达基因注释统计 (TPM > 0):\n")
                f.write(f"表达基因总数: {len(expressed_genes)}\n")
                f.write(f"表达基因中有注释的: {len(expressed_annotated)}\n")
                f.write(f"表达基因注释率: {len(expressed_annotated) / len(expressed_genes) * 100:.2f}%\n\n")

            # 韦恩图数据
            f.write("注释重叠统计:\n")
            f.write(
                f"仅InterProScan注释: {len(set(interproscan_data.keys()) - set(kegg_data.keys()) - set(cazy_data.keys()))}\n")
            f.write(
                f"仅KEGG注释: {len(set(kegg_data.keys()) - set(interproscan_data.keys()) - set(cazy_data.keys()))}\n")
            f.write(
                f"仅CAZy注释: {len(set(cazy_data.keys()) - set(interproscan_data.keys()) - set(kegg_data.keys()))}\n")
            f.write(
                f"InterProScan+KEGG: {len(set(interproscan_data.keys()) & set(kegg_data.keys()) - set(cazy_data.keys()))}\n")
            f.write(
                f"InterProScan+CAZy: {len(set(interproscan_data.keys()) & set(cazy_data.keys()) - set(kegg_data.keys()))}\n")
            f.write(
                f"KEGG+CAZy: {len(set(kegg_data.keys()) & set(cazy_data.keys()) - set(interproscan_data.keys()))}\n")
            f.write(
                f"三个数据库都注释: {len(set(interproscan_data.keys()) & set(kegg_data.keys()) & set(cazy_data.keys()))}\n")
            f.write(f"未注释基因: {total_genes - genes_with_any_annotation}\n")

        LOG.info(f"样本 {sample} 注释整合完成:")
        LOG.info(f"  总基因数: {total_genes}")
        LOG.info(f"  整体注释率: {genes_with_any_annotation / total_genes * 100:.2f}%")
        LOG.info(f"  InterProScan: {interproscan_genes / total_genes * 100:.2f}%")
        LOG.info(f"  KEGG: {kegg_genes / total_genes * 100:.2f}%")
        LOG.info(f"  CAZy: {cazy_genes / total_genes * 100:.2f}%")

        status_tracker.update_step_status(sample, 'integrate', True, integrated_file)
        return True

    except Exception as e:
        LOG.error(f"整合注释失败: {e}")
        status_tracker.update_step_status(sample, 'integrate', False)
        return False


def generate_overall_report(output_base, samples):
    """生成整体报告"""
    LOG.info("生成整体注释率报告")

    report_file = os.path.join(output_base, "annotation_results", "overall_annotation_report.txt")

    try:
        with open(report_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("整体注释率统计报告\n")
            f.write("=" * 80 + "\n\n")

            total_genes_all = 0
            annotated_genes_all = 0
            sample_stats = []

            for sample in samples:
                rate_file = os.path.join(output_base, "annotation_results", "integrated", sample,
                                         f"{sample}.annotation_rates.txt")

                if os.path.exists(rate_file):
                    with open(rate_file, 'r') as rf:
                        lines = rf.readlines()
                        # 提取关键信息
                        total_genes = 0
                        annotated_genes = 0

                        for line in lines:
                            if "总基因数:" in line:
                                try:
                                    total_genes = int(line.split(":")[1].strip())
                                except:
                                    pass
                            elif "获得至少一个注释的基因数:" in line:
                                try:
                                    annotated_genes = int(line.split(":")[1].strip())
                                except:
                                    pass

                        total_genes_all += total_genes
                        annotated_genes_all += annotated_genes

                        rate = annotated_genes / total_genes * 100 if total_genes > 0 else 0
                        sample_stats.append({
                            'sample': sample,
                            'total': total_genes,
                            'annotated': annotated_genes,
                            'rate': rate
                        })

            # 写入样本级别统计
            f.write("样本级别注释率:\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'样本':<20} {'总基因数':>12} {'注释基因数':>12} {'注释率':>10}\n")
            f.write("-" * 80 + "\n")

            for stat in sample_stats:
                f.write(f"{stat['sample']:<20} {stat['total']:>12,} {stat['annotated']:>12,} {stat['rate']:>9.2f}%\n")

            # 写入总体统计
            f.write("\n" + "=" * 80 + "\n")
            f.write("总体统计:\n")
            f.write("-" * 80 + "\n")
            overall_rate = annotated_genes_all / total_genes_all * 100 if total_genes_all > 0 else 0
            f.write(f"总基因数: {total_genes_all:,}\n")
            f.write(f"总注释基因数: {annotated_genes_all:,}\n")
            f.write(f"整体注释率: {overall_rate:.2f}%\n")
            f.write(f"未注释基因数: {total_genes_all - annotated_genes_all:,}\n")

            # 计算平均值
            if sample_stats:
                avg_rate = sum([s['rate'] for s in sample_stats]) / len(sample_stats)
                f.write(f"平均样本注释率: {avg_rate:.2f}%\n")

            # 注释率分布
            f.write("\n注释率分布:\n")
            f.write("-" * 80 + "\n")
            ranges = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 60), (60, 70), (70, 80), (80, 90),
                      (90, 100)]
            for low, high in ranges:
                count = sum(1 for s in sample_stats if low <= s['rate'] < high)
                f.write(f"{low:>3}-{high:>3}%: {count:>3} 个样本\n")

        LOG.info(f"整体报告已生成: {report_file}")
        return True
    except Exception as e:
        LOG.error(f"生成整体报告失败: {e}")
        return False


def run_annotation_pipeline(sample, output_base, steps, db_paths, status_tracker, force=False):
    """运行单个样本的注释流程"""
    LOG.info(f"开始处理样本: {sample}")
    results = {}

    try:
        if 'interproscan' in steps:
            results['interproscan'] = run_interproscan_annotation(sample, output_base, db_paths, status_tracker, force)

        if 'kegg' in steps:
            results['kegg'] = run_kegg_annotation(sample, output_base, db_paths, status_tracker, force)

        if 'cazy' in steps:
            results['cazy'] = run_cazy_annotation(sample, output_base, db_paths, status_tracker, force)

        if 'abundance' in steps:
            results['abundance'] = calculate_gene_abundance(sample, output_base, db_paths, status_tracker, force)

        # 代谢通路分析需要KEGG注释和基因丰度数据
        if 'pathway' in steps:
            # 检查依赖项是否成功
            kegg_success = results.get('kegg', False) or status_tracker.check_step_completed(sample, 'kegg',
                                                                                             check_file=True)
            abundance_success = results.get('abundance', False) or status_tracker.check_step_completed(sample,
                                                                                                       'abundance',
                                                                                                       check_file=True)

            if kegg_success and abundance_success:
                results['pathway'] = run_pathway_abundance_analysis(sample, output_base, db_paths, status_tracker,
                                                                    force)
            else:
                LOG.warning(f"样本 {sample} 跳过代谢通路分析，因为缺少KEGG注释或基因丰度数据")
                results['pathway'] = False

        # 整合注释结果（需要至少一个注释完成）
        if 'integrate' in steps:
            # 检查是否有任何注释完成
            any_annotation_complete = any(results.get(step, False) for step in ['interproscan', 'kegg', 'cazy'])
            any_annotation_complete = any_annotation_complete or any(
                status_tracker.check_step_completed(sample, step, check_file=True)
                for step in ['interproscan', 'kegg', 'cazy'])

            if any_annotation_complete:
                results['integrate'] = integrate_annotations(sample, output_base, status_tracker, force)
            else:
                LOG.warning(f"样本 {sample} 跳过注释整合，因为没有任何注释结果")
                results['integrate'] = False

        LOG.info(f"样本 {sample} 处理完成: {results}")
        return sample, results

    except Exception as e:
        LOG.error(f"样本 {sample} 处理过程中发生错误: {e}")
        return sample, {'error': str(e)}


def generate_summary_report(status_tracker, output_base, steps):
    """生成摘要报告"""
    report_file = os.path.join(output_base, "annotation_summary_report.txt")

    try:
        with open(report_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("注释流程摘要报告\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"总样本数: {len(status_tracker.status)}\n")
            f.write(f"分析步骤: {', '.join(steps)}\n\n")

            # 按步骤统计
            step_stats = {}
            for step in steps:
                step_stats[step] = {'success': 0, 'failed': 0, 'total': 0}

            failed_samples_by_step = {step: [] for step in steps}

            for sample, sample_status in status_tracker.status.items():
                for step in steps:
                    if step in sample_status:
                        step_stats[step]['total'] += 1
                        if sample_status[step].get('success', False):
                            step_stats[step]['success'] += 1
                        else:
                            step_stats[step]['failed'] += 1
                            failed_samples_by_step[step].append(sample)

            # 输出步骤统计
            f.write("步骤完成情况统计:\n")
            f.write("-" * 60 + "\n")
            for step in steps:
                stats = step_stats[step]
                if stats['total'] > 0:
                    success_rate = (stats['success'] / stats['total']) * 100
                    f.write(f"{step}: 成功 {stats['success']}/{stats['total']} ({success_rate:.1f}%)\n")
                else:
                    f.write(f"{step}: 未执行\n")

            # 输出失败样本
            f.write("\n失败样本列表:\n")
            f.write("-" * 60 + "\n")
            for step in steps:
                if failed_samples_by_step[step]:
                    f.write(f"\n{step} 失败的样本 ({len(failed_samples_by_step[step])}个):\n")
                    for i, sample in enumerate(failed_samples_by_step[step]):
                        f.write(f"  {i + 1}. {sample}\n")

            # 输出建议
            f.write("\n" + "=" * 80 + "\n")
            f.write("建议:\n")
            f.write("=" * 80 + "\n")
            f.write("1. 重新运行命令: python main_control.py --resume\n")
            f.write("2. 强制重新运行: python main_control.py --force\n")
            f.write("3. 重新运行特定步骤: python main_control.py --steps kegg cazy --resume\n")
            f.write("4. 重新运行特定样本: python main_control.py --samples sample1 sample2 --resume\n")

        LOG.info(f"摘要报告已生成: {report_file}")
        return True
    except Exception as e:
        LOG.error(f"生成摘要报告失败: {e}")
        return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="修正版并行注释流程主控脚本（集成蛋白序列清理和代谢通路分析）",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("-o", "--output", default="/home/zjw/zjwdata/",
                        help="输出主目录 (默认: /home/zjw/zjwdata/)")
    parser.add_argument("--steps", nargs="+",
                        choices=['interproscan', 'kegg', 'cazy', 'abundance', 'pathway', 'integrate', 'all'],
                        default=['all'],
                        help="要运行的步骤 (默认: all)")
    parser.add_argument("--samples", nargs="+",
                        help="指定要处理的样本 (默认: 处理所有样本)")
    parser.add_argument("--threads", type=int, default=1,
                        help="并行处理的样本数 (默认: 1)")
    parser.add_argument("--force", action="store_true",
                        help="强制重新运行所有步骤，忽略状态记录")
    parser.add_argument("--resume", action="store_true",
                        help="从上次中断的地方继续，跳过已完成的步骤")
    parser.add_argument("--summary", action="store_true",
                        help="生成摘要报告后退出，不运行任何分析")

    args = parser.parse_args()

    # 设置日志
    log_file = setup_logging(args.output)

    # 设置环境
    setup_environment()

    # 检查数据库路径
    db_paths = check_database_paths()

    # 验证和创建目录结构
    required_dirs = validate_directories(args.output)

    # 初始化状态跟踪器
    status_tracker = StatusTracker(args.output)

    # 如果只需要生成摘要报告
    if args.summary:
        if args.steps == ['all']:
            steps = ['interproscan', 'kegg', 'cazy', 'abundance', 'pathway', 'integrate']
        else:
            steps = args.steps
        generate_summary_report(status_tracker, args.output, steps)
        return

    # 获取样本列表
    if args.samples:
        samples = args.samples
    else:
        samples = get_sample_directories()

    if not samples:
        LOG.error("未找到任何样本")
        sys.exit(1)

    # 解析步骤
    if 'all' in args.steps:
        steps = ['interproscan', 'kegg', 'cazy', 'abundance', 'pathway', 'integrate']
    else:
        steps = args.steps

    # 如果使用--resume，跳过已完成的步骤
    if args.resume and not args.force:
        incomplete_samples = status_tracker.get_incomplete_samples(samples, steps)
        if incomplete_samples:
            samples_to_process = []
            for sample, incomplete_steps in incomplete_samples:
                LOG.info(f"样本 {sample} 需要处理的步骤: {incomplete_steps}")
                samples_to_process.append(sample)

            if not samples_to_process:
                LOG.info("所有样本的所有步骤均已完成，无需运行")
                generate_summary_report(status_tracker, args.output, steps)
                return

            # 只处理有未完成步骤的样本
            samples = samples_to_process
        else:
            LOG.info("所有样本的所有步骤均已完成，无需运行")
            generate_summary_report(status_tracker, args.output, steps)
            return

    LOG.info(f"开始并行注释流程")
    LOG.info(f"样本数量: {len(samples)}")
    LOG.info(f"处理步骤: {', '.join(steps)}")
    LOG.info(f"并行样本数: {args.threads}")
    LOG.info(f"每个工具线程数: 32")
    LOG.info(f"JVM内存: 96GB")
    LOG.info(f"输出目录: {args.output}")
    LOG.info(f"日志文件: {log_file}")
    LOG.info(f"强制模式: {args.force}")
    LOG.info(f"恢复模式: {args.resume}")

    # 并行处理样本
    success_count = 0
    failed_samples = []

    LOG.info(f"开始并行处理 {len(samples)} 个样本...")

    with ProcessPoolExecutor(max_workers=args.threads) as executor:
        # 提交所有任务
        future_to_sample = {
            executor.submit(run_annotation_pipeline, sample, args.output, steps, db_paths, status_tracker,
                            args.force): sample
            for sample in samples
        }

        # 收集结果
        for future in as_completed(future_to_sample):
            sample = future_to_sample[future]
            try:
                sample, results = future.result()
                if 'error' not in results:
                    success_count += 1
                    LOG.info(f"样本 {sample} 成功完成")
                else:
                    failed_samples.append(sample)
                    LOG.error(f"样本 {sample} 失败: {results['error']}")
            except Exception as e:
                failed_samples.append(sample)
                LOG.error(f"样本 {sample} 执行异常: {e}")

    # 生成最终报告
    LOG.info("=" * 60)
    LOG.info(f"注释流程完成!")
    LOG.info(f"成功处理: {success_count}/{len(samples)} 个样本")
    LOG.info(f"注释结果目录: {required_dirs['annotation_results']}")
    LOG.info(f"清理蛋白序列目录: {required_dirs['cleaned_proteins']}")
    LOG.info(f"日志文件: {log_file}")

    # 生成摘要报告
    generate_summary_report(status_tracker, args.output, steps)

    # 如果运行了整合步骤，生成整体报告
    if 'integrate' in steps:
        LOG.info("开始生成整体注释率报告...")
        generate_overall_report(args.output, samples)

    if failed_samples:
        LOG.info(f"失败的样本: {', '.join(failed_samples)}")
        # 保存失败样本列表
        with open(os.path.join(args.output, "failed_samples.txt"), 'w') as f:
            for sample in failed_samples:
                f.write(f"{sample}\n")

    # 提供后续步骤建议
    LOG.info("后续步骤建议:")
    LOG.info("1. 检查各样本的输出目录确认结果完整性")
    LOG.info("2. 查看摘要报告: cat annotation_summary_report.txt")
    LOG.info("3. 查看整体注释率报告: cat annotation_results/overall_annotation_report.txt")
    LOG.info("4. 重新运行失败步骤: python main_control.py --resume")
    LOG.info("5. 检查代谢通路丰度分析结果")
    LOG.info("6. 查看基因丰度质量控制报告")
    LOG.info("7. 查看丰度分布图")
    LOG.info("8. 查看通路丰度分布图")


if __name__ == "__main__":
    main()