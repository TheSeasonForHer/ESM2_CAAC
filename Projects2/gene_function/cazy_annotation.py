#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
独立CAZy注释脚本
使用diamond进行CAZy数据库比对
用于第二阶段功能注释中的CAZy部分
适配主控脚本调用方式
添加注释率统计功能
添加过滤步骤
修复参数传递问题
修复总基因数统计问题 - 修正注释率100%问题
添加--blast参数指定比对文件
"""

import os
import sys
import argparse
import logging
import subprocess
import glob
import re
from collections import defaultdict
import pandas as pd

LOG = logging.getLogger(__name__)
__version__ = "1.3.4"
__author__ = ("Xingguo Zhang",)
__email__ = "invicoun@foxmail.com"


def setup_logging(log_file=None):
    """设置日志格式，支持输出到文件"""
    if log_file:
        # 创建日志目录
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        handlers = [
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    else:
        handlers = [logging.StreamHandler(sys.stdout)]

    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=handlers
    )

    if log_file:
        LOG.info(f"日志文件: {log_file}")


def run_command(cmd, description=""):
    """运行命令行工具"""
    LOG.info(f"运行: {description}")
    LOG.info(f"命令: {cmd}")

    try:
        # 实时输出，避免被PIPE阻塞
        result = subprocess.run(cmd, shell=True, check=True)
        LOG.info(f"{description} 完成")
        return True
    except subprocess.CalledProcessError as e:
        LOG.error(f"{description} 失败: {e}")
        return False


def run_diamond_cazy_single(protein_file, output_file, cazy_db, threads=32, evalue=1e-5):
    """运行单个diamond CAZy比对"""
    LOG.info(f"开始CAZy注释: {protein_file}")

    cmd = f"""
    diamond blastp \
      --query {protein_file} \
      --db {cazy_db} \
      --outfmt 6 qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore qlen slen stitle \
      --max-target-seqs 5 \
      --evalue {evalue} \
      --threads {threads} \
      --out {output_file}
    """

    success = run_command(cmd, f"diamond CAZy注释")

    if success:
        if os.path.exists(output_file):
            file_size = os.path.getsize(output_file)
            if file_size > 0:
                LOG.info(f"CAZy注释完成: {output_file} ({file_size} 字节)")
                return True
            else:
                LOG.warning(f"CAZy输出文件为空: {output_file}")
                return True  # 仍然返回True，因为可能是真的没有比对结果
        else:
            LOG.error(f"CAZy输出文件不存在: {output_file}")
            return False
    else:
        LOG.error(f"CAZy注释失败")
        return False


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


def count_sequences(fasta_file):
    """统计FASTA文件中的序列数量"""
    if not fasta_file or not os.path.exists(fasta_file):
        LOG.error(f"文件不存在: {fasta_file}")
        return 0

    count = 0
    try:
        with open(fasta_file, 'r') as f:
            for line in f:
                if line.startswith('>'):
                    count += 1
        LOG.info(f"文件 {fasta_file} 包含 {count} 个序列")
        return count
    except Exception as e:
        LOG.error(f"无法统计序列数量 {fasta_file}: {e}")
        return 0


def parse_cazy_annotations(blast_file, output_prefix, evalue=1e-5, coverage=30, protein_file=None):
    """解析CAZy注释结果，添加注释率统计 - 修正注释率统计逻辑"""
    LOG.info(f"解析CAZy注释: {blast_file}")

    if not os.path.exists(blast_file):
        LOG.error(f"BLAST文件不存在: {blast_file}")
        return False, 0, 0

    file_size = os.path.getsize(blast_file)
    if file_size == 0:
        LOG.warning(f"BLAST文件为空: {blast_file}")
        # 创建空的输出文件
        create_empty_cazy_outputs(output_prefix)
        return True, 0, 0

    # 统计总基因数 - 修正：必须从蛋白文件统计，否则无法计算准确注释率
    total_genes = 0
    if protein_file and os.path.exists(protein_file):
        total_genes = count_sequences(protein_file)
        LOG.info(f"从蛋白文件统计总基因数: {total_genes} ({protein_file})")
    else:
        LOG.error(f"蛋白文件不存在或未提供: {protein_file}")
        LOG.error("无法计算准确注释率，因为缺少总基因数信息")
        LOG.error("注释率统计将跳过，只生成注释结果")
        # 设置总基因数为0，表示无法计算注释率
        total_genes = 0

    # 解析BLAST结果
    cazy_stats = defaultdict(int)
    gene_annotations = {}
    best_hits = {}
    line_count = 0

    try:
        with open(blast_file, 'r') as f:
            for line in f:
                line_count += 1
                parts = line.strip().split('\t')
                if len(parts) < 14:
                    LOG.warning(f"第{line_count}行字段不足: {line.strip()}")
                    continue

                qseqid, sseqid, pident, length, mismatch, gapopen, qstart, qend, sstart, send, evalue_val, bitscore, qlen, slen = parts[
                                                                                                                                  :14]
                stitle = parts[14] if len(parts) > 14 else ""

                # 计算覆盖度
                try:
                    qcov = (int(qend) - int(qstart) + 1) / int(qlen) * 100
                except (ValueError, ZeroDivisionError):
                    continue

                # 应用过滤条件
                if float(evalue_val) > evalue or qcov < coverage:
                    continue

                # 提取CAZy家族信息
                cazy_family = extract_cazy_family(stitle)

                if cazy_family:
                    # 统计CAZy家族
                    cazy_stats[cazy_family] += 1

                    # 选择每个基因的最佳hit
                    if qseqid not in best_hits or float(bitscore) > float(best_hits[qseqid]['bitscore']):
                        best_hits[qseqid] = {
                            'cazy_family': cazy_family,
                            'description': stitle,
                            'pident': pident,
                            'evalue': evalue_val,
                            'bitscore': bitscore,
                            'coverage': f"{qcov:.1f}%"
                        }

        LOG.info(f"成功解析 {line_count} 行数据")

        # 转换为基因注释
        for gene_id, hit in best_hits.items():
            gene_annotations[gene_id] = hit

        # 计算注释率 - 修正：只有在有总基因数时才能计算注释率
        annotated_count = len(gene_annotations)
        annotation_rate = 0

        if total_genes > 0:
            annotation_rate = (annotated_count / total_genes) * 100
            LOG.info(f"CAZy注释统计:")
            LOG.info(f"  总基因数: {total_genes}")
            LOG.info(f"  注释基因数: {annotated_count}")
            LOG.info(f"  注释率: {annotation_rate:.2f}%")
        else:
            # 没有总基因数信息，无法计算注释率
            LOG.warning(f"CAZy注释统计 (无法计算注释率):")
            LOG.warning(f"  总基因数: 未知 (缺少蛋白文件)")
            LOG.warning(f"  注释基因数: {annotated_count}")
            LOG.warning(f"  注释率: 无法计算 (缺少总基因数)")

        LOG.info(f"  发现 {len(cazy_stats)} 个不同的CAZy家族")
        LOG.info(f"  发现 {annotated_count} 个有注释的基因")

        # 生成统计报告
        write_cazy_statistics(cazy_stats, output_prefix, total_genes, annotated_count, annotation_rate)

        # 生成基因注释表
        write_cazy_gene_annotations(gene_annotations, output_prefix)

        # 生成分类统计
        write_cazy_classification(gene_annotations, output_prefix)

        LOG.info(f"CAZy注释解析完成: {annotated_count} 个基因获得注释")
        return True, annotated_count, annotation_rate

    except Exception as e:
        LOG.error(f"解析CAZy注释失败: {e}")
        # 创建空的输出文件
        create_empty_cazy_outputs(output_prefix)
        return False, 0, 0


def create_empty_cazy_outputs(output_prefix):
    """创建空的CAZy输出文件"""
    LOG.info("创建空的CAZy输出文件")

    # 空的注释率报告
    rate_file = f"{output_prefix}.annotation_rate.txt"
    with open(rate_file, 'w') as f:
        f.write("CAZy注释率统计报告\n")
        f.write("=" * 50 + "\n")
        f.write(f"总基因数: 0\n")
        f.write(f"注释基因数: 0\n")
        f.write(f"注释率: 0.00%\n")
        f.write(f"未注释基因数: 0\n")
        f.write("\n注意: CAZy注释结果为空\n")

    # 空的基因注释表
    anno_file = f"{output_prefix}.cazy_annotations.tsv"
    with open(anno_file, 'w') as f:
        f.write("GeneID\tCAZy_Family\tDescription\tIdentity\tEvalue\tBitscore\tCoverage\n")

    # 空的家族统计
    family_file = f"{output_prefix}.cazy_family_stats.tsv"
    with open(family_file, 'w') as f:
        f.write("CAZy_Family\tCount\tPercentage\n")

    # 空的分类统计
    class_file = f"{output_prefix}.cazy_class_stats.tsv"
    with open(class_file, 'w') as f:
        f.write("Class\tCount\tPercentage\n")

    LOG.info("空的CAZy输出文件创建完成")


def extract_cazy_family(stitle):
    """从标题中提取CAZy家族信息"""
    if not stitle:
        return None

    # 常见的CAZy家族模式
    cazy_patterns = [
        "GH", "GT", "PL", "CE", "AA", "CBM"
    ]

    for pattern in cazy_patterns:
        # 查找模式后跟数字的模式
        match = re.search(f"{pattern}\\d+", stitle)
        if match:
            return match.group()

    return None


def write_cazy_statistics(cazy_stats, output_prefix, total_genes, annotated_count, annotation_rate):
    """写入CAZy统计信息，包括注释率"""

    # CAZy家族统计
    family_file = f"{output_prefix}.cazy_family_stats.tsv"
    with open(family_file, 'w') as f:
        f.write("CAZy_Family\tCount\tPercentage\n")
        total_family = sum(cazy_stats.values())
        for cazy_family, count in sorted(cazy_stats.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_family) * 100 if total_family > 0 else 0
            f.write(f"{cazy_family}\t{count}\t{percentage:.2f}%\n")
    LOG.info(f"CAZy家族统计写入: {family_file}")

    # 注释率报告
    rate_file = f"{output_prefix}.annotation_rate.txt"
    with open(rate_file, 'w') as f:
        f.write("CAZy注释率统计报告\n")
        f.write("=" * 50 + "\n")

        if total_genes > 0:
            f.write(f"总基因数: {total_genes}\n")
            f.write(f"注释基因数: {annotated_count}\n")
            f.write(f"注释率: {annotation_rate:.2f}%\n")
            f.write(f"未注释基因数: {total_genes - annotated_count}\n")
            f.write(f"\n注意: 注释率基于提供的蛋白文件中的 {total_genes} 个基因计算\n")
        else:
            f.write(f"总基因数: 未知 (未提供或无法读取蛋白文件)\n")
            f.write(f"注释基因数: {annotated_count}\n")
            f.write(f"注释率: 无法计算 (缺少总基因数信息)\n")
            f.write(f"未注释基因数: 未知\n")
            f.write(f"\n警告: 无法计算准确注释率，因为缺少蛋白文件中的总基因数信息\n")
            f.write(f"请确保在运行时提供正确的蛋白文件路径 (-i 参数)\n")

        f.write("\nCAZy家族统计 (Top 20):\n")
        total_family = sum(cazy_stats.values())
        for cazy_family, count in sorted(cazy_stats.items(), key=lambda x: x[1], reverse=True)[:20]:
            percentage = (count / total_family) * 100 if total_family > 0 else 0
            f.write(f"{cazy_family}: {count} 个基因 ({percentage:.2f}%)\n")

        if total_genes > 0:
            f.write(f"\n基于 {annotated_count}/{total_genes} 个基因获得CAZy注释\n")
        else:
            f.write(f"\n基于 {annotated_count} 个基因获得CAZy注释 (总基因数未知)\n")

    LOG.info(f"注释率报告写入: {rate_file}")
    LOG.info("CAZy家族统计文件生成完成")


def write_cazy_gene_annotations(gene_annotations, output_prefix):
    """生成CAZy基因注释表"""

    anno_file = f"{output_prefix}.cazy_annotations.tsv"
    with open(anno_file, 'w') as f:
        f.write("GeneID\tCAZy_Family\tDescription\tIdentity\tEvalue\tBitscore\tCoverage\n")

        for gene_id, annotation in gene_annotations.items():
            f.write(f"{gene_id}\t")
            f.write(f"{annotation['cazy_family']}\t")
            f.write(f"{annotation['description']}\t")
            f.write(f"{annotation['pident']}\t")
            f.write(f"{annotation['evalue']}\t")
            f.write(f"{annotation['bitscore']}\t")
            f.write(f"{annotation['coverage']}\n")

    LOG.info(f"CAZy基因注释表生成完成: {anno_file} ({len(gene_annotations)} 个基因)")


def write_cazy_classification(gene_annotations, output_prefix):
    """生成CAZy分类统计"""

    # 按CAZy大类分类
    cazy_classes = {
        "GH": "Glycoside Hydrolases",
        "GT": "GlycosylTransferases",
        "PL": "Polysaccharide Lyases",
        "CE": "Carbohydrate Esterases",
        "AA": "Auxiliary Activities",
        "CBM": "Carbohydrate-Binding Modules"
    }

    class_stats = defaultdict(int)

    for gene_id, annotation in gene_annotations.items():
        cazy_family = annotation['cazy_family']
        # 提取大类前缀
        for prefix, class_name in cazy_classes.items():
            if cazy_family.startswith(prefix):
                class_stats[class_name] += 1
                break
        else:
            # 如果没有匹配到已知大类，归类为Other
            class_stats["Other"] += 1

    # 分类统计
    class_file = f"{output_prefix}.cazy_class_stats.tsv"
    with open(class_file, 'w') as f:
        f.write("Class\tCount\tPercentage\n")
        total_genes = len(gene_annotations)

        for class_name, count in sorted(class_stats.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_genes) * 100 if total_genes > 0 else 0
            f.write(f"{class_name}\t{count}\t{percentage:.2f}%\n")

    LOG.info(f"CAZy分类统计生成完成: {class_file}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="独立CAZy注释脚本 - 适配主控脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("-i", "--input", required=True,
                        help="输入蛋白文件 (FASTA格式)")
    parser.add_argument("-b", "--blast",
                        help="BLAST比对文件 (在解析模式下使用，如果不提供则根据输出目录和前缀自动构建)")
    parser.add_argument("--cazy_db", default="/home/databases/cazy_db/cazy.dmnd",
                        help="CAZy数据库路径 (默认: /home/databases/cazy_db/cazy.dmnd)")
    parser.add_argument("-t", "--threads", type=int, default=32,
                        help="diamond使用的线程数 (默认: 32)")
    parser.add_argument("-e", "--evalue", type=float, default=1e-5,
                        help="E-value阈值 (默认: 1e-5)")
    parser.add_argument("-c", "--coverage", type=float, default=30,
                        help="查询覆盖度阈值 (默认: 30)")
    parser.add_argument("--parse_only", action="store_true",
                        help="仅解析现有结果，不运行diamond")
    parser.add_argument("--filter_only", action="store_true",
                        help="仅过滤现有结果，不运行diamond和解析")
    parser.add_argument("--log", help="日志文件路径")

    # 输出参数
    parser.add_argument("-o", "--output", required=True,
                        help="输出目录")
    parser.add_argument("-p", "--prefix", required=True,
                        help="输出文件前缀")

    args = parser.parse_args()

    # 设置日志
    setup_logging(args.log)

    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)

    # 设置输出文件路径
    raw_output = os.path.join(args.output, f"{args.prefix}.cazy.m6")
    filtered_output = os.path.join(args.output, f"{args.prefix}.cazy.filtered.m6")
    output_prefix = os.path.join(args.output, args.prefix)

    if args.filter_only:
        LOG.info("仅过滤模式")
        if not os.path.exists(raw_output):
            LOG.error(f"原始CAZy文件不存在: {raw_output}")
            sys.exit(1)

        # 只运行过滤步骤
        if not filter_blast_results(raw_output, filtered_output, args.evalue, args.coverage):
            LOG.error("结果过滤失败")
            sys.exit(1)

        LOG.info("过滤步骤完成!")
        sys.exit(0)

    elif args.parse_only:
        LOG.info("仅解析模式")

        # 确定比对文件路径
        if args.blast:
            # 使用用户指定的比对文件
            blast_file = args.blast
            LOG.info(f"使用指定的比对文件: {blast_file}")
        else:
            # 自动构建比对文件路径
            blast_file = filtered_output
            LOG.info(f"自动使用过滤后的比对文件: {blast_file}")

            # 如果过滤后的文件不存在，尝试使用原始比对文件
            if not os.path.exists(blast_file):
                LOG.warning(f"过滤后的比对文件不存在: {blast_file}")
                LOG.warning(f"尝试使用原始比对文件: {raw_output}")
                blast_file = raw_output

        # 检查比对文件是否存在
        if not os.path.exists(blast_file):
            LOG.error(f"CAZy比对文件不存在: {blast_file}")
            LOG.error("请确保比对文件存在，或使用--blast参数指定正确的比对文件路径")
            sys.exit(1)

        # 检查蛋白文件是否存在
        if not os.path.exists(args.input):
            LOG.error(f"蛋白文件不存在，无法计算总基因数: {args.input}")
            LOG.error("注释率统计将不准确")
    else:
        # 正常模式，检查数据库和输入文件
        if not os.path.exists(args.cazy_db):
            LOG.error(f"CAZy数据库不存在: {args.cazy_db}")
            sys.exit(1)

        if not os.path.exists(args.input):
            LOG.error(f"输入蛋白文件不存在: {args.input}")
            sys.exit(1)

        # 步骤1: 运行diamond CAZy
        LOG.info("开始CAZy注释流程")
        if not run_diamond_cazy_single(args.input, raw_output, args.cazy_db, args.threads, args.evalue):
            LOG.error("diamond CAZy运行失败")
            sys.exit(1)

        # 步骤2: 过滤结果
        if not filter_blast_results(raw_output, filtered_output, args.evalue, args.coverage):
            LOG.error("结果过滤失败")
            sys.exit(1)

        # 在正常模式下，使用过滤后的文件作为比对文件
        blast_file = filtered_output

    # 步骤3: 解析和统计
    success, annotated_count, annotation_rate = parse_cazy_annotations(
        blast_file, output_prefix, args.evalue, args.coverage, args.input
    )

    if not success:
        LOG.error("结果解析失败")
        sys.exit(1)

    LOG.info("CAZy注释流程完成!")
    LOG.info(f"主要输出文件:")
    LOG.info(f"  - 蛋白文件: {args.input}")
    LOG.info(f"  - 比对文件: {blast_file}")
    LOG.info(f"  - 基因注释: {output_prefix}.cazy_annotations.tsv")
    LOG.info(f"  - 家族统计: {output_prefix}.cazy_family_stats.tsv")
    LOG.info(f"  - 分类统计: {output_prefix}.cazy_class_stats.tsv")
    LOG.info(f"  - 注释率报告: {output_prefix}.annotation_rate.txt")


if __name__ == "__main__":
    main()