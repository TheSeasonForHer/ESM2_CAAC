#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
独立KEGG注释脚本
使用kofam_scan进行KEGG注释
用于第二阶段功能注释中的KEGG部分
适配主控脚本调用方式
添加注释率统计功能
修复总基因数统计问题
修复kofam_scan输出格式解析问题
更新：支持新版KEGG格式
修复：适配实际KEGG文件格式（6列）
简化版：使用简单KO映射
"""

import os
import sys
import argparse
import logging
import subprocess
import csv
from collections import defaultdict
import pandas as pd

LOG = logging.getLogger(__name__)
__version__ = "1.5.1"
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
        datefmt='%Y-%m-%d %H:%M:%S',  # 修复日期格式
        handlers=handlers
    )

    if log_file:
        LOG.info(f"日志文件: {log_file}")


def run_command(cmd, description=""):
    """运行命令行工具"""
    LOG.info(f"运行: {description}")
    LOG.info(f"命令: {cmd}")

    try:
        # 实时输出
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, universal_newlines=True)

        # 实时读取输出
        for line in process.stdout:
            line = line.strip()
            if line:
                LOG.info(f"kofam_scan: {line}")

        # 读取错误输出
        stderr_lines = []
        for line in process.stderr:
            line = line.strip()
            if line:
                stderr_lines.append(line)
                if 'error' in line.lower() or 'exception' in line.lower():
                    LOG.error(f"kofam_scan错误: {line}")

        # 等待进程完成
        returncode = process.wait()

        if returncode == 0:
            LOG.info(f"{description} 完成")
            return True
        else:
            LOG.error(f"{description} 失败，返回码: {returncode}")
            if stderr_lines:
                LOG.error(f"最后10行错误输出:")
                for line in stderr_lines[-10:]:
                    LOG.error(f"  {line}")
            return False
    except Exception as e:
        LOG.error(f"{description} 执行异常: {e}")
        return False


def run_kofam_scan_single(protein_file, output_file, profile_db, ko_list, threads=32, evalue=1e-5):
    """运行单个kofam_scan任务"""
    LOG.info(f"开始KEGG注释: {protein_file}")

    cmd = f"""
    exec_annotation -o {output_file} \
      --profile {profile_db} \
      --ko-list {ko_list} \
      --cpu {threads} \
      --e-value {evalue} \
      --format detail-tsv \
      {protein_file}
    """

    success = run_command(cmd, "kofam_scan KEGG注释")

    if success:
        if os.path.exists(output_file):
            file_size = os.path.getsize(output_file)
            if file_size > 0:
                LOG.info(f"KEGG注释完成: {output_file} ({file_size} 字节)")
                return True
            else:
                LOG.warning(f"KEGG输出文件为空: {output_file}")
                return True  # 仍然返回True，因为可能是真的没有注释
        else:
            LOG.error(f"KEGG输出文件不存在: {output_file}")
            return False
    else:
        LOG.error(f"KEGG注释失败")
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


def parse_simple_kegg_format(kegg_hierarchy_file):
    """简化版KEGG解析，只提取KO信息"""
    pathway_info = {}

    LOG.info(f"简化版KEGG解析: {kegg_hierarchy_file}")

    try:
        with open(kegg_hierarchy_file, 'r') as f:
            # 跳过表头
            header = f.readline()
            LOG.info(f"KEGG文件表头: {header.strip()}")

            line_count = 0
            ko_count = 0

            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split('\t')

                # 查找包含KO号的部分
                if len(parts) >= 5:  # 至少要有5列
                    ko_id = parts[4]  # KO_id在第5列
                    pathway_desc = parts[3] if len(parts) > 3 else ""  # 通路描述在第4列

                    # 清理通路描述，移除[PATH:...]部分
                    if '[PATH:' in pathway_desc:
                        pathway_desc = pathway_desc.split('[PATH:')[0].strip()

                    if ko_id and ko_id.startswith('K'):
                        # 为每个KO创建一个简化的通路信息
                        if pathway_desc:
                            # 获取通路ID
                            pathway_id = parts[2] if len(parts) > 2 else ""
                            if not pathway_id.startswith('ko'):
                                pathway_id = f"ko{pathway_id}"
                            pathway_info[ko_id] = f"{pathway_id} {pathway_desc}"
                        else:
                            pathway_info[ko_id] = f"KO:{ko_id}"
                        ko_count += 1

                    line_count += 1

                # 每读取10000行输出一次进度
                if line_count > 0 and line_count % 10000 == 0:
                    LOG.info(f"已处理 {line_count} 行，找到 {ko_count} 个KO")

        LOG.info(f"简化版KEGG解析完成: {line_count} 行数据，{ko_count} 个KO映射")
        return pathway_info

    except Exception as e:
        LOG.error(f"简化版KEGG解析失败: {e}")
        import traceback
        LOG.error(traceback.format_exc())
        return {}


def parse_old_kegg_format(kegg_hierarchy_file):
    """解析旧版KEGG格式文件"""
    pathway_info = {}

    LOG.info(f"解析旧版KEGG格式: {kegg_hierarchy_file}")

    try:
        with open(kegg_hierarchy_file, 'r') as f:
            current_pathway = ""
            line_count = 0
            pathway_count = 0

            for line in f:
                line = line.strip()
                if not line:
                    continue

                line_count += 1

                if line.startswith('C') and '\t' in line:
                    # 通路行: C00010  Glycolysis / Gluconeogenesis
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        pathway_id = parts[0]
                        pathway_name = parts[1]
                        current_pathway = f"{pathway_id} {pathway_name}"
                elif line.startswith('D') and '\t' in line and current_pathway:
                    # KO行: D      K00844  HK; hexokinase [EC:2.7.1.1]
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        ko_id = parts[1]
                        pathway_info[ko_id] = current_pathway
                        pathway_count += 1

            LOG.info(f"成功解析旧版KEGG格式: {line_count} 行数据，{pathway_count} 个通路映射")
            return pathway_info

    except Exception as e:
        LOG.error(f"解析旧版KEGG格式失败: {e}")
        return {}


def parse_kofam_output(kegg_file, output_prefix, ko_list_file=None, kegg_hierarchy_file=None, input_file=None):
    """解析kofam_scan输出文件，添加注释率统计"""
    LOG.info(f"解析KEGG注释输出: {kegg_file}")

    # 检查输入文件是否存在
    if not os.path.exists(kegg_file):
        LOG.error(f"KEGG注释文件不存在: {kegg_file}")
        return False, 0, 0

    file_size = os.path.getsize(kegg_file)
    if file_size == 0:
        LOG.warning(f"KEGG注释文件为空: {kegg_file}")
        # 创建空的输出文件
        create_empty_outputs(output_prefix)
        return True, 0, 0

    # 统计总基因数
    total_genes = 0
    if input_file and os.path.exists(input_file):
        total_genes = count_sequences(input_file)
        LOG.info(f"从输入文件统计总基因数: {total_genes}")
    else:
        LOG.warning("未提供输入文件，无法准确计算注释率")
        # 尝试从输出文件中估计
        try:
            with open(kegg_file, 'r') as f:
                gene_set = set()
                for line in f:
                    if line.startswith('#') or not line.strip():
                        continue
                    # kofam_scan输出格式：*  gene_id  KO  score  threshold  evalue  description
                    parts = line.strip().split('\t')
                    if len(parts) >= 3:
                        gene_id = parts[1]
                        if gene_id:
                            gene_set.add(gene_id)
                total_genes = len(gene_set)
                LOG.info(f"从输出文件估计总基因数: {total_genes}")
        except:
            LOG.warning("无法从输出文件估计总基因数")

    # 读取KO列表信息
    ko_info = {}
    if ko_list_file and os.path.exists(ko_list_file):
        LOG.info(f"读取KO列表: {ko_list_file}")
        try:
            with open(ko_list_file, 'r') as f:
                for line in f:
                    if line.startswith('#') or not line.strip():
                        continue
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        ko_id = parts[0]
                        ko_name = parts[1]
                        ko_info[ko_id] = ko_name
            LOG.info(f"读取 {len(ko_info)} 个KO信息")
        except Exception as e:
            LOG.warning(f"读取KO列表失败: {e}")

    # 读取KEGG层次结构信息 - 使用简化版解析
    pathway_info = {}
    if kegg_hierarchy_file and os.path.exists(kegg_hierarchy_file):
        LOG.info(f"读取KEGG层次结构: {kegg_hierarchy_file}")

        # 先尝试读取文件第一行判断格式
        try:
            with open(kegg_hierarchy_file, 'r') as f:
                first_line = f.readline().strip()

            # 判断文件格式
            if first_line.startswith("Pathway_maps"):
                LOG.info("检测到新版KEGG格式（带表头）")
                pathway_info = parse_simple_kegg_format(kegg_hierarchy_file)
            elif first_line.startswith("C") or first_line.startswith("D"):
                LOG.info("检测到旧版KEGG格式（无表头）")
                pathway_info = parse_old_kegg_format(kegg_hierarchy_file)
            else:
                LOG.warning("无法识别KEGG格式，尝试简化版解析")
                pathway_info = parse_simple_kegg_format(kegg_hierarchy_file)
        except Exception as e:
            LOG.warning(f"判断KEGG格式失败: {e}")
            # 尝试简化版解析
            pathway_info = parse_simple_kegg_format(kegg_hierarchy_file)
    else:
        LOG.warning(f"KEGG层次结构文件不存在: {kegg_hierarchy_file}")

    LOG.info(f"读取 {len(pathway_info)} 个通路映射")

    # 解析kofam_scan输出 - 修复kofam_scan输出格式
    ko_stats = defaultdict(int)
    gene_annotations = {}
    line_count = 0
    parsed_count = 0

    try:
        with open(kegg_file, 'r') as f:
            for line in f:
                line_count += 1
                if line.startswith('#') or not line.strip():
                    continue

                # kofam_scan输出格式：*  gene_id  KO  score  threshold  evalue  description
                # 使用csv读取器处理带引号的字段
                reader = csv.reader([line], delimiter='\t', quotechar='"')
                parts = next(reader)

                if len(parts) < 3:
                    LOG.warning(f"第{line_count}行字段不足: {line.strip()}")
                    continue

                # 解析字段 - kofam_scan详细格式
                # 格式: flag, gene_name, KO, score, threshold, evalue, description
                flag = parts[0]  # 通常是'*'，表示通过阈值
                gene_id = parts[1] if len(parts) > 1 else ""
                ko_id = parts[2] if len(parts) > 2 else ""
                score = parts[3] if len(parts) > 3 else ""
                threshold = parts[4] if len(parts) > 4 else ""
                evalue = parts[5] if len(parts) > 5 else ""
                description = parts[6] if len(parts) > 6 else ""

                # 只处理通过阈值的结果（flag为'*'）
                if flag == '*' and gene_id and ko_id:
                    parsed_count += 1
                    # 统计KO分布
                    ko_stats[ko_id] += 1

                    # 收集基因注释（每个基因只保留一个最佳KO - 基于score）
                    if gene_id not in gene_annotations:
                        gene_annotations[gene_id] = {
                            'KO': ko_id,
                            'KO_name': ko_info.get(ko_id, description),
                            'pathway': pathway_info.get(ko_id, 'Unknown'),
                            'score': score,
                            'evalue': evalue,
                            'threshold': threshold,
                            'description': description
                        }
                    else:
                        # 如果已经有注释，比较score（越高越好）
                        try:
                            current_score = float(gene_annotations[gene_id]['score']) if gene_annotations[gene_id][
                                'score'] else 0
                            new_score = float(score) if score else 0
                            if new_score > current_score:
                                gene_annotations[gene_id] = {
                                    'KO': ko_id,
                                    'KO_name': ko_info.get(ko_id, description),
                                    'pathway': pathway_info.get(ko_id, 'Unknown'),
                                    'score': score,
                                    'evalue': evalue,
                                    'threshold': threshold,
                                    'description': description
                                }
                        except ValueError:
                            # 如果score不是数字，保留第一个
                            pass

        LOG.info(f"成功解析 {line_count} 行数据，其中 {parsed_count} 行包含注释")

        # 计算注释率
        annotated_count = len(gene_annotations)
        annotation_rate = (annotated_count / total_genes) * 100 if total_genes > 0 else 0

        LOG.info(f"KEGG注释统计:")
        LOG.info(f"  总基因数: {total_genes}")
        LOG.info(f"  注释基因数: {annotated_count}")
        LOG.info(f"  注释率: {annotation_rate:.2f}%")
        LOG.info(f"  发现 {len(ko_stats)} 个不同的KO")
        LOG.info(f"  发现 {len(gene_annotations)} 个有注释的基因")

        # 生成统计报告
        write_kegg_statistics(ko_stats, ko_info, pathway_info, output_prefix, total_genes, annotated_count,
                              annotation_rate)

        # 生成基因注释表
        write_gene_annotations(gene_annotations, output_prefix)

        # 生成通路统计
        write_pathway_statistics(gene_annotations, output_prefix)

        LOG.info(f"KEGG注释解析完成: {len(gene_annotations)} 个基因获得注释")
        return True, annotated_count, annotation_rate

    except Exception as e:
        LOG.error(f"解析KEGG注释失败: {e}")
        import traceback
        LOG.error(traceback.format_exc())
        # 创建空的输出文件
        create_empty_outputs(output_prefix)
        return False, 0, 0


def create_empty_outputs(output_prefix):
    """创建空的输出文件"""
    LOG.info("创建空的输出文件")

    # 空的注释率报告
    rate_file = f"{output_prefix}.annotation_rate.txt"
    with open(rate_file, 'w') as f:
        f.write("KEGG注释率统计报告\n")
        f.write("=" * 50 + "\n")
        f.write(f"总基因数: 0\n")
        f.write(f"注释基因数: 0\n")
        f.write(f"注释率: 0.00%\n")
        f.write(f"未注释基因数: 0\n")
        f.write("\n注意: KEGG注释结果为空\n")

    # 空的基因注释表
    anno_file = f"{output_prefix}.kegg_annotations.tsv"
    with open(anno_file, 'w') as f:
        f.write("GeneID\tKO\tKO_Name\tPathway\tScore\tEvalue\tThreshold\tDescription\n")

    # 空的KO统计
    ko_file = f"{output_prefix}.ko_stats.tsv"
    with open(ko_file, 'w') as f:
        f.write("KO_ID\tKO_Name\tCount\tPathway\tPercentage\n")

    # 空的通路统计
    pathway_file = f"{output_prefix}.pathway_stats.tsv"
    with open(pathway_file, 'w') as f:
        f.write("Pathway\tGene_Count\tPercentage\tGenes\n")

    LOG.info("空的输出文件创建完成")


def write_kegg_statistics(ko_stats, ko_info, pathway_info, output_prefix, total_genes, annotated_count,
                          annotation_rate):
    """写入KEGG统计信息，包括注释率"""

    # KO统计
    ko_file = f"{output_prefix}.ko_stats.tsv"
    with open(ko_file, 'w') as f:
        f.write("KO_ID\tKO_Name\tCount\tPathway\tPercentage\n")
        total_ko = sum(ko_stats.values())
        for ko_id, count in sorted(ko_stats.items(), key=lambda x: x[1], reverse=True):
            ko_name = ko_info.get(ko_id, "Unknown")
            pathway = pathway_info.get(ko_id, "Unknown")
            percentage = (count / total_ko) * 100 if total_ko > 0 else 0
            f.write(f"{ko_id}\t{ko_name}\t{count}\t{pathway}\t{percentage:.2f}%\n")
    LOG.info(f"KO统计写入: {ko_file}")

    # 注释率报告
    rate_file = f"{output_prefix}.annotation_rate.txt"
    with open(rate_file, 'w') as f:
        f.write("KEGG注释率统计报告\n")
        f.write("=" * 50 + "\n")
        f.write(f"总基因数: {total_genes}\n")
        f.write(f"注释基因数: {annotated_count}\n")
        f.write(f"注释率: {annotation_rate:.2f}%\n")
        f.write(f"未注释基因数: {total_genes - annotated_count}\n")
        f.write("\nKO注释统计 (Top 20):\n")
        total_ko = sum(ko_stats.values())
        for ko_id, count in sorted(ko_stats.items(), key=lambda x: x[1], reverse=True)[:20]:
            ko_name = ko_info.get(ko_id, "Unknown")
            percentage = (count / total_ko) * 100 if total_ko > 0 else 0
            f.write(f"{ko_id} ({ko_name}): {count} 个基因 ({percentage:.2f}%)\n")
    LOG.info(f"注释率报告写入: {rate_file}")

    LOG.info("KO统计文件生成完成")


def write_gene_annotations(gene_annotations, output_prefix):
    """生成基因注释表"""

    anno_file = f"{output_prefix}.kegg_annotations.tsv"
    with open(anno_file, 'w') as f:
        f.write("GeneID\tKO\tKO_Name\tPathway\tScore\tEvalue\tThreshold\tDescription\n")

        for gene_id, annotation in gene_annotations.items():
            f.write(f"{gene_id}\t")
            f.write(f"{annotation['KO']}\t")
            f.write(f"{annotation['KO_name']}\t")
            f.write(f"{annotation['pathway']}\t")
            f.write(f"{annotation['score']}\t")
            f.write(f"{annotation['evalue']}\t")
            f.write(f"{annotation['threshold']}\t")
            f.write(f"{annotation.get('description', '')}\n")

    LOG.info(f"基因注释表生成完成: {anno_file} ({len(gene_annotations)} 个基因)")


def write_pathway_statistics(gene_annotations, output_prefix):
    """生成通路统计"""

    pathway_stats = defaultdict(int)
    pathway_genes = defaultdict(list)

    for gene_id, annotation in gene_annotations.items():
        pathway = annotation['pathway']
        if pathway and pathway != "Unknown":
            # 提取通路ID（第一个词）
            pathway_id = pathway.split()[0] if ' ' in pathway else pathway
            pathway_stats[pathway_id] += 1
            pathway_genes[pathway_id].append(gene_id)

    # 通路统计
    pathway_file = f"{output_prefix}.pathway_stats.tsv"
    with open(pathway_file, 'w') as f:
        f.write("Pathway\tPathway_Name\tGene_Count\tPercentage\tGenes\n")
        total_genes = len(gene_annotations)

        for pathway_id, count in sorted(pathway_stats.items(), key=lambda x: x[1], reverse=True):
            # 获取完整的通路名称
            pathway_name = ""
            for gene_id in pathway_genes[pathway_id][:1]:  # 从第一个基因获取通路名称
                annotation = gene_annotations[gene_id]
                if annotation['pathway'] != "Unknown":
                    # 提取完整的通路名称（去掉通路ID）
                    pathway_full = annotation['pathway']
                    if pathway_full.startswith(pathway_id + " "):
                        pathway_name = pathway_full[len(pathway_id) + 1:]
                    else:
                        pathway_name = pathway_full
                    break

            percentage = (count / total_genes) * 100 if total_genes > 0 else 0
            genes_sample = ';'.join(pathway_genes[pathway_id][:5])  # 只显示前5个基因作为示例
            if len(pathway_genes[pathway_id]) > 5:
                genes_sample += f"...(共{len(pathway_genes[pathway_id])}个基因)"
            f.write(f"{pathway_id}\t{pathway_name}\t{count}\t{percentage:.2f}%\t{genes_sample}\n")

    LOG.info(f"通路统计生成完成: {pathway_file} ({len(pathway_stats)} 个通路)")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="独立KEGG注释脚本 - 适配主控脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("-i", "--input", required=True,
                        help="输入蛋白文件 (FASTA格式)")
    parser.add_argument("--profile_db", default="/mnt/databases/kegg/2025/profiles",
                        help="kofam_scan profile数据库路径")
    parser.add_argument("--ko_list", default="/mnt/databases/kegg/2025/ko_list",
                        help="KO列表文件路径")
    parser.add_argument("--kegg_hierarchy", default="/mnt/databases/kegg/2025/ko00001.tsv",
                        help="KEGG层次结构文件路径")
    parser.add_argument("-t", "--threads", type=int, default=32,
                        help="kofam_scan使用的线程数 (默认: 32)")
    parser.add_argument("-e", "--evalue", type=float, default=1e-5,
                        help="E-value阈值 (默认: 1e-5)")
    parser.add_argument("--parse_only", action="store_true",
                        help="仅解析现有结果，不运行kofam_scan")
    parser.add_argument("--log", help="日志文件路径")

    # 输出参数
    parser.add_argument("-o", "--output", required=True,
                        help="输出目录")
    parser.add_argument("-p", "--prefix", required=True,
                        help="输出文件前缀")

    args = parser.parse_args()

    # 设置日志
    setup_logging(args.log)

    # 检查输入文件
    if not os.path.exists(args.input):
        LOG.error(f"输入文件不存在: {args.input}")
        sys.exit(1)

    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)

    # 设置输出文件路径
    kegg_output = os.path.join(args.output, f"{args.prefix}.kegg.out")
    output_prefix = os.path.join(args.output, args.prefix)

    if args.parse_only:
        LOG.info("仅解析模式")
        if not os.path.exists(kegg_output):
            LOG.error(f"KEGG输出文件不存在: {kegg_output}")
            sys.exit(1)
    else:
        # 检查数据库
        for db_file, db_name in [(args.profile_db, "profile数据库"),
                                 (args.ko_list, "KO列表")]:
            if not os.path.exists(db_file):
                LOG.error(f"{db_name}不存在: {db_file}")
                sys.exit(1)

        # 步骤1: 运行kofam_scan
        LOG.info("开始KEGG注释流程")
        if not run_kofam_scan_single(args.input, kegg_output, args.profile_db,
                                     args.ko_list, args.threads, args.evalue):
            LOG.error("kofam_scan运行失败")
            sys.exit(1)

    # 步骤2: 解析和统计
    success, annotated_count, annotation_rate = parse_kofam_output(
        kegg_output, output_prefix, args.ko_list, args.kegg_hierarchy, args.input
    )

    if not success:
        LOG.error("结果解析失败")
        sys.exit(1)

    LOG.info("KEGG注释流程完成!")
    LOG.info(f"主要输出文件:")
    LOG.info(f"  - 原始注释: {kegg_output}")
    LOG.info(f"  - 基因注释: {output_prefix}.kegg_annotations.tsv")
    LOG.info(f"  - KO统计: {output_prefix}.ko_stats.tsv")
    LOG.info(f"  - 通路统计: {output_prefix}.pathway_stats.tsv")
    LOG.info(f"  - 注释率报告: {output_prefix}.annotation_rate.txt")


if __name__ == "__main__":
    main()