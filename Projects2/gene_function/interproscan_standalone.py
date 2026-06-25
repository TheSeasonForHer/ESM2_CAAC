#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
修正版独立InterProScan注释脚本
用于第二阶段功能注释中的InterProScan部分
修复InterProScan卡在99%的问题
添加超时控制、更好的错误处理和进度监控
添加注释率统计功能
取消超时设置，使用无限等待
"""

import os
import sys
import argparse
import logging
import subprocess
import glob
import time
import signal
from collections import defaultdict

LOG = logging.getLogger(__name__)
__version__ = "1.3.0"
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


def run_command_without_timeout(cmd, description=""):
    """运行命令行工具，无超时控制"""
    LOG.info(f"运行: {description}")
    LOG.info(f"命令: {cmd}")

    try:
        # 使用Popen以便更好地控制进程
        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            bufsize=1
        )

        # 实时输出处理
        stdout_lines = []
        stderr_lines = []

        # 读取输出，避免缓冲区阻塞
        def read_output():
            while True:
                # 读取标准输出
                stdout_line = process.stdout.readline()
                if stdout_line:
                    stdout_lines.append(stdout_line)
                    # 过滤并记录重要信息
                    line_clean = stdout_line.strip()
                    if line_clean:
                        # 记录重要进度信息
                        if any(keyword in line_clean.lower() for keyword in
                               ['completed', 'running', 'loading', 'uploaded', 'progress',
                                'analysing', 'processing', 'signature', 'database']):
                            LOG.info(f"InterProScan进度: {line_clean}")
                        # 每1000行记录一次摘要
                        elif len(stdout_lines) % 1000 == 0:
                            LOG.info(f"已处理 {len(stdout_lines)} 行输出")

                # 读取标准错误
                stderr_line = process.stderr.readline()
                if stderr_line:
                    stderr_lines.append(stderr_line)
                    line_clean = stderr_line.strip()
                    if line_clean:
                        if 'error' in line_clean.lower() or 'exception' in line_clean.lower():
                            LOG.error(f"InterProScan错误: {line_clean}")
                        else:
                            LOG.warning(f"InterProScan警告: {line_clean}")

                # 检查进程是否结束
                if process.poll() is not None:
                    # 读取剩余输出
                    remaining_stdout, remaining_stderr = process.communicate()
                    if remaining_stdout:
                        for line in remaining_stdout.split('\n'):
                            if line.strip():
                                stdout_lines.append(line + '\n')
                                if any(keyword in line.lower() for keyword in ['completed', 'finished', 'done']):
                                    LOG.info(f"InterProScan: {line.strip()}")
                    if remaining_stderr:
                        for line in remaining_stderr.split('\n'):
                            if line.strip():
                                stderr_lines.append(line + '\n')
                                if 'error' in line.lower():
                                    LOG.error(f"InterProScan错误: {line.strip()}")
                    break

                # 短暂休眠避免CPU占用过高
                time.sleep(0.1)

        # 在后台线程中读取输出
        import threading
        output_thread = threading.Thread(target=read_output)
        output_thread.daemon = True
        output_thread.start()

        # 等待进程结束
        process.wait()
        output_thread.join(timeout=10)

        if process.returncode == 0:
            LOG.info(f"{description} 完成")
            return True
        else:
            LOG.error(f"{description} 失败，返回码: {process.returncode}")
            if stderr_lines:
                LOG.error(f"最后10行错误输出:")
                for line in stderr_lines[-10:]:
                    LOG.error(f"  {line.strip()}")
            return False

    except Exception as e:
        LOG.error(f"{description} 执行异常: {e}")
        return False


def setup_interproscan_environment_fixed():
    """修正InterProScan环境设置"""
    # 清理可能干扰的环境变量
    if 'JAVA_TOOL_OPTIONS' in os.environ:
        del os.environ['JAVA_TOOL_OPTIONS']
        LOG.info("已清理 JAVA_TOOL_OPTIONS 环境变量")

    # 设置InterProScan特定的环境变量
    os.environ['INTERPROSCAN_HOME'] = '/root/tools/interproscan/interproscan-5.28-67.0'

    # 设置JVM内存参数 - 增加到96GB
    os.environ['JAVA_OPTS'] = '-Xmx96g -Xms32g -XX:ParallelGCThreads=32 -XX:+UseParallelGC -XX:+UseParallelOldGC'

    # 添加Java 8到PATH
    java_path = '/usr/lib/jvm/java-8-openjdk-amd64/jre/bin'
    if java_path not in os.environ.get('PATH', ''):
        os.environ['PATH'] = f"{java_path}:{os.environ.get('PATH', '')}"
        LOG.info(f"已添加Java 8到PATH: {java_path}")


def check_interproscan_installation():
    """检查InterProScan安装状态"""
    LOG.info("检查InterProScan安装...")

    interproscan_path = "/root/tools/interproscan/interproscan-5.28-67.0/interproscan.sh"

    if not os.path.exists(interproscan_path):
        LOG.error(f"InterProScan未找到: {interproscan_path}")
        return False

    LOG.info(f"InterProScan脚本存在: {interproscan_path}")

    # 检查主要数据库文件
    required_dbs = {
        'Pfam': 'data/pfam/31.0/pfam_a.hmm',
        'SMART': 'data/smart/7.1/smart.HMMs',
        'TIGRFAM': 'data/tigrfam/15.0/TIGRFAMs_HMM.LIB'
    }

    base_path = "/root/tools/interproscan/interproscan-5.28-67.0"
    missing_dbs = []

    for db_name, db_path in required_dbs.items():
        full_path = os.path.join(base_path, db_path)
        if not os.path.exists(full_path):
            missing_dbs.append(db_name)
            LOG.error(f"数据库 {db_name} 未找到: {full_path}")
        else:
            LOG.info(f"数据库 {db_name} 检查通过")

    if missing_dbs:
        LOG.warning(f"缺少数据库: {', '.join(missing_dbs)}")
        LOG.warning("InterProScan可能仍能运行，但某些分析将不可用")

    LOG.info("InterProScan安装检查完成")
    return True


def count_sequences(fasta_file):
    """统计FASTA文件中的序列数量"""
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


def estimate_interproscan_time(sequence_count):
    """根据序列数量估算运行时间"""
    # 优化估算：考虑32线程并行处理
    estimated_seconds = sequence_count * 0.5  # 每序列0.5秒（保守估计）
    hours = estimated_seconds // 3600
    minutes = (estimated_seconds % 3600) // 60

    LOG.info(f"序列数量: {sequence_count:,}")
    LOG.info(f"预计运行时间: {int(hours)}小时 {int(minutes)}分钟 (使用32线程)")

    return hours, minutes


def run_interproscan_single(input_file, output_file, threads=32):
    """运行单个InterProScan任务（无超时）"""
    LOG.info(f"开始InterProScan注释: {input_file}")

    # 统计序列数量并估算时间
    seq_count = count_sequences(input_file)
    if seq_count == 0:
        LOG.error(f"输入文件为空或无法读取: {input_file}")
        return False, seq_count

    hours, minutes = estimate_interproscan_time(seq_count)

    # 构建InterProScan命令 - 优化参数
    cmd = f"""
    /root/tools/interproscan/interproscan-5.28-67.0/interproscan.sh \
      -i {input_file} \
      -appl Pfam,TIGRFAM,SMART \
      -iprlookup \
      -goterms \
      -dp \
      --cpu {threads} \
      -t p \
      -f TSV \
      -o {output_file}
    """

    LOG.info(f"InterProScan命令将运行，预计需要 {int(hours)}小时 {int(minutes)}分钟")
    LOG.info(f"请耐心等待，大文件可能需要较长时间...")

    # 运行命令（无超时）
    success = run_command_without_timeout(
        cmd,
        f"InterProScan注释 ({seq_count} 个序列, {threads}线程)"
    )

    if success:
        # 检查输出文件是否完整
        if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
            file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
            LOG.info(f"InterProScan完成，输出文件: {output_file} ({file_size_mb:.2f} MB)")
            return True, seq_count
        else:
            LOG.error(f"输出文件不存在或为空: {output_file}")
            return False, seq_count
    else:
        LOG.error(f"InterProScan执行失败")
        return False, seq_count


def parse_interproscan_output(ipr_file, output_prefix, input_file=None):
    """解析InterProScan输出文件，添加注释率统计"""
    LOG.info(f"解析InterProScan输出: {ipr_file}")

    # 初始化统计字典
    domain_stats = defaultdict(int)
    go_stats = defaultdict(int)
    ipr_stats = defaultdict(int)

    # 统计注释的蛋白
    annotated_proteins = set()
    total_proteins = 0

    # 如果提供了输入文件，统计总蛋白数
    if input_file:
        total_proteins = count_sequences(input_file)
        LOG.info(f"从输入文件统计总蛋白数: {total_proteins}")

    # 解析TSV文件
    line_count = 0
    try:
        with open(ipr_file, 'r') as f:
            for line in f:
                line_count += 1
                if line.startswith('#') or not line.strip():
                    continue

                parts = line.strip().split('\t')
                if len(parts) < 12:
                    LOG.warning(f"第{line_count}行字段不足: {line.strip()}")
                    continue

                protein_id = parts[0]
                database = parts[3]
                signature = parts[4]
                description = parts[5]
                start = parts[6]
                end = parts[7]
                evalue = parts[8]
                status = parts[9]
                date = parts[10]
                interpro_id = parts[11] if len(parts) > 11 else ""
                interpro_description = parts[12] if len(parts) > 12 else ""
                go_terms = parts[13] if len(parts) > 13 else ""

                # 添加蛋白到注释集合
                annotated_proteins.add(protein_id)

                # 统计数据库使用情况
                domain_stats[database] += 1

                # 统计InterPro条目
                if interpro_id:
                    ipr_stats[interpro_id] += 1

                # 统计GO条目
                if go_terms:
                    for go_term in go_terms.split('|'):
                        if go_term:
                            go_stats[go_term] += 1

        LOG.info(f"成功解析 {line_count} 行数据")

    except Exception as e:
        LOG.error(f"解析InterProScan输出文件失败: {e}")
        return False, 0, 0

    # 计算注释率
    annotated_count = len(annotated_proteins)

    # 如果无法从输入文件获取总数，尝试从注释结果中估计
    if total_proteins == 0:
        # 尝试从输出文件中估计（不准确）
        LOG.warning("未提供输入文件或无法统计总基因数，使用注释基因数作为估计")
        # 这可能不准确，因为有些蛋白可能没有注释
        total_proteins = annotated_count
        annotation_rate = 100.0 if total_proteins > 0 else 0
    else:
        annotation_rate = (annotated_count / total_proteins) * 100 if total_proteins > 0 else 0

    LOG.info(f"InterProScan注释统计:")
    LOG.info(f"  总蛋白数: {total_proteins}")
    LOG.info(f"  注释蛋白数: {annotated_count}")
    LOG.info(f"  注释率: {annotation_rate:.2f}%")
    LOG.info(f"  发现 {len(domain_stats)} 个数据库的注释")
    LOG.info(f"  发现 {len(ipr_stats)} 个InterPro条目")
    LOG.info(f"  发现 {len(go_stats)} 个GO术语")

    # 生成统计报告
    write_statistics(domain_stats, go_stats, ipr_stats, output_prefix, total_proteins, annotated_count, annotation_rate)

    # 生成简化注释表
    write_simplified_annotation(ipr_file, output_prefix)

    return True, annotated_count, annotation_rate


def write_statistics(domain_stats, go_stats, ipr_stats, output_prefix, total_proteins, annotated_count,
                     annotation_rate):
    """写入统计信息，包括注释率"""

    # 数据库使用统计
    domain_file = f"{output_prefix}.domain_stats.tsv"
    with open(domain_file, 'w') as f:
        f.write("Database\tCount\tPercentage\n")
        total_annotations = sum(domain_stats.values())
        for db, count in sorted(domain_stats.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_annotations) * 100 if total_annotations > 0 else 0
            f.write(f"{db}\t{count}\t{percentage:.2f}%\n")
    LOG.info(f"数据库统计写入: {domain_file}")

    # GO术语统计（取前100个）
    go_file = f"{output_prefix}.go_stats.tsv"
    with open(go_file, 'w') as f:
        f.write("GO_Term\tCount\n")
        for go, count in sorted(go_stats.items(), key=lambda x: x[1], reverse=True)[:100]:
            f.write(f"{go}\t{count}\n")
    LOG.info(f"GO统计写入: {go_file}")

    # InterPro条目统计（取前100个）
    ipr_file = f"{output_prefix}.ipr_stats.tsv"
    with open(ipr_file, 'w') as f:
        f.write("InterPro_ID\tCount\n")
        for ipr, count in sorted(ipr_stats.items(), key=lambda x: x[1], reverse=True)[:100]:
            f.write(f"{ipr}\t{count}\n")
    LOG.info(f"InterPro统计写入: {ipr_file}")

    # 注释率报告
    rate_file = f"{output_prefix}.annotation_rate.txt"
    with open(rate_file, 'w') as f:
        f.write("InterProScan注释率统计报告\n")
        f.write("=" * 50 + "\n")
        f.write(f"总蛋白数: {total_proteins}\n")
        f.write(f"注释蛋白数: {annotated_count}\n")
        f.write(f"注释率: {annotation_rate:.2f}%\n")
        f.write(f"未注释蛋白数: {total_proteins - annotated_count}\n")
        f.write("\n数据库使用情况:\n")
        total_annotations = sum(domain_stats.values())
        for db, count in sorted(domain_stats.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_annotations) * 100 if total_annotations > 0 else 0
            f.write(f"{db}: {count} 个注释 ({percentage:.2f}%)\n")
    LOG.info(f"注释率报告写入: {rate_file}")

    LOG.info("所有统计文件生成完成")


def write_simplified_annotation(ipr_file, output_prefix):
    """生成简化注释表"""
    LOG.info(f"生成简化注释表")

    protein_annotations = defaultdict(list)
    line_count = 0

    try:
        with open(ipr_file, 'r') as f:
            for line in f:
                line_count += 1
                if line.startswith('#') or not line.strip():
                    continue

                parts = line.strip().split('\t')
                if len(parts) < 12:
                    continue

                protein_id = parts[0]
                database = parts[3]
                signature = parts[4]
                description = parts[5]
                interpro_id = parts[11] if len(parts) > 11 else ""
                interpro_description = parts[12] if len(parts) > 12 else ""
                go_terms = parts[13] if len(parts) > 13 else ""

                # 收集每个蛋白的注释信息
                annotation = {
                    'database': database,
                    'signature': signature,
                    'description': description,
                    'interpro_id': interpro_id,
                    'interpro_description': interpro_description,
                    'go_terms': go_terms
                }
                protein_annotations[protein_id].append(annotation)

        # 写入简化注释表
        simplified_file = f"{output_prefix}.simplified_annotations.tsv"
        with open(simplified_file, 'w') as f:
            f.write("ProteinID\tDatabases\tSignatures\tDescriptions\tInterPro_IDs\tGO_Terms\n")

            for protein_id, annotations in protein_annotations.items():
                databases = set()
                signatures = set()
                descriptions = set()
                interpro_ids = set()
                go_terms = set()

                for ann in annotations:
                    databases.add(ann['database'])
                    signatures.add(ann['signature'])
                    if ann['description']:
                        descriptions.add(ann['description'])
                    if ann['interpro_id']:
                        interpro_ids.add(ann['interpro_id'])
                    if ann['go_terms']:
                        for go in ann['go_terms'].split('|'):
                            if go:
                                go_terms.add(go)

                f.write(f"{protein_id}\t")
                f.write(f"{';'.join(sorted(databases))}\t")
                f.write(f"{';'.join(sorted(signatures))}\t")
                f.write(f"{';'.join(sorted(descriptions))}\t")
                f.write(f"{';'.join(sorted(interpro_ids))}\t")
                f.write(f"{';'.join(sorted(go_terms))}\n")

        LOG.info(f"简化注释表生成完成: {simplified_file} ({len(protein_annotations)} 个蛋白)")

    except Exception as e:
        LOG.error(f"生成简化注释表失败: {e}")
        return False

    return True


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="修正版独立InterProScan注释脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("-i", "--input", required=True,
                        help="输入蛋白文件 (FASTA格式)")
    parser.add_argument("-o", "--output", required=True,
                        help="输出目录")
    parser.add_argument("-p", "--prefix", required=True,
                        help="输出文件前缀")
    parser.add_argument("-t", "--threads", type=int, default=32,
                        help="InterProScan使用的线程数 (默认: 32)")
    parser.add_argument("--timeout", type=int, default=0,
                        help="超时时间(秒)，0表示无超时 (默认: 0)")
    parser.add_argument("--parse_only", action="store_true",
                        help="仅解析现有结果，不运行InterProScan")
    parser.add_argument("--log", help="日志文件路径")

    args = parser.parse_args()

    # 设置日志
    setup_logging(args.log)

    # 检查输入文件
    if not os.path.exists(args.input):
        LOG.error(f"输入文件不存在: {args.input}")
        sys.exit(1)

    # 设置环境
    setup_interproscan_environment_fixed()

    # 检查安装
    if not check_interproscan_installation():
        LOG.warning("InterProScan安装检查发现问题，继续运行...")

    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)

    output_file = os.path.join(args.output, f"{args.prefix}.ipr.out")

    if args.parse_only:
        LOG.info("仅解析模式")
        if not os.path.exists(output_file):
            LOG.error(f"输出文件不存在: {output_file}")
            sys.exit(1)
        total_genes = 0  # 在仅解析模式下无法获取总基因数
    else:
        # 运行InterProScan
        LOG.info(f"开始InterProScan注释，线程数: {args.threads}")
        success, total_genes = run_interproscan_single(args.input, output_file, args.threads)
        if not success:
            LOG.error("InterProScan运行失败")
            sys.exit(1)

    # 解析结果
    LOG.info("开始解析InterProScan结果...")
    output_prefix = os.path.join(args.output, args.prefix)

    try:
        success, annotated_count, annotation_rate = parse_interproscan_output(output_file, output_prefix, args.input)
        if success:
            LOG.info(f"结果解析完成: {output_prefix}.*")
            LOG.info(f"注释率: {annotation_rate:.2f}% ({annotated_count}/{total_genes})")
        else:
            LOG.error("结果解析失败")
            sys.exit(1)
    except Exception as e:
        LOG.error(f"结果解析失败: {e}")
        sys.exit(1)

    LOG.info("InterProScan注释流程完成!")


if __name__ == "__main__":
    main()