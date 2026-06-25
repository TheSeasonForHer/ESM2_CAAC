#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import logging
from datetime import datetime


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def get_all_samples(output_base):
    """从integrated目录获取所有样本"""
    integrated_dir = os.path.join(output_base, "annotation_results", "integrated")
    if not os.path.exists(integrated_dir):
        logging.error(f"integrated目录不存在: {integrated_dir}")
        return []
    samples = [d for d in os.listdir(integrated_dir)
               if os.path.isdir(os.path.join(integrated_dir, d))]
    return sorted(samples)


def generate_overall_report(output_base, samples):
    """生成整体注释率报告"""
    report_file = os.path.join(output_base, "annotation_results", "overall_annotation_report.txt")
    os.makedirs(os.path.dirname(report_file), exist_ok=True)

    try:
        with open(report_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("整体注释率统计报告\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")

            total_genes_all = 0
            annotated_genes_all = 0
            sample_stats = []
            missing_rate_files = []

            for sample in samples:
                rate_file = os.path.join(output_base, "annotation_results", "integrated", sample,
                                         f"{sample}.annotation_rates.txt")
                if not os.path.exists(rate_file):
                    missing_rate_files.append(sample)
                    continue

                with open(rate_file, 'r') as rf:
                    lines = rf.readlines()
                    total_genes = 0
                    annotated_genes = 0
                    for line in lines:
                        if "总基因数:" in line:
                            total_genes = int(line.split(":")[1].strip())
                        elif "获得至少一个注释的基因数:" in line:
                            annotated_genes = int(line.split(":")[1].strip())

                    if total_genes == 0:
                        logging.warning(f"样本 {sample} 总基因数为0，跳过")
                        continue

                    total_genes_all += total_genes
                    annotated_genes_all += annotated_genes
                    rate = annotated_genes / total_genes * 100
                    sample_stats.append({
                        'sample': sample,
                        'total': total_genes,
                        'annotated': annotated_genes,
                        'rate': rate
                    })

            if not sample_stats:
                f.write("没有找到任何有效的样本统计信息。\n")
                if missing_rate_files:
                    f.write("\n缺失annotation_rates.txt的样本: \n")
                    for s in missing_rate_files:
                        f.write(f"  - {s}\n")
                logging.warning("未生成任何统计信息，请先运行integrate步骤生成各样本的annotation_rates.txt")
                return False

            # 写入样本级别统计
            f.write("样本级别注释率:\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'样本':<20} {'总基因数':>12} {'注释基因数':>12} {'注释率':>10}\n")
            f.write("-" * 80 + "\n")
            for stat in sample_stats:
                f.write(f"{stat['sample']:<20} {stat['total']:>12,} {stat['annotated']:>12,} {stat['rate']:>9.2f}%\n")

            # 缺失文件样本提示
            if missing_rate_files:
                f.write("\n" + "-" * 80 + "\n")
                f.write("以下样本缺失.annotation_rates.txt文件，未纳入统计:\n")
                for s in missing_rate_files:
                    f.write(f"  - {s}\n")
                f.write("请先运行: python parallel_annotation_pipeline.py --steps integrate --resume\n")

            # 写入总体统计
            f.write("\n" + "=" * 80 + "\n")
            f.write("总体统计:\n")
            f.write("-" * 80 + "\n")
            overall_rate = annotated_genes_all / total_genes_all * 100 if total_genes_all > 0 else 0
            f.write(f"总基因数: {total_genes_all:,}\n")
            f.write(f"总注释基因数: {annotated_genes_all:,}\n")
            f.write(f"整体注释率: {overall_rate:.2f}%\n")
            f.write(f"未注释基因数: {total_genes_all - annotated_genes_all:,}\n")

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

        logging.info(f"整体报告已重新生成: {report_file}")
        if missing_rate_files:
            logging.warning(f"{len(missing_rate_files)} 个样本缺失注释率文件，请先运行integrate步骤")
        return True
    except Exception as e:
        logging.error(f"生成整体报告失败: {e}")
        return False


def main():
    setup_logging()
    output_base = "/mnt/zjwdata/2/gene_function/"  # 根据实际路径修改

    samples = get_all_samples(output_base)
    if not samples:
        logging.error(f"未在 {output_base}/annotation_results/integrated/ 下找到任何样本目录")
        sys.exit(1)

    logging.info(f"找到 {len(samples)} 个样本: {samples}")
    success = generate_overall_report(output_base, samples)
    if success:
        logging.info("整体报告补全完成！")
    else:
        logging.error("生成报告失败，请检查日志。")


if __name__ == "__main__":
    main()