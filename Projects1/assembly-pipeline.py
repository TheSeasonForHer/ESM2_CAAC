#!/usr/bin/env python3
"""
玉米青贮宏基因组组装流程 - 增量并行版
修复文件检查卡死问题，跳过不必要的完整性验证
"""

import subprocess
import concurrent.futures
from pathlib import Path
import sys
import json
import os
import shutil
import time
import threading
from datetime import datetime, timedelta
import psutil
import math
import re


class FailureAnalyzer:
    """失败分析器"""

    @staticmethod
    def analyze_megahit_failure(log_file_path):
        if not os.path.exists(log_file_path):
            return "日志文件不存在", [], []

        try:
            with open(log_file_path, 'r') as f:
                log_content = f.read()
        except:
            return "无法读取日志文件", [], []

        errors = []
        warnings = []

        error_patterns = {
            '内存不足': [r'MemoryError', r'out of memory', r'Killed', r'SIGKILL'],
            '磁盘空间不足': [r'No space left on device', r'Disk quota exceeded'],
            '输入文件问题': [r'Invalid input file', r'File not found', r'empty file'],
            '权限问题': [r'Permission denied', r'Access denied'],
            'k-mer范围问题': [r'kmer size', r'k-mer.*range'],
            '线程问题': [r'thread', r'parallel', r'pthread'],
        }

        for error_type, patterns in error_patterns.items():
            for pattern in patterns:
                if re.search(pattern, log_content, re.IGNORECASE):
                    if error_type not in errors:
                        errors.append(error_type)

        warning_patterns = {
            '低质量数据': [r'low quality', r'poor quality'],
            '数据量不足': [r'insufficient data', r'not enough data'],
            '高复杂度': [r'high complexity', r'complex'],
        }

        for warning_type, patterns in warning_patterns.items():
            for pattern in patterns:
                if re.search(pattern, log_content, re.IGNORECASE):
                    if warning_type not in warnings:
                        warnings.append(warning_type)

        lines = log_content.strip().split('\n')
        last_lines = lines[-10:] if len(lines) > 10 else lines

        return "\n".join(last_lines), errors, warnings

    @staticmethod
    def get_system_status():
        status = {
            'disk_free_gb': None,
            'memory_available_gb': None,
            'memory_percent': None,
            'cpu_percent': None,
        }

        try:
            disk = psutil.disk_usage('/')
            status['disk_free_gb'] = disk.free / (1024 ** 3)

            memory = psutil.virtual_memory()
            status['memory_available_gb'] = memory.available / (1024 ** 3)
            status['memory_percent'] = memory.percent

            status['cpu_percent'] = psutil.cpu_percent(interval=1)

        except Exception as e:
            print(f"获取系统状态失败: {e}")

        return status

    @staticmethod
    def check_file_integrity(file_path):
        """快速检查文件完整性 - 只检查存在性和非空，不验证格式"""
        path = Path(file_path)
        file_path_str = str(file_path)

        if not path.exists():
            return False, "文件不存在"

        try:
            size = path.stat().st_size
            if size == 0:
                return False, "文件为空"

            # 对于.gz文件，只检查最后几个字节（gzip尾标）而不是解压
            if file_path_str.endswith('.gz'):
                # 读取最后8个字节检查gzip尾标
                with open(file_path_str, 'rb') as f:
                    f.seek(-8, 2)
                    tail = f.read()
                    # gzip尾标: 最后4字节是CRC32，再前4字节是原始大小
                    if len(tail) == 8:
                        return True, f"文件正常 ({size/1024/1024:.1f} MB)"
                    else:
                        return False, "gzip文件不完整"

            return True, f"文件正常 ({size/1024/1024:.1f} MB)"

        except Exception as e:
            return False, f"检查文件时出错: {e}"


class ProgressMonitor:
    """进度监控器"""

    def __init__(self):
        self.start_time = None
        self.sample_status = {}
        self.lock = threading.Lock()

    def start_sample(self, sample_name):
        with self.lock:
            self.sample_status[sample_name] = {
                'status': 'waiting',
                'start_time': None,
                'end_time': None,
                'current_stage': '等待开始',
                'progress': 0,
                'elapsed_time': '0:00:00',
                'error_message': None,
                'diagnosis': None
            }
            if self.start_time is None:
                self.start_time = datetime.now()

    def update_sample(self, sample_name, stage, progress):
        with self.lock:
            if sample_name in self.sample_status:
                self.sample_status[sample_name]['current_stage'] = stage
                self.sample_status[sample_name]['progress'] = progress
                if self.sample_status[sample_name]['status'] == 'running' and self.sample_status[sample_name][
                    'start_time']:
                    elapsed = datetime.now() - self.sample_status[sample_name]['start_time']
                    self.sample_status[sample_name]['elapsed_time'] = str(elapsed).split('.')[0]

    def start_running(self, sample_name):
        with self.lock:
            if sample_name in self.sample_status:
                self.sample_status[sample_name]['status'] = 'running'
                self.sample_status[sample_name]['start_time'] = datetime.now()

    def complete_sample(self, sample_name, success=True, error_message=None, diagnosis=None):
        with self.lock:
            if sample_name in self.sample_status:
                self.sample_status[sample_name]['status'] = 'completed' if success else 'failed'
                self.sample_status[sample_name]['progress'] = 100 if success else 0
                self.sample_status[sample_name]['end_time'] = datetime.now()
                self.sample_status[sample_name]['error_message'] = error_message
                self.sample_status[sample_name]['diagnosis'] = diagnosis
                if self.sample_status[sample_name]['start_time']:
                    elapsed = self.sample_status[sample_name]['end_time'] - self.sample_status[sample_name][
                        'start_time']
                    self.sample_status[sample_name]['elapsed_time'] = str(elapsed).split('.')[0]

    def get_progress_summary(self):
        with self.lock:
            if not self.sample_status:
                return "等待开始..."

            total = len(self.sample_status)
            completed = sum(1 for s in self.sample_status.values() if s['status'] == 'completed')
            running = sum(1 for s in self.sample_status.values() if s['status'] == 'running')
            failed = sum(1 for s in self.sample_status.values() if s['status'] == 'failed')
            waiting = sum(1 for s in self.sample_status.values() if s['status'] == 'waiting')

            elapsed = datetime.now() - self.start_time if self.start_time else timedelta(0)

            if completed > 0 and running == 0 and waiting == 0:
                avg_time_per_sample = elapsed / completed
                eta_str = "全部完成"
            elif completed > 0 and running > 0:
                avg_time_per_sample = elapsed / (completed + running)
                remaining_samples = waiting
                eta = avg_time_per_sample * remaining_samples
                eta_str = f"预计剩余: {str(eta).split('.')[0]}"
            else:
                eta_str = "计算中..."

            summary = f"\n=== 玉米青贮宏基因组组装进度监控 ===\n"
            summary += f"总样本: {total} | 完成: {completed} | 运行中: {running} | 失败: {failed} | 等待: {waiting}\n"
            summary += f"总运行时间: {str(elapsed).split('.')[0]} | {eta_str}\n"

            system_status = FailureAnalyzer.get_system_status()
            if system_status['disk_free_gb']:
                summary += f"磁盘剩余: {system_status['disk_free_gb']:.1f}GB | "
                summary += f"内存可用: {system_status['memory_available_gb']:.1f}GB ({system_status['memory_percent']:.1f}%使用)\n"

            summary += "-" * 70 + "\n"

            running_samples = [s for s in self.sample_status.items() if s[1]['status'] == 'running']
            if running_samples:
                summary += "当前运行:\n"
                for sample_name, status in running_samples:
                    summary += f"  🔵 {sample_name}: {status['current_stage']} - {status['progress']}% (已运行: {status['elapsed_time']})\n"
                summary += "-" * 70 + "\n"

            failed_samples = [(k, v) for k, v in self.sample_status.items() if v['status'] == 'failed']
            if failed_samples:
                summary += "失败样本:\n"
                for sample_name, status in failed_samples:
                    summary += f"  ❌ {sample_name}: {status['error_message']}\n"
                summary += "-" * 70 + "\n"

            completed_samples = [(k, v) for k, v in self.sample_status.items() if v['status'] == 'completed']
            if completed_samples:
                summary += "已完成:\n"
                for sample_name, status in completed_samples:
                    summary += f"  ✅ {sample_name}: 完成 (用时: {status['elapsed_time']})\n"

            waiting_samples = [s for s in self.sample_status.items() if s[1]['status'] == 'waiting']
            if waiting_samples:
                summary += "-" * 70 + "\n"
                summary += "等待中:\n"
                for sample_name, status in waiting_samples[:5]:
                    summary += f"  ⏳ {sample_name}: 等待运行\n"
                if len(waiting_samples) > 5:
                    summary += f"  ... 还有 {len(waiting_samples) - 5} 个样本等待\n"

            return summary

    def get_total_runtime(self):
        if self.start_time:
            elapsed = datetime.now() - self.start_time
            return str(elapsed).split('.')[0]
        return "0:00:00"


class AssemblyPipeline:
    def __init__(self, qc_data_path, output_dir, cpu_limit=80, parallel_samples=2):
        self.qc_data_path = Path(qc_data_path)
        self.output_dir = Path(output_dir)
        self.samples = []
        self.monitor = ProgressMonitor()
        self.cpu_limit = cpu_limit
        self.parallel_samples = parallel_samples

        # k-mer 设置
        self.kmer_list = "87,113,141"
        self.min_contig_len = 500

        # 计算静态线程分配
        self.total_cpus = os.cpu_count() or 96
        self.available_cpus = max(2, math.floor(self.total_cpus * self.cpu_limit / 100))
        self.threads_per_sample = max(2, self.available_cpus // self.parallel_samples)

        print(f"🚀 玉米青贮宏基因组组装流程 (增量并行版)")
        print(f"QC数据路径: {self.qc_data_path}")
        print(f"输出路径: {self.output_dir}")
        print(f"k-mer参数: {self.kmer_list}")
        print(f"Contig长度: {self.min_contig_len}bp")
        print(f"系统总CPU: {self.total_cpus} | 使用率限制: {self.cpu_limit}%")
        print(f"并行样本数: {self.parallel_samples} | 每样本线程: {self.threads_per_sample}")

        # 创建输出目录
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / "logs").mkdir(exist_ok=True)
        (self.output_dir / "assemblies").mkdir(exist_ok=True)
        (self.output_dir / "genes").mkdir(exist_ok=True)
        (self.output_dir / "proteins").mkdir(exist_ok=True)
        (self.output_dir / "progress").mkdir(exist_ok=True)
        (self.output_dir / "diagnosis").mkdir(exist_ok=True)

    def start_progress_monitor(self):
        """启动进度监控线程"""

        def monitor_loop():
            last_update = datetime.now()
            while True:
                with self.monitor.lock:
                    running = sum(1 for s in self.monitor.sample_status.values() if s['status'] == 'running')
                    waiting = sum(1 for s in self.monitor.sample_status.values() if s['status'] == 'waiting')

                # 只有还有任务在跑或等待时才继续监控
                if running == 0 and waiting == 0 and self.monitor.start_time:
                    print(self.monitor.get_progress_summary())
                    print("\n🎉 所有样本处理完成!")
                    break

                if (datetime.now() - last_update).seconds >= 10:
                    print(self.monitor.get_progress_summary())
                    self.save_progress_to_file()
                    last_update = datetime.now()

                time.sleep(2)

        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
        return monitor_thread

    def save_progress_to_file(self):
        """保存进度到JSON文件"""
        progress_data = {
            'timestamp': datetime.now().isoformat(),
            'samples': self.monitor.sample_status,
            'summary': {
                'total': len(self.monitor.sample_status),
                'completed': sum(1 for s in self.monitor.sample_status.values() if s['status'] == 'completed'),
                'running': sum(1 for s in self.monitor.sample_status.values() if s['status'] == 'running'),
                'failed': sum(1 for s in self.monitor.sample_status.values() if s['status'] == 'failed'),
                'waiting': sum(1 for s in self.monitor.sample_status.values() if s['status'] == 'waiting')
            },
            'kmer_parameters': self.kmer_list,
            'contig_length': self.min_contig_len,
            'cpu_usage': f"{self.cpu_limit}%",
            'threads_per_sample': self.threads_per_sample,
            'parallel_samples': self.parallel_samples
        }

        progress_file = self.output_dir / "progress" / "assembly_progress.json"
        with open(progress_file, 'w') as f:
            json.dump(progress_data, f, indent=2, default=str)

    def discover_samples(self):
        """发现所有质控后的样本 - 快速检查，不验证格式"""
        print("\n=== Discovering QCed Samples ===")

        r1_files = list(self.qc_data_path.glob("*_R1_clean.fq.gz"))
        print(f"找到 R1 文件: {len(r1_files)} 个")

        all_samples = []
        to_process = []
        skipped_all = 0
        skipped_asm = 0

        for r1_file in sorted(r1_files):
            sample_name = r1_file.name.replace('_R1_clean.fq.gz', '')
            r2_file = self.qc_data_path / f"{sample_name}_R2_clean.fq.gz"

            if not r2_file.exists():
                print(f"❌ {sample_name}: R2文件缺失")
                all_samples.append({
                    'name': sample_name,
                    'status': 'failed',
                    'reason': 'R2文件缺失',
                    'to_process': False
                })
                continue

            # 快速检查：只验证文件存在且非空
            r1_size = r1_file.stat().st_size
            r2_size = r2_file.stat().st_size

            if r1_size == 0 or r2_size == 0:
                print(f"❌ {sample_name}: 输入文件为空")
                all_samples.append({
                    'name': sample_name,
                    'status': 'failed',
                    'reason': '输入文件为空',
                    'to_process': False
                })
                continue

            # 检查组装是否已完成
            asm_dir = self.output_dir / "assemblies" / f"{sample_name}_megahit_k87_141"
            contig_file = asm_dir / "final.contigs.fa"
            assembly_done = contig_file.exists() and contig_file.stat().st_size > 1000

            # 检查基因预测是否已完成
            protein_file = self.output_dir / "proteins" / f"{sample_name}_proteins.faa"
            gene_file = self.output_dir / "genes" / f"{sample_name}_genes.fna"
            genes_done = protein_file.exists() and gene_file.exists() and protein_file.stat().st_size > 0

            if assembly_done and genes_done:
                skipped_all += 1
                all_samples.append({
                    'name': sample_name,
                    'status': 'completed',
                    'to_process': False
                })
            elif assembly_done and not genes_done:
                skipped_asm += 1
                sample_info = {
                    'name': sample_name,
                    'R1': str(r1_file),
                    'R2': str(r2_file),
                    'size_gb': r1_size / 1024 / 1024 / 1024,
                    'assembly_done': True,
                    'contig_file': str(contig_file),
                    'status': 'waiting',
                    'to_process': True
                }
                to_process.append(sample_info)
                all_samples.append({
                    'name': sample_name,
                    'status': 'waiting',
                    'to_process': True
                })
            else:
                sample_info = {
                    'name': sample_name,
                    'R1': str(r1_file),
                    'R2': str(r2_file),
                    'size_gb': r1_size / 1024 / 1024 / 1024,
                    'assembly_done': False,
                    'contig_file': None,
                    'status': 'waiting',
                    'to_process': True
                }
                to_process.append(sample_info)
                all_samples.append({
                    'name': sample_name,
                    'status': 'waiting',
                    'to_process': True
                })

        print(f"\n=== 样本统计 ===")
        print(f"全部完成跳过: {skipped_all} 个")
        print(f"需要处理: {len(to_process)} 个 (其中组装已完成仅需基因预测: {skipped_asm} 个)")
        if len(r1_files) - len(to_process) - skipped_all > 0:
            print(f"问题样本: {len(r1_files) - len(to_process) - skipped_all} 个")

        self.samples = to_process
        return all_samples

    def run_megahit_assembly(self, sample):
        """使用MEGAHIT进行宏基因组组装 - 纯函数"""
        sample_name = sample['name']

        if sample.get('assembly_done'):
            return sample_name, True, sample['contig_file'], "组装结果已存在，跳过"

        output_dir = self.output_dir / "assemblies" / f"{sample_name}_megahit_k87_141"
        log_file = self.output_dir / "logs" / f"{sample_name}_megahit.log"

        if output_dir.exists():
            shutil.rmtree(output_dir, ignore_errors=True)

        size_gb = sample['size_gb']
        if size_gb < 5:
            memory_setting = "0.5"
        elif size_gb < 20:
            memory_setting = "0.7"
        else:
            memory_setting = "0.8"

        cmd = [
            'megahit',
            '-1', sample['R1'],
            '-2', sample['R2'],
            '-o', str(output_dir),
            '--k-list', self.kmer_list,
            '--min-contig-len', str(self.min_contig_len),
            '--min-count', '2',
            '--memory', memory_setting,
            '--num-cpu-threads', str(self.threads_per_sample),
        ]

        try:
            with open(log_file, 'w') as log:
                log.write(f"Command: {' '.join(cmd)}\n")
                log.write(f"Sample: {sample_name}, Size: {size_gb:.2f}GB\n")
                log.write(f"Threads: {self.threads_per_sample}, Memory: {memory_setting}\n")
                log.write("-" * 50 + "\n")

                process = subprocess.Popen(
                    cmd,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True
                )
                process.wait()
                returncode = process.returncode

            if returncode == 0:
                contig_file = output_dir / "final.contigs.fa"
                if contig_file.exists() and contig_file.stat().st_size > 1000:
                    try:
                        count_cmd = f"grep -c '>' {contig_file}"
                        contig_count = subprocess.check_output(count_cmd, shell=True).decode().strip()
                    except:
                        contig_count = "N/A"

                    return sample_name, True, str(contig_file), f"组装完成 | Contigs: {contig_count}"
                else:
                    return sample_name, False, None, "Contig文件未生成或过小"
            else:
                return sample_name, False, None, f"MEGAHIT返回码: {returncode}"

        except Exception as e:
            return sample_name, False, None, f"异常: {e}"

    def predict_genes(self, sample_name, contig_file):
        """使用Prodigal预测基因 - 纯函数"""
        protein_file = self.output_dir / "proteins" / f"{sample_name}_proteins.faa"
        gene_file = self.output_dir / "genes" / f"{sample_name}_genes.fna"

        if protein_file.exists() and gene_file.exists() and protein_file.stat().st_size > 0:
            try:
                with open(gene_file, 'r') as f:
                    gene_count = sum(1 for line in f if line.startswith('>'))
            except:
                gene_count = 0
            return sample_name, True, gene_count, "基因预测结果已存在，跳过"

        cmd = [
            'prodigal',
            '-i', contig_file,
            '-a', str(protein_file),
            '-d', str(gene_file),
            '-p', 'meta',
            '-q',
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0 and gene_file.exists():
                gene_count = 0
                with open(gene_file, 'r') as f:
                    for line in f:
                        if line.startswith('>'):
                            gene_count += 1

                return sample_name, True, gene_count, f"基因预测完成 | 基因数: {gene_count}"
            else:
                return sample_name, False, 0, f"Prodigal失败: {result.stderr}"

        except Exception as e:
            return sample_name, False, 0, f"异常: {e}"

    def process_single_sample(self, sample):
        """处理单个样本：组装 + 基因预测"""
        sample_name = sample['name']

        # 1. 组装
        asm_name, asm_ok, contig_file, asm_msg = self.run_megahit_assembly(sample)

        if not asm_ok:
            return {
                'name': asm_name,
                'success': False,
                'stage': 'assembly',
                'message': asm_msg,
                'gene_count': 0
            }

        # 2. 基因预测
        gene_name, gene_ok, gene_count, gene_msg = self.predict_genes(sample_name, contig_file)

        if not gene_ok:
            return {
                'name': gene_name,
                'success': False,
                'stage': 'gene_prediction',
                'message': gene_msg,
                'gene_count': 0
            }

        return {
            'name': sample_name,
            'success': True,
            'stage': 'complete',
            'message': f"{asm_msg} | {gene_msg}",
            'gene_count': gene_count
        }

    def run_assembly_pipeline(self):
        """运行完整的组装流程"""
        print("\n" + "=" * 60)
        print(f"启动增量并行处理 | 并行数: {self.parallel_samples} | 每样本线程: {self.threads_per_sample}")
        print("=" * 60)

        try:
            # 1. 发现样本（不操作监控）
            all_samples = self.discover_samples()

            # 2. 初始化所有样本的监控状态
            print("\n=== 初始化监控状态 ===")
            for sample_info in all_samples:
                self.monitor.start_sample(sample_info['name'])
                if sample_info['status'] == 'completed':
                    self.monitor.complete_sample(sample_info['name'], True)
                elif sample_info['status'] == 'failed':
                    self.monitor.complete_sample(sample_info['name'], False, sample_info.get('reason', ''))

            # 如果没有需要处理的样本，直接结束
            if not self.samples:
                print("✅ 所有样本均已处理完成，无需运行")
                self.save_progress_to_file()
                return self.save_assembly_manifest()

            # 3. 启动监控线程（此时所有样本状态已就绪）
            monitor_thread = self.start_progress_monitor()
            time.sleep(1)

            # 4. 使用线程池并行处理
            results = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.parallel_samples) as executor:
                future_to_sample = {
                    executor.submit(self.process_single_sample, sample): sample
                    for sample in self.samples
                }

                for future in concurrent.futures.as_completed(future_to_sample):
                    sample = future_to_sample[future]
                    sample_name = sample['name']

                    # 标记为运行中
                    self.monitor.start_running(sample_name)

                    try:
                        result = future.result()

                        # 主线程同步更新状态
                        if result['success']:
                            self.monitor.complete_sample(sample_name, True, result['message'])
                            print(f"✅ {sample_name}: {result['message']}")
                        else:
                            self.monitor.complete_sample(sample_name, False, result['message'])
                            print(f"❌ {sample_name}: {result['stage']} 失败 - {result['message']}")

                        results.append(result)

                    except Exception as e:
                        self.monitor.complete_sample(sample_name, False, str(e))
                        print(f"❌ {sample_name}: 处理异常 - {e}")
                        results.append({
                            'name': sample_name,
                            'success': False,
                            'stage': 'exception',
                            'message': str(e),
                            'gene_count': 0
                        })

            # 等待监控线程结束
            monitor_thread.join(timeout=30)

            # 保存结果
            manifest_file = self.save_assembly_manifest(results)
            self.generate_failure_report(results)

            print("\n" + "=" * 60)
            success_count = sum(1 for r in results if r['success'])
            print(f"🎉 处理完成! 成功: {success_count}/{len(self.samples)} 个样本")
            print(f"总运行时间: {self.monitor.get_total_runtime()}")
            return manifest_file

        except Exception as e:
            print(f"\n❌ 组装流程失败 - {e}")
            import traceback
            traceback.print_exc()
            return False

    def generate_failure_report(self, results):
        """生成失败分析报告"""
        failed = [r for r in results if not r['success']]

        if not failed:
            return

        report = {
            "timestamp": datetime.now().isoformat(),
            "failed_samples": len(failed),
            "detailed_analysis": []
        }

        for result in failed:
            sample_name = result['name']
            sample_report = {
                "sample_name": sample_name,
                "error_message": result['message'],
                "failed_stage": result['stage'],
                "log_file": str(self.output_dir / "logs" / f"{sample_name}_megahit.log"),
                "system_status": FailureAnalyzer.get_system_status()
            }
            log_file = self.output_dir / "logs" / f"{sample_name}_megahit.log"
            if log_file.exists():
                last_lines, errors, warnings = FailureAnalyzer.analyze_megahit_failure(log_file)
                sample_report["log_analysis"] = {
                    "last_lines": last_lines,
                    "detected_errors": errors,
                    "detected_warnings": warnings
                }
            report["detailed_analysis"].append(sample_report)

        report_file = self.output_dir / "diagnosis" / "failure_analysis_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"📊 失败分析报告已保存: {report_file}")

    def save_assembly_manifest(self, results=None):
        """保存组装结果清单"""
        all_samples = {}
        for name, status in self.monitor.sample_status.items():
            all_samples[name] = {
                'status': status['status'],
                'elapsed': status['elapsed_time'],
                'error': status['error_message']
            }

        if results:
            for result in results:
                all_samples[result['name']]['result'] = result['message']
                all_samples[result['name']]['gene_count'] = result.get('gene_count', 0)

        manifest = {
            "timestamp": datetime.now().isoformat(),
            "samples": all_samples,
            "output_dir": str(self.output_dir),
            "kmer_parameters": {"k_list": self.kmer_list},
            "contig_length": self.min_contig_len,
            "threads_per_sample": self.threads_per_sample,
            "parallel_samples": self.parallel_samples,
            "version": "增量并行版: 线程池 + 静态线程 + 主线程状态同步"
        }

        manifest_file = self.output_dir / "assembly_manifest_all_samples.json"
        with open(manifest_file, 'w') as f:
            json.dump(manifest, f, indent=2)

        print(f"组装清单保存: {manifest_file}")
        return manifest_file


def main():
    qc_data_path = "/mnt/zjwdata/1/corn_silage_qc_analysis/cleaned_data"
    output_dir = "/mnt/zjwdata/1/assembly_analysis"

    cpu_limit = 80
    parallel_samples = 2

    # 解析命令行参数
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == "--cpu-limit" and i + 1 < len(sys.argv):
            cpu_limit = int(sys.argv[i + 1])
            i += 2
        elif arg == "--parallel" and i + 1 < len(sys.argv):
            parallel_samples = int(sys.argv[i + 1])
            i += 2
        elif arg == "--help":
            print("用法: python assembly-pipeline.py [选项]")
            print("  --cpu-limit N     CPU使用率限制 (10-100, 默认80)")
            print("  --parallel N      并行处理样本数 (默认2)")
            print("  --help            显示帮助")
            sys.exit(0)
        else:
            i += 1

    print(f"🚀 启动参数: CPU限制={cpu_limit}%, 并行样本={parallel_samples}")

    pipeline = AssemblyPipeline(
        qc_data_path,
        output_dir,
        cpu_limit=cpu_limit,
        parallel_samples=parallel_samples
    )

    manifest_file = pipeline.run_assembly_pipeline()

    if manifest_file:
        print(f"\n🎉 处理完成! 清单文件: {manifest_file}")
        sys.exit(0)
    else:
        print("\n❌ 处理失败!")
        sys.exit(1)


if __name__ == "__main__":
    main()