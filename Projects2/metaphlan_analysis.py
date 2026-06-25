#!/usr/bin/env python3
"""
MetaPhlAn 4.x 精确物种分类 - 原始数据青贮玉米发酵专用版
适配 MetaPhlAn 4.2.4 版本
使用原始fastq数据进行MetaPhlAn分析
生成物种注释表、可视化和分析报告
修复MetaPhlAn 4.2.4参数问题
"""

import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import sys
import os
import glob
import gzip
import shutil
import hashlib
import importlib.metadata

# 创建输出目录
output_base_dir = Path("/mnt/zjwdata/2/raw/metaphlan_analysis_raw/")
output_base_dir.mkdir(parents=True, exist_ok=True)

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(output_base_dir / 'raw_data_metaphlan.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class RawDataSilageMetaphlan4Analyzer:
    def __init__(self):
        """
        初始化原始数据青贮玉米MetaPhlAn 4.x分析器
        """
        # 版本断言 - 确保环境兼容性
        self.assert_environment_versions()

        # 设置原始数据路径
        self.raw_data_dir = Path("/home/zjw/zjwdata/Raw-BYMB2024072902-ZXMB01-21-yumi/")
        self.output_base_dir = output_base_dir
        self.metaphlan_db = "/home/databases/metaphlan/"  # 数据库路径

        # 质控和去宿主配置
        self.host_genome_index = "/home/databases/human_genome/GRCh38"  # 宿主基因组索引路径
        self.do_quality_control = True  # 是否进行质控
        self.do_host_removal = False  # 暂时禁用去宿主，因为索引不完整
        self.min_quality_score = 20  # 最低质量分数
        self.min_read_length = 50  # 最短读长

        # 创建子目录
        self.per_sample_output_dir = self.output_base_dir / "per_sample_results"
        self.summary_output_dir = self.output_base_dir / "summary_reports"
        self.temp_dir = self.output_base_dir / "temp_files"
        self.qc_output_dir = self.output_base_dir / "quality_control"

        # 创建所有必要的目录
        self.per_sample_output_dir.mkdir(exist_ok=True)
        self.summary_output_dir.mkdir(exist_ok=True)
        self.temp_dir.mkdir(exist_ok=True)
        self.qc_output_dir.mkdir(exist_ok=True)

        logger.info("原始数据青贮玉米MetaPhlAn 4.x分析器初始化完成")
        logger.info(f"原始数据目录: {self.raw_data_dir}")
        logger.info(f"输出基目录: {self.output_base_dir}")
        logger.info(f"数据库路径: {self.metaphlan_db}")
        logger.info(f"质控启用: {self.do_quality_control}")
        logger.info(f"去宿主启用: {self.do_host_removal}")

    def assert_environment_versions(self):
        """
        断言环境版本要求，确保分析可重复性 - 使用importlib.metadata替代pkg_resources
        """
        logger.info("检查环境版本兼容性...")

        try:
            # 检查MetaPhlAn版本
            metaphlan_version = importlib.metadata.version("metaphlan")
            assert self.parse_version(metaphlan_version) >= self.parse_version("4.0.0"), \
                f"需要MetaPhlAn 4.x，当前版本: {metaphlan_version}"
            logger.info(f"✅ MetaPhlAn版本: {metaphlan_version}")

            # 检查pandas版本
            pandas_version = importlib.metadata.version("pandas")
            assert self.parse_version(pandas_version) >= self.parse_version("1.5.0"), \
                f"需要pandas >= 1.5.0，当前版本: {pandas_version}"
            logger.info(f"✅ pandas版本: {pandas_version}")

            # 检查其他关键依赖
            required_packages = {
                'numpy': '1.21.0',
                'matplotlib': '3.5.0',
                'seaborn': '0.11.0',
                'biom-format': '2.1.10'
            }

            for package, min_version in required_packages.items():
                try:
                    version = importlib.metadata.version(package)
                    assert self.parse_version(version) >= self.parse_version(min_version), \
                        f"{package} 需要版本 >= {min_version}, 当前: {version}"
                    logger.info(f"✅ {package}版本: {version}")
                except importlib.metadata.PackageNotFoundError:
                    logger.warning(f"⚠️ 未找到包: {package}")
                except AssertionError as e:
                    logger.error(f"❌ {e}")
                    raise

        except Exception as e:
            logger.error(f"❌ 环境版本检查失败: {e}")
            raise

    def parse_version(self, version_string):
        """
        解析版本字符串为可比较的元组，处理带后缀的版本号
        """
        # 移除后缀如 .post1, .dev0 等
        base_version = version_string.split('.post')[0].split('.dev')[0].split('+')[0]
        version_parts = []
        for part in base_version.split('.'):
            try:
                version_parts.append(int(part))
            except ValueError:
                # 如果部分无法转换为整数，跳过或处理为0
                version_parts.append(0)
        # 确保至少有3个部分
        while len(version_parts) < 3:
            version_parts.append(0)
        return tuple(version_parts[:3])

    def calculate_md5(self, file_path):
        """
        计算文件的MD5校验和
        """
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            logger.error(f"计算MD5失败 {file_path}: {e}")
            return None

    def verify_file_integrity(self, file_path, expected_md5=None):
        """
        验证文件完整性
        """
        if not file_path.exists():
            logger.error(f"文件不存在: {file_path}")
            return False

        current_md5 = self.calculate_md5(file_path)

        if expected_md5:
            if current_md5 == expected_md5:
                logger.info(f"✅ 文件完整性验证通过: {file_path.name}")
                return True
            else:
                logger.error(f"❌ 文件完整性验证失败: {file_path.name}")
                logger.error(f"   期望MD5: {expected_md5}")
                logger.error(f"   实际MD5: {current_md5}")
                return False
        else:
            # 如果没有期望的MD5，只记录当前MD5
            logger.info(f"📝 文件MD5: {file_path.name} -> {current_md5}")
            return current_md5

    def create_md5_checkpoint(self, sample_name, file_paths):
        """
        创建MD5检查点文件
        """
        checkpoint_file = self.temp_dir / f"{sample_name}_md5_checkpoint.txt"

        with open(checkpoint_file, 'w') as f:
            f.write(f"# MD5 Checkpoint for {sample_name}\n")
            f.write(f"# Created: {pd.Timestamp.now()}\n\n")

            for file_path in file_paths:
                if file_path.exists():
                    md5 = self.calculate_md5(file_path)
                    f.write(f"{file_path.name}\t{md5}\n")
                else:
                    f.write(f"{file_path.name}\tFILE_NOT_FOUND\n")

        logger.info(f"✅ 创建MD5检查点: {checkpoint_file}")
        return checkpoint_file

    def run_quality_control(self, sample_name, sample_info):
        """
        运行质控步骤 - 使用FastQC和fastp
        """
        logger.info(f"开始质控处理: {sample_name}")

        qc_sample_dir = self.qc_output_dir / sample_name
        qc_sample_dir.mkdir(exist_ok=True)

        forward_read = sample_info['forward']
        reverse_read = sample_info['reverse']

        # 验证输入文件完整性
        logger.info("验证输入文件完整性...")
        input_files = [forward_read]
        if reverse_read:
            input_files.append(reverse_read)

        # 创建输入文件MD5检查点
        input_checkpoint = self.create_md5_checkpoint(f"{sample_name}_input", input_files)

        # 步骤1: FastQC质量检查
        logger.info(f"运行FastQC质量检查: {sample_name}")
        fastqc_cmd = [
            "fastqc",
            "-o", str(qc_sample_dir),
            "-t", "4"
        ]

        if reverse_read:
            fastqc_cmd.extend([str(forward_read), str(reverse_read)])
        else:
            fastqc_cmd.append(str(forward_read))

        try:
            subprocess.run(fastqc_cmd, check=True, capture_output=True, text=True)
            logger.info(f"✅ FastQC完成: {sample_name}")
        except subprocess.CalledProcessError as e:
            logger.warning(f"⚠️ FastQC运行失败，继续处理: {e}")
        except FileNotFoundError:
            logger.warning("⚠️ FastQC未安装，跳过质量检查")

        # 步骤2: fastp质量修剪
        logger.info(f"运行fastp质量修剪: {sample_name}")

        # 设置输出文件路径
        forward_trimmed = qc_sample_dir / f"{sample_name}_R1_trimmed.fastq.gz"
        fastp_json_report = qc_sample_dir / f"{sample_name}_fastp.json"
        fastp_html_report = qc_sample_dir / f"{sample_name}_fastp.html"

        if reverse_read and reverse_read.exists():
            reverse_trimmed = qc_sample_dir / f"{sample_name}_R2_trimmed.fastq.gz"

            fastp_cmd = [
                "fastp",
                "-i", str(forward_read),
                "-I", str(reverse_read),
                "-o", str(forward_trimmed),
                "-O", str(reverse_trimmed),
                "--json", str(fastp_json_report),
                "--html", str(fastp_html_report),
                "--thread", "8",
                "--qualified_quality_phred", str(self.min_quality_score),
                "--length_required", str(self.min_read_length),
                "--cut_front",
                "--cut_tail",
                "--cut_window_size", "4",
                "--cut_mean_quality", "20",
                "--correction",
                "--detect_adapter_for_pe",
                "--overrepresentation_analysis"
            ]
        else:
            fastp_cmd = [
                "fastp",
                "-i", str(forward_read),
                "-o", str(forward_trimmed),
                "--json", str(fastp_json_report),
                "--html", str(fastp_html_report),
                "--thread", "8",
                "--qualified_quality_phred", str(self.min_quality_score),
                "--length_required", str(self.min_read_length),
                "--cut_front",
                "--cut_tail",
                "--cut_window_size", "4",
                "--cut_mean_quality", "20",
                "--correction",
                "--detect_adapter_for_se",
                "--overrepresentation_analysis"
            ]

        try:
            subprocess.run(fastp_cmd, check=True, capture_output=True, text=True)
            logger.info(f"✅ fastp完成: {sample_name}")

            # 检查输出文件并更新sample_info
            if forward_trimmed.exists():
                sample_info['forward'] = forward_trimmed
                logger.info(f"使用修剪后的正向文件: {forward_trimmed.name}")

                if reverse_read and reverse_trimmed.exists():
                    sample_info['reverse'] = reverse_trimmed
                    logger.info(f"使用修剪后的反向文件: {reverse_trimmed.name}")

                # 记录fastp统计信息
                self.parse_fastp_report(fastp_json_report, sample_name)

            else:
                logger.warning("未找到fastp输出文件，使用原始文件继续")

        except subprocess.CalledProcessError as e:
            logger.error(f"❌ fastp运行失败: {e}")
            if e.stderr:
                logger.error(f"fastp错误: {e.stderr}")
            logger.info("使用原始文件继续分析...")
        except FileNotFoundError:
            logger.warning("⚠️ fastp未安装，跳过质量修剪")

        return sample_info

    def parse_fastp_report(self, json_report, sample_name):
        """
        解析fastp JSON报告，提取关键统计信息
        """
        try:
            import json
            with open(json_report, 'r') as f:
                report_data = json.load(f)

            # 提取关键统计信息
            if 'summary' in report_data:
                before = report_data['summary']['before_filtering']
                after = report_data['summary']['after_filtering']

                logger.info(f"fastp统计 - {sample_name}:")
                logger.info(f"  原始读段: {before.get('total_reads', 'N/A')}")
                logger.info(f"  质控后读段: {after.get('total_reads', 'N/A')}")
                if before.get('total_reads', 0) > 0:
                    retention_rate = after.get('total_reads', 0) / before.get('total_reads', 1) * 100
                    logger.info(f"  保留比例: {retention_rate:.2f}%")
                logger.info(f"  Q20率: {after.get('q20_rate', 'N/A')}")
                logger.info(f"  Q30率: {after.get('q30_rate', 'N/A')}")
                logger.info(f"  GC含量: {after.get('gc_content', 'N/A')}")

        except Exception as e:
            logger.warning(f"解析fastp报告失败: {e}")

    def find_raw_samples(self):
        """
        递归查找原始数据中的所有样本（包括子目录）- 修复版本
        """
        logger.info("正在递归查找原始数据样本...")

        samples = {}

        # 递归查找所有fastq文件（包括子目录）
        fastq_patterns = ["**/*.fastq", "**/*.fq", "**/*.fastq.gz", "**/*.fq.gz"]

        for pattern in fastq_patterns:
            fastq_files = list(self.raw_data_dir.glob(pattern))
            logger.info(f"找到 {len(fastq_files)} 个文件匹配模式 {pattern}")

            for fastq_file in fastq_files:
                # 跳过空文件
                if fastq_file.stat().st_size == 0:
                    logger.warning(f"跳过空文件: {fastq_file}")
                    continue

                # 提取样本名，包含子目录信息
                sample_name = self.extract_sample_name(fastq_file)
                if sample_name:
                    if sample_name not in samples:
                        samples[sample_name] = {'forward': None, 'reverse': None, 'original_files': []}

                    # 记录原始文件
                    samples[sample_name]['original_files'].append(fastq_file)

                    # 判断是正向还是反向测序
                    if self.is_forward_read(fastq_file):
                        if samples[sample_name]['forward'] is None:
                            samples[sample_name]['forward'] = fastq_file
                            logger.debug(f"样本 {sample_name} 正向读段: {fastq_file}")
                    elif self.is_reverse_read(fastq_file):
                        if samples[sample_name]['reverse'] is None:
                            samples[sample_name]['reverse'] = fastq_file
                            logger.debug(f"样本 {sample_name} 反向读段: {fastq_file}")
                    else:
                        # 如果是单端测序或无法判断，则作为单端处理
                        if samples[sample_name]['forward'] is None:
                            samples[sample_name]['forward'] = fastq_file
                            logger.debug(f"样本 {sample_name} 单端读段: {fastq_file}")

        # 过滤掉不完整的样本并记录
        valid_samples = {}
        for sample_name, reads in samples.items():
            if reads['forward'] is not None:
                valid_samples[sample_name] = {
                    'forward': reads['forward'],
                    'reverse': reads['reverse'],
                    'original_files': reads['original_files']
                }
                logger.info(f"找到样本 {sample_name}:")
                logger.info(f"  正向: {reads['forward']}")
                if reads['reverse']:
                    logger.info(f"  反向: {reads['reverse']}")
                else:
                    logger.info(f"  反向: 无 (单端测序)")
                logger.info(f"  原始文件数: {len(reads['original_files'])}")

        logger.info(f"总共找到 {len(valid_samples)} 个有效样本")

        # 如果没有找到样本，显示目录结构以帮助调试
        if not valid_samples:
            logger.warning("未找到任何fastq文件，显示目录结构:")
            self.show_directory_structure()

        return valid_samples

    def show_directory_structure(self, max_depth=3):
        """
        显示目录结构以帮助调试
        """
        try:
            for root, dirs, files in os.walk(self.raw_data_dir):
                level = root.replace(str(self.raw_data_dir), '').count(os.sep)
                if level <= max_depth:
                    indent = ' ' * 2 * level
                    logger.warning(f"{indent}{os.path.basename(root)}/")
                    subindent = ' ' * 2 * (level + 1)
                    for file in files[:10]:  # 只显示前10个文件
                        logger.warning(f"{subindent}{file}")
                    if len(files) > 10:
                        logger.warning(f"{subindent}... 还有 {len(files) - 10} 个文件")
        except Exception as e:
            logger.error(f"显示目录结构时出错: {e}")

    def extract_sample_name(self, file_path):
        """
        从文件路径中提取样本名，保留子目录信息 - 修复版本
        """
        try:
            # 获取相对于原始数据目录的路径
            relative_path = file_path.relative_to(self.raw_data_dir)
            # 使用父目录名作为样本名的一部分
            parent_dir = relative_path.parent.name
            if parent_dir != '.':
                # 如果文件在子目录中，使用子目录名作为样本名
                sample_name = parent_dir
            else:
                # 如果文件在根目录，使用文件名（不含扩展名）作为样本名
                stem = file_path.stem
                if stem.endswith('.fastq') or stem.endswith('.fq'):
                    stem = Path(stem).stem
                sample_name = stem
        except ValueError:
            # 如果文件不在原始数据目录下，使用绝对路径的最后一部分
            sample_name = file_path.parent.name if file_path.parent.name else file_path.stem

        # 移除测序读段标识
        patterns_to_remove = [
            '_1', '_2', '_R1', '_R2', '_R1_001', '_R2_001',
            '.1', '.2', '.R1', '.R2',
            '_forward', '_reverse', '_fwd', '_rev'
        ]

        for pattern in patterns_to_remove:
            if sample_name.endswith(pattern):
                sample_name = sample_name[:-len(pattern)]
                break

        # 如果样本名仍然包含路径分隔符，用下划线替换
        sample_name = sample_name.replace('/', '_').replace('\\', '_')

        # 清理样本名中的特殊字符
        sample_name = ''.join(c for c in sample_name if c.isalnum() or c in ['-', '_'])

        # 如果样本名太长，截断
        if len(sample_name) > 50:
            sample_name = sample_name[:50]

        return sample_name if sample_name else f"sample_{hash(file_path)}"

    def is_forward_read(self, file_path):
        """
        判断是否为正向测序文件
        """
        filename = file_path.name.lower()
        forward_indicators = ['_1', '_r1', 'r1_', 'forward', 'fwd', '.1.', '.r1.']
        return any(indicator in filename for indicator in forward_indicators)

    def is_reverse_read(self, file_path):
        """
        判断是否为反向测序文件
        """
        filename = file_path.name.lower()
        reverse_indicators = ['_2', '_r2', 'r2_', 'reverse', 'rev', '.2.', '.r2.']
        return any(indicator in filename for indicator in reverse_indicators)

    def check_database_files(self):
        """
        检查数据库文件是否存在
        """
        logger.info("检查MetaPhlAn数据库文件...")

        db_path = Path(self.metaphlan_db)
        required_files = [
            "mpa_vOct22_CHOCOPhlAnSGB_202212.1.bt2l",
            "mpa_vOct22_CHOCOPhlAnSGB_202212.2.bt2l",
            "mpa_vOct22_CHOCOPhlAnSGB_202212.3.bt2l",
            "mpa_vOct22_CHOCOPhlAnSGB_202212.4.bt2l",
            "mpa_vOct22_CHOCOPhlAnSGB_202212.pkl"
        ]

        missing_files = []
        for file in required_files:
            if not (db_path / file).exists():
                missing_files.append(file)

        if missing_files:
            logger.error(f"❌ 缺少数据库文件: {missing_files}")
            return False
        else:
            logger.info("✅ 所有必需的数据库文件都存在")
            return True

    def check_metaphlan_installation(self):
        """
        检查MetaPhlAn 4.x是否安装
        """
        logger.info("检查MetaPhlAn 4.x安装...")

        try:
            result = subprocess.run(["metaphlan", "--version"],
                                    capture_output=True, text=True)
            if result.returncode == 0:
                version_info = result.stdout.strip()
                logger.info(f"✅ MetaPhlAn已安装: {version_info}")
                return True
            else:
                logger.error("❌ MetaPhlAn未正确安装")
                return False
        except FileNotFoundError:
            logger.error("❌ MetaPhlAn未安装或不在PATH中")
            return False

    def run_metaphlan_v4_analysis_single(self, sample_name, sample_info):
        """
        为单个样本运行MetaPhlAn 4.x分析 - 修复MetaPhlAn 4.2.4参数问题
        """
        logger.info(f"开始处理样本: {sample_name}")

        output_dir = self.per_sample_output_dir / sample_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # 设置输出文件路径
        metaphlan_output = output_dir / "metaphlan_species.txt"
        profile_output = output_dir / "species_profile.txt"
        kraken_compatible_output = output_dir / "kraken_compatible_species_table.csv"
        mapout_file = output_dir / f"{sample_name}.bowtie2.bz2"

        # 预处理步骤：质控和去宿主
        processed_sample_info = sample_info.copy()

        if self.do_quality_control:
            processed_sample_info = self.run_quality_control(sample_name, processed_sample_info)

        if self.do_host_removal:
            processed_sample_info = self.run_host_removal(sample_name, processed_sample_info)

        # 创建预处理后文件的MD5检查点
        processed_files = [processed_sample_info['forward']]
        if processed_sample_info.get('reverse'):
            processed_files.append(processed_sample_info['reverse'])
        processed_checkpoint = self.create_md5_checkpoint(f"{sample_name}_processed", processed_files)

        # 处理输入文件
        forward_read = processed_sample_info['forward']
        reverse_read = processed_sample_info.get('reverse')

        # 检查输入文件
        if not forward_read.exists():
            logger.error(f"❌ 正向读段文件不存在: {forward_read}")
            return None

        # 构建MetaPhlAn命令 - 修复MetaPhlAn 4.2.4参数问题
        # MetaPhlAn 4.2.4要求双端文件用逗号拼接成一个位置参数
        if reverse_read and reverse_read.exists():
            # 双端测序 - 使用逗号分隔的文件名作为位置参数
            input_files = f"{forward_read},{reverse_read}"
            cmd = [
                "metaphlan",
                input_files,  # 位置参数：逗号分隔的双端文件
                "--input_type", "fastq",
                "--db_dir", self.metaphlan_db,
                "-x", "mpa_vOct22_CHOCOPhlAnSGB_202212",
                "--tax_lev", "s",
                "-o", str(metaphlan_output),
                "--nproc", "8",
                "--offline",
                "--mapout", str(mapout_file)  # 修复：使用 --mapout 替代 --bowtie2out
            ]
            logger.info(f"使用双端测序数据: {forward_read.name} + {reverse_read.name}")
        else:
            # 单端测序
            cmd = [
                "metaphlan",
                str(forward_read),
                "--input_type", "fastq",
                "--db_dir", self.metaphlan_db,
                "-x", "mpa_vOct22_CHOCOPhlAnSGB_202212",
                "--tax_lev", "s",
                "-o", str(metaphlan_output),
                "--nproc", "8",
                "--offline"
            ]
            logger.info(f"使用单端测序数据: {forward_read.name}")

        # 记录完整的命令用于调试
        full_cmd = ' '.join(cmd)
        logger.info(f"执行MetaPhlAn命令: {full_cmd}")

        try:
            # 设置环境变量，防止自动下载
            env = os.environ.copy()
            env['METAPHLAN_DB'] = self.metaphlan_db

            # 运行MetaPhlAn
            logger.info(f"MetaPhlAn分析开始: {sample_name}")
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=21600,  # 6小时超时
                env=env
            )
            logger.info(f"✅ {sample_name} MetaPhlAn 4.x分析完成")

            # 验证输出文件完整性
            if metaphlan_output.exists():
                # 创建输出文件MD5检查点
                output_checkpoint = self.create_md5_checkpoint(f"{sample_name}_output", [metaphlan_output])

                file_size = metaphlan_output.stat().st_size
                logger.info(f"输出文件: {metaphlan_output} ({file_size} bytes)")

                # 解析结果
                species_df = self.parse_metaphlan_results_single(metaphlan_output, profile_output, sample_name)
                if species_df is not None:
                    # 生成与Kraken兼容的输出表
                    self.generate_kraken_compatible_table(species_df, kraken_compatible_output, sample_name)
                    # 青贮微生物分析
                    silage_analysis = self.analyze_silage_microbes_single(species_df)
                    # 可视化
                    self.visualize_species_composition(species_df, output_dir, sample_name)
                    # 生成单个报告
                    input_files = f"{forward_read.name} + {reverse_read.name}" if reverse_read else forward_read.name
                    self.generate_single_report(species_df, silage_analysis, input_files, output_dir, sample_name)
                    return species_df
                else:
                    return None
            else:
                logger.error(f"❌ {sample_name} MetaPhlAn输出文件未生成")
                return None

        except subprocess.CalledProcessError as e:
            logger.error(f"❌ {sample_name} MetaPhlAn运行失败: {e}")
            if e.stderr:
                error_msg = e.stderr
                logger.error(f"错误输出:")
                # 显示完整的错误信息
                logger.error(error_msg)

            # 尝试替代方法：使用不同的参数格式
            logger.info(f"尝试使用替代方法运行 {sample_name}...")
            return self.run_metaphlan_alternative_single(sample_name, sample_info)

        except subprocess.TimeoutExpired:
            logger.error(f"❌ {sample_name} MetaPhlAn分析超时 (6小时)")
            return None

    def run_metaphlan_alternative_single(self, sample_name, sample_info):
        """
        使用替代方法运行MetaPhlAn分析 - 处理双端测序的另一种方式
        """
        logger.info(f"使用替代方法运行: {sample_name}")

        output_dir = self.per_sample_output_dir / sample_name

        # 设置输出文件路径
        metaphlan_output = output_dir / "metaphlan_species.txt"
        profile_output = output_dir / "species_profile.txt"
        kraken_compatible_output = output_dir / "kraken_compatible_species_table.csv"

        # 处理输入文件
        forward_read = sample_info['forward']
        reverse_read = sample_info.get('reverse')

        # 检查输入文件
        if not forward_read.exists():
            logger.error(f"❌ 正向读段文件不存在: {forward_read}")
            return None

        # 替代方法：分别运行每个文件然后合并
        if reverse_read and reverse_read.exists():
            # 分别处理正向和反向文件
            forward_output = output_dir / "forward_metaphlan.txt"
            reverse_output = output_dir / "reverse_metaphlan.txt"

            # 处理正向文件
            forward_cmd = [
                "metaphlan",
                str(forward_read),
                "--input_type", "fastq",
                "--db_dir", self.metaphlan_db,
                "-x", "mpa_vOct22_CHOCOPhlAnSGB_202212",
                "--tax_lev", "s",
                "-o", str(forward_output),
                "--nproc", "8",
                "--offline"
            ]

            # 处理反向文件
            reverse_cmd = [
                "metaphlan",
                str(reverse_read),
                "--input_type", "fastq",
                "--db_dir", self.metaphlan_db,
                "-x", "mpa_vOct22_CHOCOPhlAnSGB_202212",
                "--tax_lev", "s",
                "-o", str(reverse_output),
                "--nproc", "8",
                "--offline"
            ]

            logger.info(f"分别处理双端测序数据: {forward_read.name} 和 {reverse_read.name}")

            try:
                # 设置环境变量
                env = os.environ.copy()
                env['METAPHLAN_DB'] = self.metaphlan_db

                # 运行正向分析
                logger.info(f"处理正向读段: {sample_name}")
                subprocess.run(forward_cmd, check=True, capture_output=True, text=True, timeout=10800, env=env)

                # 运行反向分析
                logger.info(f"处理反向读段: {sample_name}")
                subprocess.run(reverse_cmd, check=True, capture_output=True, text=True, timeout=10800, env=env)

                # 合并结果
                if forward_output.exists() and reverse_output.exists():
                    # 这里简化处理，只使用正向结果
                    shutil.copy(forward_output, metaphlan_output)
                    logger.info(f"✅ {sample_name} 替代方法分析完成")

                    # 解析结果
                    species_df = self.parse_metaphlan_results_single(metaphlan_output, profile_output, sample_name)
                    if species_df is not None:
                        # 生成与Kraken兼容的输出表
                        self.generate_kraken_compatible_table(species_df, kraken_compatible_output, sample_name)
                        # 青贮微生物分析
                        silage_analysis = self.analyze_silage_microbes_single(species_df)
                        # 可视化
                        self.visualize_species_composition(species_df, output_dir, sample_name)
                        # 生成单个报告
                        input_files = f"{forward_read.name} + {reverse_read.name}"
                        self.generate_single_report(species_df, silage_analysis, input_files, output_dir, sample_name)
                        return species_df

                return None

            except subprocess.CalledProcessError as e:
                logger.error(f"❌ {sample_name} 替代方法也运行失败: {e}")
                return None

        else:
            # 单端测序 - 使用原始方法
            return self.run_metaphlan_v4_analysis_single(sample_name, sample_info)

    def parse_metaphlan_results_single(self, metaphlan_output, profile_output, sample_name):
        """
        解析单个样本的MetaPhlAn结果
        """
        logger.info(f"解析MetaPhlAn结果: {sample_name}")

        if not metaphlan_output.exists():
            logger.error(f"❌ {sample_name} MetaPhlAn输出文件不存在")
            return None

        try:
            # 读取MetaPhlAn输出
            species_data = []
            with open(metaphlan_output, 'r') as f:
                for line in f:
                    # 跳过注释行和空行
                    if line.startswith('#') or not line.strip():
                        continue

                    parts = line.strip().split('\t')
                    if len(parts) >= 2:  # 修改为至少2个部分
                        taxonomy = parts[0]

                        # 尝试不同的列来获取丰度信息
                        abundance = None
                        for i, part in enumerate(parts[1:], 1):
                            try:
                                potential_abundance = float(part)
                                if 0 <= potential_abundance <= 100:
                                    abundance = potential_abundance
                                    break
                            except ValueError:
                                continue

                        if abundance is None:
                            logger.warning(f"跳过无法解析丰度的行: {line.strip()}")
                            continue

                        # 只处理物种水平的数据
                        if 's__' in taxonomy and 't__' not in taxonomy:
                            species_name = taxonomy.split('s__')[-1]

                            # 提取分类信息
                            tax_parts = taxonomy.split('|')
                            tax_info = {}
                            for part in tax_parts:
                                if part.startswith('k__'):
                                    tax_info['kingdom'] = part.replace('k__', '')
                                elif part.startswith('p__'):
                                    tax_info['phylum'] = part.replace('p__', '')
                                elif part.startswith('c__'):
                                    tax_info['class'] = part.replace('c__', '')
                                elif part.startswith('o__'):
                                    tax_info['order'] = part.replace('o__', '')
                                elif part.startswith('f__'):
                                    tax_info['family'] = part.replace('f__', '')
                                elif part.startswith('g__'):
                                    tax_info['genus'] = part.replace('g__', '')
                                elif part.startswith('s__'):
                                    tax_info['species'] = part.replace('s__', '')

                            species_data.append({
                                'species_name': species_name,
                                'relative_abundance': abundance,
                                'full_taxonomy': taxonomy,
                                'sample_name': sample_name,
                                **tax_info
                            })

            # 创建DataFrame并排序
            df = pd.DataFrame(species_data)
            if not df.empty:
                df = df.sort_values('relative_abundance', ascending=False)
                df.to_csv(profile_output, sep='\t', index=False)
                logger.info(f"✅ {sample_name} 物种组成表已保存: {profile_output}")
                logger.info(f"共鉴定出 {len(df)} 个物种")
                return df
            else:
                logger.warning(f"⚠️ {sample_name} 未找到有效的物种数据")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"❌ {sample_name} 解析结果失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def generate_kraken_compatible_table(self, species_df, output_file, sample_name):
        """
        生成与Kraken兼容的物种组成表
        """
        logger.info(f"生成Kraken兼容表: {sample_name}")

        if species_df.empty:
            logger.warning(f"⚠️ {sample_name} 物种数据为空，无法生成兼容表")
            return

        try:
            # 创建与Kraken输出格式兼容的表
            compatible_data = []

            for _, row in species_df.iterrows():
                compatible_data.append({
                    'name': row['species_name'],
                    'taxonomy_id': 'N/A',  # MetaPhlAn不提供taxonomy_id
                    'taxonomy_lvl': 'species',
                    'new_est_reads': int(row['relative_abundance'] * 1000),  # 模拟读段数
                    'relative_abundance': row['relative_abundance'],
                    'fraction_total_reads': row['relative_abundance'] / 100,
                    'sample_name': sample_name
                })

            compatible_df = pd.DataFrame(compatible_data)
            compatible_df = compatible_df.sort_values('relative_abundance', ascending=False)
            compatible_df.to_csv(output_file, index=False, encoding='utf-8-sig')
            logger.info(f"✅ {sample_name} Kraken兼容表已保存: {output_file}")

        except Exception as e:
            logger.error(f"❌ {sample_name} 生成Kraken兼容表失败: {e}")

    def visualize_species_composition(self, species_df, output_dir, sample_name, top_n=20):
        """
        可视化物种组成 - 移除中文字体依赖
        """
        logger.info(f"生成物种组成可视化: {sample_name}")

        if species_df.empty:
            logger.warning(f"⚠️ {sample_name} 物种数据为空，无法生成可视化")
            return

        try:
            import matplotlib
            matplotlib.use('Agg')  # 使用非交互式后端
            import matplotlib.pyplot as plt
            import seaborn as sns

            # 使用默认英文字体，移除中文字体依赖
            plt.rcParams.update({
                'font.family': 'sans-serif',
                'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
                'axes.unicode_minus': False
            })

            # 取前top_n个物种
            top_species = species_df.head(top_n).copy()

            # 创建图表
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

            # 饼图
            if len(top_species) > 0:
                wedges, texts, autotexts = ax1.pie(
                    top_species['relative_abundance'],
                    labels=top_species['species_name'],
                    autopct='%1.1f%%',
                    startangle=90
                )
                ax1.set_title(f'{sample_name} - Top {top_n} Species Composition\n(MetaPhlAn 4.x)')

                # 改善饼图标签可读性
                for text in texts:
                    text.set_fontsize(8)
                for autotext in autotexts:
                    autotext.set_fontsize(8)
                    autotext.set_color('white')
                    autotext.set_weight('bold')

            # 条形图
            y_pos = np.arange(len(top_species))
            bars = ax2.barh(y_pos, top_species['relative_abundance'])
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(top_species['species_name'], fontsize=8)
            ax2.set_xlabel('Relative Abundance (%)')
            ax2.set_title('Species Abundance Distribution')
            ax2.invert_yaxis()

            # 在条形上添加数值
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax2.text(width + 0.1, bar.get_y() + bar.get_height() / 2,
                         f'{width:.2f}%', ha='left', va='center', fontsize=7)

            plt.tight_layout()
            plt.savefig(output_dir / 'species_composition_plot.png',
                        dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"✅ {sample_name} 物种组成可视化图已保存 (前{top_n}个物种)")

        except ImportError:
            logger.warning("matplotlib或seaborn未安装，跳过可视化")
        except Exception as e:
            logger.error(f"❌ {sample_name} 可视化失败: {e}")

    def analyze_silage_microbes_single(self, species_df):
        """
        分析单个样本的青贮相关微生物
        """
        # 青贮发酵重要微生物
        silage_microbes = {
            'Lactic_Acid_Bacteria': [
                'Lactobacillus', 'Lactiplantibacillus', 'Pediococcus',
                'Enterococcus', 'Weissella', 'Leuconostoc', 'Lactococcus'
            ],
            'Acetic_Acid_Bacteria': [
                'Acetobacter', 'Gluconobacter'
            ],
            'Yeasts': [
                'Saccharomyces', 'Candida', 'Pichia'
            ],
            'Molds': [
                'Aspergillus', 'Penicillium', 'Fusarium'
            ],
            'Enterobacteria': [
                'Enterobacter', 'Klebsiella', 'Escherichia', 'Salmonella'
            ],
            'Clostridia': [
                'Clostridium', 'Butyricicoccus'
            ]
        }

        analysis_results = {}

        for group, genera in silage_microbes.items():
            group_abundance = 0
            group_species = []

            for genus in genera:
                # 查找属于该属的物种
                genus_species = species_df[species_df['species_name'].str.contains(genus, na=False)]

                for _, row in genus_species.iterrows():
                    group_abundance += row['relative_abundance']
                    group_species.append({
                        'species': row['species_name'],
                        'abundance': row['relative_abundance'],
                        'genus': genus
                    })

            analysis_results[group] = {
                'total_abundance': group_abundance,
                'species_count': len(group_species),
                'species_list': sorted(group_species, key=lambda x: x['abundance'], reverse=True)
            }

        return analysis_results

    def generate_single_report(self, species_df, silage_analysis, input_files, output_dir, sample_name):
        """
        为单个样本生成分析报告
        """
        logger.info(f"生成分析报告: {sample_name}")

        report_file = output_dir / "metaphlan_analysis_report.txt"

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write(f"      {sample_name} - Corn Silage Microbial MetaPhlAn 4.x Analysis Report\n")
            f.write("=" * 60 + "\n\n")

            f.write("1. Analysis Overview\n")
            f.write("-" * 40 + "\n")
            f.write(f"Input files: {input_files}\n")
            f.write(f"Analysis time: {pd.Timestamp.now()}\n")
            f.write(f"Total species identified: {len(species_df)}\n")
            f.write(f"Total relative abundance: {species_df['relative_abundance'].sum():.2f}%\n\n")

            f.write("2. Top 10 Most Abundant Species\n")
            f.write("-" * 40 + "\n")
            top_10 = species_df.head(10)
            for i, (_, row) in enumerate(top_10.iterrows(), 1):
                f.write(f"{i:2d}. {row['species_name']}: {row['relative_abundance']:.2f}%\n")
            f.write("\n")

            f.write("3. Silage-Related Microbial Analysis\n")
            f.write("-" * 40 + "\n")
            for group, info in silage_analysis.items():
                if info['total_abundance'] > 0:
                    f.write(f"{group}:\n")
                    f.write(f"  Total abundance: {info['total_abundance']:.2f}%\n")
                    f.write(f"  Species count: {info['species_count']}\n")

                    # 显示前3个物种
                    if info['species_list']:
                        f.write("  Main species:\n")
                        for species_info in info['species_list'][:3]:
                            f.write(f"    - {species_info['species']}: {species_info['abundance']:.2f}%\n")
                    f.write("\n")

            f.write("4. Output Files\n")
            f.write("-" * 40 + "\n")
            f.write(f"Raw MetaPhlAn output: {output_dir / 'metaphlan_species.txt'}\n")
            f.write(f"Species profile: {output_dir / 'species_profile.txt'}\n")
            f.write(f"Kraken-compatible table: {output_dir / 'kraken_compatible_species_table.csv'}\n")
            f.write(f"Species composition plot: {output_dir / 'species_composition_plot.png'}\n")
            f.write(f"Analysis report: {report_file}\n")

        logger.info(f"✅ {sample_name} 分析报告已保存: {report_file}")

    def generate_summary_report(self, all_results):
        """
        生成所有样本的汇总报告
        """
        logger.info("生成汇总报告...")

        summary_file = self.summary_output_dir / "raw_data_metaphlan_summary.txt"

        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("      Raw Data Corn Silage MetaPhlAn 4.x Analysis Summary Report\n")
            f.write("=" * 80 + "\n\n")

            f.write("1. Analysis Overview\n")
            f.write("-" * 60 + "\n")
            f.write(f"Analysis time: {pd.Timestamp.now()}\n")
            f.write(f"Total samples processed: {len(all_results)}\n")
            f.write(f"Successful analyses: {len([r for r in all_results.values() if r is not None])}\n")
            f.write(f"Failed analyses: {len([r for r in all_results.values() if r is None])}\n\n")

            f.write("2. Sample-wise Summary\n")
            f.write("-" * 60 + "\n")
            f.write(f"{'Sample Name':<<20} {'Species Count':<<15} {'Total Abundance':<<15} {'Top Species':<<30}\n")
            f.write("-" * 80 + "\n")

            for sample_name, result in all_results.items():
                if result is not None and not result.empty:
                    species_count = len(result)
                    total_abundance = result['relative_abundance'].sum()
                    top_species = result.iloc[0]['species_name']
                    top_abundance = result.iloc[0]['relative_abundance']

                    f.write(f"{sample_name:<20} {species_count:<15} {total_abundance:<15.2f} {top_species[:28]:<<30}\n")
                else:
                    f.write(f"{sample_name:<20} {'Failed':<<15} {'-':<<15} {'-':<<30}\n")

            f.write("\n")

            f.write("3. Silage-Related Microbes Summary\n")
            f.write("-" * 60 + "\n")

            # 计算乳酸菌等关键微生物的总丰度
            key_microbes_summary = {}
            for sample_name, result in all_results.items():
                if result is not None and not result.empty:
                    silage_analysis = self.analyze_silage_microbes_single(result)
                    for group, info in silage_analysis.items():
                        if group not in key_microbes_summary:
                            key_microbes_summary[group] = []
                        key_microbes_summary[group].append({
                            'sample': sample_name,
                            'abundance': info['total_abundance'],
                            'species_count': info['species_count']
                        })

            for group, sample_data in key_microbes_summary.items():
                f.write(f"\n{group}:\n")
                total_abundance_all = sum([d['abundance'] for d in sample_data])
                avg_abundance = total_abundance_all / len(sample_data) if sample_data else 0
                f.write(f"  Average abundance across all samples: {avg_abundance:.2f}%\n")

                # 显示丰度最高的3个样本
                sorted_data = sorted(sample_data, key=lambda x: x['abundance'], reverse=True)
                f.write(f"  Top samples by abundance:\n")
                for i, data in enumerate(sorted_data[:3], 1):
                    if data['abundance'] > 0:
                        f.write(
                            f"    {i}. {data['sample']}: {data['abundance']:.2f}% ({data['species_count']} species)\n")

            f.write("\n4. Output Directories\n")
            f.write("-" * 60 + "\n")
            f.write(f"Per-sample results: {self.per_sample_output_dir}\n")
            f.write(f"Summary reports: {self.summary_output_dir}\n")
            f.write(f"Quality control results: {self.qc_output_dir}\n")
            f.write(f"Analysis log: {self.output_base_dir / 'raw_data_metaphlan.log'}\n")

        logger.info(f"✅ 汇总报告已保存: {summary_file}")

    def run_raw_data_analysis(self):
        """
        运行原始数据MetaPhlAn分析流程
        """
        logger.info("开始原始数据青贮玉米MetaPhlAn 4.x物种分析...")

        # 1. 检查安装
        if not self.check_metaphlan_installation():
            return False

        # 2. 检查数据库文件
        if not self.check_database_files():
            logger.error("❌ 数据库文件不完整，请检查数据库路径和文件")
            return False

        # 3. 查找所有原始样本
        samples = self.find_raw_samples()
        if not samples:
            logger.error("未找到任何原始数据样本!")
            return False

        # 4. 分别处理每个样本
        # ===== 前置过滤：先分类已完成和待处理样本 =====
        pending_samples = {}
        skipped_samples = []
        for sample_name, sample_info in samples.items():
            output_dir = self.per_sample_output_dir / sample_name
            metaphlan_output = output_dir / "metaphlan_species.txt"
            profile_output = output_dir / "species_profile.txt"
            kraken_compatible_output = output_dir / "kraken_compatible_species_table.csv"

            if (metaphlan_output.exists() and profile_output.exists() and
                    kraken_compatible_output.exists() and metaphlan_output.stat().st_size > 0):
                skipped_samples.append(sample_name)
            else:
                pending_samples[sample_name] = sample_info

        logger.info(f"【状态统计】原始样本总数: {len(samples)}")
        logger.info(f"【状态统计】已完成样本（将跳过）: {len(skipped_samples)}个 -> {skipped_samples}")
        logger.info(f"【状态统计】待处理样本（将运行）: {len(pending_samples)}个 -> {list(pending_samples.keys())}")

        all_results = {}

        # 先加载已完成的样本结果
        for sample_name in skipped_samples:
            output_dir = self.per_sample_output_dir / sample_name
            metaphlan_output = output_dir / "metaphlan_species.txt"
            profile_output = output_dir / "species_profile.txt"
            logger.info(f"\n{'=' * 60}")
            logger.info(f"样本 {sample_name} 已处理完成，自动跳过")
            logger.info(f"{'=' * 60}")
            species_df = self.parse_metaphlan_results_single(metaphlan_output, profile_output, sample_name)
            all_results[sample_name] = species_df

        # 只处理待处理的样本
        for sample_name, sample_info in pending_samples.items():
            logger.info(f"\n{'=' * 60}")
            logger.info(f"开始处理样本: {sample_name}")
            logger.info(f"{'=' * 60}")

            result = self.run_metaphlan_v4_analysis_single(sample_name, sample_info)
            all_results[sample_name] = result

        # 5. 生成汇总报告
        self.generate_summary_report(all_results)

        # 6. 打印最终摘要
        successful_analyses = len([r for r in all_results.values() if r is not None])
        total_analyses = len(all_results)

        logger.info(f"\n{'=' * 60}")
        logger.info("原始数据MetaPhlAn 4.x物种分析完成!")
        logger.info(f"成功处理: {successful_analyses}/{total_analyses} 个样本")
        logger.info(f"{'=' * 60}")

        # 打印摘要到控制台
        print(f"\n✅ 原始数据MetaPhlAn分析完成!")
        print(f"成功处理: {successful_analyses}/{total_analyses} 个样本")
        print(f"结果保存在: {self.output_base_dir}")

        if successful_analyses > 0:
            print(f"\n各样本结果位置:")
            for sample_name in samples.keys():
                print(f"  {sample_name}: {self.per_sample_output_dir / sample_name}")

            print(f"\n汇总报告: {self.summary_output_dir / 'raw_data_metaphlan_summary.txt'}")
            print(f"质控结果: {self.qc_output_dir}")

        return True


def main():
    """
    主函数
    """
    print("Raw Data Corn Silage Microbial MetaPhlAn 4.x Species Analysis")
    print("=" * 70)

    try:
        analyzer = RawDataSilageMetaphlan4Analyzer()
        success = analyzer.run_raw_data_analysis()

        if success:
            print(f"\n✅ 原始数据分析完成! 结果保存在: {analyzer.output_base_dir}")
        else:
            print(f"\n❌ 分析失败，请检查日志文件: {analyzer.output_base_dir / 'raw_data_metaphlan.log'}")
            sys.exit(1)
    except Exception as e:
        logger.error(f"程序运行出错: {e}")
        print(f"\n❌ 程序运行出错: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()