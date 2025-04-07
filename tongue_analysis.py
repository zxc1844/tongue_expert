import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import sys
import logging
import io
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False     # 用来正常显示负号

class TongueDataAnalyzer:
    """舌诊数据分析类"""
    
    def __init__(self, data_dir="data/TonguExpertDatabase"):
        """
        初始化函数
        
        参数:
            data_dir (str): 数据目录路径
        """
        self.data_dir = Path(data_dir)
        logger.info(f"使用数据目录: {self.data_dir.absolute()}")
        
        if not self.data_dir.exists():
            logger.error(f"数据目录不存在: {self.data_dir.absolute()}")
            raise FileNotFoundError(f"数据目录不存在: {self.data_dir.absolute()}")
            
        self.phenotypes_dir = self.data_dir / "Phenotypes"
        if not self.phenotypes_dir.exists():
            logger.error(f"Phenotypes目录不存在: {self.phenotypes_dir.absolute()}")
            raise FileNotFoundError(f"Phenotypes目录不存在: {self.phenotypes_dir.absolute()}")
            
        self.manual_labels = None
        self.predict_labels = None
        self.features = None
        self.analysis_result = io.StringIO()
        
    def log(self, message):
        """记录分析结果到字符串缓冲区，并输出到日志"""
        logger.info(message)
        self.analysis_result.write(str(message) + "\n")
        
    def load_data(self):
        """加载所有相关数据文件"""
        self.log("正在加载数据...")
        
        # 加载手动标注的标签
        manual_labels_path = self.phenotypes_dir / "L1_Labels_Manual.txt"
        if not manual_labels_path.exists():
            logger.error(f"手动标注文件不存在: {manual_labels_path}")
            raise FileNotFoundError(f"手动标注文件不存在: {manual_labels_path}")
            
        self.manual_labels = pd.read_csv(
            manual_labels_path,
            sep='\t'
        )
        self.log(f"手动标注数据形状: {self.manual_labels.shape}")
        
        # 加载预测的标签
        predict_labels_path = self.phenotypes_dir / "L2_Labels_Predict.txt"
        if not predict_labels_path.exists():
            logger.error(f"预测标签文件不存在: {predict_labels_path}")
            raise FileNotFoundError(f"预测标签文件不存在: {predict_labels_path}")
            
        self.predict_labels = pd.read_csv(
            predict_labels_path,
            sep='\t'
        )
        self.log(f"预测标签数据形状: {self.predict_labels.shape}")
        
        # 加载特征数据
        feature_files = {
            'color': 'P11_Tg_Color.txt',
            'shape': 'P12_Tg_Shape.txt',
            'texture': 'P13_Tg_Texture.txt',
            'cnn': 'P14_Tg_CNN.txt'
        }
        
        self.features = {}
        for feature_type, filename in feature_files.items():
            feature_path = self.phenotypes_dir / filename
            try:
                if not feature_path.exists():
                    logger.warning(f"特征文件不存在: {feature_path}")
                    continue
                    
                self.features[feature_type] = pd.read_csv(
                    feature_path,
                    sep='\t'
                )
                self.log(f"{feature_type}特征数据形状: {self.features[feature_type].shape}")
            except Exception as e:
                logger.error(f"加载{feature_type}特征数据时出错: {str(e)}")
    
    def analyze_labels_distribution(self):
        """分析标签分布情况"""
        self.log("\n=== 标签分布分析 ===")
        
        if self.manual_labels is not None:
            self.log("\n手动标注的标签分布:")
            for col in self.manual_labels.columns[1:]:  # 跳过SID列
                self.log(f"\n{col}的分布:")
                self.log(self.manual_labels[col].value_counts(dropna=False))
        else:
            logger.warning("手动标注数据未加载，无法分析分布")
        
        if self.predict_labels is not None:
            self.log("\n预测标签的分布:")
            for col in self.predict_labels.columns[1:]:  # 跳过SID列
                self.log(f"\n{col}的分布:")
                self.log(self.predict_labels[col].value_counts(dropna=False))
        else:
            logger.warning("预测标签数据未加载，无法分析分布")
    
    def analyze_features(self):
        """分析特征数据的基本统计信息"""
        self.log("\n=== 特征数据分析 ===")
        
        if not self.features:
            logger.warning("特征数据未加载或为空，无法分析特征")
            return
            
        for feature_type, df in self.features.items():
            self.log(f"\n{feature_type}特征的基本统计信息:")
            self.log(df.describe())
    
    def save_analysis_results(self, output_dir="./results"):
        """保存分析结果到文件"""
        # 创建输出目录
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_path / f"tongue_analysis_{timestamp}.txt"
        
        # 保存结果
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(self.analysis_result.getvalue())
            
        logger.info(f"分析结果已保存到: {output_file}")
        return output_file
           
    
    def run_analysis(self):
        """运行完整的分析流程"""
        try:
            self.load_data()
            self.analyze_labels_distribution()
            self.analyze_features()
            output_file = self.save_analysis_results()
            logger.info(f"分析完成！结果已保存到: {output_file}")
        except Exception as e:
            logger.error(f"分析过程中出错: {str(e)}")
            raise

def main():
    """主函数"""
    try:
        analyzer = TongueDataAnalyzer()
        analyzer.run_analysis()
    except Exception as e:
        logger.error(f"程序运行出错: {str(e)}")
        sys.exit(1)
    
if __name__ == "__main__":
    main()