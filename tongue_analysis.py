#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
舌诊数据探索分析脚本
作者: AI助手
日期: 2024-04-26
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False     # 用来正常显示负号

class TongueDataAnalyzer:
    """舌诊数据分析类"""
    
    def __init__(self, data_dir="tonge/data/TonguExpertDatabase"):
        """
        初始化函数
        
        参数:
            data_dir (str): 数据目录路径
        """
        self.data_dir = Path(data_dir)
        self.phenotypes_dir = self.data_dir / "Phenotypes"
        self.manual_labels = None
        self.predict_labels = None
        self.features = None
        
    def load_data(self):
        """加载所有相关数据文件"""
        print("正在加载数据...")
        
        # 加载手动标注的标签
        self.manual_labels = pd.read_csv(
            self.phenotypes_dir / "L1_Labels_Manual.txt",
            sep='\t'
        )
        print(f"手动标注数据形状: {self.manual_labels.shape}")
        
        # 加载预测的标签
        self.predict_labels = pd.read_csv(
            self.phenotypes_dir / "L2_Labels_Predict.txt",
            sep='\t'
        )
        print(f"预测标签数据形状: {self.predict_labels.shape}")
        
        # 加载特征数据
        feature_files = {
            'color': 'P11_Tg_Color.txt',
            'shape': 'P12_Tg_Shape.txt',
            'texture': 'P13_Tg_Texture.txt',
            'cnn': 'P14_Tg_CNN.txt'
        }
        
        self.features = {}
        for feature_type, filename in feature_files.items():
            try:
                self.features[feature_type] = pd.read_csv(
                    self.phenotypes_dir / filename,
                    sep='\t'
                )
                print(f"{feature_type}特征数据形状: {self.features[feature_type].shape}")
            except FileNotFoundError:
                print(f"警告: {filename}文件未找到")
    
    def analyze_labels_distribution(self):
        """分析标签分布情况"""
        print("\n=== 标签分布分析 ===")
        
        if self.manual_labels is not None:
            print("\n手动标注的标签分布:")
            for col in self.manual_labels.columns[1:]:  # 跳过SID列
                print(f"\n{col}的分布:")
                print(self.manual_labels[col].value_counts(dropna=False))
        
        if self.predict_labels is not None:
            print("\n预测标签的分布:")
            for col in self.predict_labels.columns[1:]:  # 跳过SID列
                print(f"\n{col}的分布:")
                print(self.predict_labels[col].value_counts(dropna=False))
    
    def analyze_features(self):
        """分析特征数据的基本统计信息"""
        print("\n=== 特征数据分析 ===")
        
        for feature_type, df in self.features.items():
            print(f"\n{feature_type}特征的基本统计信息:")
            print(df.describe())
            
           
    
    def run_analysis(self):
        """运行完整的分析流程"""
        self.load_data()
        self.analyze_labels_distribution()
        self.analyze_features()

def main():
    """主函数"""
    analyzer = TongueDataAnalyzer()
    analyzer.run_analysis()
    
if __name__ == "__main__":
    main()