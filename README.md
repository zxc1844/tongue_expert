# 中医舌诊AI辅助分析项目

## 项目介绍

本项目旨在利用人工智能（尤其是视觉大模型）对中医舌诊进行自动分析，提升诊断准确性并辅助医生进行疾病诊断。

当前阶段的任务主要包括：

1. 收集并分析现有的舌诊数据集。
2. 探索图像预处理方法（剪裁定位、纠偏、光线亮度调整等），提高视觉大模型诊断的准确性。
3. 使用阿里云通义千问 VL-MAX 模型对舌诊图像进行基准测试，评估现有模型在识别舌诊指标方面的能力。

## 数据集来源

[TonguExpert](https://www.biosino.org/TonguExpert/index)

## 数据集说明

数据集包含两个核心文件：

### 一级标签（人工标注）`L1_Labels_Manual.txt`

- `labels_tai`：舌苔颜色
- `labels_zhi`：舌体颜色
- `labels_fissure`：舌裂情况
- `labels_tooth_mk`：齿痕状况

### 二级标签（算法预测）`L2_Labels_Predict.txt`

- `coating_label`：舌苔的厚薄程度
- `tai_label`：舌苔颜色
- `zhi_label`：舌体颜色
- `fissure_label`：舌裂情况
- `tooth_mk_label`：齿痕状况

其他特征数据主要用于深度学习模型训练。

## 项目结构

```
.
├── data/                      # 数据目录
│   └── TonguExpertDatabase/   # 舌诊数据集
│       ├── Phenotypes/        # 表型数据
│       └── TongueImage/       # 舌头图像
│           ├── Raw/           # 原始图像
│           └── Mask/          # 掩码图像
├── src/                       # 源代码
│   ├── tongue_analysis.py     # 数据分析脚本
│   └── baseline_test.py       # 基准测试脚本
├── out_put/                   # 输出目录
│   ├── baseline_results/      # 基准测试结果
│   └── label_analysis_results/# 标签分析结果
├── run_baseline_test.py       # 运行基准测试的脚本
├── requirements.txt           # 项目依赖
├── .env.template              # 环境变量模板
└── README.md                  # 项目说明
```

## 基准测试

我们使用阿里云通义千问 VL-MAX 视觉大模型对舌诊图像进行识别，评估其在舌诊指标识别方面的能力。该测试使用 OpenAI 兼容接口调用阿里云的通义千问 VL-MAX 模型。

### 测试方法

1. 从数据集中读取舌头图像和对应的标签（coating_label, tai_label, zhi_label, fissure_label, tooth_mk_label）
2. 将图像转换为 Base64 编码
3. 通过通义千问的 OpenAI 兼容接口调用 VL-MAX API 对图像进行分析
4. 提取模型预测的标签
5. 计算模型在各项舌诊指标上的准确率等性能指标
6. 生成测试报告

### 阿里云通义千问 API 接口

本项目使用阿里云通义千问的 OpenAI 兼容接口调用 VL-MAX 模型。通义千问提供了与 OpenAI 兼容的接口，可以直接使用 OpenAI Python SDK 进行调用，只需更改 base_url。

```python
client = openai.OpenAI(
    api_key=os.environ.get("DASHCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)
```

### 运行测试

1. 安装必要的依赖：
```bash
pip install -r requirements.txt
```

2. 设置API密钥：
   - 复制 `.env.template` 文件为 `.env`
   - 在 `.env` 文件中填入您的阿里云通义千问 API 密钥（DASHCOPE_API_KEY）

3. 运行基准测试：
```bash
# 默认测试10个样本
python run_baseline_test.py

# 测试100个样本
python run_baseline_test.py --sample 100

# 测试所有样本
python run_baseline_test.py --sample -1

# 指定不同的模型（如果通义千问提供了其他模型）
python run_baseline_test.py --model other-vl-model

# 查看所有可用选项
python run_baseline_test.py --help
```

### 测试结果

测试结果将保存在 `out_put/baseline_results/` 目录下，包括：
- 预测结果的 JSON 文件
- 性能指标的 JSON 文件
- 可读性的 Markdown 格式测试报告

## 已完成工作

- 完成数据集的获取与初步分析，编写了对应的数据探索脚本，分析了标签分布和基本特征。
- 实现了使用阿里云通义千问 VL-MAX 模型进行舌诊图像识别的基准测试功能。

## 下一步计划

- **探索图像预处理方法**：
  - 图像裁剪与舌体区域定位
  - 角度纠偏技术
  - 光线、亮度及对比度调整
- **实验设计**：对比原始图像与经过不同预处理方法后的图像在视觉大模型上的表现，找到有效提升模型准确率的方法。