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
│   ├── preprocessing_results/ # 图像预处理结果和对比
│   ├── TonguExpertDatabase/   # 舌诊数据集
│   │   ├── Phenotypes/        # 表型数据
│   │   └── TongueImage/       # 舌头图像
│   │       ├── Raw/           # 原始图像
│   │       └── Mask/          # 掩码图像
│   ├── test.txt               # 测试集数据
│   └── train.jsonl            # 训练集数据
├── src/                       # 源代码
│   ├── tongue_analysis.py     # 数据分析脚本
│   ├── baseline_test.py       # 基准测试脚本
│   ├── image_preprocessing.py # 图像预处理模块
│   └── test_preprocessing.py  # 预处理测试脚本
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

### 并发处理功能

为了更高效地处理大量图像，项目现已实现并发API调用功能，显著减少处理5000多张图片所需的时间。

#### 并发处理特性

- **并行API调用**：同时处理多张图片，大幅提高处理速度
- **智能速率限制**：自动控制API调用频率，避免超出服务提供商的限制
- **灵活配置**：可调整并发工作线程数和API调用速率
- **完善错误处理**：单个图片处理失败不会影响整体流程，同时记录失败的图片ID
- **进度监控**：实时显示处理进度和估计完成时间

#### 使用并发处理

运行具有并发处理能力的测试脚本：

```bash
# 使用默认设置（5个工作线程，每秒2个API调用）处理10个样本
python run_concurrent_baseline.py

# 处理所有样本，使用10个工作线程和每秒3个API调用
python run_concurrent_baseline.py --sample -1 --workers 10 --rate 3

# 自定义输出目录
python run_concurrent_baseline.py --output "out_put/concurrent_results"
```

#### 可用命令行参数

- `--sample`：要处理的样本数量。设置为-1表示处理所有样本
- `--workers`：并发工作线程的最大数量（默认为5）
- `--rate`：每秒最大API调用次数（默认为2）
- `--model`：要使用的模型名称（默认为"qwen-vl-max"）
- `--output`：结果输出目录

#### 性能优化建议

根据阿里云通义千问的服务限制，建议以下并发设置：

- 对于 `qwen-vl-max` 模型，最佳设置为 `--workers 5 --rate 2`
- 处理更多图像时，可以增加工作线程数，但应保持API调用速率不变
- 如果遇到API限制错误，请减少 `--rate` 参数值

#### 错误处理和恢复

脚本会记录处理失败的图片ID，保存到输出目录中的JSON文件。可以使用这些ID重新处理失败的图片：

```bash
# 处理特定图片（未实现，仅作示例）
# python run_concurrent_baseline.py --sids failed_sids.json
```

### 测试结果

测试结果将保存在 `out_put/baseline_results/` 目录下，包括：

- 预测结果的 JSON 文件
- 性能指标的 JSON 文件
- 可读性的 Markdown 格式测试报告

## 已完成工作

- 完成数据集的获取与初步分析，编写了对应的数据探索脚本，分析了标签分布和基本特征。
- 实现了使用阿里云通义千问 VL-MAX 模型进行舌诊图像识别的基准测试功能。
- 完成图像预处理模块的开发与集成，提升舌诊图像的质量。

## 下一步计划

- **探索图像预处理方法**：
  - 图像裁剪与舌体区域定位
  - 角度纠偏技术
  - 光线、亮度及对比度调整
- **实验设计**：对比原始图像与经过不同预处理方法后的图像在视觉大模型上的表现，找到有效提升模型准确率的方法。

# Tongue Expert Dataset Preparation

This repository contains code and data for the Tongue Expert project, which involves creating training and test datasets for fine-tuning a model for tongue diagnosis in Traditional Chinese Medicine (TCM).

## Dataset Structure

The original dataset is stored in the following locations:

- Images: `data/TonguExpertDatabase/TongueImage/Raw/`
- Labels: `data/TonguExpertDatabase/Phenotypes/L2_Labels_Predict.txt`

## Dataset Splitting

The dataset has been split into training and test sets as follows:

1. **Test Set**:

   - 500 randomly sampled images with stratified sampling to maintain class distribution
   - Saved to `data/test.txt`
   - Format: Tab-separated values with columns for SID and 5 diagnostic labels
2. **Training Set**:

   - All remaining samples (~5,492 images)
   - Saved to `data/train.jsonl`
   - Format: JSONL file with each line containing a complete training example in the format required for fine-tuning

## Implementation Details

- Random seed of 42 was used for reproducibility
- Stratified sampling was attempted on a composite label to maintain distribution across all categories
- All 5 diagnostic labels are included: coating_label, tai_label, zhi_label, fissure_label, tooth_mk_label
- Empty values in the original dataset were handled appropriately (converted to "None" in the output)
- Image paths in the training data refer to the original image files in the Raw directory

## Scripts

- `split_dataset.py`: Main script to split the dataset and create the test.txt and train.jsonl files
- `clean_test_file.py`: Script to clean the test.txt file by removing the composite_label column

## Dataset Statistics

The dataset includes the following label distributions:

- `coating_label`: greasy (~89%), greasy_thick (~9%), non_greasy (~2%)
- `tai_label`: white (~56%), light_yellow (~38%), yellow (~6%)
- `zhi_label`: regular (~50%), dark (~27%), light (~23%)
- `fissure_label`: light (~21%), severe (~12%), None (~67%)
- `tooth_mk_label`: light (~32%), severe (~12%), None (~56%)

## Usage

The test.txt file can be used for evaluation, while the train.jsonl file can be used directly for fine-tuning the model according to the specifications in the demand document.

## 图像预处理模块

新增的图像预处理模块提供了舌诊图像增强功能，以提高后续AI识别的准确性。

### 功能特点

该预处理模块包含以下增强功能：

1. **自动白平衡**：调整图像的白平衡，使舌头颜色更加自然。
2. **光照归一化**：确保舌头图像亮度均匀，消除不同光照条件带来的影响。
3. **色彩校正**：标准化颜色，修正不同设备或光源造成的色差。
4. **阴影去除与增强**：增强舌头阴影区域的细节。
5. **其他增强**：包括图像对比度增强和轻度降噪处理。

### 使用方法

#### 预处理单张图像

```python
from image_preprocessing import load_and_preprocess_image

# 加载并预处理图像
processed_image = load_and_preprocess_image('图像路径.jpg')
```

#### 应用特定的预处理步骤

```python
import cv2
from image_preprocessing import white_balance, light_normalization, preprocess_image

# 加载图像
image = cv2.imread('图像路径.jpg')

# 仅应用白平衡
balanced_image = white_balance(image)

# 仅应用光照归一化
normalized_image = light_normalization(image)

# 应用完整的预处理流程
fully_processed_image = preprocess_image(image)
```

#### 测试预处理效果

可以使用提供的测试脚本来可视化比较每个预处理步骤的效果，并生成HTML报告：

```bash
# 默认将结果保存到 data/preprocessing_results/[时间戳] 目录下
python src/test_preprocessing.py data/TonguExpertDatabase/TongueImage/Raw/TE0002823.jpg

# 指定自定义输出目录
python src/test_preprocessing.py 图像路径.jpg --save_dir 自定义输出目录
```

该脚本会执行以下操作：

1. 显示每个预处理步骤的处理前后对比图像
2. 将所有处理结果保存到指定目录
3. 生成HTML格式的可视化对比报告（processing_report.html）

### 技术实现

预处理流程包括以下步骤：

1. **降噪**：应用轻度降噪滤波器，减少噪点同时保留细节。
2. **白平衡**：使用灰度世界算法调整色彩通道。
3. **光照归一化**：使用CLAHE（对比度受限的自适应直方图均衡化）来规范化亮度。
4. **色彩校正**：使用白平衡技术标准化颜色。
5. **Retinex增强**：去除阴影并增强暗区域的细节。
6. **伽马校正**：调整图像的亮度。
7. **对比度增强**：改善最终图像的对比度。

### 与基准测试集成

该预处理模块已经与基准测试系统集成，在将图像发送给视觉模型之前自动应用预处理，处理流程如下：

1. 在`baseline_test.py`中，`encode_image_to_base64`函数加载图像时会首先应用预处理
2. 若预处理失败则自动回退到原始图像，确保测试流程不会中断
3. 处理后的图像再编码为Base64格式发送给视觉模型

这种集成方式允许在不修改整体测试流程的前提下，提升图像质量，提高模型识别准确率。
