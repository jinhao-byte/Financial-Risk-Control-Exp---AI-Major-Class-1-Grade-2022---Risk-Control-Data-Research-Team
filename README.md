# Financial-Risk-Control-Exp---AI-Major-Class-1-Grade-2022---Risk-Control-Data-Research-Team
## 项目简介
本小组在《机器学习》课程综合实践项目的多个场景（含信用评分检测、欺诈检测、反洗钱等）中，选定信用评分检测为实践场景，开发基于机器学习的信用评分检测系统；该系统通过对比传统与进阶算法性能，筛选优化模型方案以提升信用评分准确性与风险区分度，涵盖数据预处理、模型训练、性能优化及部署全流程，可应用于金融机构信贷审批等信用评分场景，为风控决策提供技术支撑。

## 实验目的
1. 掌握数据预处理基本原理与实现方法
2. 理解并应用k近邻、随机森林、SVM、逻辑回归等传统机器学习算法
3. 熟练使用Python及scikit-learn库构建分类模型
4. 掌握精度、AUC、ROC等模型评估指标的应用
5. 运用特征选择、集成学习等方法优化模型性能
6. 实现基于LightGBM的高性能信用评分检测模型及可视化交互系统

## 核心算法原理

### 传统机器学习模型
| 模型名称 | 核心原理 | 优缺点分析 |
|---------|---------|-----------|
| 逻辑回归 | 基于Sigmoid函数的线性分类模型 | 简单易解释，但对非线性数据拟合能力弱 |
| 支持向量机 | 通过核函数映射实现高维空间线性分隔 | 泛化能力强，但对大规模数据训练较慢 |
| 随机森林 | 多棵决策树集成的Bagging算法 | 抗过拟合能力强，可解释性较好 |
| BP神经网络 | 多层前馈神经网络，通过反向传播优化参数 | 非线性拟合能力强，需大量数据训练 |
| 朴素贝叶斯 | 基于贝叶斯定理的概率模型，假设特征独立 | 计算高效，对特征相关性敏感 |

### 进阶模型：LightGBM
- **核心优势**：采用直方图算法、梯度单边采样(GOSS)和Leaf-wise树生长策略
- **性能优化**：针对不平衡数据设计，通过类别权重调整提升少数类识别能力
- **工程特性**：训练速度快，支持特征重要性评估，适合大规模金融数据场景

## 数据集描述
- **来源**：欧洲持卡人2013年9月交易数据（基础任务）、IEEE金融欺诈检测竞赛数据集（进阶任务）
- **特征构成**：
  - 基础特征：经PCA处理的28个匿名特征(V1-V28)
  - 交易特征：交易金额(Amount)、时间戳(Time)
- **数据规模**：包含284,807条交易记录，其中欺诈样本占比0.172%
- **预处理**：
  - 缺失值填充：使用-999填充缺失特征
  - 特征标准化：对Amount字段进行Z-score标准化
  - 类别特征编码：使用LabelEncoder处理分类型特征

## 实现步骤

### 1. 环境配置
```bash
# 核心依赖库
pip install numpy pandas scikit-learn lightgbm matplotlib seaborn
```

### 2. 数据预处理
```python
# 标准化处理示例
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
df['Amount'] = sc.fit_transform(df['Amount'].values.reshape(-1, 1))
```

### 3. 模型训练流程
1. **基础任务**：实现5种传统模型对比实验
2. **进阶任务**：构建LightGBM模型并优化
   ```python
   model = lgb.LGBMClassifier(
       n_estimators=1000,
       learning_rate=0.05,
       class_weight='balanced',
       random_state=42
   )
   ```
3. **高阶任务**：开发Tkinter可视化交互系统

## 实验结果与分析

### 模型性能对比
| 模型 | F1-Score | AUC | 核心优势 |
|-----|---------|-----|---------|
| 逻辑回归 | 0.7514 | 0.89 | 计算高效，可解释性强 |
| 随机森林 | 0.7797 | 0.92 | 抗过拟合，特征重要性可解释 |
| BP神经网络 | 0.8387 | 0.94 | 非线性拟合能力强 |
| LightGBM | - | 0.97 | 处理不平衡数据性能最优 |

### 关键发现
1. 传统模型中，BP神经网络表现最优(F1=0.8387)
2. LightGBM在AUC指标上较传统模型提升3-8%
3. 样本不平衡是影响模型性能的关键因素，需通过类别权重调整优化

## 核心代码结构
```
project/
├── data/               # 数据集目录
├── model/              # 模型保存路径
│   ├── ieee_lgbm_model.pkl
│   └── cat_encoders.pkl
├── basic_task.py       # 传统模型对比实验
├── advanced_task.py    # LightGBM模型实现
└── gui_demo.py         # Tkinter可视化系统
```

## 系统展示
![欺诈检测系统界面](https://p3-flow-imagex-sign.byteimg.com/ocean-cloud-tos/pdf/d3fc3f98134878fdebea22d4dd24605e_30_1200.jpg)

## 项目亮点
1. **多模型对比**：系统评估5种传统算法性能，为模型选择提供依据
2. **性能优化**：LightGBM模型AUC达0.97，优于传统模型
3. **工程实现**：完整的模型训练-评估-部署流程，包含数据预处理工具链
4. **可视化交互**：基于Tkinter构建直观的欺诈概率展示界面

## 贡献指南
1. Fork本仓库
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add some amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 创建Pull Request

## 许可证
[MIT License](LICENSE) - 详情参见LICENSE文件

## 联系方式
- 项目维护者：吕子恒,施以慷,王攀登,李超俊,段浩
- 联系邮箱：[your-email@example.com]
