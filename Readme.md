<div align='center'><font size='6'>NBA 运动员各项数据分析</font></div>

## NBA Analytics

#### 创建一个虚拟环境，隔离所有的 python 包，便于封装所有内容，避免依赖冲突

> python -m venv nbaanalytics
> .\nbaanalytics\Scripts\activate

### streamlit

- **快速使用 python 构建 web 应用**

```
pip install streamlit
pip install pandas-profiling #弃用
pip install ydata_profiling
```

### EDA

##### EDA 是探索性数据分析（Exploratory Data Analysis）,它是数据分析的一个关键步骤，通常在正式的数据处理和建模之前进行。EDA 的目的是使用统计图表和其他数据可视化手段来发现数据的主要特征、模式、异常值、数据结构和内在关系。

- 数据总结：了解数据集的基本信息，如样本大小、特征数量、特征的数据类型等。
- 单变量分析：分析每个单独变量的分布情况，包括中心趋势（如均值、中位数）和离散程度（如方差、标准差）。
- 双变量或多变量分析：探索变量之间的关系，比如使用散点图来查看两个连续变量之间的相关性，或使用箱形图来比较不同组别的数据分布。
- 处理缺失值和异常值：识别数据中的缺失值和异常值，并决定如何处理它们。
- 数据可视化：使用图表和图形来直观展示数据和分析结果，例如条形图、直方图、箱形图、饼图、热图和散点图等。
  > 详情见 EDA_NBA.ipynb

### 以下是本次 NBA 分析中使用的软件包的详细说明：

#### 核心数据操作和可视化：

- numpy：提供高效的数值计算和数组操作来处理 NBA 数据。

- pandas：创建和操作 DataFrame，从而轻松进行数据探索和清理。

- matplotlib.pyplot：生成信息可视化，例如绘图和图表，以可视化数据中的趋势和关系。

- seaborn：增强视觉清晰度和美观性，基于 matplotlib 构建具有视觉吸引力的统计图形。
  机器学习：

- bubbly ：绘制气泡图

  >  [bubbly](https://github.com/AashitaK/bubbly/pull/4/commits/6fb8dfe2ead180d644c96920b87e100951bca44b)>Fixed AttributeError：'DataFrame' object has no attribute 'append'

- sklearn.model_selection：提供用于将数据拆分为训练集和测试集、实施交叉验证以及评估模型性能的工具。
  sklearn.linear_model：包含各种线性回归模型，包括 LinearRegression、Lasso 和 Ridge，用于构建预测模型。
  评估和诊断：

- sklearn.metrics：提供一套评估模型性能的指标，例如均方误差、R 平方等。
  scipy.stats.mstats：包含统计测试，包括正态测试，用于检查数据分布的正态性，这是线性回归的关键假设。
  scipy.stats：提供额外的统计工具，例如用于转换非正态数据以更好地适应线性回归假设的 boxcox 函数。
  数据预处理：

- sklearn.preprocessing：包括数据缩放和转换技术，例如 StandardScaler，它将特征标准化为均值 0 和标准差 1，通常可以增强模型性能。
  PolynomialFeatures：生成多项式特征，允许探索变量之间的非线性关系。
  模型构建和细化：

- sklearn.pipeline：通过创建一系列步骤（包括预处理和模型拟合）来简化建模过程，从而使模型开发更加高效。
  sklearn.model_selection.GridSearchCV：自动调整超参数，搜索最佳模型配置以提高模型性能。
  3D 可视化：

- mpl_toolkits.mplot3d：支持创建 3D 绘图，对于以更直观和身临其境的方式可视化多个变量之间的关系特别有用。
