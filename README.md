# 共享单车出行与绿色生活方式 —— 数据挖掘分析

基于 Capital Bikeshare 系统（Washington D.C., 2011-2012）的共享单车数据，运用数据挖掘技术分析骑行行为模式，并量化共享单车对碳减排的贡献。

## 分析任务

| 序号 | 任务 | 方法 |
|------|------|------|
| 1 | 工作日与节假日骑行模式对比 | 描述性统计、24h时序分析、Welch t检验、Mann-Whitney U检验 |
| 2 | 天气/温湿度对骑行量的影响回归与分类 | 线性回归、岭回归、决策树(回归+分类)、逻辑回归、随机森林、梯度提升 |
| 3 | 用户类型时序行为聚类 + 关联规则 | K-Means聚类、轮廓系数法、Apriori关联规则 |
| 4 | 共享单车与碳排放减少量化估算 | 碳减排计算、月度趋势、季节贡献、全球规模外推 |

## 项目结构

```
├── data/
│   └── bike+sharing+dataset/
│       ├── day.csv          # 日级数据 (731条)
│       ├── hour.csv         # 小时级数据 (17379条)
│       └── Readme.txt       # 数据集说明
├── models/                  # 模型存储目录
├── app.py                   # Flask + ECharts Web 展示系统
├── sample.ipynb             # 主分析 Notebook
├── requirements.txt         # Python 依赖
├── .gitignore
└── README.md
```

## 数据说明

数据来自 [UCI Bike Sharing Dataset](https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset)，主要字段：

| 字段 | 说明 |
|------|------|
| `season` | 季节 (1春/2夏/3秋/4冬) |
| `weathersit` | 天气 (1晴/2雾/3小雪雨/4大雨冰雹) |
| `temp` | 归一化温度 (÷41) |
| `hum` | 归一化湿度 (÷100) |
| `casual` | 临时用户骑行量 |
| `registered` | 注册用户骑行量 |
| `cnt` | 总骑行量 |
| `workingday` | 是否工作日 (1/0) |

## 环境配置

```bash
pip install -r requirements.txt
```

主要依赖：pandas, numpy, matplotlib, seaborn, scikit-learn, scipy, flask

## 运行

### Jupyter Notebook
在 PyCharm 或 Jupyter 中打开 `sample.ipynb`，顺序执行所有 cell。

### Web 展示系统 (Flask + ECharts)
```bash
python app.py
```
启动后浏览器访问 http://localhost:5000，顶部导航栏切换4个分析页面，所有图表使用 ECharts 渲染，支持交互式缩放和数据提示。

## 算法清单

| 类型 | 算法 | 参数调优 |
|------|------|----------|
| 回归 | 线性回归、岭回归、决策树回归、随机森林、梯度提升 | 决策树max_depth 7档对比 |
| 分类 | 决策树分类、逻辑回归 | max_depth 6档 + 5折交叉验证 |
| 聚类 | K-Means | K=2~8 肘部法 + 轮廓系数法 |
| 关联规则 | Apriori思想 | 最小支持度、最小置信度可调 |

## 核心发现

- **工作日**骑行呈早晚高峰双峰模式（通勤驱动），**非工作日**呈午后单峰模式（休闲驱动）
- **温度**是影响骑行量最强的正向因子，湿度和恶劣天气显著抑制骑行
- 注册用户行为可清晰聚类为「通勤型」和「休闲型」两种模式
- 夏季+晴天+高温 => 骑行量高（关联规则提升度显著大于1）
- 共享单车具有显著碳减排效益，是实现双碳目标的有效途径

## 参考文献

1. Fanaee-T, H., & Gama, J. (2013). Event labeling combining ensemble detectors and background knowledge. *Progress in Artificial Intelligence*, 1-15.
2. Capital Bikeshare System Data. http://capitalbikeshare.com/system-data
