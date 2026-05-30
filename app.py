"""
共享单车出行与绿色生活方式 —— 数据挖掘分析系统
Flask + ECharts Web 应用
运行命令: python app.py
然后浏览器访问: http://localhost:5000
"""

import csv
import json
import os
from flask import Flask, render_template_string
from collections import Counter, defaultdict

app = Flask(__name__)

# ========== 数据加载 (使用 csv 模块，不依赖 pandas) ==========
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data', 'bike+sharing+dataset')


def load_csv(filename):
    path = os.path.join(DATA_DIR, filename)
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader)


def to_float(val, default=0.0):
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def to_int(val, default=0):
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


# 加载数据
day_data = load_csv('day.csv')
hour_data = load_csv('hour.csv')

# ========== 数据预处理与分析 ==========

# 季节/天气映射
season_map = {'1': '春', '2': '夏', '3': '秋', '4': '冬'}
weather_map = {'1': '晴/少云', '2': '雾/阴', '3': '小雪/雨', '4': '大雨/冰雹'}

# --- 任务一: 工作日 vs 节假日 ---
def analyze_task1():
    workday_cnt = []
    nonworkday_cnt = []
    for row in day_data:
        cnt = to_int(row['cnt'])
        if row['workingday'] == '1':
            workday_cnt.append(cnt)
        else:
            nonworkday_cnt.append(cnt)

    # 24小时模式
    hourly_work = defaultdict(list)
    hourly_nonwork = defaultdict(list)
    hourly_work_reg = defaultdict(list)
    hourly_nonwork_reg = defaultdict(list)
    hourly_work_cas = defaultdict(list)
    hourly_nonwork_cas = defaultdict(list)
    for row in hour_data:
        hr = to_int(row['hr'])
        cnt = to_int(row['cnt'])
        reg = to_int(row['registered'])
        cas = to_int(row['casual'])
        if row['workingday'] == '1':
            hourly_work[hr].append(cnt)
            hourly_work_reg[hr].append(reg)
            hourly_work_cas[hr].append(cas)
        else:
            hourly_nonwork[hr].append(cnt)
            hourly_nonwork_reg[hr].append(reg)
            hourly_nonwork_cas[hr].append(cas)

    hours = list(range(24))
    avg = lambda d: [round(sum(d[h]) / len(d[h]), 1) if d[h] else 0 for h in hours]

    return {
        'workday_mean': round(sum(workday_cnt) / len(workday_cnt)),
        'nonworkday_mean': round(sum(nonworkday_cnt) / len(nonworkday_cnt)),
        'workday_count': len(workday_cnt),
        'nonworkday_count': len(nonworkday_cnt),
        'hours': hours,
        'hourly_cnt_work': avg(hourly_work),
        'hourly_cnt_nonwork': avg(hourly_nonwork),
        'hourly_reg_work': avg(hourly_work_reg),
        'hourly_reg_nonwork': avg(hourly_nonwork_reg),
        'hourly_cas_work': avg(hourly_work_cas),
        'hourly_cas_nonwork': avg(hourly_nonwork_cas),
    }


# --- 任务二: 回归分析 ---
def analyze_task2():
    # 相关性计算
    fields = ['temp', 'atemp', 'hum', 'windspeed', 'weathersit', 'season', 'cnt']
    data_dict = {f: [to_float(row[f]) for row in day_data] for f in fields}
    n = len(day_data)

    def pearson_corr(x, y):
        mx, my = sum(x)/n, sum(y)/n
        num = sum((xi-mx)*(yi-my) for xi, yi in zip(x, y))
        dx = sum((xi-mx)**2 for xi in x) ** 0.5
        dy = sum((yi-my)**2 for yi in y) ** 0.5
        return round(num / (dx * dy), 3) if dx * dy > 0 else 0

    corr_matrix = []
    for f1 in fields:
        row = []
        for f2 in fields:
            row.append(pearson_corr(data_dict[f1], data_dict[f2]))
        corr_matrix.append(row)

    # 温度 vs 骑行量散点
    temp_scatter = [[round(to_float(row['temp'])*41, 1), to_int(row['cnt'])] for row in day_data]

    # 天气对骑行量的影响
    weather_cnt = defaultdict(list)
    for row in day_data:
        w = weather_map.get(row['weathersit'], '未知')
        weather_cnt[w].append(to_int(row['cnt']))
    weather_labels = list(weather_cnt.keys())
    weather_means = [round(sum(v)/len(v)) for v in weather_cnt.values()]

    # 季节对骑行量的影响
    season_cnt = defaultdict(list)
    for row in day_data:
        s = season_map.get(row['season'], '未知')
        season_cnt[s].append(to_int(row['cnt']))
    season_order = ['春', '夏', '秋', '冬']
    season_means = [round(sum(season_cnt[s])/len(season_cnt[s])) for s in season_order]

    return {
        'corr_matrix': corr_matrix,
        'corr_labels': ['温度', '体感温度', '湿度', '风速', '天气', '季节', '骑行量'],
        'temp_scatter': temp_scatter,
        'weather_labels': weather_labels,
        'weather_means': weather_means,
        'season_labels': season_order,
        'season_means': season_means,
    }


# --- 任务三: 聚类分析 ---
def analyze_task3():
    # 构建每小时骑行分布
    casual_by_day = defaultdict(lambda: defaultdict(int))
    registered_by_day = defaultdict(lambda: defaultdict(int))
    for row in hour_data:
        d = row['dteday']
        hr = to_int(row['hr'])
        casual_by_day[d][hr] += to_int(row['casual'])
        registered_by_day[d][hr] += to_int(row['registered'])

    def get_profiles(by_day):
        profiles = {}
        for d, hours in by_day.items():
            total = sum(hours.values())
            if total > 0:
                profiles[d] = [hours.get(h, 0) / total for h in range(24)]
        return profiles

    casual_profiles = get_profiles(casual_by_day)
    registered_profiles = get_profiles(registered_by_day)

    # 简单 K-Means (手动实现)
    import random
    random.seed(42)

    def simple_kmeans(data, k=3, max_iter=20):
        keys = list(data.keys())
        vectors = [data[k] for k in keys]
        n = len(vectors)
        dim = len(vectors[0])

        # 随机初始化
        indices = random.sample(range(n), k)
        centers = [vectors[i][:] for i in indices]

        for _ in range(max_iter):
            # 分配
            clusters = [[] for _ in range(k)]
            for i, v in enumerate(vectors):
                dists = []
                for c in centers:
                    d = sum((v[j]-c[j])**2 for j in range(dim)) ** 0.5
                    dists.append(d)
                clusters[dists.index(min(dists))].append(i)

            # 更新中心
            new_centers = []
            for cluster in clusters:
                if cluster:
                    center = [0] * dim
                    for i in cluster:
                        for j in range(dim):
                            center[j] += vectors[i][j]
                    center = [x / len(cluster) for x in center]
                    new_centers.append(center)
                else:
                    new_centers.append([0] * dim)
            centers = new_centers

        # 最终分配
        labels = []
        for v in vectors:
            dists = [sum((v[j]-c[j])**2 for j in range(dim)) ** 0.5 for c in centers]
            labels.append(dists.index(min(dists)))

        cluster_sizes = [labels.count(i) for i in range(k)]
        return centers, cluster_sizes

    c_centers, c_sizes = simple_kmeans(casual_profiles, k=3)
    r_centers, r_sizes = simple_kmeans(registered_profiles, k=3)

    return {
        'hours': list(range(24)),
        'casual_centers': [[round(x, 4) for x in c] for c in c_centers],
        'casual_sizes': c_sizes,
        'registered_centers': [[round(x, 4) for x in c] for c in r_centers],
        'registered_sizes': r_sizes,
    }


# --- 任务四: 碳减排估算 ---
def analyze_task4():
    AVG_TRIP_KM = 3.5
    CAR_CO2 = 120  # g/km
    BUS_CO2 = 80
    TREE_CO2 = 22000  # g/year

    total_trips = sum(to_int(row['cnt']) for row in day_data)
    total_co2_car = total_trips * AVG_TRIP_KM * CAR_CO2 / 1000  # kg
    total_co2_bus = total_trips * AVG_TRIP_KM * BUS_CO2 / 1000

    # 月度趋势
    monthly = defaultdict(int)
    monthly_co2 = defaultdict(float)
    for row in day_data:
        month = row['dteday'][:7]  # YYYY-MM
        cnt = to_int(row['cnt'])
        monthly[month] += cnt
        monthly_co2[month] += cnt * AVG_TRIP_KM * CAR_CO2 / 1000

    months = sorted(monthly.keys())

    # 季节贡献
    season_co2 = defaultdict(float)
    for row in day_data:
        s = season_map.get(row['season'], '未知')
        season_co2[s] += to_int(row['cnt']) * AVG_TRIP_KM * CAR_CO2 / 1000

    return {
        'total_trips': total_trips,
        'total_km': round(total_trips * AVG_TRIP_KM),
        'co2_car_kg': round(total_co2_car),
        'co2_car_ton': round(total_co2_car / 1000, 1),
        'co2_bus_kg': round(total_co2_bus),
        'co2_bus_ton': round(total_co2_bus / 1000, 1),
        'trees_car': round(total_co2_car / (TREE_CO2 / 1000)),
        'trees_bus': round(total_co2_bus / (TREE_CO2 / 1000)),
        'months': months,
        'monthly_trips': [monthly[m] for m in months],
        'monthly_co2': [round(monthly_co2[m], 1) for m in months],
        'season_labels': ['春', '夏', '秋', '冬'],
        'season_co2': [round(season_co2[s], 1) for s in ['春', '夏', '秋', '冬']],
    }


# ========== 预计算所有分析结果 ==========
task1 = analyze_task1()
task2 = analyze_task2()
task3 = analyze_task3()
task4 = analyze_task4()


# ========== HTML 模板 ==========
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>共享单车数据挖掘分析系统</title>
    <script src="https://cdn.bootcdn.net/ajax/libs/echarts/5.4.3/echarts.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Microsoft YaHei', sans-serif; background: #f0f2f5; color: #333; }
        .header {
            background: linear-gradient(135deg, #1a73e8, #0d47a1);
            color: white; padding: 30px; text-align: center;
        }
        .header h1 { font-size: 28px; margin-bottom: 8px; }
        .header p { font-size: 14px; opacity: 0.85; }
        .nav {
            background: white; padding: 0 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            display: flex; justify-content: center; gap: 0; position: sticky; top: 0; z-index: 100;
        }
        .nav a {
            padding: 14px 24px; text-decoration: none; color: #555; font-size: 14px;
            border-bottom: 3px solid transparent; transition: all 0.3s;
        }
        .nav a:hover, .nav a.active { color: #1a73e8; border-bottom-color: #1a73e8; }
        .container { max-width: 1200px; margin: 20px auto; padding: 0 20px; }
        .section {
            background: white; border-radius: 12px; padding: 24px; margin-bottom: 20px;
            box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        }
        .section h2 { font-size: 20px; color: #1a73e8; margin-bottom: 16px; padding-bottom: 8px; border-bottom: 2px solid #e8eaed; }
        .chart { width: 100%; height: 400px; margin: 10px 0; }
        .chart-half { width: 48%; height: 350px; display: inline-block; }
        .metrics { display: flex; gap: 16px; flex-wrap: wrap; margin: 16px 0; }
        .metric {
            flex: 1; min-width: 150px; background: #f8f9fa; border-radius: 8px;
            padding: 16px; text-align: center; border-left: 4px solid #1a73e8;
        }
        .metric .value { font-size: 24px; font-weight: bold; color: #1a73e8; }
        .metric .label { font-size: 12px; color: #666; margin-top: 4px; }
        .insight {
            background: #e8f5e9; border-left: 4px solid #4caf50; padding: 12px 16px;
            margin: 12px 0; border-radius: 4px; font-size: 14px;
        }
        table { width: 100%; border-collapse: collapse; margin: 12px 0; }
        th, td { padding: 10px 12px; text-align: center; border: 1px solid #e0e0e0; }
        th { background: #f5f5f5; font-weight: 600; }
        tr:hover { background: #f0f7ff; }
        .footer { text-align: center; padding: 20px; color: #999; font-size: 12px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>共享单车出行与绿色生活方式 —— 数据挖掘分析系统</h1>
        <p>Capital Bikeshare, Washington D.C. (2011-2012) | 绿色发展 · 双碳目标 · 低碳生活</p>
    </div>
    <div class="nav">
        <a href="javascript:void(0)" class="active" onclick="return showSection('task1', this)">工作日vs节假日</a>
        <a href="javascript:void(0)" onclick="return showSection('task2', this)">回归与分类</a>
        <a href="javascript:void(0)" onclick="return showSection('task3', this)">聚类分析</a>
        <a href="javascript:void(0)" onclick="return showSection('task4', this)">碳减排估算</a>
    </div>
    <div class="container">
        <!-- 任务一 -->
        <div id="task1" class="section">
            <h2>任务一：工作日与节假日骑行模式对比</h2>
            <div class="metrics">
                <div class="metric"><div class="value">{{ task1.workday_mean }}</div><div class="label">工作日均骑行量</div></div>
                <div class="metric"><div class="value">{{ task1.nonworkday_mean }}</div><div class="label">非工作日均骑行量</div></div>
                <div class="metric"><div class="value">{{ task1.workday_count }}</div><div class="label">工作日天数</div></div>
                <div class="metric"><div class="value">{{ task1.nonworkday_count }}</div><div class="label">非工作日天数</div></div>
            </div>
            <div id="chart1" class="chart"></div>
            <div class="insight">工作日呈早晚高峰双峰模式(7-9时、16-19时)，典型通勤特征；非工作日呈午后单峰模式(10-17时)，体现休闲骑行特征。</div>
        </div>

        <!-- 任务二 -->
        <div id="task2" class="section" style="display:none">
            <h2>任务二：天气/温湿度对骑行量的影响回归</h2>
            <div id="chart2_corr" class="chart"></div>
            <div id="chart2_temp" class="chart"></div>
            <div style="display:flex; gap:20px;">
                <div id="chart2_weather" class="chart-half"></div>
                <div id="chart2_season" class="chart-half"></div>
            </div>
            <div class="insight">温度是影响骑行量最重要的正向因子(r={{ task2.corr_matrix[0][6] }})，湿度和恶劣天气显著抑制骑行。</div>
        </div>

        <!-- 任务三 -->
        <div id="task3" class="section" style="display:none">
            <h2>任务三：用户类型时序行为聚类分析</h2>
            <h3 style="margin:16px 0 8px;">临时用户聚类中心 (K=3)</h3>
            <div id="chart3_casual" class="chart"></div>
            <h3 style="margin:16px 0 8px;">注册用户聚类中心 (K=3)</h3>
            <div id="chart3_registered" class="chart"></div>
            <div class="insight">注册用户可清晰聚类为「工作日通勤型」(双峰)和「周末休闲型」(单峰)，临时用户行为分布较均匀。</div>
        </div>

        <!-- 任务四 -->
        <div id="task4" class="section" style="display:none">
            <h2>任务四：共享单车碳减排量化估算</h2>
            <div class="metrics">
                <div class="metric"><div class="value">{{ "{:,}".format(task4.total_trips) }}</div><div class="label">总骑行次数</div></div>
                <div class="metric"><div class="value">{{ task4.co2_car_ton }} 吨</div><div class="label">替代汽车碳减排</div></div>
                <div class="metric"><div class="value">{{ task4.co2_bus_ton }} 吨</div><div class="label">替代公交碳减排</div></div>
                <div class="metric"><div class="value">{{ "{:,}".format(task4.trees_car) }} 棵</div><div class="label">等效植树</div></div>
            </div>
            <div id="chart4_trend" class="chart"></div>
            <div id="chart4_season" class="chart"></div>
            <div class="insight">共享单车具有显著碳减排效益，是实现"双碳"目标、推动绿色低碳生活的有效途径。</div>
        </div>
    </div>
    <div class="footer">
        <p>参考文献: Fanaee-T & Gama (2013) | Capital Bikeshare System Data</p>
    </div>

    <script>
    // 页面切换
    function showSection(id, el) {
        document.querySelectorAll('.section').forEach(function(s) { s.style.display = 'none'; });
        document.getElementById(id).style.display = 'block';
        document.querySelectorAll('.nav a').forEach(function(a) { a.classList.remove('active'); });
        el.classList.add('active');
        // 触发 resize 以重新渲染图表
        setTimeout(function() { window.dispatchEvent(new Event('resize')); }, 100);
        return false;
    }

    // ===== 任务一: 24小时骑行模式 =====
    var chart1 = echarts.init(document.getElementById('chart1'));
    chart1.setOption({
        title: { text: '24小时平均骑行模式对比', left: 'center' },
        tooltip: { trigger: 'axis' },
        legend: { data: ['工作日-总', '非工作日-总', '工作日-注册', '非工作日-注册', '工作日-临时', '非工作日-临时'], bottom: 0 },
        xAxis: { type: 'category', data: {{ task1.hours }}, name: '小时' },
        yAxis: { type: 'value', name: '平均骑行量' },
        series: [
            { name: '工作日-总', type: 'line', data: {{ task1.hourly_cnt_work }}, lineStyle: {width: 3}, itemStyle: {color: '#2196F3'} },
            { name: '非工作日-总', type: 'line', data: {{ task1.hourly_cnt_nonwork }}, lineStyle: {width: 3, type: 'dashed'}, itemStyle: {color: '#FF9800'} },
            { name: '工作日-注册', type: 'line', data: {{ task1.hourly_reg_work }}, lineStyle: {width: 2}, itemStyle: {color: '#4CAF50'}, show: false },
            { name: '非工作日-注册', type: 'line', data: {{ task1.hourly_reg_nonwork }}, lineStyle: {width: 2, type: 'dashed'}, itemStyle: {color: '#8BC34A'} },
            { name: '工作日-临时', type: 'line', data: {{ task1.hourly_cas_work }}, lineStyle: {width: 2}, itemStyle: {color: '#F44336'} },
            { name: '非工作日-临时', type: 'line', data: {{ task1.hourly_cas_nonwork }}, lineStyle: {width: 2, type: 'dashed'}, itemStyle: {color: '#FF5722'} },
        ]
    });

    // ===== 任务二: 相关系数热力图 =====
    var chart2_corr = echarts.init(document.getElementById('chart2_corr'));
    var corrData = {{ task2.corr_matrix }};
    var corrLabels = {{ task2.corr_labels }};
    var heatData = [];
    for (var i = 0; i < corrData.length; i++) {
        for (var j = 0; j < corrData[i].length; j++) {
            heatData.push([j, i, corrData[i][j]]);
        }
    }
    chart2_corr.setOption({
        title: { text: '环境因素与骑行量相关系数矩阵', left: 'center' },
        tooltip: { formatter: function(p) { return corrLabels[p.data[1]] + ' vs ' + corrLabels[p.data[0]] + ': ' + p.data[2]; } },
        xAxis: { type: 'category', data: corrLabels, splitArea: {show: true} },
        yAxis: { type: 'category', data: corrLabels, splitArea: {show: true} },
        visualMap: { min: -1, max: 1, calculable: true, orient: 'horizontal', left: 'center', bottom: 0, inRange: {color: ['#d73027', '#fee08b', '#1a9850']} },
        series: [{ type: 'heatmap', data: heatData, label: {show: true, formatter: function(p) { return p.data[2].toFixed(2); }}, emphasis: {itemStyle: {shadowBlur: 10}} }]
    });

    // 温度散点图
    var chart2_temp = echarts.init(document.getElementById('chart2_temp'));
    chart2_temp.setOption({
        title: { text: '温度 vs 日骑行量', left: 'center' },
        tooltip: { trigger: 'item', formatter: function(p) { return '温度: ' + p.data[0] + '°C<br>骑行量: ' + p.data[1]; } },
        xAxis: { name: '温度 (°C)', type: 'value' },
        yAxis: { name: '骑行量', type: 'value' },
        series: [{ type: 'scatter', data: {{ task2.temp_scatter }}, symbolSize: 6, itemStyle: {color: '#2196F3', opacity: 0.5} }]
    });

    // 天气柱状图
    var chart2_weather = echarts.init(document.getElementById('chart2_weather'));
    chart2_weather.setOption({
        title: { text: '天气状况 vs 平均骑行量', left: 'center' },
        xAxis: { type: 'category', data: {{ task2.weather_labels }} },
        yAxis: { type: 'value', name: '平均骑行量' },
        series: [{ type: 'bar', data: {{ task2.weather_means }}, itemStyle: {color: '#4CAF50'}, barWidth: '50%', label: {show: true, position: 'top'} }]
    });

    // 季节柱状图
    var chart2_season = echarts.init(document.getElementById('chart2_season'));
    chart2_season.setOption({
        title: { text: '季节 vs 平均骑行量', left: 'center' },
        xAxis: { type: 'category', data: {{ task2.season_labels }} },
        yAxis: { type: 'value', name: '平均骑行量' },
        series: [{ type: 'bar', data: {{ task2.season_means }}, itemStyle: {color: '#FF9800'}, barWidth: '50%', label: {show: true, position: 'top'} }]
    });

    // ===== 任务三: 聚类中心 =====
    var chart3_c = echarts.init(document.getElementById('chart3_casual'));
    var c_centers = {{ task3.casual_centers }};
    var c_sizes = {{ task3.casual_sizes }};
    chart3_c.setOption({
        title: { text: '临时用户聚类中心 (24小时骑行占比%)', left: 'center' },
        tooltip: { trigger: 'axis' },
        legend: { bottom: 0 },
        xAxis: { type: 'category', data: {{ task3.hours }}, name: '小时' },
        yAxis: { type: 'value', name: '骑行占比 (%)' },
        series: [
            { name: '簇0 (n=' + c_sizes[0] + ')', type: 'bar', data: c_centers[0].map(function(v) { return (v*100).toFixed(2); }) },
            { name: '簇1 (n=' + c_sizes[1] + ')', type: 'bar', data: c_centers[1].map(function(v) { return (v*100).toFixed(2); }) },
            { name: '簇2 (n=' + c_sizes[2] + ')', type: 'bar', data: c_centers[2].map(function(v) { return (v*100).toFixed(2); }) },
        ]
    });

    var chart3_r = echarts.init(document.getElementById('chart3_registered'));
    var r_centers = {{ task3.registered_centers }};
    var r_sizes = {{ task3.registered_sizes }};
    chart3_r.setOption({
        title: { text: '注册用户聚类中心 (24小时骑行占比%)', left: 'center' },
        tooltip: { trigger: 'axis' },
        legend: { bottom: 0 },
        xAxis: { type: 'category', data: {{ task3.hours }}, name: '小时' },
        yAxis: { type: 'value', name: '骑行占比 (%)' },
        series: [
            { name: '簇0 (n=' + r_sizes[0] + ')', type: 'bar', data: r_centers[0].map(function(v) { return (v*100).toFixed(2); }) },
            { name: '簇1 (n=' + r_sizes[1] + ')', type: 'bar', data: r_centers[1].map(function(v) { return (v*100).toFixed(2); }) },
            { name: '簇2 (n=' + r_sizes[2] + ')', type: 'bar', data: r_centers[2].map(function(v) { return (v*100).toFixed(2); }) },
        ]
    });

    // ===== 任务四: 碳减排 =====
    var chart4_trend = echarts.init(document.getElementById('chart4_trend'));
    chart4_trend.setOption({
        title: { text: '月度骑行量与碳减排趋势', left: 'center' },
        tooltip: { trigger: 'axis' },
        legend: { data: ['月骑行量', '碳减排(kg CO₂)'], bottom: 0 },
        xAxis: { type: 'category', data: {{ task4.months }}, axisLabel: {rotate: 45} },
        yAxis: [
            { type: 'value', name: '骑行量' },
            { type: 'value', name: '碳减排 (kg CO₂)', position: 'right' }
        ],
        series: [
            { name: '月骑行量', type: 'bar', data: {{ task4.monthly_trips }}, itemStyle: {color: '#4CAF50'}, yAxisIndex: 0 },
            { name: '碳减排(kg CO₂)', type: 'line', data: {{ task4.monthly_co2 }}, itemStyle: {color: '#F44336'}, yAxisIndex: 1, lineStyle: {width: 2} }
        ]
    });

    var chart4_season = echarts.init(document.getElementById('chart4_season'));
    chart4_season.setOption({
        title: { text: '各季节碳减排贡献', left: 'center' },
        tooltip: { trigger: 'item', formatter: '{b}: {c} kg ({d}%)' },
        series: [{
            type: 'pie', radius: ['35%', '65%'],
            data: [
                {value: {{ task4.season_co2[0] }}, name: '春', itemStyle: {color: '#66BB6A'}},
                {value: {{ task4.season_co2[1] }}, name: '夏', itemStyle: {color: '#FFA726'}},
                {value: {{ task4.season_co2[2] }}, name: '秋', itemStyle: {color: '#EF5350'}},
                {value: {{ task4.season_co2[3] }}, name: '冬', itemStyle: {color: '#42A5F5'}},
            ],
            label: { formatter: '{b}: {d}%' }
        }]
    });

    // 响应式
    window.addEventListener('resize', function() {
        [chart1, chart2_corr, chart2_temp, chart2_weather, chart2_season, chart3_c, chart3_r, chart4_trend, chart4_season].forEach(function(c) { c.resize(); });
    });
    </script>
</body>
</html>
"""


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE,
                                  task1=task1, task2=task2, task3=task3, task4=task4)


if __name__ == '__main__':
    print("=" * 50)
    print("  共享单车数据挖掘分析系统")
    print("  访问地址: http://localhost:5000")
    print("=" * 50)
    app.run(debug=False, host='0.0.0.0', port=5000)
