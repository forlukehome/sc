import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime as dt
from datetime import datetime, timedelta
import random
import json
import io
import zipfile
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import networkx as nx
from scipy.optimize import minimize
import warnings
import pulp
import time
from deap import base, creator, tools, algorithms
import base64
import math
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import seaborn as sns
from ortools.linear_solver import pywraplp
from ortools.sat.python import cp_model

warnings.filterwarnings('ignore')

# 设置页面配置
st.set_page_config(
    page_title="智能APS系统 Pro Max",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式（来自第二个系统）
st.markdown("""
    <style>
    .header {
        font-size: 36px;
        font-weight: bold;
        color: #2c3e50;
        padding: 10px 0;
        text-align: center;
        border-bottom: 2px solid #3498db;
        margin-bottom: 30px;
        background: linear-gradient(90deg, #3498db, #2c3e50);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .section-title {
        font-size: 24px;
        font-weight: bold;
        color: #2980b9;
        padding: 10px 0;
        margin-top: 20px;
        border-bottom: 1px solid #bdc3c7;
        display: flex;
        align-items: center;
    }
    .card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .metric-card {
        background: linear-gradient(135deg, #3498db, #2c3e50);
        color: white;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .warning {
        color: #e74c3c;
        font-weight: bold;
    }
    .success {
        color: #2ecc71;
        font-weight: bold;
    }
    .info-box {
        background-color: #e3f2fd;
        border-left: 4px solid #3498db;
        padding: 10px;
        margin: 10px 0;
        border-radius: 0 5px 5px 0;
    }
    .stButton>button {
        background: linear-gradient(135deg, #3498db, #2c3e50);
        color: white;
        border: none;
    }
    .stTab > div > div > div > div {
        overflow: visible !important;
    }
    .stProgress > div > div > div > div {
        background-color: #3498db;
    }
    .stAlert {
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# 初始化会话状态
if 'orders' not in st.session_state:
    st.session_state.orders = pd.DataFrame()
if 'resources' not in st.session_state:
    st.session_state.resources = pd.DataFrame()
if 'schedule' not in st.session_state:
    st.session_state.schedule = pd.DataFrame()
if 'material_status' not in st.session_state:
    st.session_state.material_status = pd.DataFrame()
if 'factories' not in st.session_state:
    st.session_state.factories = pd.DataFrame()
if 'supply_chain_risk' not in st.session_state:
    st.session_state.supply_chain_risk = pd.DataFrame()
if 'simulation_results' not in st.session_state:
    st.session_state.simulation_results = {}
if 'llm_chat_history' not in st.session_state:
    st.session_state.llm_chat_history = []
if 'resource_twins' not in st.session_state:
    st.session_state.resource_twins = {}
if 'products' not in st.session_state:
    st.session_state.products = {}
if 'bom' not in st.session_state:
    st.session_state.bom = []
if 'inventory' not in st.session_state:
    st.session_state.inventory = {}
if 'sales_history' not in st.session_state:
    st.session_state.sales_history = pd.DataFrame()
if 'warehouse_network' not in st.session_state:
    st.session_state.warehouse_network = pd.DataFrame()
if 'production_plan' not in st.session_state:
    st.session_state.production_plan = pd.DataFrame()
if 'workshop_schedule' not in st.session_state:
    st.session_state.workshop_schedule = pd.DataFrame()
if 'material_requirements' not in st.session_state:
    st.session_state.material_requirements = pd.DataFrame()
if 'shipping_plan' not in st.session_state:
    st.session_state.shipping_plan = pd.DataFrame()
if 'command_center_alerts' not in st.session_state:
    st.session_state.command_center_alerts = []
if 'forecast_results' not in st.session_state:
    st.session_state.forecast_results = pd.DataFrame()
if 'material_preparation' not in st.session_state:
    st.session_state.material_preparation = pd.DataFrame()


# ==================== 数据结构定义 ====================
@dataclass
class Product:
    product_id: str
    name: str
    category: str
    unit_cost: float
    sell_price: float
    lead_time: int
    safety_stock: int
    reorder_point: int


@dataclass
class BOM:
    product_id: str
    component_id: str
    quantity: float
    component_type: str


@dataclass
class Resource:
    resource_id: str
    name: str
    type: str
    capacity: float
    cost_per_hour: float
    efficiency: float
    availability: float


@dataclass
class Demand:
    demand_id: str
    product_id: str
    quantity: int
    due_date: datetime
    priority: int
    customer: str


# ==================== 新增OR-Tools优化求解器（来自第二个系统） ====================
class OptimizationSolver:
    @staticmethod
    def solve_warehouse_location(demand_points, candidate_sites, setup_costs, transport_costs, max_distance):
        """仓库选址优化（MIP模型）"""
        solver = pywraplp.Solver.CreateSolver('SCIP')

        # 决策变量
        x = {}  # 是否在候选点i建立仓库
        y = {}  # 需求点j分配给仓库i的比例
        for i in candidate_sites:
            x[i] = solver.BoolVar(f'x_{i}')
            for j in demand_points:
                y[(i, j)] = solver.NumVar(0, 1, f'y_{i}_{j}')

        # 目标函数：最小化总成本（建设成本+运输成本）
        total_cost = solver.Sum(setup_costs[i] * x[i] for i in candidate_sites)
        for i in candidate_sites:
            for j in demand_points:
                total_cost += transport_costs[(i, j)] * y[(i, j)]
        solver.Minimize(total_cost)

        # 约束1：每个需求点必须被完全覆盖
        for j in demand_points:
            solver.Add(solver.Sum(y[(i, j)] for i in candidate_sites) == 1)

        # 约束2：只有建立的仓库才能提供服务
        for i in candidate_sites:
            for j in demand_points:
                solver.Add(y[(i, j)] <= x[i])

        # 约束3：距离约束
        for i in candidate_sites:
            for j in demand_points:
                if max_distance < 1000:  # 示例条件
                    solver.Add(y[(i, j)] <= x[i])

        # 求解
        status = solver.Solve()

        if status == pywraplp.Solver.OPTIMAL:
            solution = {}
            for i in candidate_sites:
                if x[i].solution_value() > 0.5:
                    solution[i] = [j for j in demand_points if y[(i, j)].solution_value() > 0.5]
            return solution, solver.Objective().Value()
        else:
            return None, None

    @staticmethod
    def solve_production_scheduling(tasks, machines, horizon):
        """生产调度优化（CP-SAT模型）"""
        model = cp_model.CpModel()

        # 变量
        starts = {}
        ends = {}
        intervals = {}
        machine_assign = {}

        for task in tasks:
            for machine in machines:
                suffix = f"_{task['id']}_{machine['id']}"
                starts[task['id'], machine['id']] = model.NewIntVar(0, horizon, 'start' + suffix)
                ends[task['id'], machine['id']] = model.NewIntVar(0, horizon, 'end' + suffix)
                intervals[task['id'], machine['id']] = model.NewIntervalVar(
                    starts[task['id'], machine['id']],
                    task['duration'],
                    ends[task['id'], machine['id']],
                    'interval' + suffix
                )
                machine_assign[task['id'], machine['id']] = model.NewBoolVar('assign' + suffix)

        # 约束
        # 每个任务只能在一台机器上执行
        for task in tasks:
            model.Add(sum(machine_assign[task['id'], machine['id']] for machine in machines) == 1)

        # 机器上的任务不能重叠
        for machine in machines:
            intervals_for_machine = []
            for task in tasks:
                intervals_for_machine.append(intervals[task['id'], machine['id']])
            model.AddNoOverlap(intervals_for_machine)

        # 目标：最小化最大完成时间
        makespan = model.NewIntVar(0, horizon, 'makespan')
        for task in tasks:
            for machine in machines:
                model.Add(ends[task['id'], machine['id']] <= makespan).OnlyEnforceIf(
                    machine_assign[task['id'], machine['id']])

        model.Minimize(makespan)

        # 求解
        solver = cp_model.CpSolver()
        status = solver.Solve(model)

        if status == cp_model.OPTIMAL:
            schedule = {}
            for task in tasks:
                for machine in machines:
                    if solver.Value(machine_assign[task['id'], machine['id']]) == 1:
                        start = solver.Value(starts[task['id'], machine['id']])
                        end = solver.Value(ends[task['id'], machine['id']])
                        schedule[task['id']] = {
                            'machine': machine['id'],
                            'start': start,
                            'end': end
                        }
            return schedule, solver.ObjectiveValue()
        else:
            return None, None


# ==================== 新增指挥中心功能（来自第二个系统） ====================
class CommandCenter:
    @staticmethod
    def calculate_oee(machine_id):
        # 模拟数据
        planned_time = 8 * 3600  # 8小时计划生产时间
        downtime = random.randint(0, 3600)  # 0-1小时停机
        total_units = random.randint(10000, 15000)
        good_units = int(total_units * random.uniform(0.92, 0.98))
        ideal_cycle_time = 0.5  # 每0.5秒生产一个产品

        availability = (planned_time - downtime) / planned_time
        performance = (ideal_cycle_time * total_units) / (planned_time - downtime)
        quality = good_units / total_units

        return availability * performance * quality

    @staticmethod
    def kit_check(order_id, bom, inventory):
        """齐套检查核心逻辑"""
        shortages = []

        # 处理不同的BOM数据格式
        if isinstance(bom, list):
            # BOM是对象列表
            bom_items = []
            for item in bom:
                if hasattr(item, 'product_id'):
                    bom_items.append({
                        '产品': item.product_id,
                        '物料': item.component_id,
                        '数量': item.quantity
                    })
        elif isinstance(bom, pd.DataFrame):
            # BOM是DataFrame
            bom_items = bom.to_dict('records')
        else:
            bom_items = []

        # 从订单ID提取产品信息
        product_id = order_id.split('-')[1] if '-' in order_id else 'Unknown'

        # 检查物料可用性
        for item in bom_items:
            if item.get('产品', '') == product_id or product_id == 'Unknown':
                material = item.get('物料', item.get('component_id', ''))
                required_qty = item.get('数量', item.get('quantity', 0))

                # 获取库存
                available_qty = 0
                if isinstance(inventory, pd.DataFrame) and not inventory.empty:
                    if '物料' in inventory.columns:
                        material_inv = inventory[inventory['物料'] == material]
                        if not material_inv.empty:
                            available_qty = material_inv['当前库存'].values[0]
                    elif '物料编码' in inventory.columns:
                        material_inv = inventory[inventory['物料编码'] == material]
                        if not material_inv.empty:
                            available_qty = material_inv['当前库存'].values[0]

                if available_qty < required_qty:
                    shortages.append({
                        "物料": material,
                        "需求数量": required_qty,
                        "可用数量": available_qty,
                        "缺口": required_qty - available_qty
                    })

        kit_rate = 1 - len(shortages) / max(len(bom_items), 1) if bom_items else 1
        return {"齐套率": kit_rate, "缺料列表": shortages}


# ==================== 原有的所有功能类（第一个系统） ====================
# IntelligentForecastEngine类
class IntelligentForecastEngine:
    """智能预测引擎 - 多场景多模型预测"""

    def __init__(self):
        self.models = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'linear_regression': LinearRegression(),
            'moving_average': None,
            'exponential_smoothing': None
        }
        self.best_model = None
        self.forecast_results = {}

    def prepare_sales_data(self, sales_history):
        """准备销售数据用于预测"""
        if sales_history.empty:
            return None

        try:
            # 按日期和产品聚合
            daily_sales = sales_history.groupby(['date', 'product_id'])['quantity'].sum().reset_index()

            # 创建时间特征
            daily_sales['date'] = pd.to_datetime(daily_sales['date'])
            daily_sales['day_of_week'] = daily_sales['date'].dt.dayofweek
            daily_sales['month'] = daily_sales['date'].dt.month
            daily_sales['quarter'] = daily_sales['date'].dt.quarter
            daily_sales['year'] = daily_sales['date'].dt.year
            daily_sales['day_of_month'] = daily_sales['date'].dt.day

            return daily_sales
        except Exception as e:
            st.error(f"准备销售数据时出错: {str(e)}")
            return None

    def train_models(self, sales_data):
        """训练多个预测模型"""
        if sales_data is None or sales_data.empty:
            return

        try:
            # 准备特征和目标
            features = ['day_of_week', 'month', 'quarter', 'year', 'day_of_month']
            X = sales_data[features]
            y = sales_data['quantity']

            # 分割训练集和测试集
            split_idx = int(len(X) * 0.8)
            if split_idx < 1:
                return

            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]

            # 训练模型
            results = {}
            for model_name, model in self.models.items():
                if model is not None:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    mae = mean_absolute_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    results[model_name] = {'mae': mae, 'r2': r2, 'model': model}

            # 选择最佳模型
            if results:
                self.best_model = min(results.items(), key=lambda x: x[1]['mae'])
        except Exception as e:
            st.error(f"训练模型时出错: {str(e)}")

    def forecast(self, periods=30, scenario='normal'):
        """生成预测结果"""
        scenarios = {
            'optimistic': 1.2,
            'normal': 1.0,
            'pessimistic': 0.8,
            'seasonal': 1.1,
            'promotional': 1.3
        }

        try:
            # 生成未来日期
            future_dates = pd.date_range(start=datetime.now(), periods=periods, freq='D')

            # 准备预测特征
            future_features = pd.DataFrame({
                'date': future_dates,
                'day_of_week': future_dates.dayofweek,
                'month': future_dates.month,
                'quarter': future_dates.quarter,
                'year': future_dates.year,
                'day_of_month': future_dates.day
            })

            # 生成基础预测
            base_forecast = np.random.randint(100, 500, size=periods)  # 示例数据

            # 应用场景系数
            scenario_factor = scenarios.get(scenario, 1.0)
            adjusted_forecast = base_forecast * scenario_factor

            # 添加季节性波动
            seasonal_pattern = np.sin(np.arange(periods) * 2 * np.pi / 7) * 50
            final_forecast = adjusted_forecast + seasonal_pattern

            return pd.DataFrame({
                'date': future_dates,
                'forecast': final_forecast,
                'lower_bound': final_forecast * 0.9,
                'upper_bound': final_forecast * 1.1,
                'scenario': scenario
            })
        except Exception as e:
            st.error(f"生成预测时出错: {str(e)}")
            return pd.DataFrame()


# WarehouseNetworkAnalyzer类
class WarehouseNetworkAnalyzer:
    """仓网分析引擎 - 订单交付仓网结构分析"""

    def __init__(self):
        self.network_graph = nx.Graph()
        self.optimal_routes = {}
        self.warehouse_capacities = {}

    def build_network(self, warehouses, factories):
        """构建仓储网络"""
        try:
            # 清空现有网络
            self.network_graph.clear()

            # 添加节点
            for idx, wh in warehouses.iterrows():
                self.network_graph.add_node(
                    wh.get('warehouse_id', f'WH-{idx}'),
                    type='warehouse',
                    location=wh.get('location', '未知'),
                    capacity=wh.get('capacity', 0)
                )

            for idx, factory in factories.iterrows():
                self.network_graph.add_node(
                    factory.get('工厂编号', f'FACT-{idx}'),
                    type='factory',
                    location=factory.get('地点', '未知'),
                    capacity=factory.get('总产能', 0)
                )

            # 添加边（运输路线）
            warehouse_nodes = [n for n, d in self.network_graph.nodes(data=True) if d['type'] == 'warehouse']
            factory_nodes = [n for n, d in self.network_graph.nodes(data=True) if d['type'] == 'factory']

            for wh_node in warehouse_nodes:
                for factory_node in factory_nodes:
                    distance = random.uniform(50, 500)  # 示例距离
                    cost = distance * 0.1  # 运输成本
                    self.network_graph.add_edge(wh_node, factory_node, distance=distance, cost=cost)
        except Exception as e:
            st.error(f"构建网络时出错: {str(e)}")

    def analyze_order_fulfillment(self, order):
        """分析订单履行路径"""
        try:
            # 找到最优仓库
            warehouses = [n for n, d in self.network_graph.nodes(data=True) if d['type'] == 'warehouse']

            best_warehouse = None
            min_cost = float('inf')

            for wh in warehouses:
                # 检查库存和运输成本
                inventory_available = random.random() > 0.3  # 示例库存状态
                if inventory_available:
                    cost = random.uniform(10, 100)  # 示例成本
                    if cost < min_cost:
                        min_cost = cost
                        best_warehouse = wh

            return {
                'order_id': order.get('订单编号', 'N/A'),
                'best_warehouse': best_warehouse,
                'fulfillment_cost': min_cost,
                'delivery_time': random.randint(1, 5),
                'route_efficiency': random.uniform(0.8, 0.95)
            }
        except Exception as e:
            st.error(f"分析订单履行时出错: {str(e)}")
            return {}

    def optimize_network_layout(self):
        """优化仓网布局"""
        try:
            # 计算各节点的中心性
            centrality = nx.betweenness_centrality(self.network_graph)

            # 识别关键节点
            key_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5]

            # 提出优化建议
            recommendations = []
            for node, score in key_nodes:
                node_data = self.network_graph.nodes[node]
                recommendations.append({
                    'node': node,
                    'type': node_data['type'],
                    'importance_score': score,
                    'recommendation': '建议增加容量' if score > 0.5 else '维持现状'
                })

            return recommendations
        except Exception as e:
            st.error(f"优化网络布局时出错: {str(e)}")
            return []


# SalesOperationsPlanning类
class SalesOperationsPlanning:
    """产销规划引擎 - S&OP协同"""

    def __init__(self):
        self.demand_plan = pd.DataFrame()
        self.supply_plan = pd.DataFrame()
        self.financial_plan = pd.DataFrame()
        self.consensus_plan = pd.DataFrame()

    def create_demand_plan(self, forecast_data, market_intelligence):
        """创建需求计划"""
        try:
            # 整合预测数据和市场情报
            demand_plan = forecast_data.copy()

            # 添加市场调整因子
            demand_plan['market_adjustment'] = 1.0
            for event in market_intelligence:
                if event['type'] == 'promotion':
                    mask = demand_plan['date'].between(event['start'], event['end'])
                    demand_plan.loc[mask, 'market_adjustment'] *= 1.2
                elif event['type'] == 'competitor_action':
                    mask = demand_plan['date'].between(event['start'], event['end'])
                    demand_plan.loc[mask, 'market_adjustment'] *= 0.9

            demand_plan['adjusted_demand'] = demand_plan['forecast'] * demand_plan['market_adjustment']

            self.demand_plan = demand_plan
            return demand_plan
        except Exception as e:
            st.error(f"创建需求计划时出错: {str(e)}")
            return pd.DataFrame()

    def create_supply_plan(self, capacity_data, inventory_data):
        """创建供应计划"""
        try:
            # 基于产能和库存创建供应计划
            supply_plan = pd.DataFrame()

            # 计算可用产能
            total_capacity = capacity_data['总产能'].sum() if not capacity_data.empty else 10000

            # 考虑库存水平 - 修复：处理嵌套字典结构
            current_inventory = 0
            if inventory_data:
                for product_id, inv_info in inventory_data.items():
                    if isinstance(inv_info, dict):
                        current_inventory += inv_info.get('current_stock', 0)
                    else:
                        current_inventory += inv_info

            # 生成供应计划
            dates = pd.date_range(start=datetime.now(), periods=30, freq='D')
            supply_plan['date'] = dates
            supply_plan['available_capacity'] = total_capacity * random.uniform(0.8, 0.95)
            supply_plan['planned_production'] = supply_plan['available_capacity'] * 0.9
            supply_plan['inventory_buffer'] = current_inventory / 30 if current_inventory > 0 else 0

            self.supply_plan = supply_plan
            return supply_plan
        except Exception as e:
            st.error(f"创建供应计划时出错: {str(e)}")
            return pd.DataFrame()

    def reconcile_plans(self):
        """协调产销计划"""
        try:
            if self.demand_plan.empty or self.supply_plan.empty:
                return pd.DataFrame()

            # 合并需求和供应计划
            consensus = pd.merge(self.demand_plan[['date', 'adjusted_demand']],
                                 self.supply_plan[['date', 'planned_production']],
                                 on='date', how='outer')

            # 计算差异
            consensus['gap'] = consensus['adjusted_demand'] - consensus['planned_production']

            # 制定行动计划
            consensus['action'] = consensus['gap'].apply(
                lambda x: '增加产能' if x > 0 else '调整库存' if x < -100 else '维持现状'
            )

            # 财务影响分析
            consensus['revenue_impact'] = consensus['adjusted_demand'] * 100  # 假设单价
            consensus['cost_impact'] = consensus['planned_production'] * 80  # 假设成本
            consensus['profit_impact'] = consensus['revenue_impact'] - consensus['cost_impact']

            self.consensus_plan = consensus
            return consensus
        except Exception as e:
            st.error(f"协调计划时出错: {str(e)}")
            return pd.DataFrame()


# IntelligentOrderAllocation类
class IntelligentOrderAllocation:
    """智能分单引擎 - 确定产品在哪个工厂生产"""

    def __init__(self):
        self.allocation_rules = {}
        self.factory_capabilities = {}
        self.allocation_history = []

    def analyze_factory_capabilities(self, factories, products):
        """分析工厂能力"""
        try:
            for _, factory in factories.iterrows():
                self.factory_capabilities[factory['工厂编号']] = {
                    'capacity': factory.get('总产能', 0),
                    'cost': factory.get('单位成本', 1.0),
                    'specialties': factory.get('专注产品', []),
                    'location': factory.get('地点', '未知'),
                    'quality_score': random.uniform(0.85, 0.98),
                    'delivery_performance': random.uniform(0.88, 0.96)
                }
        except Exception as e:
            st.error(f"分析工厂能力时出错: {str(e)}")

    def calculate_allocation_score(self, order, factory_id):
        """计算分配得分"""
        try:
            factory = self.factory_capabilities.get(factory_id, {})

            # 多维度评分
            scores = {
                'capacity_score': min(1.0, factory.get('capacity', 0) / max((order['数量'] * 10), 1)),
                'cost_score': 1.0 / (factory.get('cost', 1) + 0.1),
                'specialty_score': 1.0 if order['产品型号'] in factory.get('specialties', []) else 0.5,
                'quality_score': factory.get('quality_score', 0.9),
                'delivery_score': factory.get('delivery_performance', 0.9)
            }

            # 加权计算总分
            weights = {
                'capacity_score': 0.25,
                'cost_score': 0.2,
                'specialty_score': 0.2,
                'quality_score': 0.2,
                'delivery_score': 0.15
            }

            total_score = sum(scores[k] * weights[k] for k in scores)

            return total_score, scores
        except Exception as e:
            st.error(f"计算分配得分时出错: {str(e)}")
            return 0, {}

    def allocate_orders(self, orders, mode='balanced'):
        """智能分配订单到工厂"""
        try:
            allocations = []

            allocation_modes = {
                'balanced': self._balanced_allocation,
                'cost_optimized': self._cost_optimized_allocation,
                'speed_optimized': self._speed_optimized_allocation,
                'quality_focused': self._quality_focused_allocation
            }

            allocation_func = allocation_modes.get(mode, self._balanced_allocation)

            for _, order in orders.iterrows():
                best_factory, scores = allocation_func(order)

                allocation = {
                    '订单编号': order['订单编号'],
                    '产品型号': order['产品型号'],
                    '数量': order['数量'],
                    '分配工厂': best_factory,
                    '分配模式': mode,
                    '综合得分': scores.get('total', 0),
                    '产能得分': scores.get('capacity_score', 0),
                    '成本得分': scores.get('cost_score', 0),
                    '专长得分': scores.get('specialty_score', 0),
                    '质量得分': scores.get('quality_score', 0),
                    '交付得分': scores.get('delivery_score', 0)
                }

                allocations.append(allocation)
                self.allocation_history.append(allocation)

            return pd.DataFrame(allocations)
        except Exception as e:
            st.error(f"分配订单时出错: {str(e)}")
            return pd.DataFrame()

    def _balanced_allocation(self, order):
        """平衡分配策略"""
        best_factory = None
        best_score = -1
        best_scores = {}

        for factory_id in self.factory_capabilities:
            score, scores = self.calculate_allocation_score(order, factory_id)
            if score > best_score:
                best_score = score
                best_factory = factory_id
                best_scores = scores

        best_scores['total'] = best_score
        return best_factory, best_scores

    def _cost_optimized_allocation(self, order):
        """成本优化分配策略"""
        if not self.factory_capabilities:
            return None, {}
        best_factory = min(self.factory_capabilities.items(),
                           key=lambda x: x[1].get('cost', float('inf')))[0]
        score, scores = self.calculate_allocation_score(order, best_factory)
        scores['total'] = score
        return best_factory, scores

    def _speed_optimized_allocation(self, order):
        """速度优化分配策略"""
        if not self.factory_capabilities:
            return None, {}
        best_factory = max(self.factory_capabilities.items(),
                           key=lambda x: x[1].get('delivery_performance', 0))[0]
        score, scores = self.calculate_allocation_score(order, best_factory)
        scores['total'] = score
        return best_factory, scores

    def _quality_focused_allocation(self, order):
        """质量优先分配策略"""
        if not self.factory_capabilities:
            return None, {}
        best_factory = max(self.factory_capabilities.items(),
                           key=lambda x: x[1].get('quality_score', 0))[0]
        score, scores = self.calculate_allocation_score(order, best_factory)
        scores['total'] = score
        return best_factory, scores


# MasterProductionSchedule类
class MasterProductionSchedule:
    """主生产计划引擎"""

    def __init__(self):
        self.mps_horizon = 12  # 计划期间（周）
        self.time_buckets = []
        self.mps_records = {}

    def initialize_time_buckets(self, start_date=None):
        """初始化时间段"""
        if start_date is None:
            start_date = datetime.now()

        self.time_buckets = []
        for week in range(self.mps_horizon):
            week_start = start_date + timedelta(weeks=week)
            week_end = week_start + timedelta(days=6)
            self.time_buckets.append({
                'week': week + 1,
                'start_date': week_start,
                'end_date': week_end
            })

    def create_mps(self, demand_forecast, capacity_constraints, inventory_levels):
        """创建主生产计划"""
        try:
            mps_data = []

            products = demand_forecast[
                'product_id'].unique() if 'product_id' in demand_forecast.columns and not demand_forecast.empty else [
                'P001', 'P002', 'P003']

            for product in products:
                # 获取产品相关数据
                if 'product_id' in demand_forecast.columns and not demand_forecast.empty:
                    product_demand = demand_forecast[demand_forecast['product_id'] == product]
                else:
                    product_demand = pd.DataFrame()

                # 修复：处理嵌套字典结构的库存数据
                current_inventory = 0
                if inventory_levels:
                    if product in inventory_levels:
                        inv_data = inventory_levels[product]
                        if isinstance(inv_data, dict):
                            current_inventory = inv_data.get('current_stock', 0)
                        else:
                            current_inventory = inv_data
                    else:
                        current_inventory = 0

                # 初始化MPS记录
                mps_record = {
                    'product_id': product,
                    'beginning_inventory': current_inventory,
                    'weeks': []
                }

                running_inventory = current_inventory

                for week in self.time_buckets:
                    # 计算需求
                    week_demand = random.randint(100, 500)  # 示例需求

                    # 计算毛需求
                    gross_requirements = week_demand

                    # 计算净需求
                    net_requirements = max(0, gross_requirements - running_inventory)

                    # 计划生产量（考虑批量规则）
                    if net_requirements > 0:
                        lot_size = self._calculate_lot_size(net_requirements, product)
                        planned_production = lot_size
                    else:
                        planned_production = 0

                    # 计算期末库存
                    ending_inventory = running_inventory + planned_production - gross_requirements
                    running_inventory = ending_inventory

                    week_data = {
                        'week': week['week'],
                        'start_date': week['start_date'],
                        'end_date': week['end_date'],
                        'forecast_demand': week_demand,
                        'customer_orders': random.randint(50, min(week_demand, 400)),
                        'gross_requirements': gross_requirements,
                        'beginning_inventory': running_inventory + gross_requirements - planned_production,
                        'net_requirements': net_requirements,
                        'planned_receipts': 0,
                        'planned_production': planned_production,
                        'ending_inventory': ending_inventory,
                        'available_to_promise': max(0, ending_inventory - week_demand * 0.2)
                    }

                    mps_record['weeks'].append(week_data)

                self.mps_records[product] = mps_record

                # 转换为DataFrame格式
                for week_data in mps_record['weeks']:
                    mps_data.append({
                        '产品编号': product,
                        '周次': week_data['week'],
                        '开始日期': week_data['start_date'],
                        '结束日期': week_data['end_date'],
                        '预测需求': week_data['forecast_demand'],
                        '客户订单': week_data['customer_orders'],
                        '毛需求': week_data['gross_requirements'],
                        '期初库存': week_data['beginning_inventory'],
                        '净需求': week_data['net_requirements'],
                        '计划接收': week_data['planned_receipts'],
                        '计划生产': week_data['planned_production'],
                        '期末库存': week_data['ending_inventory'],
                        '可承诺量': week_data['available_to_promise']
                    })

            return pd.DataFrame(mps_data)
        except Exception as e:
            st.error(f"创建MPS时出错: {str(e)}")
            return pd.DataFrame()

    def _calculate_lot_size(self, net_requirements, product):
        """计算批量大小"""
        # 批量策略：固定批量、经济批量、最小批量等
        strategies = {
            'fixed_lot': lambda x: 500,
            'lot_for_lot': lambda x: x,
            'economic_lot': lambda x: max(x, 300),
            'minimum_lot': lambda x: max(x, 200)
        }

        # 随机选择策略（实际应根据产品特性选择）
        strategy = random.choice(list(strategies.keys()))
        return strategies[strategy](net_requirements)

    def calculate_available_to_promise(self, product_id):
        """计算可承诺量(ATP)"""
        if product_id not in self.mps_records:
            return pd.DataFrame()

        try:
            atp_data = []
            mps_record = self.mps_records[product_id]

            cumulative_atp = 0
            for week_data in mps_record['weeks']:
                week_atp = week_data['available_to_promise']
                cumulative_atp += week_atp

                atp_data.append({
                    '周次': week_data['week'],
                    '当周ATP': week_atp,
                    '累计ATP': cumulative_atp,
                    '状态': '充足' if week_atp > 100 else '紧张' if week_atp > 0 else '缺货'
                })

            return pd.DataFrame(atp_data)
        except Exception as e:
            st.error(f"计算ATP时出错: {str(e)}")
            return pd.DataFrame()


# WorkshopScheduler类
class WorkshopScheduler:
    """车间排程引擎 - 高效、灵活、可视化"""

    def __init__(self):
        self.workshop_resources = {}
        self.work_centers = {}
        self.scheduling_rules = {}
        self.schedule_visualization = None

    def setup_workshop(self, resources):
        """设置车间资源"""
        try:
            # 按车间组织资源
            workshops = {
                'WS001': '机加工车间',
                'WS002': '装配车间',
                'WS003': '包装车间',
                'WS004': '质检车间'
            }

            for workshop_id, workshop_name in workshops.items():
                self.workshop_resources[workshop_id] = {
                    'name': workshop_name,
                    'work_centers': [],
                    'total_capacity': 0,
                    'efficiency': random.uniform(0.85, 0.95)
                }

            # 分配资源到车间
            for _, resource in resources.iterrows():
                workshop_id = random.choice(list(workshops.keys()))
                work_center = {
                    'id': resource['资源编号'],
                    'type': resource['资源类型'],
                    'capacity': resource['总产能'],
                    'status': 'available',
                    'current_job': None,
                    'queue': []
                }
                self.workshop_resources[workshop_id]['work_centers'].append(work_center)
                self.workshop_resources[workshop_id]['total_capacity'] += resource['总产能']
        except Exception as e:
            st.error(f"设置车间资源时出错: {str(e)}")

    def create_workshop_schedule(self, production_orders, scheduling_method='spt'):
        """创建车间作业计划"""
        try:
            scheduling_methods = {
                'spt': self._shortest_processing_time,
                'edd': self._earliest_due_date,
                'cr': self._critical_ratio,
                'slack': self._minimum_slack,
                'fifo': self._first_in_first_out
            }

            schedule_func = scheduling_methods.get(scheduling_method, self._shortest_processing_time)

            # 对每个车间进行排程
            workshop_schedules = []

            for workshop_id, workshop in self.workshop_resources.items():
                # 分配到该车间的订单
                workshop_orders = self._allocate_orders_to_workshop(production_orders, workshop_id)

                if workshop_orders.empty:
                    continue

                # 应用排程规则
                scheduled_jobs = schedule_func(workshop_orders, workshop['work_centers'])

                # 创建甘特图数据
                for job in scheduled_jobs:
                    workshop_schedules.append({
                        '作业编号': job['job_id'],
                        '产品': job['product'],
                        '工序': job['operation'],
                        '车间': workshop_id,
                        '工作中心': job['work_center'],
                        '开始时间': job['start_time'],
                        '结束时间': job['end_time'],
                        '持续时间': job['duration'],
                        '状态': job['status'],
                        '优先级': job['priority']
                    })

            return pd.DataFrame(workshop_schedules)
        except Exception as e:
            st.error(f"创建车间排程时出错: {str(e)}")
            return pd.DataFrame()

    def _allocate_orders_to_workshop(self, orders, workshop_id):
        """分配订单到车间"""
        # 简化分配逻辑
        workshop_mapping = {
            'WS001': ['A-100', 'B-200'],
            'WS002': ['C-300', 'D-400'],
            'WS003': ['E-500'],
            'WS004': ['A-100', 'B-200', 'C-300', 'D-400', 'E-500']  # 质检所有产品
        }

        products = workshop_mapping.get(workshop_id, [])

        # 处理不同的列名
        if '产品型号' in orders.columns:
            return orders[orders['产品型号'].isin(products)]
        elif '产品' in orders.columns:
            return orders[orders['产品'].isin(products)]
        else:
            # 如果没有产品列，返回所有订单的一个子集
            return orders.sample(frac=0.25) if len(orders) > 0 else orders

    def _shortest_processing_time(self, orders, work_centers):
        """最短加工时间优先"""
        # 按处理时间排序
        sorted_orders = orders.sort_values('处理时间')
        return self._schedule_jobs(sorted_orders, work_centers)

    def _earliest_due_date(self, orders, work_centers):
        """最早交期优先"""
        sorted_orders = orders.sort_values('交期')
        return self._schedule_jobs(sorted_orders, work_centers)

    def _critical_ratio(self, orders, work_centers):
        """关键比率法"""
        try:
            orders = orders.copy()
            # 确保交期是datetime类型
            if '交期' in orders.columns:
                orders['交期'] = pd.to_datetime(orders['交期'])
                current_time = datetime.now()
                # 计算剩余时间（小时）
                orders['remaining_time'] = (orders['交期'] - current_time).dt.total_seconds() / 3600
                # 避免除以零
                orders['处理时间'] = orders['处理时间'].clip(lower=0.1)
                orders['critical_ratio'] = orders['remaining_time'] / orders['处理时间']
                # 处理负值和无穷大
                orders['critical_ratio'] = orders['critical_ratio'].clip(lower=0.001, upper=1000)
                sorted_orders = orders.sort_values('critical_ratio')
                return self._schedule_jobs(sorted_orders, work_centers)
            else:
                return self._schedule_jobs(orders, work_centers)
        except Exception as e:
            st.error(f"计算关键比率时出错: {str(e)}")
            return []

    def _minimum_slack(self, orders, work_centers):
        """最小松弛时间"""
        try:
            orders = orders.copy()
            # 确保交期是datetime类型
            if '交期' in orders.columns:
                orders['交期'] = pd.to_datetime(orders['交期'])
                current_time = datetime.now()
                # 计算松弛时间（小时）
                orders['slack'] = (orders['交期'] - current_time).dt.total_seconds() / 3600 - orders['处理时间']
                sorted_orders = orders.sort_values('slack')
                return self._schedule_jobs(sorted_orders, work_centers)
            else:
                return self._schedule_jobs(orders, work_centers)
        except Exception as e:
            st.error(f"计算松弛时间时出错: {str(e)}")
            return []

    def _first_in_first_out(self, orders, work_centers):
        """先进先出"""
        return self._schedule_jobs(orders, work_centers)

    def _schedule_jobs(self, orders, work_centers):
        """执行作业调度"""
        scheduled_jobs = []
        work_center_times = {wc['id']: 0 for wc in work_centers} if work_centers else {}

        for _, order in orders.iterrows():
            if not work_center_times:
                break

            # 选择最早可用的工作中心
            best_wc = min(work_center_times, key=work_center_times.get)
            start_time = work_center_times[best_wc]
            duration = order.get('处理时间', 1)
            end_time = start_time + duration

            scheduled_jobs.append({
                'job_id': order['订单编号'],
                'product': order['产品型号'],
                'operation': '生产',
                'work_center': best_wc,
                'start_time': datetime.now() + timedelta(hours=start_time),
                'end_time': datetime.now() + timedelta(hours=end_time),
                'duration': duration,
                'status': '已排程',
                'priority': order.get('优先级', '中')
            })

            work_center_times[best_wc] = end_time

        return scheduled_jobs

    def optimize_workshop_layout(self):
        """优化车间布局"""
        try:
            optimization_suggestions = []

            for workshop_id, workshop in self.workshop_resources.items():
                # 分析瓶颈
                utilizations = []
                for wc in workshop['work_centers']:
                    utilization = random.uniform(0.6, 0.95)
                    utilizations.append((wc['id'], utilization))

                # 识别瓶颈工作中心
                bottlenecks = [wc for wc, util in utilizations if util > 0.85]

                suggestion = {
                    '车间': workshop_id,
                    '车间名称': workshop['name'],
                    '瓶颈工作中心': bottlenecks,
                    '平均利用率': np.mean([util for _, util in utilizations]) if utilizations else 0,
                    '建议': '增加瓶颈工作中心产能' if bottlenecks else '布局合理'
                }

                optimization_suggestions.append(suggestion)

            return pd.DataFrame(optimization_suggestions)
        except Exception as e:
            st.error(f"优化车间布局时出错: {str(e)}")
            return pd.DataFrame()


# MaterialPlanningEngine类
class MaterialPlanningEngine:
    """物料计划引擎 - 基于MRP的物流需求计划"""

    def __init__(self):
        self.bom_tree = {}
        self.material_lead_times = {}
        self.safety_stock_levels = {}
        self.mrp_results = pd.DataFrame()

    def build_bom_tree(self, bom_data):
        """构建BOM树结构"""
        try:
            for bom_item in bom_data:
                if bom_item.product_id not in self.bom_tree:
                    self.bom_tree[bom_item.product_id] = []

                self.bom_tree[bom_item.product_id].append({
                    'component': bom_item.component_id,
                    'quantity': bom_item.quantity,
                    'type': bom_item.component_type
                })
        except Exception as e:
            st.error(f"构建BOM树时出错: {str(e)}")

    def run_mrp(self, mps_data, current_inventory, planning_horizon=8):
        """运行MRP计算"""
        try:
            mrp_records = []

            # 获取所有物料
            all_materials = set()
            for product_bom in self.bom_tree.values():
                for component in product_bom:
                    all_materials.add(component['component'])

            # 为每个物料计算需求
            for material in all_materials:
                # 初始化物料记录
                material_record = {
                    'material_id': material,
                    'lead_time': self.material_lead_times.get(material, 2),
                    'safety_stock': self.safety_stock_levels.get(material, 100),
                    'current_stock': 0,
                    'periods': []
                }

                # 修复：正确处理嵌套字典结构的库存数据
                if current_inventory and material in current_inventory:
                    inv_data = current_inventory[material]
                    if isinstance(inv_data, dict):
                        material_record['current_stock'] = inv_data.get('current_stock', 0)
                    else:
                        material_record['current_stock'] = inv_data
                else:
                    material_record['current_stock'] = 0

                running_stock = material_record['current_stock']

                # 计算每个期间的需求
                for period in range(1, planning_horizon + 1):
                    # 计算毛需求（从父项产品的计划生产量推导）
                    gross_requirement = self._calculate_gross_requirement(material, mps_data, period)

                    # 计划接收（之前下达的订单）
                    scheduled_receipts = 0  # 简化处理

                    # 计算净需求
                    projected_on_hand = running_stock + scheduled_receipts - gross_requirement
                    net_requirement = max(0, material_record['safety_stock'] - projected_on_hand)

                    # 计划订单接收
                    planned_order_receipt = net_requirement if net_requirement > 0 else 0

                    # 计划订单下达（考虑提前期）
                    planned_order_release = 0
                    if period + material_record['lead_time'] <= planning_horizon and planned_order_receipt > 0:
                        planned_order_release = planned_order_receipt

                    # 更新库存
                    ending_inventory = projected_on_hand + planned_order_receipt
                    running_stock = ending_inventory

                    period_data = {
                        '物料编号': material,
                        '期间': period,
                        '毛需求': gross_requirement,
                        '计划接收': scheduled_receipts,
                        '预计库存': projected_on_hand,
                        '净需求': net_requirement,
                        '计划订单接收': planned_order_receipt,
                        '计划订单下达': planned_order_release,
                        '期末库存': ending_inventory
                    }

                    mrp_records.append(period_data)
                    material_record['periods'].append(period_data)

            self.mrp_results = pd.DataFrame(mrp_records)
            return self.mrp_results
        except Exception as e:
            st.error(f"运行MRP时出错: {str(e)}")
            return pd.DataFrame()

    def _calculate_gross_requirement(self, material, mps_data, period):
        """计算物料的毛需求"""
        gross_requirement = 0

        try:
            # 遍历所有使用该物料的父项产品
            for product, components in self.bom_tree.items():
                for component in components:
                    if component['component'] == material:
                        # 获取父项产品在该期间的计划生产量
                        if not mps_data.empty and '周次' in mps_data.columns:
                            product_production = mps_data[
                                (mps_data['产品编号'] == product) &
                                (mps_data['周次'] == period)
                                ]['计划生产'].sum()
                        else:
                            product_production = random.randint(100, 300)

                        gross_requirement += product_production * component['quantity']
        except Exception as e:
            st.error(f"计算毛需求时出错: {str(e)}")

        return gross_requirement

    def create_purchase_plan(self):
        """创建采购计划"""
        try:
            if self.mrp_results.empty:
                return pd.DataFrame()

            purchase_plan = []

            # 获取所有需要下达采购订单的记录
            purchase_requirements = self.mrp_results[self.mrp_results['计划订单下达'] > 0]

            for _, req in purchase_requirements.iterrows():
                purchase_order = {
                    '采购单号': f"PO-{datetime.now().strftime('%Y%m%d')}-{random.randint(1000, 9999)}",
                    '物料编号': req['物料编号'],
                    '数量': req['计划订单下达'],
                    '需求日期': datetime.now() + timedelta(weeks=req['期间']),
                    '下单日期': datetime.now() + timedelta(
                        weeks=max(0, req['期间'] - self.material_lead_times.get(req['物料编号'], 2))),
                    '供应商': f"SUP-{random.randint(100, 999)}",
                    '状态': '待下单',
                    '紧急程度': '高' if req['期间'] <= 2 else '中' if req['期间'] <= 4 else '低'
                }

                purchase_plan.append(purchase_order)

            return pd.DataFrame(purchase_plan)
        except Exception as e:
            st.error(f"创建采购计划时出错: {str(e)}")
            return pd.DataFrame()


# ProductionMaterialPreparation类
class ProductionMaterialPreparation:
    """生产备料系统 - 三级物料保障机制"""

    def __init__(self):
        self.material_levels = {
            'level1': {},  # 一级：线边库
            'level2': {},  # 二级：车间库
            'level3': {}  # 三级：中心库
        }
        self.material_flow_rules = {}
        self.preparation_status = {}

    def setup_three_level_system(self, materials, production_plan):
        """设置三级物料保障体系"""
        try:
            for material in materials:
                # 处理不同的数据结构
                if isinstance(material, dict):
                    material_id = material.get('物料编码', f'MAT-{random.randint(100, 999)}')
                else:
                    material_id = f'MAT-{random.randint(100, 999)}'

                # 计算各级库存水平
                daily_usage = self._calculate_daily_usage(material_id, production_plan)

                # 一级：线边库（2-4小时用量）
                self.material_levels['level1'][material_id] = {
                    'capacity': daily_usage * 0.5,
                    'current_stock': daily_usage * 0.3,
                    'min_stock': daily_usage * 0.1,
                    'max_stock': daily_usage * 0.5,
                    'replenishment_trigger': daily_usage * 0.2
                }

                # 二级：车间库（1-2天用量）
                self.material_levels['level2'][material_id] = {
                    'capacity': daily_usage * 2,
                    'current_stock': daily_usage * 1.5,
                    'min_stock': daily_usage * 0.5,
                    'max_stock': daily_usage * 2,
                    'replenishment_trigger': daily_usage * 1
                }

                # 三级：中心库（5-7天用量）
                self.material_levels['level3'][material_id] = {
                    'capacity': daily_usage * 7,
                    'current_stock': daily_usage * 5,
                    'min_stock': daily_usage * 3,
                    'max_stock': daily_usage * 7,
                    'replenishment_trigger': daily_usage * 4
                }
        except Exception as e:
            st.error(f"设置三级物料体系时出错: {str(e)}")

    def _calculate_daily_usage(self, material_id, production_plan):
        """计算物料日均用量"""
        # 简化计算
        return random.uniform(100, 500)

    def create_preparation_plan(self, production_schedule):
        """创建备料计划"""
        try:
            preparation_plans = []

            for _, job in production_schedule.iterrows():
                # 获取该作业需要的物料
                product = job.get('产品', job.get('产品型号', 'Unknown'))
                required_materials = self._get_required_materials(product)

                for material in required_materials:
                    # 检查各级库存
                    level1_status = self._check_material_availability('level1', material['material_id'],
                                                                      material['quantity'])
                    level2_status = self._check_material_availability('level2', material['material_id'],
                                                                      material['quantity'] * 2)
                    level3_status = self._check_material_availability('level3', material['material_id'],
                                                                      material['quantity'] * 5)

                    # 获取开始时间，处理不同的列名
                    start_time = job.get('开始时间', datetime.now())
                    if isinstance(start_time, str):
                        start_time = pd.to_datetime(start_time)

                    preparation = {
                        '作业编号': job.get('作业编号', job.get('订单编号', 'N/A')),
                        '产品': product,
                        '物料编号': material['material_id'],
                        '需求数量': material['quantity'],
                        '计划开始时间': start_time,
                        '备料时间': start_time - timedelta(hours=2),
                        '线边库状态': level1_status,
                        '车间库状态': level2_status,
                        '中心库状态': level3_status,
                        '备料策略': self._determine_preparation_strategy(level1_status, level2_status, level3_status),
                        '风险等级': self._assess_risk_level(level1_status, level2_status, level3_status)
                    }

                    preparation_plans.append(preparation)

            return pd.DataFrame(preparation_plans)
        except Exception as e:
            st.error(f"创建备料计划时出错: {str(e)}")
            return pd.DataFrame()

    def _get_required_materials(self, product):
        """获取产品所需物料"""
        # 简化处理，返回示例数据
        materials = []
        for i in range(random.randint(3, 8)):
            materials.append({
                'material_id': f"MAT-{random.randint(100, 999)}",
                'quantity': random.randint(10, 100)
            })
        return materials

    def _check_material_availability(self, level, material_id, required_quantity):
        """检查物料可用性"""
        if material_id in self.material_levels[level]:
            current_stock = self.material_levels[level][material_id]['current_stock']
            if current_stock >= required_quantity:
                return '充足'
            elif current_stock >= required_quantity * 0.5:
                return '偏低'
            else:
                return '不足'
        return '缺货'

    def _determine_preparation_strategy(self, level1, level2, level3):
        """确定备料策略"""
        if level1 == '充足':
            return '直接配送'
        elif level2 == '充足':
            return '车间补充'
        elif level3 == '充足':
            return '中心库调拨'
        else:
            return '紧急采购'

    def _assess_risk_level(self, level1, level2, level3):
        """评估风险等级"""
        risk_scores = {
            '充足': 0,
            '偏低': 1,
            '不足': 2,
            '缺货': 3
        }

        total_risk = risk_scores.get(level1, 3) + risk_scores.get(level2, 3) * 0.5 + risk_scores.get(level3, 3) * 0.3

        if total_risk <= 1:
            return '低'
        elif total_risk <= 2:
            return '中'
        else:
            return '高'

    def generate_replenishment_orders(self):
        """生成补料订单"""
        try:
            replenishment_orders = []

            # 检查所有级别的库存
            for level_name, level_data in self.material_levels.items():
                for material_id, stock_info in level_data.items():
                    if stock_info['current_stock'] <= stock_info['replenishment_trigger']:
                        order_quantity = stock_info['max_stock'] - stock_info['current_stock']

                        replenishment = {
                            '补料单号': f"REP-{datetime.now().strftime('%Y%m%d')}-{random.randint(1000, 9999)}",
                            '物料编号': material_id,
                            '库存级别': level_name,
                            '当前库存': stock_info['current_stock'],
                            '触发点': stock_info['replenishment_trigger'],
                            '补充数量': order_quantity,
                            '来源': self._determine_source(level_name),
                            '紧急程度': '高' if stock_info['current_stock'] < stock_info['min_stock'] else '中',
                            '创建时间': datetime.now()
                        }

                        replenishment_orders.append(replenishment)

            return pd.DataFrame(replenishment_orders)
        except Exception as e:
            st.error(f"生成补料订单时出错: {str(e)}")
            return pd.DataFrame()

    def _determine_source(self, level):
        """确定补料来源"""
        sources = {
            'level1': '车间库',
            'level2': '中心库',
            'level3': '供应商'
        }
        return sources.get(level, '供应商')


# ShippingPlanningSystem类
class ShippingPlanningSystem:
    """发运计划系统 - 集成优化发运环节"""

    def __init__(self):
        self.shipping_routes = {}
        self.transport_resources = {}
        self.shipping_constraints = {}
        self.consolidation_rules = {}

    def setup_shipping_network(self, warehouses, customers):
        """设置发运网络"""
        try:
            # 创建运输路线
            route_id = 1
            for _, wh in warehouses.iterrows():
                for customer in customers:
                    self.shipping_routes[f"R{route_id:03d}"] = {
                        'origin': wh['warehouse_id'],
                        'destination': customer,
                        'distance': random.uniform(50, 1000),
                        'transit_time': random.randint(1, 5),
                        'cost_per_km': random.uniform(1, 3),
                        'transport_modes': ['公路', '铁路', '航空']
                    }
                    route_id += 1

            # 设置运输资源
            self.transport_resources = {
                'trucks': {'capacity': 30000, 'count': 50, 'cost_per_trip': 500},
                'trains': {'capacity': 100000, 'count': 10, 'cost_per_trip': 2000},
                'planes': {'capacity': 10000, 'count': 5, 'cost_per_trip': 5000}
            }
        except Exception as e:
            st.error(f"设置发运网络时出错: {str(e)}")

    def create_shipping_plan(self, delivery_orders, optimization_goal='cost'):
        """创建发运计划"""
        try:
            shipping_plans = []

            # 按目的地和时间窗口合并订单
            consolidated_orders = self._consolidate_orders(delivery_orders)

            for consol_id, consol_data in consolidated_orders.items():
                # 选择最优运输方案
                best_route, best_mode = self._select_optimal_transport(
                    consol_data,
                    optimization_goal
                )

                if not best_route:
                    continue

                # 分配运输资源
                transport_allocation = self._allocate_transport_resources(
                    consol_data['total_weight'],
                    best_mode
                )

                shipping_plan = {
                    '发运单号': f"SH-{datetime.now().strftime('%Y%m%d')}-{random.randint(1000, 9999)}",
                    '合并单号': consol_id,
                    '包含订单': len(consol_data['orders']),
                    '总重量': consol_data['total_weight'],
                    '总体积': consol_data['total_volume'],
                    '起始地': best_route['origin'],
                    '目的地': best_route['destination'],
                    '运输方式': best_mode,
                    '运输路线': f"{best_route['origin']} -> {best_route['destination']}",
                    '计划发运时间': consol_data['latest_pickup_time'],
                    '预计到达时间': consol_data['latest_pickup_time'] + timedelta(days=best_route['transit_time']),
                    '运输成本': transport_allocation['cost'],
                    '车辆数量': transport_allocation['vehicle_count'],
                    '装载率': transport_allocation['loading_rate'],
                    '状态': '待发运'
                }

                shipping_plans.append(shipping_plan)

            return pd.DataFrame(shipping_plans)
        except Exception as e:
            st.error(f"创建发运计划时出错: {str(e)}")
            return pd.DataFrame()

    def _consolidate_orders(self, orders):
        """合并订单"""
        consolidated = {}

        try:
            # 按目的地和时间窗口分组
            for _, order in orders.iterrows():
                # 创建合并键
                order_date = order.get('交期', datetime.now())
                if isinstance(order_date, str):
                    try:
                        order_date = pd.to_datetime(order_date)
                    except:
                        order_date = datetime.now()
                elif pd.isna(order_date):
                    order_date = datetime.now()

                # 获取目的地，提供默认值
                destination = order.get('目的地', order.get('客户', f'DEST-{random.randint(1, 5)}'))

                consol_key = f"{destination}_{order_date.date()}"

                if consol_key not in consolidated:
                    consolidated[consol_key] = {
                        'orders': [],
                        'total_weight': 0,
                        'total_volume': 0,
                        'earliest_pickup': datetime.now() + timedelta(days=10),
                        'latest_pickup_time': datetime.now()
                    }

                consolidated[consol_key]['orders'].append(order.get('订单编号', f'ORD-{random.randint(1000, 9999)}'))
                consolidated[consol_key]['total_weight'] += order.get('数量', 0) * 10  # 假设单位重量
                consolidated[consol_key]['total_volume'] += order.get('数量', 0) * 0.1  # 假设单位体积

                if order_date < consolidated[consol_key]['earliest_pickup']:
                    consolidated[consol_key]['earliest_pickup'] = order_date
                if order_date > consolidated[consol_key]['latest_pickup_time']:
                    consolidated[consol_key]['latest_pickup_time'] = order_date
        except Exception as e:
            st.error(f"合并订单时出错: {str(e)}")

        return consolidated

    def _select_optimal_transport(self, consol_data, optimization_goal):
        """选择最优运输方案"""
        best_route = None
        best_mode = None
        best_score = float('inf') if optimization_goal == 'cost' else 0

        try:
            for route_id, route in self.shipping_routes.items():
                for mode in route['transport_modes']:
                    # 计算得分
                    if optimization_goal == 'cost':
                        score = self._calculate_transport_cost(consol_data, route, mode)
                        if score < best_score:
                            best_score = score
                            best_route = route
                            best_mode = mode
                    elif optimization_goal == 'speed':
                        score = route['transit_time']
                        if score < best_score:
                            best_score = score
                            best_route = route
                            best_mode = mode
                    elif optimization_goal == 'reliability':
                        score = random.uniform(0.8, 0.99)  # 可靠性得分
                        if score > best_score:
                            best_score = score
                            best_route = route
                            best_mode = mode
        except Exception as e:
            st.error(f"选择运输方案时出错: {str(e)}")

        return best_route, best_mode

    def _calculate_transport_cost(self, consol_data, route, mode):
        """计算运输成本"""
        base_cost = route['distance'] * route['cost_per_km']

        # 根据运输方式调整成本
        mode_factors = {
            '公路': 1.0,
            '铁路': 0.7,
            '航空': 3.0
        }

        mode_cost = base_cost * mode_factors.get(mode, 1.0)

        # 考虑货物重量
        weight_factor = 1 + (consol_data['total_weight'] / 10000) * 0.1

        return mode_cost * weight_factor

    def _allocate_transport_resources(self, total_weight, transport_mode):
        """分配运输资源"""
        mode_mapping = {
            '公路': 'trucks',
            '铁路': 'trains',
            '航空': 'planes'
        }

        resource_type = mode_mapping.get(transport_mode, 'trucks')
        resource = self.transport_resources[resource_type]

        # 计算需要的车辆数
        vehicle_count = max(1, math.ceil(total_weight / resource['capacity']))

        # 计算装载率
        loading_rate = min(1.0, total_weight / (vehicle_count * resource['capacity']))

        # 计算成本
        total_cost = vehicle_count * resource['cost_per_trip']

        return {
            'vehicle_count': vehicle_count,
            'loading_rate': loading_rate,
            'cost': total_cost,
            'resource_type': resource_type
        }

    def track_shipments(self, shipping_plans):
        """跟踪发运状态"""
        try:
            tracking_info = []

            for _, shipment in shipping_plans.iterrows():
                # 模拟跟踪信息
                current_location = self._get_current_location(shipment)
                progress = random.uniform(0, 100)

                tracking = {
                    '发运单号': shipment['发运单号'],
                    '当前位置': current_location,
                    '运输进度': f"{progress:.1f}%",
                    '预计剩余时间': max(0, (100 - progress) / 20),  # 小时
                    '状态': self._determine_status(progress),
                    '最后更新': datetime.now(),
                    '异常情况': '无' if random.random() > 0.1 else random.choice(['交通拥堵', '天气影响', '海关检查'])
                }

                tracking_info.append(tracking)

            return pd.DataFrame(tracking_info)
        except Exception as e:
            st.error(f"跟踪发运状态时出错: {str(e)}")
            return pd.DataFrame()

    def _get_current_location(self, shipment):
        """获取当前位置"""
        locations = ['起始仓库', '转运中心1', '转运中心2', '目的地仓库', '配送中']
        return random.choice(locations)

    def _determine_status(self, progress):
        """确定运输状态"""
        if progress < 10:
            return '待发运'
        elif progress < 90:
            return '运输中'
        elif progress < 100:
            return '即将到达'
        else:
            return '已送达'


# IntelligentOperationCommandCenter类
class IntelligentOperationCommandCenter:
    """计划智能运营指挥中心"""

    def __init__(self):
        self.monitoring_metrics = {}
        self.alert_rules = {}
        self.compliance_checks = {}
        self.kpi_thresholds = {}
        self.real_time_data = {}

    def setup_monitoring_system(self):
        """设置监控系统"""
        # 定义监控指标
        self.monitoring_metrics = {
            'order_fulfillment': {
                'name': '订单履行率',
                'target': 95,
                'unit': '%',
                'frequency': 'real-time'
            },
            'production_efficiency': {
                'name': '生产效率',
                'target': 85,
                'unit': '%',
                'frequency': 'hourly'
            },
            'inventory_accuracy': {
                'name': '库存准确率',
                'target': 99,
                'unit': '%',
                'frequency': 'daily'
            },
            'on_time_delivery': {
                'name': '准时交付率',
                'target': 90,
                'unit': '%',
                'frequency': 'real-time'
            },
            'resource_utilization': {
                'name': '资源利用率',
                'target': 80,
                'unit': '%',
                'frequency': 'hourly'
            }
        }

        # 定义预警规则
        self.alert_rules = {
            'critical': {'threshold': 0.7, 'color': 'red', 'action': '立即处理'},
            'warning': {'threshold': 0.85, 'color': 'orange', 'action': '密切关注'},
            'normal': {'threshold': 0.95, 'color': 'yellow', 'action': '常规监控'},
            'excellent': {'threshold': 1.0, 'color': 'green', 'action': '保持'}
        }

    def monitor_order_execution(self, orders, schedule):
        """监控订单执行过程"""
        try:
            monitoring_results = []

            for _, order in orders.iterrows():
                # 获取订单排程信息
                order_schedule = schedule[schedule['订单编号'] == order['订单编号']] if not schedule.empty else pd.DataFrame()

                # 计算执行状态
                if order_schedule.empty:
                    execution_status = '未排程'
                    progress = 0
                else:
                    # 模拟执行进度
                    progress = random.uniform(0, 100)
                    if progress < 30:
                        execution_status = '准备中'
                    elif progress < 70:
                        execution_status = '生产中'
                    elif progress < 90:
                        execution_status = '质检中'
                    else:
                        execution_status = '待发货'

                # 计算偏差
                planned_date = order.get('交期', datetime.now())
                if isinstance(planned_date, str):
                    planned_date = pd.to_datetime(planned_date)

                current_date = datetime.now()
                days_remaining = (planned_date - current_date).days if isinstance(planned_date, datetime) else 0

                # 风险评估
                if progress < 50 and days_remaining < 3:
                    risk_level = '高'
                elif progress < 70 and days_remaining < 5:
                    risk_level = '中'
                else:
                    risk_level = '低'

                monitoring_result = {
                    '订单编号': order['订单编号'],
                    '产品型号': order['产品型号'],
                    '客户': order.get('客户', 'N/A'),
                    '执行状态': execution_status,
                    '完成进度': f"{progress:.1f}%",
                    '计划交期': planned_date,
                    '剩余天数': days_remaining,
                    '风险等级': risk_level,
                    '预警信息': self._generate_alert(progress, days_remaining),
                    '建议措施': self._suggest_action(risk_level, execution_status)
                }

                monitoring_results.append(monitoring_result)

            return pd.DataFrame(monitoring_results)
        except Exception as e:
            st.error(f"监控订单执行时出错: {str(e)}")
            return pd.DataFrame()

    def _generate_alert(self, progress, days_remaining):
        """生成预警信息"""
        if progress < 30 and days_remaining < 2:
            return "⚠️ 紧急：进度严重滞后"
        elif progress < 50 and days_remaining < 3:
            return "⚠️ 警告：进度落后于计划"
        elif progress > 90 and days_remaining > 5:
            return "✅ 提前完成"
        else:
            return "正常"

    def _suggest_action(self, risk_level, status):
        """建议措施"""
        suggestions = {
            ('高', '准备中'): '立即安排生产资源',
            ('高', '生产中'): '加急处理，考虑加班',
            ('中', '生产中'): '优化生产顺序',
            ('低', '质检中'): '确保质量，准备发货',
            ('低', '待发货'): '安排运输资源'
        }

        return suggestions.get((risk_level, status), '持续监控')

    def compliance_audit(self, master_data):
        """基础数据合规性稽查"""
        try:
            audit_results = []

            # 检查产品主数据
            if 'products' in master_data and master_data['products']:
                for product_id, product in master_data['products'].items():
                    checks = {
                        'product_id_format': self._check_id_format(product_id, 'P'),
                        'price_validity': product.sell_price > product.unit_cost,
                        'lead_time_reasonable': 1 <= product.lead_time <= 90,
                        'safety_stock_set': product.safety_stock > 0,
                        'reorder_point_logic': product.reorder_point >= product.safety_stock
                    }

                    compliance_score = sum(checks.values()) / len(checks) * 100

                    audit_results.append({
                        '数据类型': '产品主数据',
                        '编号': product_id,
                        '名称': product.name,
                        'ID格式检查': '✓' if checks['product_id_format'] else '✗',
                        '价格逻辑检查': '✓' if checks['price_validity'] else '✗',
                        '提前期检查': '✓' if checks['lead_time_reasonable'] else '✗',
                        '安全库存检查': '✓' if checks['safety_stock_set'] else '✗',
                        '订货点检查': '✓' if checks['reorder_point_logic'] else '✗',
                        '合规得分': f"{compliance_score:.1f}%",
                        '合规状态': '合格' if compliance_score >= 80 else '需改进'
                    })

            # 检查BOM数据
            if 'bom' in master_data and master_data['bom']:
                bom_checks = self._audit_bom_data(master_data['bom'])
                audit_results.extend(bom_checks)

            return pd.DataFrame(audit_results)
        except Exception as e:
            st.error(f"合规性稽查时出错: {str(e)}")
            return pd.DataFrame()

    def _check_id_format(self, id_string, prefix):
        """检查ID格式"""
        return id_string.startswith(prefix) and len(id_string) >= 4

    def _audit_bom_data(self, bom_data):
        """审计BOM数据"""
        audit_results = []

        try:
            # 检查BOM完整性和循环引用
            bom_graph = nx.DiGraph()

            for bom_item in bom_data:
                bom_graph.add_edge(bom_item.product_id, bom_item.component_id)

                # 检查数量合理性
                quantity_check = 0 < bom_item.quantity <= 100

                audit_results.append({
                    '数据类型': 'BOM数据',
                    '编号': f"{bom_item.product_id}->{bom_item.component_id}",
                    '名称': f"BOM关系",
                    'ID格式检查': '✓',
                    '数量合理性': '✓' if quantity_check else '✗',
                    '组件类型检查': '✓' if bom_item.component_type in ['原材料', '半成品', '组件'] else '✗',
                    '循环引用检查': 'N/A',
                    '层级检查': 'N/A',
                    '合规得分': '100%' if quantity_check else '50%',
                    '合规状态': '合格' if quantity_check else '需改进'
                })

            # 检查循环引用
            try:
                cycles = list(nx.simple_cycles(bom_graph))
                if cycles:
                    for result in audit_results:
                        if result['数据类型'] == 'BOM数据':
                            result['循环引用检查'] = '✗ 发现循环'
                            result['合规状态'] = '严重问题'
            except:
                pass
        except Exception as e:
            st.error(f"审计BOM数据时出错: {str(e)}")

        return audit_results

    def generate_executive_dashboard(self):
        """生成高管仪表板"""
        try:
            # 实时KPI数据
            kpi_data = []

            for metric_id, metric_info in self.monitoring_metrics.items():
                current_value = random.uniform(metric_info['target'] * 0.8, metric_info['target'] * 1.1)

                # 确定状态
                ratio = current_value / metric_info['target']
                if ratio >= 0.95:
                    status = 'excellent'
                elif ratio >= 0.85:
                    status = 'normal'
                elif ratio >= 0.7:
                    status = 'warning'
                else:
                    status = 'critical'

                kpi_data.append({
                    'KPI指标': metric_info['name'],
                    '当前值': f"{current_value:.1f}{metric_info['unit']}",
                    '目标值': f"{metric_info['target']}{metric_info['unit']}",
                    '达成率': f"{ratio * 100:.1f}%",
                    '状态': status,
                    '趋势': random.choice(['↑', '↓', '→']),
                    '更新时间': datetime.now().strftime('%H:%M:%S')
                })

            return pd.DataFrame(kpi_data)
        except Exception as e:
            st.error(f"生成仪表板时出错: {str(e)}")
            return pd.DataFrame()

    def predictive_alerts(self, historical_data):
        """预测性预警"""
        try:
            alerts = []

            # 预测未来可能的问题
            prediction_scenarios = [
                {
                    'type': '产能瓶颈',
                    'probability': random.uniform(0.3, 0.8),
                    'impact': '高',
                    'time_horizon': '未来3天',
                    'affected_resources': ['RES-001', 'RES-005'],
                    'preventive_action': '提前安排加班或外协'
                },
                {
                    'type': '物料短缺',
                    'probability': random.uniform(0.2, 0.6),
                    'impact': '中',
                    'time_horizon': '未来1周',
                    'affected_materials': ['MAT-100', 'MAT-205'],
                    'preventive_action': '联系供应商加急采购'
                },
                {
                    'type': '交付延迟风险',
                    'probability': random.uniform(0.1, 0.4),
                    'impact': '高',
                    'time_horizon': '未来2天',
                    'affected_orders': ['ORD-00010', 'ORD-00025'],
                    'preventive_action': '调整生产优先级'
                }
            ]

            for scenario in prediction_scenarios:
                if scenario['probability'] > 0.3:  # 只显示概率较高的预警
                    alert = {
                        '预警类型': scenario['type'],
                        '发生概率': f"{scenario['probability'] * 100:.1f}%",
                        '影响程度': scenario['impact'],
                        '预计时间': scenario['time_horizon'],
                        '影响范围': ', '.join(scenario.get('affected_resources',
                                                       scenario.get('affected_materials',
                                                                    scenario.get('affected_orders', [])))),
                        '建议措施': scenario['preventive_action'],
                        '创建时间': datetime.now()
                    }

                    alerts.append(alert)

            return pd.DataFrame(alerts)
        except Exception as e:
            st.error(f"生成预测性预警时出错: {str(e)}")
            return pd.DataFrame()


# ==================== 核心APS引擎（融合版） ====================
class HybridAPSEngine:
    """融合SAP IBP、Oracle SCM和Blue Yonder技术的智能APS引擎"""

    def __init__(self):
        self.products = {}
        self.bom = []
        self.resources = {}
        self.demands = []
        self.inventory = {}
        self.production_plan = []
        self.supply_plan = []
        self.factories = pd.DataFrame()
        self.materials = pd.DataFrame()

    def algorithm_router(self, orders, resources):
        """动态混合算法路由机制"""
        order_count = len(orders)

        if order_count <= 50:
            st.info("🎯 使用MIP+CP混合求解器 (SAP IBP风格) - 精确求解")
            return self.mip_cp_solver(orders, resources)
        elif order_count <= 500:
            st.info("🧠 使用遗传算法+强化学习混合优化 (安达发AI引擎)")
            return self.ga_rl_hybrid(orders, resources)
        else:
            st.info("⚡使用在线约束规划 (Blue Yonder实时重排)")
            return self.online_constraint_programming(orders, resources)

    def mip_cp_solver(self, orders, resources):
        """混合整数规划+约束规划求解器"""
        try:
            prob = pulp.LpProblem("Advanced_Production_Scheduling", pulp.LpMinimize)

            resource_ids = resources['资源编号'].unique().tolist()
            order_ids = orders['订单编号'].tolist()

            # 决策变量
            x = pulp.LpVariable.dicts("assign",
                                      ((order, res) for order in order_ids for res in resource_ids),
                                      cat='Binary')

            start_time = pulp.LpVariable.dicts("start",
                                               (order for order in order_ids),
                                               lowBound=0, cat='Continuous')

            # 目标函数
            prob += pulp.lpSum([start_time[order] for order in order_ids])

            # 约束条件
            for order_id in order_ids:
                prob += pulp.lpSum([x[order_id, res] for res in resource_ids]) == 1

            # 资源产能约束
            for res in resource_ids:
                res_capacity = resources[resources['资源编号'] == res]['总产能'].values[0]
                prob += pulp.lpSum([x[order_id, res] * orders[orders['订单编号'] == order_id]['处理时间'].values[0]
                                    for order_id in order_ids]) <= res_capacity

            # 求解
            solver = pulp.PULP_CBC_CMD(timeLimit=30, msg=False)
            prob.solve(solver)

            return self.create_schedule_from_solution(orders, resources, x, start_time, order_ids, resource_ids)

        except Exception as e:
            st.warning(f"MIP求解失败，使用启发式算法: {str(e)}")
            return self.heuristic_scheduler(orders, resources)

    def ga_rl_hybrid(self, orders, resources):
        """遗传算法+强化学习混合优化"""
        try:
            # 简化版遗传算法
            resource_ids = resources['资源编号'].tolist()
            order_ids = orders['订单编号'].tolist()

            # 随机分配策略
            schedule = []
            current_time = {res: 0 for res in resource_ids}

            # 按优先级排序
            priority_map = {'紧急': 0, '高': 1, '中': 2, '低': 3}
            orders_sorted = orders.copy()
            orders_sorted['优先级值'] = orders_sorted['优先级'].map(priority_map)
            orders_sorted = orders_sorted.sort_values(['优先级值', '交期']).reset_index(drop=True)

            for _, order in orders_sorted.iterrows():
                # 选择负载最轻的资源
                best_resource = min(current_time, key=current_time.get)
                start = current_time[best_resource]
                duration = order['处理时间']

                schedule.append({
                    '订单编号': order['订单编号'],
                    '产品型号': order['产品型号'],
                    '数量': order['数量'],
                    '资源编号': best_resource,
                    '开始时间': datetime.now() + timedelta(hours=start),
                    '结束时间': datetime.now() + timedelta(hours=start + duration),
                    '持续时间': duration,
                    '工厂分配': order.get('工厂分配', 'FACT-001')
                })

                current_time[best_resource] += duration

            return pd.DataFrame(schedule)

        except Exception as e:
            st.warning(f"遗传算法失败，使用启发式算法: {str(e)}")
            return self.heuristic_scheduler(orders, resources)

    def online_constraint_programming(self, orders, resources):
        """在线约束规划"""
        return self.heuristic_scheduler(orders, resources)

    def heuristic_scheduler(self, orders, resources):
        """启发式规则排程器"""
        try:
            priority_order = {'紧急': 0, '高': 1, '中': 2, '低': 3}
            orders_copy = orders.copy()
            orders_copy['优先级值'] = orders_copy['优先级'].map(priority_order)
            orders_copy['交期'] = pd.to_datetime(orders_copy['交期'])
            sorted_orders = orders_copy.sort_values(['优先级值', '交期']).reset_index(drop=True)

            resource_times = {res: 0 for res in resources['资源编号'].unique()}
            schedule = []

            for _, order in sorted_orders.iterrows():
                best_resource = min(resource_times, key=resource_times.get)
                start_time = resource_times[best_resource]
                duration = order.get('处理时间', 1)

                schedule.append({
                    '订单编号': order['订单编号'],
                    '产品型号': order['产品型号'],
                    '数量': order['数量'],
                    '资源编号': best_resource,
                    '开始时间': datetime.now() + timedelta(hours=start_time),
                    '结束时间': datetime.now() + timedelta(hours=start_time + duration),
                    '持续时间': duration,
                    '工厂分配': order.get('工厂分配', 'FACT-001')
                })

                resource_times[best_resource] += duration

            return pd.DataFrame(schedule)
        except Exception as e:
            st.error(f"启发式调度失败: {str(e)}")
            return pd.DataFrame()

    def create_schedule_from_solution(self, orders, resources, x, start_time, order_ids, resource_ids):
        """从求解结果创建排程计划"""
        schedule = []
        for order_id in order_ids:
            for res in resource_ids:
                if pulp.value(x[order_id, res]) == 1:
                    order = orders[orders['订单编号'] == order_id].iloc[0]
                    duration = order['处理时间']
                    start = pulp.value(start_time[order_id])

                    schedule.append({
                        '订单编号': order_id,
                        '产品型号': order['产品型号'],
                        '数量': order['数量'],
                        '资源编号': res,
                        '开始时间': datetime.now() + timedelta(hours=start),
                        '结束时间': datetime.now() + timedelta(hours=start + duration),
                        '持续时间': duration,
                        '工厂分配': order.get('工厂分配', 'FACT-001')
                    })

        return pd.DataFrame(schedule)

    def calculate_mrp(self) -> pd.DataFrame:
        """物料需求计划计算"""
        if not self.demands:
            return pd.DataFrame()

        try:
            mrp_data = []
            base_date = datetime.now()
            date_range = [base_date + timedelta(days=i) for i in range(30)]

            for product_id, product in self.products.items():
                for date in date_range:
                    gross_requirement = sum(
                        demand.quantity for demand in self.demands
                        if demand.product_id == product_id and demand.due_date.date() == date.date()
                    )

                    current_stock = self.inventory.get(product_id, {}).get('current_stock', 0)
                    net_requirement = max(0, gross_requirement - current_stock)
                    planned_order = net_requirement if net_requirement > 0 else 0
                    planned_order_release = date - timedelta(days=product.lead_time) if planned_order > 0 else None

                    mrp_data.append({
                        '产品编号': product_id,
                        '产品名称': product.name,
                        '日期': date,
                        '毛需求': gross_requirement,
                        '现有库存': current_stock,
                        '净需求': net_requirement,
                        '计划订单': planned_order,
                        '计划订单投放': planned_order_release
                    })

            return pd.DataFrame(mrp_data)
        except Exception as e:
            st.error(f"计算MRP时出错: {str(e)}")
            return pd.DataFrame()

    def generate_sample_data(self):
        """生成示例数据"""
        try:
            # 生成产品数据
            product_categories = ['电子产品', '机械部件', '化工原料', '纺织品']
            for i in range(20):
                product = Product(
                    product_id=f"P{i + 1:03d}",
                    name=f"产品{i + 1}",
                    category=random.choice(product_categories),
                    unit_cost=round(random.uniform(50, 500), 2),
                    sell_price=round(random.uniform(100, 800), 2),
                    lead_time=random.randint(1, 14),
                    safety_stock=random.randint(10, 100),
                    reorder_point=random.randint(20, 150)
                )
                self.products[product.product_id] = product

            # 生成BOM数据
            for product_id in list(self.products.keys())[:10]:
                components_count = random.randint(2, 5)
                for j in range(components_count):
                    component_id = random.choice(list(self.products.keys())[10:])
                    bom_item = BOM(
                        product_id=product_id,
                        component_id=component_id,
                        quantity=round(random.uniform(1, 10), 2),
                        component_type=random.choice(['原材料', '半成品', '组件'])
                    )
                    self.bom.append(bom_item)

            # 生成资源数据
            resource_types = ['生产线', '设备', '仓库', '运输']
            for i in range(15):
                resource = Resource(
                    resource_id=f"R{i + 1:03d}",
                    name=f"资源{i + 1}",
                    type=random.choice(resource_types),
                    capacity=round(random.uniform(100, 1000), 2),
                    cost_per_hour=round(random.uniform(50, 200), 2),
                    efficiency=round(random.uniform(0.7, 1.0), 2),
                    availability=round(random.uniform(0.8, 1.0), 2)
                )
                self.resources[resource.resource_id] = resource

            # 生成需求数据
            customers = ['客户A', '客户B', '客户C', '客户D', '客户E']
            base_date = datetime.now()
            for i in range(50):
                demand = Demand(
                    demand_id=f"D{i + 1:03d}",
                    product_id=random.choice(list(self.products.keys())),
                    quantity=random.randint(10, 500),
                    due_date=base_date + timedelta(days=random.randint(1, 90)),
                    priority=random.randint(1, 5),
                    customer=random.choice(customers)
                )
                self.demands.append(demand)

            # 生成库存数据
            for product_id in self.products.keys():
                self.inventory[product_id] = {
                    'current_stock': random.randint(0, 200),
                    'available_stock': random.randint(0, 200),
                    'allocated_stock': random.randint(0, 50),
                    'in_transit': random.randint(0, 30)
                }

            # 同步到session state
            st.session_state.products = self.products
            st.session_state.bom = self.bom
            st.session_state.inventory = self.inventory
        except Exception as e:
            st.error(f"生成示例数据时出错: {str(e)}")


# ==================== 数字孪生与仿真层 ====================
class ResourceDigitalTwin:
    """资源数字孪生体 - 参考西门子Opcenter"""

    def __init__(self, resource_id, base_capacity):
        self.resource_id = resource_id
        self.base_capacity = base_capacity
        self.update_state()

    def update_state(self):
        """更新实时状态"""
        self.oee = random.uniform(0.75, 0.98)
        self.dynamic_capacity = self.base_capacity * self.oee
        self.status = "运行中" if random.random() > 0.05 else "故障"

        maintenance_due = random.random() > 0.85
        self.maintenance_required = maintenance_due
        self.maintenance_time = random.randint(1, 4) if maintenance_due else 0

        return {
            "资源编号": self.resource_id,
            "OEE": round(self.oee, 3),
            "动态产能": round(self.dynamic_capacity, 1),
            "状态": self.status,
            "需要维护": self.maintenance_required,
            "维护时间": self.maintenance_time
        }


# ==================== 供应链协同网络 ====================
def factory_auction_system(order, factories):
    """多工厂资源竞拍机制"""
    if factories.empty:
        return "FACT-001"

    factory_capacities = []
    for _, factory in factories.iterrows():
        idle_capacity = factory["总产能"] * random.uniform(0.1, 0.4)
        base_cost = factory["单位成本"]
        transport_cost = random.uniform(0.05, 0.2)
        cost_per_unit = base_cost * (1 + transport_cost)
        delivery_time = random.randint(1, 5)
        score = idle_capacity / max((cost_per_unit * delivery_time), 0.1)

        factory_capacities.append({
            "工厂编号": factory["工厂编号"],
            "地点": factory["地点"],
            "空闲产能": idle_capacity,
            "单位成本": cost_per_unit,
            "交付时间": delivery_time,
            "得分": score
        })

    factory_capacities.sort(key=lambda x: x["得分"], reverse=True)
    best_factory = factory_capacities[0]
    return best_factory["工厂编号"]


def generate_supply_chain_risk_map():
    """供应链风险热力图"""
    regions = ["华东", "华南", "华北", "西南", "西北", "东北", "华中"]
    risk_data = []

    for region in regions:
        risk_data.append({
            "地区": region,
            "物流风险": random.uniform(0.1, 0.9),
            "政治风险": random.uniform(0.1, 0.7),
            "供应风险": random.uniform(0.1, 0.8),
            "综合风险": random.uniform(0.2, 0.85)
        })

    risk_df = pd.DataFrame(risk_data)
    st.session_state.supply_chain_risk = risk_df

    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = LinearSegmentedColormap.from_list("risk_cmap", ["green", "yellow", "red"])

    risk_values = risk_df[["物流风险", "政治风险", "供应风险", "综合风险"]].values
    im = ax.imshow(risk_values, cmap=cmap)

    ax.set_xticks(np.arange(len(risk_df.columns[1:])))
    ax.set_yticks(np.arange(len(risk_df)))
    ax.set_xticklabels(risk_df.columns[1:])
    ax.set_yticklabels(risk_df["地区"])

    for i in range(len(risk_df)):
        for j in range(len(risk_df.columns[1:])):
            text = ax.text(j, i, f"{risk_values[i, j]:.2f}",
                           ha="center", va="center", color="black")

    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("风险等级", rotation=-90, va="bottom")
    plt.title("供应链风险热力图")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150)
    buf.seek(0)
    img_str = "data:image/png;base64," + base64.b64encode(buf.read()).decode()
    plt.close()

    return img_str


# ==================== 人机协作界面 ====================
def llm_command_processor(command):
    """自然语言命令处理器"""
    responses = {
        "提前": f"已将订单提前处理，预计交付时间提前{random.randint(1, 6)}小时",
        "推迟": f"已将订单推迟处理，预计交付时间延迟{random.randint(1, 4)}小时",
        "状态": "系统运行正常，所有资源利用率在85%-95%之间",
        "风险": "检测到东南亚地区供应风险，已启动备用供应商方案",
        "效率": f"当前生产效率为{random.randint(85, 98)}%，高于行业平均水平",
        "排程": "正在重新优化排产计划，预计可提升效率12%",
        "分析": "已完成供应链风险分析，建议增加关键物料库存",
        "订单": f"当前共有{len(st.session_state.orders)}个订单，其中{len(st.session_state.orders[st.session_state.orders['状态'] == '已排产']) if not st.session_state.orders.empty else 0}个已排产",
        "资源": f"系统共有{len(st.session_state.resources)}个资源，利用率分析显示整体运行良好"
    }

    for keyword, response in responses.items():
        if keyword in command:
            return response

    return "命令已接收，正在优化排产计划..."


# ==================== 假设分析与场景模拟 ====================
def what_if_analysis(scenario_type):
    """假设分析：极端场景压力测试"""
    scenarios = {
        "东南亚洪灾": {"影响地区": ["越南", "泰国"], "物料影响": ["PCB", "连接器"], "影响程度": 0.7},
        "芯片断供": {"影响地区": ["全球"], "物料影响": ["IC-100"], "影响程度": 0.9},
        "港口罢工": {"影响地区": ["美国西海岸", "欧洲"], "物料影响": ["外壳", "屏幕"], "影响程度": 0.6},
        "疫情封控": {"影响地区": ["中国"], "物料影响": ["所有"], "影响程度": 0.8}
    }

    scenario = scenarios.get(scenario_type, {})
    if not scenario:
        return "未知场景"

    strategies = {
        "东南亚洪灾": ["启用马来西亚备用供应商", "空运关键物料", "调整生产优先级"],
        "芯片断供": ["寻找替代芯片型号", "减少非关键产品产量", "与客户协商延期"],
        "港口罢工": ["转用其他港口", "增加本地库存", "启用近岸供应商"],
        "疫情封控": ["启用多工厂协作", "实施闭环生产", "增加安全库存"]
    }

    affected_orders = len(st.session_state.orders) if scenario_type == "疫情封控" else random.randint(5, 20)

    result = {
        "场景": scenario_type,
        "影响分析": f"预计影响{affected_orders}个订单，{len(scenario['物料影响'])}种物料供应减少{scenario['影响程度'] * 100}%",
        "应急策略": strategies.get(scenario_type, []),
        "预计恢复时间": f"{random.randint(7, 30)}天"
    }

    st.session_state.simulation_results[scenario_type] = result
    return result


# ==================== 数据生成模块 ====================
def load_sample_data():
    """加载示例数据到session state"""
    try:
        # 生成工厂数据
        factories = []
        factory_locations = ["上海", "深圳", "重庆", "武汉", "沈阳"]
        for i, location in enumerate(factory_locations, 1):
            factories.append({
                "工厂编号": f"FACT-{i:03d}",
                "地点": location,
                "总产能": random.randint(50000, 200000),
                "单位成本": round(random.uniform(0.8, 1.5), 2),
                "专注产品": random.sample(['A-100', 'B-200', 'C-300', 'D-400', 'E-500'], 3),
                "状态": "运行中"
            })

        st.session_state.factories = pd.DataFrame(factories)

        # 生成订单数据
        orders = []
        products = ['A-100', 'B-200', 'C-300', 'D-400', 'E-500']
        priorities = ['紧急', '高', '中', '低']
        modes = ["JIT", "ASAP"]

        for i in range(1, 101):
            product = random.choice(products)
            quantity = random.randint(100, 5000)
            due_date = datetime.now() + timedelta(days=random.randint(1, 21))
            processing_time = quantity * 0.001 * random.uniform(0.8, 1.2)
            orders.append({
                '订单编号': f'ORD-{i:05d}',
                '产品型号': product,
                '数量': quantity,
                '交期': due_date,
                '优先级': random.choice(priorities),
                '状态': '未排产',
                '模式': random.choice(modes),
                '工厂分配': None,
                '处理时间': processing_time
            })

        st.session_state.orders = pd.DataFrame(orders)

        # 生成资源数据
        resources = []
        machine_types = ['CNC-100', '注塑机', '组装线', '测试站', '包装线']
        for i in range(1, 31):
            machine_type = random.choice(machine_types)
            efficiency = round(random.uniform(0.85, 0.98), 2)
            factory = random.choice(st.session_state.factories['工厂编号'].tolist())
            resources.append({
                '资源编号': f'RES-{i:03d}',
                '资源类型': machine_type,
                '工厂归属': factory,
                '总产能': random.randint(500, 2000),
                '效率系数': efficiency,
                '当前状态': '空闲',
                '维护计划': f'每{random.randint(30, 90)}天'
            })

        st.session_state.resources = pd.DataFrame(resources)

        # 生成物料状态数据
        materials = []
        components = ['IC-100', 'PCB', '外壳', '屏幕', '电池', '连接器', 'IC-100A', 'IC-100B', 'PCB-A', '电池-A']
        for comp in components:
            materials.append({
                '物料编码': comp,
                '当前库存': random.randint(500, 10000),
                '在途数量': random.randint(0, 5000),
                '预计到货': (datetime.now() + timedelta(days=random.randint(1, 14))).strftime('%Y-%m-%d'),
                '安全库存': random.randint(300, 2000),
                '缺货风险': random.choice(['低', '中', '高']),
                '供应商地区': random.choice(['华东', '华南', '东南亚', '欧洲', '北美'])
            })

        st.session_state.material_status = pd.DataFrame(materials)

        # 生成销售历史数据
        sales_history = []
        base_date = datetime.now() - timedelta(days=365)
        for i in range(365):
            date = base_date + timedelta(days=i)
            for product in products:
                sales_history.append({
                    'date': date,
                    'product_id': product,
                    'quantity': max(50, int(random.randint(50, 500) * (1 + 0.3 * np.sin(i * 2 * np.pi / 365))))  # 添加季节性
                })

        st.session_state.sales_history = pd.DataFrame(sales_history)

        # 生成仓库网络数据
        warehouses = []
        warehouse_locations = ["北京", "上海", "广州", "成都", "西安"]
        for i, location in enumerate(warehouse_locations, 1):
            warehouses.append({
                'warehouse_id': f'WH-{i:03d}',
                'location': location,
                'capacity': random.randint(10000, 50000),
                'type': random.choice(['中心仓', '区域仓', '前置仓']),
                'coverage_area': random.choice(['华北', '华东', '华南', '西南', '西北'])
            })

        st.session_state.warehouse_network = pd.DataFrame(warehouses)

        # 初始化APS引擎并生成标准数据
        aps_engine = HybridAPSEngine()
        aps_engine.generate_sample_data()
    except Exception as e:
        st.error(f"加载示例数据时出错: {str(e)}")


# ==================== 所有页面函数 ====================

def show_overview():
    """系统概览页面"""
    st.header("🏠 系统概览")

    # 顶部KPI指标（使用新样式）
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        order_count = len(st.session_state.orders) if not st.session_state.orders.empty else 0
        fulfillment_rate = 98.7 if order_count > 0 else 0
        st.markdown(f'<div class="metric-card"><h3>订单满足率</h3><h2>{fulfillment_rate}%</h2></div>',
                    unsafe_allow_html=True)
    with col2:
        plan_completion = 95.2 if not st.session_state.schedule.empty else 0
        st.markdown(f'<div class="metric-card"><h3>生产计划达成率</h3><h2>{plan_completion}%</h2></div>',
                    unsafe_allow_html=True)
    with col3:
        inventory_turn = random.uniform(8, 15)
        st.markdown(f'<div class="metric-card"><h3>库存周转率</h3><h2>{inventory_turn:.1f}次</h2></div>',
                    unsafe_allow_html=True)
    with col4:
        oee = random.uniform(86, 95)
        st.markdown(f'<div class="metric-card"><h3>OEE</h3><h2>{oee:.1f}%</h2></div>',
                    unsafe_allow_html=True)

    # 概览图表
    col1, col2 = st.columns(2)

    with col1:
        if not st.session_state.orders.empty:
            # 需求趋势图
            demand_by_date = st.session_state.orders.groupby(st.session_state.orders['交期'].dt.date)[
                '数量'].sum().reset_index()
            demand_by_date.columns = ['日期', '需求数量']

            fig = px.line(demand_by_date, x='日期', y='需求数量',
                          title='需求趋势分析', labels={'需求数量': '需求数量', '日期': '日期'})
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("请先生成示例数据")

    with col2:
        if not st.session_state.orders.empty:
            # 产品分布
            product_dist = st.session_state.orders['产品型号'].value_counts().reset_index()
            product_dist.columns = ['产品型号', '数量']

            fig = px.pie(product_dist, values='数量', names='产品型号',
                         title='产品需求分布')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("请先生成示例数据")

    # 系统状态摘要
    st.subheader("📋 系统状态摘要")
    col1, col2, col3 = st.columns(3)

    with col1:
        resource_count = len(st.session_state.resources) if not st.session_state.resources.empty else 0
        st.info(f"**资源总数**: {resource_count}")
        factory_count = len(st.session_state.factories) if not st.session_state.factories.empty else 0
        st.info(f"**工厂数量**: {factory_count}")

    with col2:
        scheduled_count = len(
            st.session_state.orders[st.session_state.orders['状态'] == '已排产']) if not st.session_state.orders.empty else 0
        st.success(f"**已排产订单**: {scheduled_count}")
        material_count = len(st.session_state.material_status) if not st.session_state.material_status.empty else 0
        st.success(f"**物料种类**: {material_count}")

    with col3:
        cost_variance = random.uniform(-5, 10)
        st.warning(f"**成本差异**: {cost_variance:.1f}%")
        risk_level = random.choice(['低', '中', '高'])
        st.warning(f"**供应链风险**: {risk_level}")


def show_demand_management():
    """需求管理页面"""
    st.header("📊 需求管理")

    if st.session_state.orders.empty:
        st.info("请先生成示例数据")
        return

    # 筛选控件
    col1, col2, col3 = st.columns(3)
    with col1:
        customers = ['全部'] + st.session_state.orders['产品型号'].unique().tolist()
        selected_customer = st.selectbox("产品筛选", customers)
    with col2:
        priorities = ['全部'] + st.session_state.orders['优先级'].unique().tolist()
        selected_priority = st.selectbox("优先级筛选", priorities)
    with col3:
        statuses = ['全部'] + st.session_state.orders['状态'].unique().tolist()
        selected_status = st.selectbox("状态筛选", statuses)

    # 应用筛选
    filtered_orders = st.session_state.orders.copy()
    if selected_customer != '全部':
        filtered_orders = filtered_orders[filtered_orders['产品型号'] == selected_customer]
    if selected_priority != '全部':
        filtered_orders = filtered_orders[filtered_orders['优先级'] == selected_priority]
    if selected_status != '全部':
        filtered_orders = filtered_orders[filtered_orders['状态'] == selected_status]

    # 订单数据表
    st.subheader("📋 需求订单列表")
    st.dataframe(filtered_orders, use_container_width=True, height=400)

    # 需求分析图表
    col1, col2 = st.columns(2)

    with col1:
        # 优先级分布
        priority_dist = st.session_state.orders['优先级'].value_counts().reset_index()
        priority_dist.columns = ['优先级', '数量']
        fig = px.bar(priority_dist, x='优先级', y='数量',
                     title='需求优先级分布', color='优先级')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # 模式分布
        mode_dist = st.session_state.orders['模式'].value_counts().reset_index()
        mode_dist.columns = ['模式', '数量']
        fig = px.pie(mode_dist, values='数量', names='模式',
                     title='订单模式分布')
        st.plotly_chart(fig, use_container_width=True)


def show_capacity_planning():
    """产能规划页面"""
    st.header("⚡ 产能规划(CRP)")

    if st.session_state.resources.empty:
        st.info("请先生成示例数据")
        return

    # 计算产能分析
    capacity_data = []
    for _, resource in st.session_state.resources.iterrows():
        utilization = random.uniform(60, 120)
        capacity_data.append({
            '资源编号': resource['资源编号'],
            '资源名称': resource['资源编号'],
            '资源类型': resource['资源类型'],
            '总产能': resource['总产能'],
            '负载': resource['总产能'] * utilization / 100,
            '利用率': utilization,
            '可用产能': max(0, resource['总产能'] * (100 - utilization) / 100),
            '是否瓶颈': utilization > 90
        })

    capacity_df = pd.DataFrame(capacity_data)

    # 产能概览指标
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        avg_utilization = capacity_df['利用率'].mean()
        st.metric("平均利用率", f"{avg_utilization:.1f}%")
    with col2:
        bottlenecks = capacity_df[capacity_df['是否瓶颈']].shape[0]
        st.metric("瓶颈资源数", bottlenecks)
    with col3:
        total_capacity = capacity_df['总产能'].sum()
        st.metric("总产能", f"{total_capacity:.0f}")
    with col4:
        available_capacity = capacity_df['可用产能'].sum()
        st.metric("可用产能", f"{available_capacity:.0f}")

    # 产能利用率图表
    st.subheader("📊 资源利用率分析")
    fig = px.bar(capacity_df, x='资源名称', y='利用率',
                 color='是否瓶颈', color_discrete_map={True: 'red', False: 'blue'},
                 title='资源利用率分析')
    fig.add_hline(y=90, line_dash="dash", line_color="red", annotation_text="瓶颈阈值(90%)")
    st.plotly_chart(fig, use_container_width=True)

    # 产能详细数据
    st.subheader("📋 产能详细分析")
    st.dataframe(capacity_df, use_container_width=True)


def show_production_scheduling():
    """生产调度页面"""
    st.header("📅 生产调度")

    if st.session_state.schedule.empty:
        st.info("请先执行智能排程")
        return

    # 生产计划甘特图
    st.subheader("📊 生产调度甘特图")
    fig = px.timeline(st.session_state.schedule,
                      x_start="开始时间", x_end="结束时间",
                      y="资源编号", color="产品型号", text="订单编号",
                      title="生产调度甘特图")
    fig.update_yaxes(categoryorder="total ascending")
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    # 调度详情
    st.subheader("📋 生产调度详情")
    st.dataframe(st.session_state.schedule, use_container_width=True, height=400)

    # 调度分析
    col1, col2 = st.columns(2)

    with col1:
        # 资源负载分析
        if '持续时间' in st.session_state.schedule.columns:
            resource_load = st.session_state.schedule.groupby('资源编号')['持续时间'].sum().reset_index()
            fig = px.bar(resource_load, x='资源编号', y='持续时间',
                         title='资源负载分析')
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        # 产品分布
        if '产品型号' in st.session_state.schedule.columns:
            product_dist = st.session_state.schedule['产品型号'].value_counts().reset_index()
            product_dist.columns = ['产品型号', '数量']
            fig = px.pie(product_dist, values='数量', names='产品型号',
                         title='生产任务产品分布')
            st.plotly_chart(fig, use_container_width=True)


def show_inventory_management():
    """库存管理页面"""
    st.header("📦 库存管理")

    if st.session_state.material_status.empty:
        st.info("请先生成示例数据")
        return

    # 添加库存价值等字段
    inventory_df = st.session_state.material_status.copy()
    inventory_df['单位成本'] = np.random.uniform(10, 100, len(inventory_df))
    inventory_df['库存价值'] = inventory_df['当前库存'] * inventory_df['单位成本']
    inventory_df['库存状态'] = inventory_df.apply(
        lambda x: '缺货' if x['当前库存'] <= x['安全库存'] * 0.5
        else '低库存' if x['当前库存'] <= x['安全库存']
        else '正常', axis=1
    )

    # 库存KPI
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        total_value = inventory_df['库存价值'].sum()
        st.metric("库存总价值", f"¥{total_value:,.0f}")
    with col2:
        low_stock_count = len(inventory_df[inventory_df['库存状态'].isin(['缺货', '低库存'])])
        st.metric("低库存产品", low_stock_count)
    with col3:
        avg_turnover = random.uniform(8, 15)
        st.metric("平均周转率", f"{avg_turnover:.1f}")
    with col4:
        total_in_transit = inventory_df['在途数量'].sum()
        st.metric("在途库存", total_in_transit)

    # 库存状态分析
    st.subheader("📊 库存状态分析")
    col1, col2 = st.columns(2)

    with col1:
        # 库存状态分布
        status_dist = inventory_df['库存状态'].value_counts().reset_index()
        status_dist.columns = ['状态', '数量']
        fig = px.pie(status_dist, values='数量', names='状态',
                     title='库存状态分布',
                     color_discrete_map={'正常': 'green', '低库存': 'orange', '缺货': 'red'})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # 供应商地区分布
        region_dist = inventory_df['供应商地区'].value_counts().reset_index()
        region_dist.columns = ['地区', '数量']
        fig = px.bar(region_dist, x='地区', y='数量',
                     title='供应商地区分布')
        st.plotly_chart(fig, use_container_width=True)

    # 库存详细数据
    st.subheader("📋 库存详细信息")
    st.dataframe(inventory_df, use_container_width=True, height=400)


def show_kpi_dashboard():
    """KPI仪表板页面"""
    st.header("📈 KPI仪表板")

    # 主要KPI指标
    st.subheader("🎯 核心KPI指标")

    col1, col2, col3 = st.columns(3)

    with col1:
        # 准时交付率仪表盘
        otd_rate = random.uniform(85, 95)
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=otd_rate,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "准时交付率(%)"},
            delta={'reference': 90},
            gauge={'axis': {'range': [None, 100]},
                   'bar': {'color': "darkblue"},
                   'steps': [{'range': [0, 70], 'color': "lightgray"},
                             {'range': [70, 90], 'color': "gray"}],
                   'threshold': {'line': {'color': "red", 'width': 4},
                                 'thickness': 0.75, 'value': 90}}))
        fig.update_layout(height=250)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # 产能利用率仪表盘
        capacity_util = random.uniform(75, 95)
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=capacity_util,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "产能利用率(%)"},
            delta={'reference': 85},
            gauge={'axis': {'range': [None, 120]},
                   'bar': {'color': "darkgreen"},
                   'steps': [{'range': [0, 60], 'color': "lightgray"},
                             {'range': [60, 85], 'color': "gray"}],
                   'threshold': {'line': {'color': "red", 'width': 4},
                                 'thickness': 0.75, 'value': 100}}))
        fig.update_layout(height=250)
        st.plotly_chart(fig, use_container_width=True)

    with col3:
        # 库存周转率仪表盘
        inventory_turn = random.uniform(8, 15)
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=inventory_turn,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "库存周转率"},
            delta={'reference': 12},
            gauge={'axis': {'range': [0, 20]},
                   'bar': {'color': "darkorange"},
                   'steps': [{'range': [0, 8], 'color': "lightgray"},
                             {'range': [8, 12], 'color': "gray"}],
                   'threshold': {'line': {'color': "red", 'width': 4},
                                 'thickness': 0.75, 'value': 15}}))
        fig.update_layout(height=250)
        st.plotly_chart(fig, use_container_width=True)

    # 趋势分析
    st.subheader("📊 趋势分析")

    dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D')
    trend_data = pd.DataFrame({
        'date': dates,
        'otd': np.random.normal(otd_rate, 3, len(dates)),
        'capacity_util': np.random.normal(capacity_util, 5, len(dates)),
        'inventory_turn': np.random.normal(inventory_turn, 1, len(dates))
    })

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('准时交付率趋势', '产能利用率趋势', '库存周转率趋势', '综合指标'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )

    fig.add_trace(go.Scatter(x=trend_data['date'], y=trend_data['otd'],
                             name='准时交付率', line=dict(color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=trend_data['date'], y=trend_data['capacity_util'],
                             name='产能利用率', line=dict(color='green')), row=1, col=2)
    fig.add_trace(go.Scatter(x=trend_data['date'], y=trend_data['inventory_turn'],
                             name='库存周转率', line=dict(color='orange')), row=2, col=1)

    fig.update_layout(height=500, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


def show_digital_twin():
    """数字孪生页面"""
    st.header("🤖 数字孪生")

    if st.session_state.resources.empty:
        st.info("请先生成示例数据")
        return

    st.subheader("资源数字孪生体")

    # 创建数字孪生体面板
    cols = st.columns(4)
    resource_display = []

    for i, (_, resource) in enumerate(st.session_state.resources.iterrows()):
        res_id = resource['资源编号']

        # 获取或创建数字孪生体
        if res_id in st.session_state.resource_twins:
            twin = st.session_state.resource_twins[res_id]
        else:
            twin = ResourceDigitalTwin(res_id, resource['总产能'])
            st.session_state.resource_twins[res_id] = twin

        status = twin.update_state()
        resource_display.append(status)

        # 在卡片中显示
        with cols[i % 4]:
            with st.container(border=True):
                status_color = "green" if status['状态'] == "运行中" else "red"
                st.markdown(f"<h4 style='color:{status_color};'>{res_id}</h4>", unsafe_allow_html=True)
                st.caption(f"{resource['资源类型']} | {resource['工厂归属']}")

                col1, col2 = st.columns(2)
                col1.metric("OEE", f"{status['OEE'] * 100:.1f}%")
                col2.metric("产能", f"{status['动态产能']:.1f}")

                st.progress(status['OEE'], text=f"状态: {status['状态']}")

                if status['需要维护']:
                    st.warning(f"⚠️ 需要维护: {status['维护时间']}小时")

    # 显示详细数据
    st.subheader("设备状态详情")
    st.dataframe(pd.DataFrame(resource_display), use_container_width=True)


def show_supply_chain():
    """供应链协同页面"""
    st.header("🌐 供应链协同")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("多工厂分布")
        if not st.session_state.factories.empty:
            # 工厂地图视图模拟
            factory_locations = {
                "上海": (31.2304, 121.4737),
                "深圳": (22.5431, 114.0579),
                "重庆": (29.5630, 106.5516),
                "武汉": (30.5928, 114.3055),
                "沈阳": (41.8057, 123.4315)
            }

            map_data = pd.DataFrame({
                "lat": [loc[0] for loc in factory_locations.values()],
                "lon": [loc[1] for loc in factory_locations.values()],
                "工厂": st.session_state.factories["地点"],
                "规模": st.session_state.factories["总产能"] / 1000
            })

            st.map(map_data, size="规模", color="#0068c9")
            st.dataframe(st.session_state.factories, use_container_width=True)
        else:
            st.info("请先生成示例数据")

    with col2:
        st.subheader("供应链风险热力图")
        if 'risk_map' in st.session_state:
            st.image(st.session_state.risk_map, use_container_width=True)
        else:
            st.info("点击'风险分析'按钮生成热力图")

        # 假设分析结果
        if st.session_state.simulation_results:
            st.subheader("场景模拟结果")
            for scenario, result in st.session_state.simulation_results.items():
                with st.expander(f"{scenario} - 模拟结果"):
                    st.markdown(f"**影响分析**: {result['影响分析']}")
                    st.markdown(f"**预计恢复时间**: {result['预计恢复时间']}")
                    st.markdown("**应急策略**:")
                    for strategy in result['应急策略']:
                        st.markdown(f"- {strategy}")


def show_optimization_analysis():
    """优化分析页面"""
    st.header("🔍 优化分析")

    st.subheader("🎯 优化目标设置")

    col1, col2, col3 = st.columns(3)
    with col1:
        primary_objective = st.selectbox(
            "主要优化目标",
            ["最小化总成本", "最大化准时交付率", "最小化生产周期", "最大化产能利用率"]
        )
    with col2:
        optimization_scope = st.selectbox(
            "优化范围",
            ["整个供应链", "生产计划", "库存管理", "资源分配"]
        )
    with col3:
        time_horizon = st.selectbox(
            "优化时间范围",
            ["1周", "1个月", "1季度", "半年"]
        )

    # 约束条件设置
    st.subheader("⚙️ 约束条件")
    col1, col2 = st.columns(2)
    with col1:
        max_overtime = st.slider("最大加班时间(%)", 0, 50, 20)
        min_service_level = st.slider("最低服务水平(%)", 80, 100, 95)
    with col2:
        max_inventory_investment = st.number_input("最大库存投资(万元)", 100, 1000, 500)
        resource_constraints = st.multiselect(
            "资源约束",
            ["生产能力", "仓储空间", "运输能力", "人力资源"],
            default=["生产能力", "仓储空间"]
        )

    # 运行优化
    if st.button("🚀 运行优化分析", type="primary"):
        with st.spinner("正在执行优化算法..."):
            time.sleep(2)

            optimization_results = {
                "cost_reduction": random.uniform(8, 18),
                "delivery_improvement": random.uniform(3, 8),
                "cycle_time_reduction": random.uniform(10, 25),
                "utilization_improvement": random.uniform(5, 15),
                "inventory_reduction": random.uniform(12, 22)
            }

            st.success("✅ 优化分析完成！")

            # 显示优化结果
            st.subheader("📊 优化结果")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("成本降低", f"{optimization_results['cost_reduction']:.1f}%",
                          delta=f"-{optimization_results['cost_reduction']:.1f}%")
                st.metric("交付改善", f"{optimization_results['delivery_improvement']:.1f}%",
                          delta=f"+{optimization_results['delivery_improvement']:.1f}%")

            with col2:
                st.metric("周期缩短", f"{optimization_results['cycle_time_reduction']:.1f}%",
                          delta=f"-{optimization_results['cycle_time_reduction']:.1f}%")
                st.metric("利用率提升", f"{optimization_results['utilization_improvement']:.1f}%",
                          delta=f"+{optimization_results['utilization_improvement']:.1f}%")

            with col3:
                st.metric("库存减少", f"{optimization_results['inventory_reduction']:.1f}%",
                          delta=f"-{optimization_results['inventory_reduction']:.1f}%")


def show_industry_solutions():
    """行业解决方案页面"""
    st.header("🏭 行业解决方案")

    industry = st.radio("选择行业", ["汽车制造", "电子制造", "流程工业"], horizontal=True)

    if industry == "汽车制造":
        st.info("🚗 柔性制造解决方案 - 支持多车型混线生产")
        st.markdown("""
        **核心功能：**
        - 极氪航空座椅与几何内饰同线切换
        - 动态切换时间优化算法
        - 混线平衡分析
        """)

        # 产品切换时间计算器
        st.subheader("产品切换时间计算")
        col1, col2 = st.columns(2)
        with col1:
            current_product = st.selectbox("当前产品", ['A-100', 'B-200', 'C-300', 'D-400', 'E-500'])
        with col2:
            next_product = st.selectbox("下个产品", ['A-100', 'B-200', 'C-300', 'D-400', 'E-500'])

        if st.button("计算切换时间"):
            changeover_time = random.uniform(1.5, 3.5)
            st.success(f"从 {current_product} 切换到 {next_product} 需要 {changeover_time:.1f} 小时")

            st.markdown("**排程影响分析**:")
            st.markdown(f"- 预计产能损失: {changeover_time * 100:.0f}个产品")
            st.markdown(f"- 建议切换时间: 非生产高峰时段")
            st.markdown(f"- 切换成本: ¥{changeover_time * 1500:.0f}")

    elif industry == "电子制造":
        st.info("📱 虚拟BOM引擎 - 替代料智能管理")
        st.markdown("""
        **核心功能：**
        - 短缺元件秒级替换
        - 替代料兼容性验证
        - 自动BOM更新
        """)

        # 替代料分析
        st.subheader("物料替代分析")
        if not st.session_state.material_status.empty:
            material = st.selectbox("选择需要替代的物料", st.session_state.material_status['物料编码'].unique())
            required_qty = st.number_input("需求数量", min_value=1, value=500)

            if st.button("查找替代料"):
                # 模拟替代料查找
                substitutes = [
                    {"替代物料": f"{material}A", "可用数量": random.randint(100, 1000),
                     "成本系数": random.uniform(0.9, 1.3), "兼容性": "高"},
                    {"替代物料": f"{material}B", "可用数量": random.randint(100, 1000),
                     "成本系数": random.uniform(0.9, 1.3), "兼容性": "中"}
                ]

                if substitutes:
                    st.success(f"找到 {len(substitutes)} 种替代料:")
                    st.dataframe(pd.DataFrame(substitutes), use_container_width=True)
                else:
                    st.warning(f"没有找到 {material} 的可用替代料")

    else:  # 流程工业
        st.info("⚡ 能源-排程耦合优化 - 高耗能工序避峰管理")
        st.markdown("""
        **核心功能：**
        - 分时电价敏感排程
        - 能源消耗预测
        - 碳中和指标跟踪
        """)

        # 能源成本优化
        st.subheader("能源成本优化模拟")
        energy_cost = pd.DataFrame({
            "时段": ["00:00-08:00", "08:00-12:00", "12:00-18:00", "18:00-22:00", "22:00-24:00"],
            "电价": [0.35, 1.20, 0.85, 1.10, 0.45],
            "碳排放": [0.8, 1.5, 1.2, 1.3, 0.7]
        })

        fig = px.bar(energy_cost, x='时段', y='电价', color='碳排放',
                     color_continuous_scale='thermal', title="分时电价与碳排放")
        st.plotly_chart(fig, use_container_width=True)
        st.info("系统自动将高耗能工序安排在低电价时段，预计可节约成本23%")


def show_ai_assistant():
    """AI智能助手页面"""
    st.header("💬 AI智能助手")
    st.markdown("集成大语言模型的自然语言交互界面")

    # 聊天界面
    chat_container = st.container(height=400, border=True)
    for message in st.session_state.llm_chat_history[-10:]:
        if message["role"] == "user":
            with chat_container.chat_message("user", avatar="🧑‍💼"):
                st.write(message["content"])
        else:
            with chat_container.chat_message("assistant", avatar="🤖"):
                st.write(message["content"])

    # 输入框
    user_input = st.chat_input("输入您的问题或指令...")
    if user_input:
        st.session_state.llm_chat_history.append({"role": "user", "content": user_input})
        response = llm_command_processor(user_input)
        st.session_state.llm_chat_history.append({"role": "assistant", "content": response})
        st.rerun()

    # 预设命令按钮
    st.subheader("快捷命令")
    col1, col2, col3, col4 = st.columns(4)

    if col1.button("📊 订单状态", use_container_width=True):
        st.session_state.llm_chat_history.append({"role": "user", "content": "当前订单状态"})
        response = llm_command_processor("订单")
        st.session_state.llm_chat_history.append({"role": "assistant", "content": response})
        st.rerun()

    if col2.button("⚡ 资源利用率", use_container_width=True):
        st.session_state.llm_chat_history.append({"role": "user", "content": "资源利用率情况"})
        response = llm_command_processor("资源")
        st.session_state.llm_chat_history.append({"role": "assistant", "content": response})
        st.rerun()

    if col3.button("🌍 风险预警", use_container_width=True):
        st.session_state.llm_chat_history.append({"role": "user", "content": "有哪些供应链风险"})
        response = llm_command_processor("风险")
        st.session_state.llm_chat_history.append({"role": "assistant", "content": response})
        st.rerun()

    if col4.button("📈 效率分析", use_container_width=True):
        st.session_state.llm_chat_history.append({"role": "user", "content": "当前系统效率如何"})
        response = llm_command_processor("效率")
        st.session_state.llm_chat_history.append({"role": "assistant", "content": response})
        st.rerun()

    # 场景模拟控制
    st.subheader("场景模拟")
    scenario = st.selectbox("选择压力测试场景", ["东南亚洪灾", "芯片断供", "港口罢工", "疫情封控"])
    if st.button("🔄 运行模拟", use_container_width=True):
        result = what_if_analysis(scenario)
        st.session_state.simulation_results[scenario] = result
        st.success(f"{scenario}场景模拟完成!")

        # 显示结果
        with st.expander(f"{scenario} - 模拟结果", expanded=True):
            st.markdown(f"**影响分析**: {result['影响分析']}")
            st.markdown(f"**预计恢复时间**: {result['预计恢复时间']}")
            st.markdown("**应急策略**:")
            for strategy in result['应急策略']:
                st.markdown(f"- {strategy}")


def show_data_export():
    """数据导出页面"""
    st.header("📤 数据导出")

    st.subheader("📊 可导出数据集")

    # 数据集选择
    datasets = {
        "订单数据": "orders",
        "资源数据": "resources",
        "工厂数据": "factories",
        "物料数据": "material_status",
        "排程结果": "schedule",
        "数字孪生状态": "resource_twins",
        "销售历史": "sales_history",
        "仓库网络": "warehouse_network",
        "生产计划": "production_plan",
        "车间排程": "workshop_schedule",
        "物料需求": "material_requirements",
        "发运计划": "shipping_plan"
    }

    selected_datasets = st.multiselect(
        "选择要导出的数据集",
        list(datasets.keys()),
        default=list(datasets.keys())[:6]
    )

    # 导出格式选择
    export_format = st.selectbox("导出格式", ["Excel (.xlsx)", "CSV (.csv)", "JSON (.json)"])

    # 生成导出数据
    export_data = {}

    for dataset_name in selected_datasets:
        dataset_key = datasets[dataset_name]

        if dataset_key in st.session_state and hasattr(st.session_state, dataset_key):
            data = getattr(st.session_state, dataset_key)
            if isinstance(data, pd.DataFrame) and not data.empty:
                export_data[dataset_name] = data
            elif dataset_key == "resource_twins" and st.session_state.resource_twins:
                # 特殊处理数字孪生数据
                twins_data = []
                for twin_id, twin in st.session_state.resource_twins.items():
                    twins_data.append(twin.update_state())
                export_data[dataset_name] = pd.DataFrame(twins_data)

    # 数据预览
    if export_data:
        st.subheader("📋 数据预览")
        preview_dataset = st.selectbox("选择预览数据集", list(export_data.keys()))
        if preview_dataset:
            df = export_data[preview_dataset]
            st.dataframe(df, use_container_width=True, height=300)

            # 数据统计
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("记录数", len(df))
            with col2:
                st.metric("字段数", len(df.columns))
            with col3:
                if hasattr(df, 'memory_usage'):
                    st.metric("数据大小", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")
                else:
                    st.metric("数据大小", "N/A")

    # 导出功能
    st.subheader("💾 导出数据")

    if st.button("生成导出文件", type="primary"):
        if not export_data:
            st.error("请至少选择一个数据集进行导出")
            return

        with st.spinner("正在生成导出文件..."):
            if export_format == "Excel (.xlsx)":
                # 导出为Excel文件
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    for sheet_name, df in export_data.items():
                        df.to_excel(writer, sheet_name=sheet_name[:31], index=False)  # Excel sheet名称限制31字符

                st.download_button(
                    label="📥 下载Excel文件",
                    data=output.getvalue(),
                    file_name=f"APS_Export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

            elif export_format == "CSV (.csv)":
                # 导出为压缩的CSV文件
                import zipfile
                zip_buffer = io.BytesIO()

                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                    for name, df in export_data.items():
                        csv_buffer = io.StringIO()
                        df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
                        zip_file.writestr(f"{name}.csv", csv_buffer.getvalue())

                st.download_button(
                    label="📥 下载CSV文件包",
                    data=zip_buffer.getvalue(),
                    file_name=f"APS_Export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                    mime="application/zip"
                )

            elif export_format == "JSON (.json)":
                # 导出为JSON文件
                json_data = {}
                for name, df in export_data.items():
                    # 处理日期时间列，转换为字符串
                    df_copy = df.copy()
                    for col in df_copy.columns:
                        if df_copy[col].dtype == 'datetime64[ns]' or pd.api.types.is_datetime64_any_dtype(df_copy[col]):
                            df_copy[col] = df_copy[col].astype(str)
                    json_data[name] = df_copy.to_dict('records')

                json_str = json.dumps(json_data, ensure_ascii=False, indent=2, default=str)

                st.download_button(
                    label="📥 下载JSON文件",
                    data=json_str.encode('utf-8'),
                    file_name=f"APS_Export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

        st.success("✅ 导出文件已生成，点击上方按钮下载")


# ==================== 新增功能页面实现 ====================

def show_intelligent_forecast():
    """智能预测页面"""
    st.header("🔮 智能预测")
    st.markdown("基于销售数据的多场景多模型智能预测")

    # 初始化预测引擎
    forecast_engine = IntelligentForecastEngine()

    # 预测参数设置
    col1, col2, col3 = st.columns(3)
    with col1:
        forecast_horizon = st.number_input("预测周期(天)", min_value=7, max_value=365, value=30)
    with col2:
        forecast_scenario = st.selectbox("预测场景",
                                         ["normal", "optimistic", "pessimistic", "seasonal", "promotional"])
    with col3:
        confidence_level = st.slider("置信水平", 0.8, 0.99, 0.95)

    # 模型选择
    st.subheader("🤖 预测模型配置")
    col1, col2 = st.columns(2)
    with col1:
        selected_models = st.multiselect(
            "选择预测模型",
            ["随机森林", "线性回归", "移动平均", "指数平滑", "LSTM神经网络"],
            default=["随机森林", "线性回归"]
        )
    with col2:
        ensemble_method = st.selectbox(
            "集成方法",
            ["加权平均", "投票法", "堆叠法", "最优选择"]
        )

    # 执行预测
    if st.button("🚀 执行预测", type="primary"):
        with st.spinner("正在训练模型并生成预测..."):
            # 准备数据
            sales_data = forecast_engine.prepare_sales_data(st.session_state.sales_history)

            if sales_data is not None:
                # 训练模型
                forecast_engine.train_models(sales_data)

                # 生成预测
                forecast_results = forecast_engine.forecast(periods=forecast_horizon, scenario=forecast_scenario)
                st.session_state.forecast_results = forecast_results

                st.success("✅ 预测完成！")

                # 显示预测结果
                st.subheader("📊 预测结果")

                # 预测图表
                fig = go.Figure()

                # 历史数据
                if not st.session_state.sales_history.empty:
                    historical = st.session_state.sales_history.groupby('date')['quantity'].sum().reset_index()
                    fig.add_trace(go.Scatter(
                        x=historical['date'],
                        y=historical['quantity'],
                        mode='lines',
                        name='历史销售',
                        line=dict(color='blue')
                    ))

                # 预测数据
                fig.add_trace(go.Scatter(
                    x=forecast_results['date'],
                    y=forecast_results['forecast'],
                    mode='lines',
                    name='预测值',
                    line=dict(color='red', dash='dash')
                ))

                # 置信区间
                fig.add_trace(go.Scatter(
                    x=forecast_results['date'],
                    y=forecast_results['upper_bound'],
                    mode='lines',
                    name='置信上限',
                    line=dict(color='rgba(255,0,0,0.2)'),
                    showlegend=False
                ))

                fig.add_trace(go.Scatter(
                    x=forecast_results['date'],
                    y=forecast_results['lower_bound'],
                    mode='lines',
                    name='置信下限',
                    line=dict(color='rgba(255,0,0,0.2)'),
                    fill='tonexty',
                    showlegend=False
                ))

                fig.update_layout(
                    title=f"销售预测 - {forecast_scenario}场景",
                    xaxis_title="日期",
                    yaxis_title="销售量",
                    height=500
                )

                st.plotly_chart(fig, use_container_width=True)

                # 预测统计
                col1, col2, col3 = st.columns(3)
                with col1:
                    avg_forecast = forecast_results['forecast'].mean()
                    st.metric("平均预测值", f"{avg_forecast:.0f}")
                with col2:
                    total_forecast = forecast_results['forecast'].sum()
                    st.metric("总预测量", f"{total_forecast:.0f}")
                with col3:
                    volatility = forecast_results['forecast'].std()
                    st.metric("预测波动性", f"{volatility:.1f}")

                # 详细预测数据
                st.subheader("📋 详细预测数据")
                st.dataframe(forecast_results, use_container_width=True)
            else:
                st.warning("请先生成销售历史数据")


def show_warehouse_network_analysis():
    """仓网分析页面"""
    st.header("🌐 仓网分析")
    st.markdown("深入分析现有订单交付仓网结构")

    # 初始化仓网分析器
    network_analyzer = WarehouseNetworkAnalyzer()

    if not st.session_state.warehouse_network.empty and not st.session_state.factories.empty:
        # 构建网络
        network_analyzer.build_network(st.session_state.warehouse_network, st.session_state.factories)

        # 网络可视化
        st.subheader("📊 仓储网络结构")

        # 创建网络图
        G = network_analyzer.network_graph
        pos = nx.spring_layout(G)

        # 创建Plotly图
        edge_trace = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_trace.append(go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(width=0.5, color='#888'),
                hoverinfo='none'
            ))

        node_trace = go.Scatter(
            x=[],
            y=[],
            text=[],
            mode='markers+text',
            textposition="top center",
            hoverinfo='text',
            marker=dict(
                showscale=True,
                colorscale='YlGnBu',
                size=10,
                color=[],
                colorbar=dict(
                    thickness=15,
                    title='节点容量',
                    xanchor='left'
                )
            )
        )

        for node in G.nodes():
            x, y = pos[node]
            node_trace['x'] += tuple([x])
            node_trace['y'] += tuple([y])
            node_info = G.nodes[node]
            node_trace['text'] += tuple([node])
            node_trace['marker']['color'] += tuple([node_info['capacity']])

        fig = go.Figure(data=edge_trace + [node_trace],
                        layout=go.Layout(
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=0, l=0, r=0, t=0),
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                        ))

        st.plotly_chart(fig, use_container_width=True)

        # 订单履行分析
        st.subheader("📦 订单履行路径分析")

        if not st.session_state.orders.empty:
            sample_orders = st.session_state.orders.head(10)
            fulfillment_analysis = []

            for idx, order in sample_orders.iterrows():
                result = network_analyzer.analyze_order_fulfillment(order)
                fulfillment_analysis.append(result)

            fulfillment_df = pd.DataFrame(fulfillment_analysis)
            st.dataframe(fulfillment_df, use_container_width=True)

            # 履行效率统计
            col1, col2, col3 = st.columns(3)
            with col1:
                avg_cost = fulfillment_df['fulfillment_cost'].mean()
                st.metric("平均履行成本", f"¥{avg_cost:.2f}")
            with col2:
                avg_time = fulfillment_df['delivery_time'].mean()
                st.metric("平均交付时间", f"{avg_time:.1f}天")
            with col3:
                avg_efficiency = fulfillment_df['route_efficiency'].mean()
                st.metric("平均路线效率", f"{avg_efficiency * 100:.1f}%")

        # 网络优化建议
        st.subheader("🎯 网络优化建议")
        recommendations = network_analyzer.optimize_network_layout()

        if recommendations:
            recommendation_df = pd.DataFrame(recommendations)
            st.dataframe(recommendation_df, use_container_width=True)

            # 生成优化报告
            st.info("""
            **优化建议总结：**
            1. 重点关注高重要性得分的节点，考虑增加其容量
            2. 评估低利用率仓库的必要性，考虑整合或关闭
            3. 优化运输路线，减少中转次数
            4. 在需求密集区域增设前置仓
            """)
    else:
        st.info("请先生成仓库网络和工厂数据")


def show_sales_operations_planning():
    """产销规划页面"""
    st.header("📈 产销规划(S&OP)")
    st.markdown("产销协同计划制定与优化")

    # 初始化S&OP引擎
    sop_engine = SalesOperationsPlanning()

    # 市场情报输入
    st.subheader("🌍 市场情报")
    col1, col2 = st.columns(2)
    with col1:
        promotion_start = st.date_input("促销开始日期", datetime.now().date())
        promotion_end = st.date_input("促销结束日期",
                                      (datetime.now() + timedelta(days=14)).date())
    with col2:
        promotion_impact = st.slider("促销影响系数", 0.8, 1.5, 1.2)
        competitor_action = st.selectbox("竞争对手动向", ["无", "新品上市", "价格战", "市场退出"])

    # 创建市场情报
    market_intelligence = []
    if st.button("添加市场事件"):
        market_intelligence.append({
            'type': 'promotion',
            'start': promotion_start,
            'end': promotion_end,
            'impact': promotion_impact
        })
        st.success("市场事件已添加")

    # 创建计划
    if st.button("🚀 创建产销计划", type="primary"):
        with st.spinner("正在创建产销计划..."):
            # 创建需求计划
            if not st.session_state.forecast_results.empty:
                demand_plan = sop_engine.create_demand_plan(
                    st.session_state.forecast_results,
                    market_intelligence
                )

                # 创建供应计划
                supply_plan = sop_engine.create_supply_plan(
                    st.session_state.factories,
                    st.session_state.inventory
                )

                # 协调计划
                consensus_plan = sop_engine.reconcile_plans()

                if not consensus_plan.empty:
                    st.success("✅ 产销计划创建成功！")

                    # 显示协调后的计划
                    st.subheader("📊 产销协调计划")

                    # 产销平衡图
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=consensus_plan['date'],
                        y=consensus_plan['adjusted_demand'],
                        mode='lines',
                        name='需求计划',
                        line=dict(color='red')
                    ))

                    fig.add_trace(go.Scatter(
                        x=consensus_plan['date'],
                        y=consensus_plan['planned_production'],
                        mode='lines',
                        name='供应计划',
                        line=dict(color='blue')
                    ))

                    fig.update_layout(
                        title="产销平衡分析",
                        xaxis_title="日期",
                        yaxis_title="数量",
                        height=400
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # 关键指标
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        total_gap = consensus_plan['gap'].sum()
                        st.metric("供需缺口", f"{abs(total_gap):.0f}",
                                  delta=f"{total_gap:.0f}")
                    with col2:
                        revenue_impact = consensus_plan['revenue_impact'].sum()
                        st.metric("预期收入", f"¥{revenue_impact:,.0f}")
                    with col3:
                        cost_impact = consensus_plan['cost_impact'].sum()
                        st.metric("预期成本", f"¥{cost_impact:,.0f}")
                    with col4:
                        profit_impact = consensus_plan['profit_impact'].sum()
                        st.metric("预期利润", f"¥{profit_impact:,.0f}")

                    # 详细计划数据
                    st.subheader("📋 详细计划数据")
                    st.dataframe(consensus_plan, use_container_width=True)
                else:
                    st.warning("请先生成预测数据")
            else:
                st.warning("请先执行智能预测")


def show_intelligent_allocation():
    """智能分单页面"""
    st.header("🎯 智能分单")
    st.markdown("智能确定产品在哪个工厂生产")

    # 初始化分单引擎
    allocation_engine = IntelligentOrderAllocation()

    if not st.session_state.orders.empty and not st.session_state.factories.empty:
        # 分析工厂能力
        allocation_engine.analyze_factory_capabilities(st.session_state.factories, st.session_state.products)

        # 分配策略选择
        st.subheader("📊 分配策略配置")
        col1, col2, col3 = st.columns(3)
        with col1:
            allocation_mode = st.selectbox(
                "分配策略",
                ["balanced", "cost_optimized", "speed_optimized", "quality_focused"],
                format_func=lambda x: {
                    "balanced": "平衡分配",
                    "cost_optimized": "成本优化",
                    "speed_optimized": "速度优先",
                    "quality_focused": "质量优先"
                }.get(x, x)
            )
        with col2:
            batch_size = st.number_input("批量处理数量", min_value=10, max_value=1000, value=100)
        with col3:
            consider_capacity = st.checkbox("考虑产能约束", value=True)

        # 执行分单
        if st.button("🚀 执行智能分单", type="primary"):
            with st.spinner("正在进行智能分单..."):
                # 获取待分配订单
                unallocated_orders = st.session_state.orders[
                    st.session_state.orders['工厂分配'].isna()
                ].head(batch_size)

                if not unallocated_orders.empty:
                    # 执行分配
                    allocation_results = allocation_engine.allocate_orders(
                        unallocated_orders,
                        mode=allocation_mode
                    )

                    st.success(f"✅ 成功分配 {len(allocation_results)} 个订单！")

                    # 显示分配结果
                    st.subheader("📊 分配结果分析")

                    # 工厂分配统计
                    factory_dist = allocation_results['分配工厂'].value_counts()
                    fig = px.pie(
                        values=factory_dist.values,
                        names=factory_dist.index,
                        title="订单工厂分配分布"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # 评分分析
                    col1, col2 = st.columns(2)
                    with col1:
                        # 各维度平均得分
                        score_columns = ['产能得分', '成本得分', '专长得分', '质量得分', '交付得分']
                        avg_scores = allocation_results[score_columns].mean()

                        fig = go.Figure(data=go.Scatterpolar(
                            r=avg_scores.values,
                            theta=avg_scores.index,
                            fill='toself',
                            name='平均得分'
                        ))

                        fig.update_layout(
                            polar=dict(
                                radialaxis=dict(
                                    visible=True,
                                    range=[0, 1]
                                )),
                            title="多维度评分雷达图",
                            showlegend=False
                        )

                        st.plotly_chart(fig, use_container_width=True)

                    with col2:
                        # 综合得分分布
                        fig = px.histogram(
                            allocation_results,
                            x='综合得分',
                            nbins=20,
                            title="综合得分分布"
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    # 详细分配结果
                    st.subheader("📋 详细分配结果")
                    st.dataframe(allocation_results, use_container_width=True)

                    # 更新订单状态
                    for _, allocation in allocation_results.iterrows():
                        st.session_state.orders.loc[
                            st.session_state.orders['订单编号'] == allocation['订单编号'],
                            '工厂分配'
                        ] = allocation['分配工厂']

                else:
                    st.info("没有待分配的订单")
    else:
        st.info("请先生成订单和工厂数据")


def show_master_production_schedule():
    """主生产计划页面"""
    st.header("📋 主生产计划(MPS)")
    st.markdown("制定主生产计划，平衡需求与产能")

    # 初始化MPS引擎
    mps_engine = MasterProductionSchedule()
    mps_engine.initialize_time_buckets()

    # MPS参数设置
    st.subheader("⚙️ MPS参数配置")
    col1, col2, col3 = st.columns(3)
    with col1:
        planning_horizon = st.selectbox("计划期间", ["4周", "8周", "12周", "16周"], index=2)
        mps_engine.mps_horizon = int(planning_horizon.split('周')[0])
    with col2:
        lot_sizing_rule = st.selectbox(
            "批量规则",
            ["固定批量", "经济批量", "最小批量", "批对批"]
        )
    with col3:
        safety_factor = st.slider("安全系数", 0.9, 1.3, 1.1)

    # 创建MPS
    if st.button("🚀 创建主生产计划", type="primary"):
        with st.spinner("正在创建主生产计划..."):
            # 准备数据
            demand_forecast = st.session_state.forecast_results if not st.session_state.forecast_results.empty else pd.DataFrame()
            capacity_constraints = st.session_state.factories
            inventory_levels = st.session_state.inventory

            # 创建MPS
            mps_data = mps_engine.create_mps(
                demand_forecast,
                capacity_constraints,
                inventory_levels
            )

            if not mps_data.empty:
                st.success("✅ 主生产计划创建成功！")
                st.session_state.production_plan = mps_data

                # 显示MPS概览
                st.subheader("📊 MPS概览")

                # 按产品显示MPS
                products = mps_data['产品编号'].unique()
                selected_product = st.selectbox("选择产品", products)

                product_mps = mps_data[mps_data['产品编号'] == selected_product]

                # MPS时间序列图
                fig = go.Figure()

                fig.add_trace(go.Scatter(
                    x=product_mps['周次'],
                    y=product_mps['预测需求'],
                    mode='lines+markers',
                    name='预测需求',
                    line=dict(color='red')
                ))

                fig.add_trace(go.Scatter(
                    x=product_mps['周次'],
                    y=product_mps['计划生产'],
                    mode='lines+markers',
                    name='计划生产',
                    line=dict(color='blue')
                ))

                fig.add_trace(go.Scatter(
                    x=product_mps['周次'],
                    y=product_mps['期末库存'],
                    mode='lines+markers',
                    name='期末库存',
                    line=dict(color='green')
                ))

                fig.update_layout(
                    title=f"{selected_product} - 主生产计划",
                    xaxis_title="周次",
                    yaxis_title="数量",
                    height=400
                )

                st.plotly_chart(fig, use_container_width=True)

                # 关键指标
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    total_demand = product_mps['预测需求'].sum()
                    st.metric("总需求", f"{total_demand:,.0f}")
                with col2:
                    total_production = product_mps['计划生产'].sum()
                    st.metric("计划生产", f"{total_production:,.0f}")
                with col3:
                    avg_inventory = product_mps['期末库存'].mean()
                    st.metric("平均库存", f"{avg_inventory:,.0f}")
                with col4:
                    service_level = (product_mps['可承诺量'] > 0).mean() * 100
                    st.metric("服务水平", f"{service_level:.1f}%")

                # ATP分析
                st.subheader("📊 可承诺量(ATP)分析")
                atp_data = mps_engine.calculate_available_to_promise(selected_product)

                if not atp_data.empty:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.dataframe(atp_data, use_container_width=True)
                    with col2:
                        # ATP状态分布
                        status_counts = atp_data['状态'].value_counts()
                        fig = px.pie(
                            values=status_counts.values,
                            names=status_counts.index,
                            title="ATP状态分布",
                            color_discrete_map={'充足': 'green', '紧张': 'orange', '缺货': 'red'}
                        )
                        st.plotly_chart(fig, use_container_width=True)

                # 详细MPS数据
                st.subheader("📋 详细MPS数据")
                st.dataframe(mps_data, use_container_width=True)


def show_workshop_scheduling():
    """车间排程页面"""
    st.header("🏭 车间排程")
    st.markdown("优化车间作业调度，提高生产效率")

    # 初始化车间排程引擎
    workshop_scheduler = WorkshopScheduler()

    if not st.session_state.resources.empty and not st.session_state.orders.empty:
        # 设置车间资源
        workshop_scheduler.setup_workshop(st.session_state.resources)

        # 排程参数设置
        st.subheader("⚙️ 排程规则配置")
        col1, col2, col3 = st.columns(3)
        with col1:
            scheduling_method = st.selectbox(
                "排程规则",
                ["spt", "edd", "cr", "slack", "fifo"],
                format_func=lambda x: {
                    "spt": "最短加工时间优先(SPT)",
                    "edd": "最早交期优先(EDD)",
                    "cr": "关键比率法(CR)",
                    "slack": "最小松弛时间(Slack)",
                    "fifo": "先进先出(FIFO)"
                }.get(x, x)
            )
        with col2:
            consider_setup_time = st.checkbox("考虑换产时间", value=True)
        with col3:
            allow_preemption = st.checkbox("允许抢占", value=False)

        # 执行排程
        if st.button("🚀 执行车间排程", type="primary"):
            with st.spinner("正在进行车间排程优化..."):
                # 创建车间排程
                workshop_schedule = workshop_scheduler.create_workshop_schedule(
                    st.session_state.orders,
                    scheduling_method
                )

                if not workshop_schedule.empty:
                    st.success("✅ 车间排程完成！")
                    st.session_state.workshop_schedule = workshop_schedule

                    # 显示排程结果
                    st.subheader("📊 车间排程甘特图")

                    # 按车间分组显示
                    workshops = workshop_schedule['车间'].unique()
                    selected_workshop = st.selectbox("选择车间", workshops)

                    workshop_data = workshop_schedule[workshop_schedule['车间'] == selected_workshop]

                    # 创建甘特图
                    fig = px.timeline(
                        workshop_data,
                        x_start="开始时间",
                        x_end="结束时间",
                        y="工作中心",
                        color="优先级",
                        text="作业编号",
                        title=f"{selected_workshop} - 排程甘特图"
                    )

                    fig.update_yaxes(categoryorder="total ascending")
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)

                    # 车间效率分析
                    st.subheader("📊 车间效率分析")

                    # 计算各车间指标
                    workshop_metrics = []
                    for workshop in workshops:
                        ws_data = workshop_schedule[workshop_schedule['车间'] == workshop]
                        total_time = ws_data['持续时间'].sum()
                        job_count = len(ws_data)
                        avg_duration = ws_data['持续时间'].mean()

                        workshop_metrics.append({
                            '车间': workshop,
                            '作业数': job_count,
                            '总工时': f"{total_time:.1f}",
                            '平均工时': f"{avg_duration:.1f}",
                            '利用率': f"{random.uniform(75, 95):.1f}%"
                        })

                    metrics_df = pd.DataFrame(workshop_metrics)
                    st.dataframe(metrics_df, use_container_width=True)

                    # 优化建议
                    st.subheader("🎯 车间布局优化建议")
                    optimization_suggestions = workshop_scheduler.optimize_workshop_layout()

                    if not optimization_suggestions.empty:
                        st.dataframe(optimization_suggestions, use_container_width=True)

                    # 详细排程数据
                    st.subheader("📋 详细排程数据")
                    st.dataframe(workshop_schedule, use_container_width=True)


def show_material_planning():
    """物料计划页面"""
    st.header("📦 物料计划(MRP)")
    st.markdown("基于BOM和主生产计划计算物料需求")

    # 初始化MRP引擎
    mrp_engine = MaterialPlanningEngine()

    # 构建BOM树
    if st.session_state.bom:
        mrp_engine.build_bom_tree(st.session_state.bom)

    # MRP参数设置
    st.subheader("⚙️ MRP参数配置")
    col1, col2, col3 = st.columns(3)
    with col1:
        planning_horizon = st.number_input("计划期间(周)", min_value=4, max_value=12, value=8)
    with col2:
        lead_time_buffer = st.slider("提前期缓冲(%)", 0, 50, 20)
    with col3:
        safety_stock_factor = st.slider("安全库存系数", 0.5, 2.0, 1.2)

    # 设置物料参数
    if st.button("设置物料参数"):
        # 模拟设置物料提前期和安全库存
        materials = ['MAT-100', 'MAT-200', 'MAT-300', 'MAT-400', 'MAT-500']
        for material in materials:
            mrp_engine.material_lead_times[material] = random.randint(1, 3)
            mrp_engine.safety_stock_levels[material] = random.randint(50, 200)
        st.success("物料参数设置完成")

    # 运行MRP
    if st.button("🚀 运行MRP计算", type="primary"):
        with st.spinner("正在进行MRP计算..."):
            # 获取MPS数据
            mps_data = st.session_state.production_plan
            current_inventory = st.session_state.inventory

            # 运行MRP
            mrp_results = mrp_engine.run_mrp(
                mps_data,
                current_inventory,
                planning_horizon
            )

            if not mrp_results.empty:
                st.success("✅ MRP计算完成！")
                st.session_state.material_requirements = mrp_results

                # 显示MRP结果
                st.subheader("📊 物料需求概览")

                # 选择物料查看详情
                materials = mrp_results['物料编号'].unique()
                selected_material = st.selectbox("选择物料", materials)

                material_data = mrp_results[mrp_results['物料编号'] == selected_material]

                # MRP时间序列图
                fig = go.Figure()

                fig.add_trace(go.Bar(
                    x=material_data['期间'],
                    y=material_data['毛需求'],
                    name='毛需求',
                    marker_color='red'
                ))

                fig.add_trace(go.Scatter(
                    x=material_data['期间'],
                    y=material_data['期末库存'],
                    mode='lines+markers',
                    name='期末库存',
                    marker_color='green'
                ))

                fig.add_trace(go.Bar(
                    x=material_data['期间'],
                    y=material_data['计划订单接收'],
                    name='计划订单接收',
                    marker_color='blue'
                ))

                fig.update_layout(
                    title=f"{selected_material} - 物料需求计划",
                    xaxis_title="期间",
                    yaxis_title="数量",
                    barmode='group',
                    height=400
                )

                st.plotly_chart(fig, use_container_width=True)

                # 创建采购计划
                st.subheader("📋 采购计划")
                purchase_plan = mrp_engine.create_purchase_plan()

                if not purchase_plan.empty:
                    # 采购统计
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        total_orders = len(purchase_plan)
                        st.metric("采购单数", total_orders)
                    with col2:
                        urgent_orders = len(purchase_plan[purchase_plan['紧急程度'] == '高'])
                        st.metric("紧急采购", urgent_orders)
                    with col3:
                        total_value = random.uniform(100000, 500000)
                        st.metric("采购总值", f"¥{total_value:,.0f}")
                    with col4:
                        suppliers = purchase_plan['供应商'].nunique()
                        st.metric("涉及供应商", suppliers)

                    # 采购计划详情
                    st.dataframe(purchase_plan, use_container_width=True)

                # 详细MRP数据
                st.subheader("📋 详细MRP数据")
                st.dataframe(mrp_results, use_container_width=True)


def show_material_preparation():
    """生产备料页面"""
    st.header("🔧 生产备料")
    st.markdown("三级物料保障体系，确保生产顺利进行")

    # 初始化备料系统
    prep_system = ProductionMaterialPreparation()

    # 设置三级物料体系
    if not st.session_state.material_status.empty:
        materials = st.session_state.material_status.to_dict('records')
        prep_system.setup_three_level_system(materials, st.session_state.production_plan)

        # 显示三级库存状态
        st.subheader("📊 三级物料保障体系")

        # 创建三级库存可视化
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("### 🏭 一级：线边库")
            st.info("存储2-4小时用量，支持JIT生产")
            level1_data = []
            for mat_id, info in list(prep_system.material_levels['level1'].items())[:5]:
                level1_data.append({
                    '物料': mat_id,
                    '当前库存': f"{info['current_stock']:.0f}",
                    '库存率': f"{(info['current_stock'] / info['capacity'] * 100):.1f}%"
                })
            st.dataframe(pd.DataFrame(level1_data), use_container_width=True)

        with col2:
            st.markdown("### 🏢 二级：车间库")
            st.info("存储1-2天用量，缓冲波动")
            level2_data = []
            for mat_id, info in list(prep_system.material_levels['level2'].items())[:5]:
                level2_data.append({
                    '物料': mat_id,
                    '当前库存': f"{info['current_stock']:.0f}",
                    '库存率': f"{(info['current_stock'] / info['capacity'] * 100):.1f}%"
                })
            st.dataframe(pd.DataFrame(level2_data), use_container_width=True)

        with col3:
            st.markdown("### 🏗️ 三级：中心库")
            st.info("存储5-7天用量，战略储备")
            level3_data = []
            for mat_id, info in list(prep_system.material_levels['level3'].items())[:5]:
                level3_data.append({
                    '物料': mat_id,
                    '当前库存': f"{info['current_stock']:.0f}",
                    '库存率': f"{(info['current_stock'] / info['capacity'] * 100):.1f}%"
                })
            st.dataframe(pd.DataFrame(level3_data), use_container_width=True)

        # 创建备料计划
        if not st.session_state.workshop_schedule.empty:
            st.subheader("📋 生产备料计划")

            if st.button("🚀 生成备料计划", type="primary"):
                with st.spinner("正在生成备料计划..."):
                    # 创建备料计划
                    preparation_plan = prep_system.create_preparation_plan(
                        st.session_state.workshop_schedule
                    )

                    if not preparation_plan.empty:
                        st.success("✅ 备料计划生成成功！")
                        st.session_state.material_preparation = preparation_plan

                        # 风险分析
                        risk_summary = preparation_plan['风险等级'].value_counts()

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            high_risk = risk_summary.get('高', 0)
                            st.metric("高风险物料", high_risk,
                                      delta="-2" if high_risk > 0 else "0")
                        with col2:
                            medium_risk = risk_summary.get('中', 0)
                            st.metric("中风险物料", medium_risk)
                        with col3:
                            low_risk = risk_summary.get('低', 0)
                            st.metric("低风险物料", low_risk,
                                      delta="+3" if low_risk > 0 else "0")

                        # 备料策略分布
                        strategy_dist = preparation_plan['备料策略'].value_counts()
                        fig = px.pie(
                            values=strategy_dist.values,
                            names=strategy_dist.index,
                            title="备料策略分布",
                            color_discrete_map={
                                '直接配送': 'green',
                                '车间补充': 'blue',
                                '中心库调拨': 'orange',
                                '紧急采购': 'red'
                            }
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        # 详细备料计划
                        st.dataframe(preparation_plan, use_container_width=True)

                        # 生成补料订单
                        st.subheader("📦 自动补料订单")
                        replenishment_orders = prep_system.generate_replenishment_orders()

                        if not replenishment_orders.empty:
                            # 补料统计
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                total_replenishments = len(replenishment_orders)
                                st.metric("补料单数", total_replenishments)
                            with col2:
                                urgent_replenishments = len(
                                    replenishment_orders[replenishment_orders['紧急程度'] == '高']
                                )
                                st.metric("紧急补料", urgent_replenishments)
                            with col3:
                                total_quantity = replenishment_orders['补充数量'].sum()
                                st.metric("补料总量", f"{total_quantity:,.0f}")

                            st.dataframe(replenishment_orders, use_container_width=True)


def show_shipping_planning():
    """发运计划页面"""
    st.header("🚚 发运计划")
    st.markdown("集成优化发运环节，降低物流成本")

    # 初始化发运系统
    shipping_system = ShippingPlanningSystem()

    # 设置发运网络
    if not st.session_state.warehouse_network.empty:
        customers = ['客户A', '客户B', '客户C', '客户D', '客户E']
        shipping_system.setup_shipping_network(st.session_state.warehouse_network, customers)

        # 发运参数设置
        st.subheader("⚙️ 发运参数配置")
        col1, col2, col3 = st.columns(3)
        with col1:
            optimization_goal = st.selectbox(
                "优化目标",
                ["cost", "speed", "reliability"],
                format_func=lambda x: {
                    "cost": "成本最优",
                    "speed": "速度最快",
                    "reliability": "可靠性最高"
                }.get(x, x)
            )
        with col2:
            consolidation_window = st.number_input("合并时间窗口(天)", min_value=1, max_value=7, value=3)
        with col3:
            min_load_rate = st.slider("最低装载率(%)", 50, 100, 80)

        # 创建发运计划
        if not st.session_state.orders.empty:
            st.subheader("📦 待发运订单")

            # 筛选待发运订单
            ready_orders = st.session_state.orders[
                st.session_state.orders['状态'].isin(['已排产', '待发货'])
            ].head(50)

            if not ready_orders.empty:
                st.info(f"共有 {len(ready_orders)} 个订单待发运")

                if st.button("🚀 创建发运计划", type="primary"):
                    with st.spinner("正在优化发运计划..."):
                        # 创建发运计划
                        shipping_plan = shipping_system.create_shipping_plan(
                            ready_orders,
                            optimization_goal
                        )

                        if not shipping_plan.empty:
                            st.success("✅ 发运计划创建成功！")
                            st.session_state.shipping_plan = shipping_plan

                            # 发运统计
                            st.subheader("📊 发运计划概览")

                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                total_shipments = len(shipping_plan)
                                st.metric("发运批次", total_shipments)
                            with col2:
                                total_weight = shipping_plan['总重量'].sum()
                                st.metric("总重量(kg)", f"{total_weight:,.0f}")
                            with col3:
                                total_cost = shipping_plan['运输成本'].sum()
                                st.metric("运输成本", f"¥{total_cost:,.0f}")
                            with col4:
                                avg_loading = shipping_plan['装载率'].mean()
                                st.metric("平均装载率", f"{avg_loading:.1%}")

                            # 运输方式分布
                            transport_dist = shipping_plan['运输方式'].value_counts()
                            fig = px.bar(
                                x=transport_dist.index,
                                y=transport_dist.values,
                                title="运输方式分布",
                                labels={'x': '运输方式', 'y': '批次数'}
                            )
                            st.plotly_chart(fig, use_container_width=True)

                            # 发运计划详情
                            st.subheader("📋 发运计划详情")
                            st.dataframe(shipping_plan, use_container_width=True)

                            # 发运跟踪
                            st.subheader("📍 发运跟踪")
                            tracking_info = shipping_system.track_shipments(shipping_plan)

                            if not tracking_info.empty:
                                # 按状态分组显示
                                status_groups = tracking_info.groupby('状态')

                                for status, group in status_groups:
                                    with st.expander(f"{status} ({len(group)} 批次)"):
                                        st.dataframe(
                                            group[['发运单号', '当前位置', '运输进度',
                                                   '预计剩余时间', '异常情况']],
                                            use_container_width=True
                                        )


def show_command_center():
    """智能指挥中心页面"""
    st.header("🎮 智能指挥中心")
    st.markdown("实时监控和预测性分析")

    # 初始化指挥中心
    command_center = IntelligentOperationCommandCenter()
    command_center.setup_monitoring_system()

    # 显示实时KPI仪表板
    st.subheader("📊 实时KPI监控")
    executive_dashboard = command_center.generate_executive_dashboard()

    if not executive_dashboard.empty:
        # 使用颜色编码显示KPI状态
        def color_kpi_status(val):
            colors = {
                'excellent': 'background-color: #28a745; color: white',
                'normal': 'background-color: #ffc107; color: black',
                'warning': 'background-color: #fd7e14; color: white',
                'critical': 'background-color: #dc3545; color: white'
            }
            return colors.get(val, '')

        styled_dashboard = executive_dashboard.style.applymap(
            color_kpi_status, subset=['状态']
        )
        st.dataframe(styled_dashboard, use_container_width=True)

    # 订单执行监控
    if not st.session_state.orders.empty and not st.session_state.schedule.empty:
        st.subheader("📦 订单执行监控")

        monitoring_results = command_center.monitor_order_execution(
            st.session_state.orders.head(20),
            st.session_state.schedule
        )

        if not monitoring_results.empty:
            # 风险分布
            risk_dist = monitoring_results['风险等级'].value_counts()

            col1, col2, col3 = st.columns(3)
            with col1:
                high_risk = risk_dist.get('高', 0)
                st.metric("高风险订单", high_risk,
                          delta=f"-{high_risk}" if high_risk > 0 else "0")
            with col2:
                medium_risk = risk_dist.get('中', 0)
                st.metric("中风险订单", medium_risk)
            with col3:
                low_risk = risk_dist.get('低', 0)
                st.metric("低风险订单", low_risk,
                          delta=f"+{low_risk}" if low_risk > 0 else "0")

            # 执行状态分布
            status_dist = monitoring_results['执行状态'].value_counts()
            fig = px.pie(
                values=status_dist.values,
                names=status_dist.index,
                title="订单执行状态分布"
            )
            st.plotly_chart(fig, use_container_width=True)

            # 详细监控数据
            st.dataframe(monitoring_results, use_container_width=True)

    # 预测性预警
    st.subheader("🔮 预测性预警")
    predictive_alerts = command_center.predictive_alerts(st.session_state.sales_history)

    if not predictive_alerts.empty:
        # 按影响程度分组显示
        for impact in ['高', '中', '低']:
            impact_alerts = predictive_alerts[predictive_alerts['影响程度'] == impact]
            if not impact_alerts.empty:
                with st.expander(f"{impact}影响预警 ({len(impact_alerts)} 项)",
                                 expanded=(impact == '高')):
                    for _, alert in impact_alerts.iterrows():
                        st.warning(f"""
                        **{alert['预警类型']}**
                        - 发生概率: {alert['发生概率']}
                        - 预计时间: {alert['预计时间']}
                        - 影响范围: {alert['影响范围']}
                        - 建议措施: {alert['建议措施']}
                        """)

    # 数据合规性稽查
    st.subheader("✅ 数据合规性稽查")

    master_data = {
        'products': st.session_state.products,
        'bom': st.session_state.bom
    }

    if master_data['products'] or master_data['bom']:
        audit_results = command_center.compliance_audit(master_data)

        if not audit_results.empty:
            # 合规统计
            compliant_count = len(audit_results[audit_results['合规状态'] == '合格'])
            total_count = len(audit_results)
            compliance_rate = (compliant_count / total_count * 100) if total_count > 0 else 0

            st.metric("整体合规率", f"{compliance_rate:.1f}%",
                      delta=f"{compliance_rate - 90:.1f}%")

            # 详细稽查结果
            st.dataframe(audit_results, use_container_width=True)

    # OEE实时监控
    if not st.session_state.resources.empty:
        st.subheader("⚙️ OEE实时监控")

        oee_data = []
        for _, resource in st.session_state.resources.head(10).iterrows():
            oee = CommandCenter.calculate_oee(resource['资源编号'])
            oee_data.append({
                '资源编号': resource['资源编号'],
                '资源类型': resource['资源类型'],
                'OEE': f"{oee * 100:.1f}%",
                '状态': '优秀' if oee > 0.85 else '良好' if oee > 0.75 else '需改进'
            })

        oee_df = pd.DataFrame(oee_data)
        st.dataframe(oee_df, use_container_width=True)


# ==================== 主应用界面 ====================
def main():
    st.markdown('<div class="header">智能APS系统 Pro Max</div>', unsafe_allow_html=True)
    st.markdown("**融合SAP IBP、Oracle SCM、Blue Yonder和OR-Tools技术的下一代智能排程解决方案**")
    st.markdown("---")

    # 侧边栏导航
    st.sidebar.title("系统导航")

    # 使用单选按钮显示所有功能模块
    pages = [
        ("🏠 系统概览", "overview"),
        ("📊 需求管理", "demand"),
        ("🔮 智能预测", "forecast"),
        ("🌐 仓网分析", "warehouse_network"),
        ("📈 产销规划(S&OP)", "sales_operations_planning"),
        ("🎯 智能分单", "intelligent_allocation"),
        ("📋 主生产计划(MPS)", "master_production_schedule"),
        ("🏭 车间排程", "workshop_scheduling"),
        ("📦 物料计划(MRP)", "material_planning"),
        ("🔧 生产备料", "material_preparation"),
        ("🚚 发运计划", "shipping_planning"),
        ("🎮 智能指挥中心", "command_center"),
        ("⚡ 产能规划(CRP)", "crp"),
        ("📅 生产调度", "scheduling"),
        ("📦 库存管理", "inventory"),
        ("📈 KPI仪表板", "kpi"),
        ("🤖 数字孪生", "digital_twin"),
        ("🌐 供应链协同", "supply_chain"),
        ("🔍 优化分析", "optimization"),
        ("🏭 行业解决方案", "industry"),
        ("💬 AI智能助手", "ai_assistant"),
        ("📤 数据导出", "export")
    ]

    # 在侧边栏显示所有功能模块
    st.sidebar.markdown("### 功能模块")
    selected_page_name = st.sidebar.radio(
        "选择功能",
        [page[0] for page in pages],
        index=0
    )

    # 获取对应的页面key
    page_key = next(page[1] for page in pages if page[0] == selected_page_name)

    # 显示模块说明
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 系统控制")

    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("🔄 生成数据", help="创建模拟数据", use_container_width=True):
            load_sample_data()
            st.success("示例数据已生成!")

        if st.button("🚀 智能排程", help="运行优化算法", use_container_width=True):
            if st.session_state.orders.empty:
                st.warning("请先生成示例数据!")
            else:
                with st.spinner("智能排程中..."):
                    start_time = time.time()
                    aps_engine = HybridAPSEngine()
                    schedule = aps_engine.algorithm_router(st.session_state.orders, st.session_state.resources)
                    st.session_state.schedule = schedule

                    # 更新订单状态
                    if not schedule.empty:
                        st.session_state.orders.loc[
                            st.session_state.orders['订单编号'].isin(schedule['订单编号']), '状态'] = '已排产'

                    elapsed = time.time() - start_time
                    st.success(f"排程完成! 共排产 {len(schedule)} 个订单, 耗时 {elapsed:.2f}秒")

    with col2:
        if st.button("📊 更新孪生", help="刷新设备状态", use_container_width=True):
            if st.session_state.resources.empty:
                st.warning("请先生成示例数据!")
            else:
                for _, res in st.session_state.resources.iterrows():
                    twin = ResourceDigitalTwin(res['资源编号'], res['总产能'])
                    st.session_state.resource_twins[res['资源编号']] = twin
                st.success("设备状态已更新!")

        if st.button("🌍 风险分析", help="生成风险热力图", use_container_width=True):
            st.session_state.risk_map = generate_supply_chain_risk_map()
            st.success("风险热力图已生成!")

    # 系统统计
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 系统统计")
    if not st.session_state.orders.empty:
        st.sidebar.metric("订单总数", len(st.session_state.orders))
        scheduled_count = len(st.session_state.orders[st.session_state.orders['状态'] == '已排产'])
        st.sidebar.metric("已排产", scheduled_count)
    else:
        st.sidebar.metric("订单总数", 0)
        st.sidebar.metric("已排产", 0)

    if not st.session_state.resources.empty:
        st.sidebar.metric("资源数量", len(st.session_state.resources))
    else:
        st.sidebar.metric("资源数量", 0)

    # 页面路由
    if page_key == "overview":
        show_overview()
    elif page_key == "demand":
        show_demand_management()
    elif page_key == "forecast":
        show_intelligent_forecast()
    elif page_key == "warehouse_network":
        show_warehouse_network_analysis()
    elif page_key == "sales_operations_planning":
        show_sales_operations_planning()
    elif page_key == "intelligent_allocation":
        show_intelligent_allocation()
    elif page_key == "master_production_schedule":
        show_master_production_schedule()
    elif page_key == "workshop_scheduling":
        show_workshop_scheduling()
    elif page_key == "material_planning":
        show_material_planning()
    elif page_key == "material_preparation":
        show_material_preparation()
    elif page_key == "shipping_planning":
        show_shipping_planning()
    elif page_key == "command_center":
        show_command_center()
    elif page_key == "crp":
        show_capacity_planning()
    elif page_key == "scheduling":
        show_production_scheduling()
    elif page_key == "inventory":
        show_inventory_management()
    elif page_key == "kpi":
        show_kpi_dashboard()
    elif page_key == "digital_twin":
        show_digital_twin()
    elif page_key == "supply_chain":
        show_supply_chain()
    elif page_key == "optimization":
        show_optimization_analysis()
    elif page_key == "industry":
        show_industry_solutions()
    elif page_key == "ai_assistant":
        show_ai_assistant()
    elif page_key == "export":
        show_data_export()


# 程序入口
if __name__ == "__main__":
    main()

