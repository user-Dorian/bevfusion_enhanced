# -*- coding: utf-8 -*-
"""
训练数据可视化脚本
生成基线模型与当前模型的对比图表
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from matplotlib.font_manager import FontProperties

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 如果中文字体不可用，使用英文
try:
    fp = FontProperties(fname='C:/Windows/Fonts/simhei.ttf')
except:
    fp = None

# 数据目录
DATA_DIR = Path(r"d:\workbench\bev\bevfusion_enhanced\同参数训练改进对比")

# 类别名称映射 (英文 -> 中文)
CATEGORY_NAMES = {
    'car': '轿车',
    'truck': '卡车',
    'construction_vehicle': '工程车',
    'bus': '公交车',
    'trailer': '拖车',
    'barrier': '路障',
    'motorcycle': '摩托车',
    'bicycle': '自行车',
    'pedestrian': '行人',
    'traffic_cone': '交通锥'
}

# 误差指标名称映射
ERROR_METRICS = {
    'mATE': '平移误差 (mATE)',
    'mASE': '尺度误差 (mASE)',
    'mAOE': '朝向误差 (mAOE)',
    'mAVE': '速度误差 (mAVE)',
    'mAAE': '属性误差 (mAAE)'
}


def load_jsonl_data(filepath):
    """加载 JSONL 格式的训练数据"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and line != '{}':
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return data


def extract_validation_data(data):
    """提取验证数据（mode='val' 的记录）"""
    val_data = [d for d in data if d.get('mode') == 'val']
    return val_data


def extract_training_losses(data):
    """提取训练损失数据"""
    train_data = [d for d in data if d.get('mode') == 'train']
    losses = []
    iterations = []
    epochs = []
    
    for d in train_data:
        if 'loss' in d:
            losses.append(d['loss'])
            iterations.append(d.get('iter', 0))
            epochs.append(d.get('epoch', 0))
    
    return iterations, epochs, losses


def calculate_map_nds(val_data):
    """计算 mAP 和 NDS"""
    if not val_data:
        return None, None
    
    # 获取最后一个 epoch 的验证结果
    latest_val = val_data[-1]
    
    # 计算 mAP (所有类别的平均 AP)
    ap_keys = [k for k in latest_val.keys() if 'ap_dist' in k]
    if ap_keys:
        # 按距离阈值分组计算 mAP
        distances = ['0.5', '1.0', '2.0', '4.0']
        map_scores = {}
        
        for dist in distances:
            dist_aps = []
            for key in ap_keys:
                if f'dist_{dist}' in key:
                    dist_aps.append(latest_val[key])
            if dist_aps:
                map_scores[dist] = np.mean(dist_aps)
        
        # 总体 mAP 是所有距离的平均
        mAP = np.mean(list(map_scores.values())) if map_scores else 0
    else:
        mAP = 0
    
    # 计算 NDS (使用 nuScenes Detection Score 公式)
    # NDS = 0.5 * (mAP + sum(1 - error) for each error metric)
    error_metrics = ['mATE', 'mASE', 'mAOE', 'mAVE', 'mAAE']
    errors = []
    
    for metric in error_metrics:
        key = f'object/{metric}'
        if key in latest_val:
            errors.append(latest_val[key])
    
    if errors:
        # 归一化误差 (nuScenes 使用特定归一化因子)
        norm_errors = [
            min(1, errors[0]),  # mATE
            min(1, errors[1]),  # mASE
            min(1, errors[2]),  # mAOE
            min(1, errors[3]),  # mAVE
            min(1, errors[4])   # mAAE
        ]
        NDS = 0.5 * (mAP + np.mean([1 - e for e in norm_errors]))
    else:
        NDS = 0
    
    return mAP, NDS


def extract_category_aps(val_data):
    """提取各类别的 AP"""
    if not val_data:
        return {}
    
    latest_val = val_data[-1]
    category_aps = {}
    
    for key, value in latest_val.items():
        if 'ap_dist' in key:
            parts = key.split('/')
            if len(parts) >= 2:
                category = parts[1]  # e.g., 'car'
                # 提取距离信息
                dist_info = key.split('ap_dist_')[-1]
                
                if category not in category_aps:
                    category_aps[category] = {}
                category_aps[category][dist_info] = value
    
    # 计算每个类别的平均 AP
    avg_aps = {}
    for cat, dists in category_aps.items():
        avg_aps[cat] = np.mean(list(dists.values()))
    
    return avg_aps


def extract_error_metrics(val_data):
    """提取误差指标"""
    if not val_data:
        return {}
    
    latest_val = val_data[-1]
    error_metrics = {}
    
    metrics = ['mATE', 'mASE', 'mAOE', 'mAVE', 'mAAE']
    for metric in metrics:
        key = f'object/{metric}'
        if key in latest_val:
            error_metrics[metric] = latest_val[key]
    
    return error_metrics


def plot_training_losses(baseline_losses, current_losses, save_path):
    """绘制训练损失对比曲线图"""
    fig, ax = plt.subplots(figsize=(14, 8), dpi=300)
    
    baseline_iters, baseline_epochs, baseline_loss_vals = baseline_losses
    current_iters, current_epochs, current_loss_vals = current_losses
    
    ax.plot(baseline_iters, baseline_loss_vals, 'b-', linewidth=2, 
            label='Baseline Model / 基线模型', alpha=0.7)
    ax.plot(current_iters, current_loss_vals, 'r-', linewidth=2, 
            label='Current Model / 当前模型', alpha=0.7)
    
    ax.set_xlabel('Iteration / 迭代次数', fontsize=12, fontproperties=fp)
    ax.set_ylabel('Loss / 损失', fontsize=12, fontproperties=fp)
    ax.set_title('Training Loss Comparison / 训练损失对比曲线', fontsize=14, fontproperties=fp)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 已保存：{save_path}")


def plot_map_nds_comparison(baseline_mAP, baseline_NDS, current_mAP, current_NDS, save_path):
    """绘制 mAP 和 NDS 对比柱状图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    
    metrics = ['mAP', 'NDS']
    baseline_vals = [baseline_mAP * 100, baseline_NDS * 100]  # 转换为百分比
    current_vals = [current_mAP * 100, current_NDS * 100]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    # mAP 对比
    bars1 = axes[0].bar(x - width/2, baseline_vals, width, label='Baseline / 基线', 
                        color='steelblue', alpha=0.8)
    bars2 = axes[0].bar(x + width/2, current_vals, width, label='Current / 当前', 
                        color='coral', alpha=0.8)
    
    axes[0].set_ylabel('Score (%)', fontsize=12, fontproperties=fp)
    axes[0].set_title('mAP Comparison / mAP 对比', fontsize=13, fontproperties=fp)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metrics, fontsize=11)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3, axis='y', linestyle='--')
    axes[0].set_ylim(0, max(max(baseline_vals), max(current_vals)) * 1.2)
    
    # 添加数值标签
    for bar in bars1:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}%', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}%', ha='center', va='bottom', fontsize=9)
    
    # 改进幅度
    improvements = [(current_vals[i] - baseline_vals[i]) for i in range(len(metrics))]
    colors = ['green' if imp > 0 else 'red' for imp in improvements]
    
    axes[1].bar(['mAP', 'NDS'], improvements, color=colors, alpha=0.7)
    axes[1].set_ylabel('Improvement (%) / 改进幅度', fontsize=12, fontproperties=fp)
    axes[1].set_title('Performance Improvement / 性能改进', fontsize=13, fontproperties=fp)
    axes[1].grid(True, alpha=0.3, axis='y', linestyle='--')
    axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    
    # 添加数值标签
    for i, imp in enumerate(improvements):
        axes[1].text(i, imp, f'{imp:+.2f}%', ha='center', 
                    va='bottom' if imp > 0 else 'top', fontsize=10, 
                    color='darkgreen' if imp > 0 else 'darkred')
    
    plt.suptitle('mAP & NDS Comparison / mAP 和 NDS 对比', fontsize=14, fontproperties=fp, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 已保存：{save_path}")


def plot_category_ap_comparison(baseline_aps, current_aps, save_path):
    """绘制各类别 AP 对比柱状图"""
    # 获取所有类别
    all_categories = set(baseline_aps.keys()) | set(current_aps.keys())
    
    # 使用中文名称或英文名称
    categories = []
    category_labels = []
    for cat in sorted(all_categories):
        categories.append(cat)
        category_labels.append(CATEGORY_NAMES.get(cat, cat))
    
    baseline_vals = [baseline_aps.get(cat, 0) * 100 for cat in categories]
    current_vals = [current_aps.get(cat, 0) * 100 for cat in categories]
    
    fig, ax = plt.subplots(figsize=(16, 8), dpi=300)
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, baseline_vals, width, label='Baseline / 基线', 
                   color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, current_vals, width, label='Current / 当前', 
                   color='coral', alpha=0.8)
    
    ax.set_xlabel('Category / 类别', fontsize=12, fontproperties=fp)
    ax.set_ylabel('Average AP (%)', fontsize=12, fontproperties=fp)
    ax.set_title('Per-Category AP Comparison / 各类别 AP 对比', fontsize=14, fontproperties=fp)
    ax.set_xticks(x)
    ax.set_xticklabels(category_labels, fontsize=10, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.set_ylim(0, max(max(baseline_vals), max(current_vals), 10) * 1.15)
    
    # 添加数值标签
    for bar in bars1:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 已保存：{save_path}")


def plot_error_metrics_radar(baseline_errors, current_errors, save_path):
    """绘制误差指标雷达图"""
    metrics = ['mATE', 'mASE', 'mAOE', 'mAVE', 'mAAE']
    baseline_vals = [baseline_errors.get(m, 0) for m in metrics]
    current_vals = [current_errors.get(m, 0) for m in metrics]
    
    # 雷达图需要闭合
    baseline_vals += baseline_vals[:1]
    current_vals += current_vals[:1]
    
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), dpi=300, subplot_kw=dict(polar=True))
    
    # 绘制基线模型
    ax.plot(angles, baseline_vals, 'b-', linewidth=2, label='Baseline / 基线', alpha=0.7)
    ax.fill(angles, baseline_vals, 'b', alpha=0.1)
    
    # 绘制当前模型
    ax.plot(angles, current_vals, 'r-', linewidth=2, label='Current / 当前', alpha=0.7)
    ax.fill(angles, current_vals, 'r', alpha=0.1)
    
    # 设置标签
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([ERROR_METRICS.get(m, m) for m in metrics], fontsize=11)
    ax.set_ylim(0, max(max(baseline_vals), max(current_vals), 0.5))
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
    
    ax.set_title('Error Metrics Comparison / 误差指标雷达图', fontsize=14, fontproperties=fp, pad=20)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 已保存：{save_path}")


def plot_improvement_comparison(baseline_aps, current_aps, baseline_errors, current_errors, save_path):
    """绘制综合改进幅度对比图"""
    # 计算各类别改进幅度
    all_categories = set(baseline_aps.keys()) | set(current_aps.keys())
    categories = sorted(all_categories)
    
    ap_improvements = []
    category_labels = []
    for cat in categories:
        baseline_val = baseline_aps.get(cat, 0) * 100
        current_val = current_aps.get(cat, 0) * 100
        improvement = current_val - baseline_val
        ap_improvements.append(improvement)
        category_labels.append(CATEGORY_NAMES.get(cat, cat))
    
    # 计算误差改进幅度 (误差降低为正改进)
    error_metrics = ['mATE', 'mASE', 'mAOE', 'mAVE', 'mAAE']
    error_improvements = []
    error_labels = []
    for metric in error_metrics:
        baseline_val = baseline_errors.get(metric, 0)
        current_val = current_errors.get(metric, 0)
        improvement = (baseline_val - current_val) * 100  # 误差降低为正
        error_improvements.append(improvement)
        error_labels.append(ERROR_METRICS.get(metric, metric)[:15])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), dpi=300)
    
    # AP 改进
    colors_ap = ['green' if imp > 0 else 'red' if imp < 0 else 'gray' for imp in ap_improvements]
    bars1 = ax1.bar(range(len(categories)), ap_improvements, color=colors_ap, alpha=0.7)
    ax1.set_xlabel('Category / 类别', fontsize=11, fontproperties=fp)
    ax1.set_ylabel('AP Improvement (%) / AP 改进幅度', fontsize=11, fontproperties=fp)
    ax1.set_title('Per-Category AP Improvement / 各类别 AP 改进', fontsize=12, fontproperties=fp)
    ax1.set_xticks(range(len(categories)))
    ax1.set_xticklabels(category_labels, rotation=45, ha='right', fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    
    # 添加数值标签
    for i, imp in enumerate(ap_improvements):
        ax1.text(i, imp, f'{imp:+.1f}', ha='center', 
                va='bottom' if imp > 0 else 'top', fontsize=8,
                color='darkgreen' if imp > 0 else 'darkred' if imp < 0 else 'black')
    
    # 误差改进
    colors_err = ['green' if imp > 0 else 'red' if imp < 0 else 'gray' for imp in error_improvements]
    bars2 = ax2.bar(range(len(error_metrics)), error_improvements, color=colors_err, alpha=0.7)
    ax2.set_xlabel('Error Metric / 误差指标', fontsize=11, fontproperties=fp)
    ax2.set_ylabel('Error Reduction (%) / 误差降低幅度', fontsize=11, fontproperties=fp)
    ax2.set_title('Error Metrics Improvement / 误差指标改进', fontsize=12, fontproperties=fp)
    ax2.set_xticks(range(len(error_metrics)))
    ax2.set_xticklabels(error_labels, rotation=45, ha='right', fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    
    # 添加数值标签
    for i, imp in enumerate(error_improvements):
        ax2.text(i, imp, f'{imp:+.1f}', ha='center', 
                va='bottom' if imp > 0 else 'top', fontsize=8,
                color='darkgreen' if imp > 0 else 'darkred' if imp < 0 else 'black')
    
    plt.suptitle('Comprehensive Improvement Analysis / 综合改进幅度对比', 
                fontsize=14, fontproperties=fp, y=1.05)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 已保存：{save_path}")


def generate_analysis_report(baseline_data, current_data, save_path):
    """生成数据分析报告"""
    report = []
    report.append("=" * 80)
    report.append("训练改进对比数据分析报告")
    report.append("Training Improvement Comparison Analysis Report")
    report.append("=" * 80)
    report.append("")
    
    # 基线模型信息
    baseline_val = extract_validation_data(baseline_data)
    current_val = extract_validation_data(current_data)
    
    baseline_mAP, baseline_NDS = calculate_map_nds(baseline_val)
    current_mAP, current_NDS = calculate_map_nds(current_val)
    
    baseline_aps = extract_category_aps(baseline_val)
    current_aps = extract_category_aps(current_val)
    
    baseline_errors = extract_error_metrics(baseline_val)
    current_errors = extract_error_metrics(current_val)
    
    report.append("【1】主要指标对比 / Main Metrics Comparison")
    report.append("-" * 80)
    report.append(f"{'指标':<20} {'基线模型':<15} {'当前模型':<15} {'改进幅度':<15}")
    report.append(f"{'Metric':<20} {'Baseline':<15} {'Current':<15} {'Improvement':<15}")
    report.append("-" * 80)
    
    if baseline_mAP is not None and current_mAP is not None:
        map_imp = (current_mAP - baseline_mAP) * 100
        report.append(f"{'mAP':<20} {baseline_mAP*100:>12.2f}%  {current_mAP*100:>12.2f}%  {map_imp:>+12.2f}%")
    
    if baseline_NDS is not None and current_NDS is not None:
        nds_imp = (current_NDS - baseline_NDS) * 100
        report.append(f"{'NDS':<20} {baseline_NDS*100:>12.2f}%  {current_NDS*100:>12.2f}%  {nds_imp:>+12.2f}%")
    
    report.append("")
    report.append("【2】各类别 AP 对比 / Per-Category AP Comparison")
    report.append("-" * 80)
    report.append(f"{'类别':<25} {'基线 AP':<12} {'当前 AP':<12} {'改进':<12}")
    report.append(f"{'Category':<25} {'Baseline':<12} {'Current':<12} {'Improvement':<12}")
    report.append("-" * 80)
    
    all_categories = set(baseline_aps.keys()) | set(current_aps.keys())
    for cat in sorted(all_categories):
        baseline_ap = baseline_aps.get(cat, 0) * 100
        current_ap = current_aps.get(cat, 0) * 100
        improvement = current_ap - baseline_ap
        cat_name = CATEGORY_NAMES.get(cat, cat)
        report.append(f"{cat_name:<25} {baseline_ap:>10.2f}%  {current_ap:>10.2f}%  {improvement:>+10.2f}%")
    
    report.append("")
    report.append("【3】误差指标对比 / Error Metrics Comparison")
    report.append("-" * 80)
    report.append(f"{'指标':<20} {'基线值':<15} {'当前值':<15} {'改进 (降低)':<15}")
    report.append(f"{'Metric':<20} {'Baseline':<15} {'Current':<15} {'Improvement':<15}")
    report.append("-" * 80)
    
    for metric in ['mATE', 'mASE', 'mAOE', 'mAVE', 'mAAE']:
        baseline_err = baseline_errors.get(metric, 0)
        current_err = current_errors.get(metric, 0)
        improvement = (baseline_err - current_err) * 100
        metric_name = ERROR_METRICS.get(metric, metric)
        report.append(f"{metric_name:<20} {baseline_err:>12.4f}  {current_err:>12.4f}  {improvement:>+12.2f}%")
    
    report.append("")
    report.append("【4】训练损失统计 / Training Loss Statistics")
    report.append("-" * 80)
    
    baseline_losses = extract_training_losses(baseline_data)
    current_losses = extract_training_losses(current_data)
    
    if baseline_losses[2] and current_losses[2]:
        report.append(f"{'':<20} {'初始损失':<15} {'最终损失':<15} {'下降幅度':<15}")
        report.append(f"{'':<20} {'Initial':<15} {'Final':<15} {'Reduction':<15}")
        report.append(f"基线模型 Baseline:  {baseline_losses[2][0]:>12.4f}  {baseline_losses[2][-1]:>12.4f}  {(1-baseline_losses[2][-1]/baseline_losses[2][0])*100:>12.2f}%")
        report.append(f"当前模型 Current:  {current_losses[2][0]:>12.4f}  {current_losses[2][-1]:>12.4f}  {(1-current_losses[2][-1]/current_losses[2][0])*100:>12.2f}%")
    
    report.append("")
    report.append("【5】关键发现 / Key Findings")
    report.append("-" * 80)
    
    # 分析改进情况
    if baseline_mAP is not None and current_mAP is not None:
        if current_mAP > baseline_mAP:
            report.append(f"✓ mAP 提升了 {map_imp:.2f}%, 检测精度有所改善")
        elif current_mAP < baseline_mAP:
            report.append(f"✗ mAP 下降了 {abs(map_imp):.2f}%, 需要进一步优化")
        else:
            report.append("○ mAP 保持不变")
    
    if baseline_NDS is not None and current_NDS is not None:
        if current_NDS > baseline_NDS:
            report.append(f"✓ NDS 提升了 {nds_imp:.2f}%, 综合检测性能提升")
        elif current_NDS < baseline_NDS:
            report.append(f"✗ NDS 下降了 {abs(nds_imp):.2f}%, 需要关注")
        else:
            report.append("○ NDS 保持不变")
    
    # 误差分析
    total_error_reduction = sum((baseline_errors.get(m, 0) - current_errors.get(m, 0)) 
                                for m in ['mATE', 'mASE', 'mAOE', 'mAVE', 'mAAE'])
    if total_error_reduction > 0:
        report.append(f"✓ 总体误差降低了 {total_error_reduction*100:.2f}%, 预测质量提升")
    else:
        report.append(f"✗ 总体误差增加了 {abs(total_error_reduction)*100:.2f}%, 需要改进")
    
    report.append("")
    report.append("【6】建议 / Recommendations")
    report.append("-" * 80)
    
    # 根据数据提供建议
    if baseline_aps and current_aps:
        worst_category = min(all_categories, 
                            key=lambda c: current_aps.get(c, 0) - baseline_aps.get(c, 0))
        worst_cat_name = CATEGORY_NAMES.get(worst_category, worst_category)
        report.append(f"• {worst_cat_name} 的改进最小，建议重点关注该类别的优化")
    
    if baseline_errors and current_errors:
        worst_error_metric = max(['mATE', 'mASE', 'mAOE', 'mAVE', 'mAAE'],
                                key=lambda m: current_errors.get(m, 0))
        worst_error_name = ERROR_METRICS.get(worst_error_metric, worst_error_metric)
        report.append(f"• {worst_error_name} 仍然较大，建议针对性优化")
    
    report.append("")
    report.append("=" * 80)
    report.append("报告生成完成 / Report Generation Complete")
    report.append("=" * 80)
    
    # 保存报告
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print(f"✓ 已保存：{save_path}")
    return '\n'.join(report)


def main():
    """主函数"""
    print("=" * 80)
    print("开始生成训练数据可视化图表")
    print("Starting Visualization Generation")
    print("=" * 80)
    print()
    
    # 加载数据
    print("【1】加载数据 / Loading Data...")
    baseline_data = load_jsonl_data(DATA_DIR / "基线模型训练数据.json")
    current_data = load_jsonl_data(DATA_DIR / "当前模型训练数据.json")
    print(f"  基线模型数据：{len(baseline_data)} 条记录")
    print(f"  当前模型数据：{len(current_data)} 条记录")
    print()
    
    # 提取验证数据
    print("【2】提取验证数据 / Extracting Validation Data...")
    baseline_val = extract_validation_data(baseline_data)
    current_val = extract_validation_data(current_data)
    print(f"  基线模型验证记录：{len(baseline_val)} 条")
    print(f"  当前模型验证记录：{len(current_val)} 条")
    print()
    
    # 计算关键指标
    print("【3】计算关键指标 / Calculating Key Metrics...")
    baseline_mAP, baseline_NDS = calculate_map_nds(baseline_val)
    current_mAP, current_NDS = calculate_map_nds(current_val)
    print(f"  基线模型 - mAP: {baseline_mAP*100:.2f}%, NDS: {baseline_NDS*100:.2f}%")
    print(f"  当前模型 - mAP: {current_mAP*100:.2f}%, NDS: {current_NDS*100:.2f}%")
    print()
    
    baseline_aps = extract_category_aps(baseline_val)
    current_aps = extract_category_aps(current_val)
    baseline_errors = extract_error_metrics(baseline_val)
    current_errors = extract_error_metrics(current_val)
    
    # 提取训练损失
    print("【4】提取训练损失 / Extracting Training Losses...")
    baseline_losses = extract_training_losses(baseline_data)
    current_losses = extract_training_losses(current_data)
    print(f"  基线模型损失范围：[{min(baseline_losses[2]):.4f}, {max(baseline_losses[2]):.4f}]")
    print(f"  当前模型损失范围：[{min(current_losses[2]):.4f}, {max(current_losses[2]):.4f}]")
    print()
    
    # 生成图表
    print("【5】生成可视化图表 / Generating Visualizations...")
    print("-" * 80)
    
    plot_training_losses(
        baseline_losses, current_losses,
        DATA_DIR / "01_训练损失对比曲线.png"
    )
    
    plot_map_nds_comparison(
        baseline_mAP, baseline_NDS, current_mAP, current_NDS,
        DATA_DIR / "02_主要指标对比.png"
    )
    
    plot_category_ap_comparison(
        baseline_aps, current_aps,
        DATA_DIR / "03_各类别 AP 对比.png"
    )
    
    plot_error_metrics_radar(
        baseline_errors, current_errors,
        DATA_DIR / "04_误差指标雷达图.png"
    )
    
    plot_improvement_comparison(
        baseline_aps, current_aps, baseline_errors, current_errors,
        DATA_DIR / "05_综合改进幅度对比.png"
    )
    
    print()
    print("【6】生成分析报告 / Generating Analysis Report...")
    print("-" * 80)
    report = generate_analysis_report(
        baseline_data, current_data,
        DATA_DIR / "数据分析报告.txt"
    )
    
    print()
    print("=" * 80)
    print("✓ 所有图表和报告生成完成!")
    print("✓ All visualizations and reports have been generated!")
    print("=" * 80)
    print()
    print("生成的文件 / Generated Files:")
    print("  - 01_训练损失对比曲线.png")
    print("  - 02_主要指标对比.png")
    print("  - 03_各类别 AP 对比.png")
    print("  - 04_误差指标雷达图.png")
    print("  - 05_综合改进幅度对比.png")
    print("  - 数据分析报告.txt")
    print()


if __name__ == "__main__":
    main()
