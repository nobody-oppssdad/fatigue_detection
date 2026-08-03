import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams, font_manager
import warnings

warnings.filterwarnings('ignore')

# ====================== 基础配置（科研级可视化） ======================
# 1. 路径配置（使用你提供的真实 LOSS 文件）
TRAIN_LOSS_PATH = r"C:\Users\15136\PycharmProjects\pythonProject2\fatigue_detection\loss_logs\train_val_loss.json"
EVAL_RESULT_PATH = "eval_results.json"
MULTI_MODEL_CSV = "model_comparison_results/model_comparison_report.csv"
SAVE_DIR = "visualization_results"
os.makedirs(SAVE_DIR, exist_ok=True)

# 2. 科研级字体/样式配置
try:
    font_manager.fontManager.addfont('times.ttf')
    rcParams['font.family'] = 'Times New Roman'
except:
    rcParams['font.family'] = 'DejaVu Serif'
rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

# 核心样式配置
rcParams.update({
    'figure.figsize': (12, 8),
    'axes.grid': False,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 1.2,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'xtick.major.size': 5,
    'ytick.major.size': 5,
    'lines.linewidth': 2.0,
    'lines.markersize': 8,
    'patch.linewidth': 1.2,
    'savefig.dpi': 600,
    'savefig.format': 'png'
})

# 3. 配色方案
COLOR_PALETTE = {
    'train': '#1f77b4',
    'val': '#ff7f0e',
    'ap_05': '#2ca02c',
    'ap_75': '#d62728',
    'ap_all': '#9467bd',
    'small': '#8c564b',
    'medium': '#e377c2',
    'ar_10': '#bcbd22',
    'param': '#d62728',
    'infer_time': '#1f77b4'
}

# ====================== 工具函数（已修改：完全使用你的真实数据） ======================
def save_eval_results(eval_metrics, save_path=EVAL_RESULT_PATH):
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(eval_metrics, f, indent=4, ensure_ascii=False)

def load_eval_results(load_path=EVAL_RESULT_PATH):
    latest_metrics = {
        "AP@0.5:0.95": 0.542,
        "AP@0.5": 0.966,
        "AP@0.75": 0.531,
        "AP@small": 0.541,
        "AP@media": 0.537,
        "AP@large": -1.0,
        "AR@1": 0.489,
        "AR@10": 0.611,
        "AR@100": 0.611,
        "AR@small": 0.593,
        "AR@media": 0.582,
        "AR@large": -1.0
    }
    save_eval_results(latest_metrics)
    return latest_metrics

# ====================== ✅ 已修改：完全加载你提供的真实 LOSS 数据 ======================
def load_train_loss(loss_path=TRAIN_LOSS_PATH):
    """
    完全使用你提供的 train_val_loss.json
    无模拟，无修改，100% 真实数据
    """
    with open(loss_path, 'r', encoding='utf-8') as f:
        loss_data = json.load(f)

    train_loss = np.array(loss_data["train_loss"])
    val_loss = np.array(loss_data["val_loss"])
    epochs = np.array(loss_data["epoch"])

    return train_loss, val_loss, epochs

# ====================== 可视化函数 ======================
def plot_train_loss():
    """训练/验证损失曲线（使用你真实的loss）"""
    train_loss, val_loss, epochs = load_train_loss()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_loss, color=COLOR_PALETTE['train'],
            marker='o', markevery=2, label='Training Loss', zorder=3)
    ax.plot(epochs, val_loss, color=COLOR_PALETTE['val'],
            marker='s', markevery=2, label='Validation Loss', zorder=2)

    ax.set_title('Training and Validation Loss Curves (Real Training Data)', pad=15)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Average Loss')
    ax.set_xticks(np.arange(0, len(epochs)+1, 5))
    ax.set_ylim(0, 0.5)

    # 标注最终损失
    ax.text(epochs[-1], train_loss[-1], f'{train_loss[-1]:.4f}',
            ha='right', va='bottom', color=COLOR_PALETTE['train'], fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    ax.text(epochs[-1], val_loss[-1], f'{val_loss[-1]:.4f}',
            ha='right', va='top', color=COLOR_PALETTE['val'], fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    ax.legend(loc='upper right', frameon=True)
    plt.savefig(os.path.join(SAVE_DIR, 'train_loss_curve.png'),
                dpi=600, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print("✅ 真实训练损失曲线已保存！")

def plot_coco_ap_metrics():
    metrics = load_eval_results()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5), constrained_layout=True)
    bar_width = 0.6

    ap_labels = ['AP@0.5', 'AP@0.75', 'AP@0.5:0.95']
    ap_values = [metrics['AP@0.5'], metrics['AP@0.75'], metrics['AP@0.5:0.95']]
    ap_colors = [COLOR_PALETTE['ap_05'], COLOR_PALETTE['ap_75'], COLOR_PALETTE['ap_all']]
    bars1 = ax1.bar(ap_labels, ap_values, width=bar_width, color=ap_colors, edgecolor='black', linewidth=1.2)
    ax1.set_title('Core AP Metrics')
    ax1.set_ylabel('Average Precision')
    ax1.set_ylim(0, 1.05)
    for b, v in zip(bars1, ap_values):
        ax1.text(b.get_x()+b.get_width()/2, b.get_height()+0.015, f'{v:.3f}', ha='center', fontweight='bold')

    size_labels = ['Small (Eyes)', 'Medium (Mouth)']
    size_values = [metrics['AP@small'], metrics['AP@media']]
    size_colors = [COLOR_PALETTE['small'], COLOR_PALETTE['medium']]
    bars2 = ax2.bar(size_labels, size_values, width=bar_width, color=size_colors, edgecolor='black', linewidth=1.2)
    ax2.set_title('AP by Object Size')
    ax2.set_ylim(0, 0.65)
    for b, v in zip(bars2, size_values):
        ax2.text(b.get_x()+b.get_width()/2, b.get_height()+0.015, f'{v:.3f}', ha='center', fontweight='bold')

    plt.savefig(os.path.join(SAVE_DIR, 'coco_ap_metrics.png'), dpi=600, bbox_inches='tight')
    plt.close()
    print("✅ AP指标图已保存")

def plot_coco_ar_metrics():
    metrics = load_eval_results()
    ar_labels = ['AR@1', 'AR@10', 'AR@100', 'AR@small', 'AR@media']
    ar_values = [metrics['AR@1'], metrics['AR@10'], metrics['AR@100'], metrics['AR@small'], metrics['AR@media']]

    fig, ax = plt.subplots(figsize=(8,8), subplot_kw=dict(projection='polar'))
    angles = np.linspace(0, 2*np.pi, len(ar_labels), endpoint=False).tolist()
    ar_values += ar_values[:1]
    angles += angles[:1]

    ax.plot(angles, ar_values, color=COLOR_PALETTE['ar_10'], linewidth=2)
    ax.fill(angles, ar_values, color=COLOR_PALETTE['ar_10'], alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(ar_labels)
    ax.set_ylim(0, 0.7)
    ax.set_title('Average Recall (AR) Radar Chart', pad=30)
    ax.grid(True, color='#dddddd')

    plt.savefig(os.path.join(SAVE_DIR, 'coco_ar_metrics.png'), dpi=600, bbox_inches='tight')
    plt.close()
    print("✅ AR雷达图已保存")

def plot_multi_model_comparison():
    if not os.path.exists(MULTI_MODEL_CSV):
        return
    df = pd.read_csv(MULTI_MODEL_CSV)
    fig, ax1 = plt.subplots(figsize=(10,5))
    ax1.plot(df['Backbone'], df['AP@0.5'], 'o-', color=COLOR_PALETTE['ap_05'], label='AP@0.5')
    ax2 = ax1.twinx()
    ax2.plot(df['Backbone'], df['参数量(M)'], 's--', color=COLOR_PALETTE['param'], label='Params')
    ax1.set_title('Model Comparison')
    ax1.legend(loc='upper left')
    plt.savefig(os.path.join(SAVE_DIR, 'multi_model_comparison.png'), dpi=600, bbox_inches='tight')
    plt.close()

# ====================== 主函数 ======================
def main():
    print("🚀 正在生成 100% 真实训练曲线...")
    plot_train_loss()
    plot_coco_ap_metrics()
    plot_coco_ar_metrics()
    plot_multi_model_comparison()
    print(f"\n🎉 全部图表已生成在：{SAVE_DIR}")

if __name__ == "__main__":
    main()