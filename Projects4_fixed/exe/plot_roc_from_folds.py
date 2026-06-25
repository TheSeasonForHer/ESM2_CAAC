import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy.interpolate import make_interp_spline

# ==================== 配置 ====================
BASE_DIR = "/home/zjw/zjwdata/3_deep_learning/training_data/curated_v2/gnn_model/fold_results"
CLASS_NAMES = {0: 'Negative', 1: 'Positive', 2: 'Hard'}
CLASS_COLORS = {0: '#E74C3C', 1: '#3498DB', 2: '#27AE60'}
N_CLASSES = 3
N_FOLDS = 5

# ==================== 读取数据 ====================
all_folds_data = []
for fold_idx in range(1, N_FOLDS + 1):
    csv_path = os.path.join(BASE_DIR, f"fold_{fold_idx}", "test_predictions.csv")
    all_folds_data.append(pd.read_csv(csv_path))

# ==================== 高分辨率插值网格 ====================
mean_fpr = np.linspace(0, 1, 3000)

fig, ax = plt.subplots(figsize=(9, 8), dpi=300)


# 辅助函数：动态精度显示 AUC，避免 ±0.000
def fmt_auc(mean, std):
    if std < 1e-6:
        return f"{mean:.4f} ± <0.0001"
    dec = max(3, int(-np.floor(np.log10(std))) + 1)
    return f"{mean:.{dec}f} ± {std:.{dec}f}"


for cls in range(N_CLASSES):
    tprs = []
    aucs = []

    for df in all_folds_data:
        y_true = df['true_label'].values
        y_score = df[f'prob_{cls}'].values
        y_true_binary = (y_true == cls).astype(int)

        fpr, tpr, _ = roc_curve(y_true_binary, y_score)
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

        # 去除重复 FPR 点
        uniq_idx = np.unique(fpr, return_index=True)[1]
        fpr_u = fpr[uniq_idx]
        tpr_u = tpr[uniq_idx]

        # B-样条三次插值（比 Pchip 更柔和，无 overshoot）
        if len(fpr_u) >= 4:
            spl = make_interp_spline(fpr_u, tpr_u, k=3)
            interp_tpr = spl(mean_fpr)
        else:
            interp_tpr = np.interp(mean_fpr, fpr_u, tpr_u)

        interp_tpr = np.clip(interp_tpr, 0, 1)
        interp_tpr[0] = 0.0
        interp_tpr[-1] = 1.0
        tprs.append(interp_tpr)

    # Mean ± SD
    tprs_arr = np.array(tprs)
    mean_tpr = tprs_arr.mean(axis=0)
    std_tpr = tprs_arr.std(axis=0)

    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)

    # 阴影区域（更淡，不抢戏）
    ax.fill_between(
        mean_fpr,
        np.maximum(mean_tpr - std_tpr, 0),
        np.minimum(mean_tpr + std_tpr, 1),
        color=CLASS_COLORS[cls],
        alpha=0.08,
        edgecolor='none'
    )

    # 平滑曲线
    ax.plot(
        mean_fpr,
        mean_tpr,
        color=CLASS_COLORS[cls],
        linewidth=2.2,
        label=f"{CLASS_NAMES[cls]} (AUC = {fmt_auc(mean_auc, std_auc)})"
    )

# 对角线
ax.plot([0, 1], [0, 1], 'k--', linewidth=1.0, alpha=0.4, label='Random Classifier')

# ==================== 主图美化 ====================
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])
ax.set_xlabel('False Positive Rate', fontsize=13, fontweight='bold')
ax.set_ylabel('True Positive Rate', fontsize=13, fontweight='bold')
ax.set_title('Mean ROC Curves with Standard Deviation\n(5-Fold Cross-Validation)',
             fontsize=14, fontweight='bold', pad=15)
ax.legend(loc='lower right', fontsize=10, frameon=True, fancybox=True, shadow=True)
ax.grid(True, linestyle='--', alpha=0.3)
ax.set_aspect('equal')
for spine in ax.spines.values():
    spine.set_linewidth(1.1)

# ==================== 局部放大图（右上角，完全不遮挡主曲线） ====================
ax_inset = ax.inset_axes([0.58, 0.58, 0.38, 0.35])  # [x, y, w, h] in axes coords

for cls in range(N_CLASSES):
    tprs = []
    for df in all_folds_data:
        y_true = df['true_label'].values
        y_score = df[f'prob_{cls}'].values
        y_true_binary = (y_true == cls).astype(int)
        fpr, tpr, _ = roc_curve(y_true_binary, y_score)

        uniq_idx = np.unique(fpr, return_index=True)[1]
        fpr_u, tpr_u = fpr[uniq_idx], tpr[uniq_idx]
        if len(fpr_u) >= 4:
            spl = make_interp_spline(fpr_u, tpr_u, k=3)
            interp_tpr = spl(mean_fpr)
        else:
            interp_tpr = np.interp(mean_fpr, fpr_u, tpr_u)
        interp_tpr = np.clip(interp_tpr, 0, 1)
        tprs.append(interp_tpr)

    mean_tpr = np.array(tprs).mean(axis=0)
    std_tpr = np.array(tprs).std(axis=0)

    ax_inset.fill_between(mean_fpr, np.maximum(mean_tpr - std_tpr, 0),
                          np.minimum(mean_tpr + std_tpr, 1),
                          color=CLASS_COLORS[cls], alpha=0.10, edgecolor='none')
    ax_inset.plot(mean_fpr, mean_tpr, color=CLASS_COLORS[cls], linewidth=1.8)

ax_inset.set_xlim(-0.005, 0.12)
ax_inset.set_ylim(0.75, 1.005)
ax_inset.set_xticks([0, 0.05, 0.10])
ax_inset.set_yticks([0.80, 0.90, 1.00])
ax_inset.tick_params(labelsize=9)
ax_inset.grid(True, linestyle='--', alpha=0.3)

# 不画 indicate_inset_zoom 连接线，避免斜线横穿主图；用黑色细边框标示 inset 范围
for spine in ax_inset.spines.values():
    spine.set_linewidth(1.2)
    spine.set_color('gray')

# ==================== 保存 ====================
plt.tight_layout()
save_path = os.path.join(BASE_DIR, "roc_curves_final_v2.png")
plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"ROC 曲线已保存至: {save_path}")
plt.show()