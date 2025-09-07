#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from matplotlib import font_manager
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from test import build,build2,build3  # 使用 @test.py 中的 build 方法导入对象数据（支持 sheet_index）

# ---------------------- 中文字体与图形风格 ----------------------
def _resolve_chinese_font():
    candidates = [
        r"C:\\Windows\\Fonts\\msyh.ttc",
        r"C:\\Windows\\Fonts\\simhei.ttf",
        r"C:\\Windows\\Fonts\\simsun.ttc",
        r"/System/Library/Fonts/PingFang.ttc",
        r"/System/Library/Fonts/STHeiti Medium.ttc",
        r"/usr/share/fonts/truetype/arphic/ukai.ttc",
        r"/usr/share/fonts/truetype/arphic/uming.ttc",
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                font_manager.fontManager.addfont(p)
                fp = font_manager.FontProperties(fname=p)
                try:
                    fam = fp.get_name()
                    # 将识别到的字体族名置于首位，增强中文兼容
                    plt.rcParams['font.sans-serif'] = [fam] + plt.rcParams.get('font.sans-serif', [])
                except Exception:
                    pass
                print(f"使用中文字体: {p}")
                return fp
            except Exception:
                continue
    # 回退：不指定具体文件，仅依赖全局rcParams
    return None

CH_FONT = _resolve_chinese_font()
# 设置中文字体，优先使用已识别的字体
if CH_FONT is not None:
    try:
        font_name = CH_FONT.get_name()
        plt.rcParams['font.sans-serif'] = [font_name, 'Microsoft YaHei', 'SimHei', 'SimSun', 'Arial Unicode MS']
    except:
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'Arial Unicode MS']
else:
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'Arial Unicode MS']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid")

# 是否保存图片（本次需求为仅显示不保存）
SAVE_FIG = False
TOP_FEATURES_FOR_BIPLOT = 10  # 双标图上仅标注最重要的前N个特征
DRAW_HEATMAP = False  # 不需要热力图
DRAW_BIPLOT = False   # 不需要双标图
SCORE_HEALTH_THRESHOLD = 60.0  # 健康评分判定阈值
DRAW_CLASSIFICATION_PLOTS = True  # 二分类可视化开关


def load_excel_second_sheet() -> pd.DataFrame:
    """读取 Excel 第二个工作表（sheet index=1），兼容多种放置路径。"""
    script_dir = Path(__file__).resolve().parent
    base_dir = Path(__file__).resolve().parents[1]
    cwd = Path.cwd()

    candidates = [
        script_dir / "data.xlsx",                # 程序同目录（推荐）
        script_dir / "附件.xlsx",
        base_dir / "数据导入" / "data.xlsx",     # 题目给定目录1
        base_dir / "3. 数据集" / "原始数据" / "附件.xlsx",  # 题目给定目录2
        cwd / "data.xlsx",                      # 当前工作目录回退
        cwd / "附件.xlsx",
    ]

    excel_path = next((p for p in candidates if p.exists()), None)
    if excel_path is None:
        raise FileNotFoundError(
            "未找到数据文件，请将 data.xlsx/附件.xlsx 放在：程序同目录、当前工作目录、数据导入/ 或 3. 数据集/原始数据/ 下"
        )
    # 读取第二个工作表（sheet=1）
    df = pd.read_excel(excel_path, sheet_name=1)
    return df


def select_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    """选择用于PCA的数值型特征，并排除明显的ID/代码类字段。"""
    num_df = df.select_dtypes(include=[np.number]).copy()
    to_drop = [c for c in num_df.columns if ("代码" in str(c)) or ("编号" in str(c)) or ("ID" in str(c).upper())]
    num_df = num_df.drop(columns=to_drop, errors='ignore')
    # 去除全空或常数列
    keep_cols = []
    for c in num_df.columns:
        col = pd.to_numeric(num_df[c], errors='coerce')
        if col.notna().sum() >= 3 and col.nunique(dropna=True) >= 3:
            keep_cols.append(c)
    num_df = num_df[keep_cols]
    return num_df


def pca_analysis(df_feat: pd.DataFrame, explained_threshold: float = 0.85):
    """
    对特征做标准化 + PCA，自动选取累计解释方差≥阈值的成分数量；
    输出：
      - pca_full: 拟合了全部成分的PCA对象（用于方差解释）
      - pca_k:    选取前k个成分的PCA对象
      - X_scaled: 标准化后的特征矩阵
      - k:        选择的成分数
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_feat.values)
    pca_full = PCA()
    pca_full.fit(X_scaled)
    cum = np.cumsum(pca_full.explained_variance_ratio_)
    k = int(np.searchsorted(cum, explained_threshold) + 1)
    k = max(1, min(k, df_feat.shape[1]))
    pca_k = PCA(n_components=k)
    pca_k.fit(X_scaled)
    return pca_full, pca_k, X_scaled, k


def save_explained_variance_fig(pca_full: PCA):
    ratios = pca_full.explained_variance_ratio_
    plt.figure(figsize=(7, 4))
    plt.bar(range(1, len(ratios) + 1), ratios, alpha=0.6, label='单个主成分解释方差')
    plt.step(range(1, len(ratios) + 1), np.cumsum(ratios), where='mid', label='累计解释方差')
    # 标注八成(80%)与（可选）85%参考线
    plt.axhline(0.80, color='#ff7f0e', linestyle='--', linewidth=1.0)
    if CH_FONT is not None:
        plt.text(1, 0.802, '80%基准', color='#ff7f0e', fontsize=9, fontproperties=CH_FONT)
    else:
        plt.text(1, 0.802, '80% baseline', color='#ff7f0e', fontsize=9)
    plt.axhline(0.85, color='#2ca02c', linestyle=':', linewidth=1.0, alpha=0.8)
    if CH_FONT is not None:
        plt.xlabel('主成分数量', fontproperties=CH_FONT)
        plt.ylabel('解释方差比例', fontproperties=CH_FONT)
        plt.title('PCA解释方差比例', fontproperties=CH_FONT)
        plt.legend(prop=CH_FONT)
    else:
        plt.xlabel('主成分数量')
        plt.ylabel('解释方差比例')
        plt.title('PCA解释方差比例')
        plt.legend()
    plt.tight_layout()
    if SAVE_FIG:
        plt.savefig('pca_explained_variance.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("PCA解释方差图已显示")


def save_loadings_outputs(pca_k: PCA, df_feat: pd.DataFrame):
    # 载荷（原始特征与主成分的相关性近似）：components_.T * sqrt(explained_variance_)
    loadings = pca_k.components_.T * np.sqrt(pca_k.explained_variance_)
    loading_df = pd.DataFrame(loadings, index=df_feat.columns, columns=[f"PC{i+1}" for i in range(pca_k.n_components_)])
    # 保存载荷
    loading_df.round(4).to_csv('pca_loadings.csv', encoding='utf-8-sig')
    # 重要性排名（按各PC的|loading|）
    importance = loading_df.abs().sort_values(by=list(loading_df.columns), ascending=False)
    importance.head(20).round(4).to_csv('pca_top_loadings.csv', encoding='utf-8-sig')

    # 热力图可视化（按需）
    if DRAW_HEATMAP:
        plt.figure(figsize=(max(6, 0.4 * pca_k.n_components_ + 4), max(6, 0.25 * len(df_feat.columns) + 3)))
        sns.heatmap(loading_df, annot=False, cmap='coolwarm', center=0)
        if CH_FONT is not None:
            plt.title('主成分载荷热力图', fontproperties=CH_FONT)
            plt.xlabel('主成分', fontproperties=CH_FONT)
            plt.ylabel('原始特征', fontproperties=CH_FONT)
        else:
            plt.title('主成分载荷热力图')
            plt.xlabel('主成分')
            plt.ylabel('原始特征')
        plt.tight_layout()
        if SAVE_FIG:
            plt.savefig('pca_loadings_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()


def save_biplot(pca_k: PCA, X_scaled: np.ndarray, df_feat: pd.DataFrame):
    if (not DRAW_BIPLOT) or (pca_k.n_components_ < 2):
        return
    X_2d = pca_k.transform(X_scaled)[:, :2]
    plt.figure(figsize=(7, 5))
    plt.scatter(X_2d[:, 0], X_2d[:, 1], s=14, alpha=0.45, color='#1f77b4', label='样本')
    # 仅标注载荷贡献Top-N的特征，减少重叠
    vectors = pca_k.components_[:2, :].T  # shape: (n_features, 2)
    contrib = np.linalg.norm(vectors, axis=1)
    top_idx = np.argsort(contrib)[::-1][:TOP_FEATURES_FOR_BIPLOT]
    scale = 2.0
    for i in top_idx:
        feat = df_feat.columns[i]
        vx, vy = vectors[i, 0] * scale, vectors[i, 1] * scale
        plt.arrow(0, 0, vx, vy, color='#d62728', alpha=0.85, head_width=0.03, length_includes_head=True, zorder=3)
        # 根据象限调整标签对齐，避免遮挡箭头
        ha = 'left' if vx >= 0 else 'right'
        va = 'bottom' if vy >= 0 else 'top'
        txt_kwargs = dict(fontsize=8, color='#d62728', ha=ha, va=va,
                          bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1.5))
        if CH_FONT is not None:
            txt_kwargs.update(fontproperties=CH_FONT)
        plt.text(vx * 1.06, vy * 1.06, feat, **txt_kwargs)
    var = pca_k.explained_variance_ratio_
    if CH_FONT is not None:
        plt.xlabel(f'PC1 ({var[0]*100:.1f}%)', fontproperties=CH_FONT)
        plt.ylabel(f'PC2 ({var[1]*100:.1f}%)', fontproperties=CH_FONT)
        plt.title('PCA双标图（样本与特征）', fontproperties=CH_FONT)
        plt.legend(prop=CH_FONT)
    else:
        plt.xlabel(f'PC1 ({var[0]*100:.1f}%)')
        plt.ylabel(f'PC2 ({var[1]*100:.1f}%)')
        plt.title('PCA双标图（样本与特征）')
        plt.legend()
    plt.tight_layout()
    if SAVE_FIG:
        plt.savefig('pca_biplot.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("PCA双标图已显示")


def _parse_gestation_value(s):
    """将如 '12w+3' 转为数值周。无法解析返回np.nan。"""
    try:
        s = str(s).strip()
        import re
        m = re.match(r"(\d+)w(?:\+(\d+))?", s, re.IGNORECASE)
        if m:
            weeks = int(m.group(1))
            days = int(m.group(2)) if m.group(2) else 0
            return weeks + days / 7.0
        v = float(s)
        return v if np.isfinite(v) else np.nan
    except Exception:
        return np.nan


def _health_score_from_row(row: pd.Series) -> tuple[float, dict]:
    """按描述性的评分规则计算健康分（0-100）。返回 (score, details)。"""
    # 读取变量
    z13 = pd.to_numeric(row.get("染色体13_z值"), errors="coerce")
    z18 = pd.to_numeric(row.get("染色体18_z值"), errors="coerce")
    z21 = pd.to_numeric(row.get("染色体21_z值"), errors="coerce")
    xz  = pd.to_numeric(row.get("染色体X_z值"), errors="coerce") if "染色体X_z值" in row else np.nan
    bmi = pd.to_numeric(row.get("孕妇BMI"), errors="coerce")
    gc  = pd.to_numeric(row.get("GC含量"), errors="coerce")
    ga  = pd.to_numeric(row.get("孕周_数值"), errors="coerce")
    # 胎儿分数（由后续模型赋予，这里不读取原始表字段）
    ff  = np.nan

    # 归一化的扣分函数（0正常→0；异常→逐步接近1）
    def f_z(z):
        if not np.isfinite(z):
            return 0.0
        d = max(0.0, abs(z) - 3.0)
        # 平滑增长，|z|-3 为 0 起点，~3 内为0，越大越接近1
        return float(np.tanh(d / 2.5))  # d≈3→~0.76，d≈6→~0.94

    def g_bmi(v):
        if not np.isfinite(v):
            return 0.0
        # 28以下不扣，40及以上满扣，线性
        return float(np.clip((v - 28.0) / 12.0, 0.0, 1.0))

    def h_ga(v):
        if not np.isfinite(v):
            return 0.2  # 未知给轻微惩罚
        # 10周前显著扣分，10-12 有小扣，>=12不扣
        if v < 10.0:
            return float(np.clip((10.0 - v) / 4.0, 0.0, 1.0))  # 6周→1，8周→0.5
        if v < 12.0:
            return float((12.0 - v) / 4.0)  # 10~12 线性到0
        return 0.0

    def q_gc(v):
        if not np.isfinite(v):
            return 0.15  # 未知GC含量，中等惩罚
        # GC含量评分：35%~65%为正常范围，超出范围扣分
        # 35%以下：严重偏低，满扣分
        if v < 0.35:
            return 1.0
        # 35%~40%：偏低，线性扣分
        if v < 0.40:
            return float((0.40 - v) / 0.05)  # 35%→1.0, 40%→0.0
        # 40%~60%：正常范围，不扣分
        if v <= 0.60:
            return 0.0
        # 60%~65%：偏高，线性扣分
        if v <= 0.65:
            return float((v - 0.60) / 0.05)  # 60%→0.0, 65%→1.0
        # 65%以上：严重偏高，满扣分
        return 1.0

    def r_x(v):
        # 优先使用 X 染色体 z 值
        if np.isfinite(xz):
            return f_z(xz)
        # 若无 z，可对 X 浓度使用温和扣分（无参考期望，默认不扣）
        return 0.0

    def s_ff(v):
        # 胎儿分数（fetal fraction）：<4% 风险增加；4%~10%小惩罚；>=10%不扣
        if not np.isfinite(v):
            return 0.05  # 轻微惩罚（未知）
        if v < 0.04:
            return 1.0
        if v < 0.10:
            return float((0.10 - v) / 0.06)  # 4%→~1，10%→0
        return 0.0

    # 基础权重（将按存在变量重标定到总和100）
    base_weights = {
        "z13": 12.0,
        "z18": 12.0,
        "z21": 15.0,
        "bmi": 18.0,
        "ga":   8.0,
        "gc":   25.0,  # 增加GC含量权重，作为重要指标
        "x":    10.0,
        # "ff":   2.0,  # 由后续模型赋值后再启用
    }

    # 仅对存在的变量启用权重
    flags = {
        "z13": np.isfinite(z13),
        "z18": np.isfinite(z18),
        "z21": np.isfinite(z21),
        "bmi": np.isfinite(bmi),
        "ga":  np.isfinite(ga),
        "gc":  np.isfinite(gc),
        "x":   (np.isfinite(xz) or ("X染色体浓度" in row.index)),
        # "ff":  np.isfinite(ff),
    }
    active = {k: w for k, w in base_weights.items() if flags.get(k, False)}
    if not active:
        return 100.0, {"note": "无可用变量"}
    scale = 100.0 / sum(active.values())
    weights = {k: w * scale for k, w in active.items()}

    # 计算各项扣分
    penalties = {}
    if flags["z13"]:
        penalties["p13"] = weights["z13"] * f_z(z13)
    if flags["z18"]:
        penalties["p18"] = weights["z18"] * f_z(z18)
    if flags["z21"]:
        penalties["p21"] = weights["z21"] * f_z(z21)
    if flags["bmi"]:
        penalties["pBMI"] = weights["bmi"] * g_bmi(bmi)
    if flags["ga"]:
        penalties["pGA"] = weights["ga"] * h_ga(ga)
    if flags["gc"]:
        penalties["pGC"] = weights["gc"] * q_gc(gc)
    if flags["x"]:
        penalties["pX"] = weights["x"] * r_x(row.get("X染色体浓度"))
    # 若未来有模型产生胎儿分数 ff，再启用以下两行：
    # if flags["ff"]:
    #     penalties["pFF"] = weights["ff"] * s_ff(ff)

    total_penalty = float(sum(penalties.values()))
    score = float(np.clip(100.0 - total_penalty, 0.0, 100.0))
    details = {"weights": weights, "penalties": penalties}
    return score, details

def main():
    # 0) 参考 test.py 的 build() 导入对象数据（用于评分与健康判断）
    try:
        # 优先读取第0个工作表；如需第1个，请将 sheet_index 改为 1
        data = build2(sheet_index=1)
    except Exception as e:
        print(f"build() 读取对象失败: {e}. 回退到 Excel 第二工作表。")
        data = None

    # 1) 读取第二工作表（用于PCA与可视化）。若 build 可用，也可尝试从对象重建DataFrame。
    raw = None
    try:
        raw = load_excel_second_sheet()
    except Exception as e:
        print(f"Excel 读取失败: {e}")
    if raw is None and data is not None:
        # 从对象构造原始表的近似DataFrame
        raw = pd.DataFrame([getattr(o, "__dict__", {}) for o in data])
    if raw is None:
        print("无法获得原始数据，终止。")
        return
    # 2) 选择数值特征并清洗
    feat_df = select_numeric_features(raw)
    if feat_df.shape[1] < 2:
        print('可用于PCA的数值特征不足（<2）。')
        return
    # 去除含缺失过多的样本
    feat_df = feat_df.dropna(thresh=max(2, int(0.8 * feat_df.shape[1]))).fillna(feat_df.median(numeric_only=True))

    # 3) PCA分析
    pca_full, pca_k, X_scaled, k = pca_analysis(feat_df, explained_threshold=0.85)
    cum = np.cumsum(pca_full.explained_variance_ratio_)
    k80 = int(np.searchsorted(cum, 0.80) + 1)
    print(f"选择的主成分数量k(≥85%)={k}，累计解释方差={np.sum(pca_k.explained_variance_ratio_):.4f}")
    print(f"达到八成(80%)所需主成分数: k80={k80}，cum80={cum[k80-1]:.4f}")
    # 告知使用了哪些主成分（前 k 个），以及每一主成分的解释方差比例
    var = pca_full.explained_variance_ratio_
    used = [f"PC{i+1}({var[i]*100:.1f}%)" for i in range(min(k, len(var)))]
    print("本次用于后续分析的主成分：", ", ".join(used))

    # 4) 输出与可视化
    print("正在生成PCA可视化图表...")
    save_explained_variance_fig(pca_full)
    save_loadings_outputs(pca_k, feat_df)
    save_biplot(pca_k, X_scaled, feat_df)
    print("PCA图表已显示！")

    # ====================== 健康评分计算（基于描述性评分函数） ======================
    try:
        # 若有 build 对象，优先从对象做评分与判定
        if data is not None and len(data) > 0:
            rows = [getattr(o, "__dict__", {}) for o in data]
            df_for_score = pd.DataFrame(rows)
        else:
            df_for_score = raw.copy()
        # 生成/补充孕周数值
        if "孕周_数值" not in df_for_score.columns:
            df_for_score["孕周_数值"] = df_for_score.get("检测孕周", "").map(_parse_gestation_value)
        # 逐行评分
        scores = []
        for _, r in df_for_score.iterrows():
            s, _ = _health_score_from_row(r)
            scores.append(s)
        df_for_score["健康评分"] = scores
        # 不再设置胎儿分数属性
        # 使用染色体的非整倍体作为真实标签验证预测准确性
        # 有内容=异常=1，空白=正常=0
        aneuploidy_col = df_for_score.get("染色体的非整倍体", pd.Series([None]*len(df_for_score)))
        y_true = (aneuploidy_col.notna() & (aneuploidy_col.astype(str).str.strip() != '') & (aneuploidy_col.astype(str).str.strip() != 'nan')).astype(int)
        
        # 基于评分与阈值进行判定（低分=异常=1，高分=正常=0）
        score_used = df_for_score["健康评分"]
        y_pred = (score_used < SCORE_HEALTH_THRESHOLD).astype(int)  # 低分预测为异常
        
        # 计算预测准确率
        mask = y_true.notna()
        if mask.sum() >= 3 and y_true[mask].nunique() == 2:
            acc = float((y_pred[mask] == y_true[mask].astype(int)).mean())
            print(f"基于评分阈值({SCORE_HEALTH_THRESHOLD:.0f})的预测准确率: {acc*100:.2f}% (n={int(mask.sum())})")
            print(f"真实标签分布: 正常={int((y_true[mask]==0).sum())}, 异常={int((y_true[mask]==1).sum())}")
            print(f"预测标签分布: 正常={int((y_pred[mask]==0).sum())}, 异常={int((y_pred[mask]==1).sum())}")
        else:
            print("染色体的非整倍体标签不足或单一，无法计算准确率。已仅输出评分。")
            if mask.sum() > 0:
                print(f"标签分布: {y_true[mask].value_counts().to_dict()}")
        # 导出评分列表
        out_cols = ["健康评分"]
        if "染色体的非整倍体" in df_for_score.columns:
            out_cols.append("染色体的非整倍体")
        if "胎儿是否健康" in df_for_score.columns:
            out_cols.append("胎儿是否健康")
        df_for_score[out_cols].to_csv("health_scores.csv", index=False, encoding="utf-8-sig")
        print("评分列表已导出: health_scores.csv")
    except Exception as e:
        print(f"健康评分与准确率计算失败: {e}")

    # ====================== 基于染色体的非整倍体的二分类（逻辑回归） ======================
    try:
        # 使用打分后的数据作为特征来源
        df_clf = df_for_score.copy()
        
        # 先检查染色体的非整倍体字段的实际内容
        print("染色体的非整倍体字段检查：")
        aneuploidy_col = df_clf.get("染色体的非整倍体", pd.Series([None]*len(df_clf)))
        print(f"字段存在: {'染色体的非整倍体' in df_clf.columns}")
        print(f"非空值数量: {aneuploidy_col.notna().sum()}")
        print(f"空值数量: {aneuploidy_col.isna().sum()}")
        if aneuploidy_col.notna().any():
            print("非空值示例:", aneuploidy_col.dropna().head().tolist())
        print("唯一值:", aneuploidy_col.unique())
        
        # GC含量分析
        if "GC含量" in df_clf.columns:
            print("\nGC含量分析：")
            gc_data = pd.to_numeric(df_clf["GC含量"], errors='coerce')
            print(f"GC含量统计: 均值={gc_data.mean():.3f}, 标准差={gc_data.std():.3f}")
            print(f"GC含量范围: {gc_data.min():.3f} - {gc_data.max():.3f}")
            print(f"正常范围(35%-65%)样本数: {((gc_data >= 0.35) & (gc_data <= 0.65)).sum()}")
            print(f"异常范围样本数: {((gc_data < 0.35) | (gc_data > 0.65)).sum()}")
        
        # 目标标签：基于染色体的非整倍体字段（有字符串内容=异常=1，空白/空值=正常=0）
        aneuploidy = df_clf.get("染色体的非整倍体", pd.Series([None]*len(df_clf)))
        # 将非空且非空字符串的值视为异常，空值/NaN/空字符串视为正常
        try:
            y_map = (aneuploidy.notna() & (aneuploidy.astype(str).str.strip() != '') & (aneuploidy.astype(str).str.strip() != 'nan')).astype(int)
            mask_label = aneuploidy.notna() | aneuploidy.isna()  # 所有样本都参与（包括正常和异常）
        except Exception as e:
            print(f"标签处理失败: {e}")
            # 回退到简单处理
            y_map = aneuploidy.notna().astype(int)
            mask_label = pd.Series([True]*len(df_clf))
        # 选择可用特征（存在且有数值）：Z值、BMI、GC、孕周、X_Z、健康评分
        candidate_feats = [
            "染色体13_z值","染色体18_z值","染色体21_z值",
            "孕妇BMI","GC含量","孕周_数值","染色体X_z值","健康评分"
        ]
        
        # 特征工程：添加Z值的组合特征和GC含量相关特征
        if all(c in df_clf.columns for c in ["染色体13_z值","染色体18_z值","染色体21_z值"]):
            # 先转换为数值，处理None值
            z_cols = ["染色体13_z值","染色体18_z值","染色体21_z值"]
            z_df = df_clf[z_cols].apply(pd.to_numeric, errors='coerce')
            z_abs = z_df.abs()
            
            df_clf["Z值最大值"] = z_abs.max(axis=1)
            df_clf["Z值平均值"] = z_abs.mean(axis=1)
            df_clf["Z值标准差"] = z_abs.std(axis=1)
            candidate_feats.extend(["Z值最大值","Z值平均值","Z值标准差"])
        
        # GC含量相关特征工程
        if "GC含量" in df_clf.columns:
            gc_col = pd.to_numeric(df_clf["GC含量"], errors='coerce')
            # GC含量偏离正常范围的程度
            df_clf["GC含量偏离度"] = np.abs(gc_col - 0.50)  # 以50%为理想值
            # GC含量是否在正常范围内 (35%-65%)
            df_clf["GC含量正常"] = ((gc_col >= 0.35) & (gc_col <= 0.65)).astype(int)
            candidate_feats.extend(["GC含量偏离度","GC含量正常"])
        feat_cols = [c for c in candidate_feats if c in df_clf.columns]
        if len(feat_cols) >= 2:
            try:
                X_all = df_clf[feat_cols].apply(pd.to_numeric, errors="coerce")
                # 不以特征缺失剔除样本，交给 Imputer 处理；仅要求Z标签可用
                X_use = X_all.loc[mask_label]
                y_use = y_map.loc[mask_label].astype(int)
                print(f"特征矩阵形状: {X_use.shape}, 标签形状: {y_use.shape}")
            except Exception as e:
                print(f"特征矩阵构建失败: {e}")
                return
            # 需要两类标签且样本量适中
            cls_counts = y_use.value_counts() if y_use.notna().any() else pd.Series()
            if (len(X_use) >= 8) and (y_use.nunique() == 2) and (min(cls_counts) >= 3):
                strat = y_use
                X_tr, X_te, y_tr, y_te = train_test_split(X_use, y_use, test_size=0.3, random_state=42, stratify=strat)
                
                # 逻辑回归（优化参数）
                pipe_lr = Pipeline(steps=[
                    ("impute", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=2000, class_weight='balanced', C=0.1))
                ])
                pipe_lr.fit(X_tr, y_tr)
                prob_lr = pipe_lr.predict_proba(X_te)[:,1]
                pred_lr = (prob_lr >= 0.5).astype(int)
                print("\n[染色体的非整倍体二分类·Logistic] 评估（标签：有内容=异常，空白=正常）:")
                try:
                    auc_lr = roc_auc_score(y_te, prob_lr)
                    print(f"AUC={auc_lr:.4f}")
                except Exception:
                    auc_lr = float('nan')
                print(classification_report(y_te, pred_lr, digits=4, zero_division=0))
                
                # 随机森林（优化参数）
                pipe_rf = Pipeline(steps=[
                    ("impute", SimpleImputer(strategy="median")),
                    ("clf", RandomForestClassifier(
                        n_estimators=500, 
                        max_depth=10, 
                        min_samples_split=5,
                        min_samples_leaf=2,
                        max_features='sqrt',
                        random_state=42, 
                        n_jobs=-1, 
                        class_weight='balanced'
                    ))
                ])
                pipe_rf.fit(X_tr, y_tr)
                prob_rf = pipe_rf.predict_proba(X_te)[:,1]
                
                # 阈值优化：寻找最佳阈值
                from sklearn.metrics import precision_recall_curve
                precision, recall, thresholds = precision_recall_curve(y_te, prob_rf)
                f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
                best_threshold_idx = np.argmax(f1_scores)
                best_threshold = thresholds[best_threshold_idx] if best_threshold_idx < len(thresholds) else 0.5
                
                pred_rf_opt = (prob_rf >= best_threshold).astype(int)
                pred_rf_default = (prob_rf >= 0.5).astype(int)
                
                print("\n[染色体的非整倍体二分类·RandomForest] 评估:")
                try:
                    auc_rf = roc_auc_score(y_te, prob_rf)
                    print(f"AUC={auc_rf:.4f}")
                    print(f"最佳阈值={best_threshold:.3f} (基于F1-score)")
                except Exception:
                    auc_rf = float('nan')
                
                print("\n默认阈值(0.5)分类报告:")
                print(classification_report(y_te, pred_rf_default, digits=4, zero_division=0))
                print("\n优化阈值分类报告:")
                print(classification_report(y_te, pred_rf_opt, digits=4, zero_division=0))
                
                # 特征重要性
                try:
                    importances = pipe_rf.named_steps["clf"].feature_importances_
                    # 确保特征名称和重要性数组长度匹配
                    actual_feat_cols = feat_cols[:len(importances)]  # 截取匹配的长度
                    imp_series = pd.Series(importances, index=actual_feat_cols).sort_values(ascending=False)
                    print("随机森林 特征重要性:")
                    print(imp_series.round(4))
                    
                    # 特征重要性可视化
                    if DRAW_CLASSIFICATION_PLOTS:
                        plt.figure(figsize=(10, 6))
                        imp_series.plot(kind='barh', color='skyblue', alpha=0.7)
                        if CH_FONT is not None:
                            plt.title('随机森林特征重要性', fontproperties=CH_FONT)
                            plt.xlabel('重要性', fontproperties=CH_FONT)
                            plt.ylabel('特征', fontproperties=CH_FONT)
                        else:
                            plt.title('RandomForest Feature Importance')
                            plt.xlabel('Importance')
                            plt.ylabel('Features')
                        plt.grid(True, alpha=0.3)
                        plt.tight_layout()
                        plt.show()
                        print("特征重要性图已显示！")
                except Exception as e:
                    print(f"特征重要性分析失败: {e}")
                    pass
                
                # ROC 图和 Precision-Recall 图
                if DRAW_CLASSIFICATION_PLOTS:
                    try:
                        # ROC 图
                        plt.figure(figsize=(12,5))
                        plt.subplot(1,2,1)
                        # Logistic ROC
                        fpr_lr, tpr_lr, _ = roc_curve(y_te, prob_lr)
                        plt.plot(fpr_lr, tpr_lr, label=f"Logistic AUC={auc_lr:.3f}", linewidth=2, color='blue')
                        # RandomForest ROC
                        fpr_rf, tpr_rf, _ = roc_curve(y_te, prob_rf)
                        plt.plot(fpr_rf, tpr_rf, label=f"RandomForest AUC={auc_rf:.3f}", linewidth=2, color='red')
                        plt.plot([0,1],[0,1], 'k--', alpha=0.5, label='Random')
                        plt.xlabel('FPR')
                        plt.ylabel('TPR')
                        if CH_FONT is not None:
                            plt.title('ROC 曲线', fontproperties=CH_FONT)
                        else:
                            plt.title('ROC Curve')
                        plt.legend()
                        plt.grid(True, alpha=0.3)
                        
                        # Precision-Recall 图
                        plt.subplot(1,2,2)
                        precision_lr, recall_lr, _ = precision_recall_curve(y_te, prob_lr)
                        plt.plot(recall_lr, precision_lr, label=f"Logistic", linewidth=2, color='blue')
                        plt.plot(recall, precision, label=f"RandomForest", linewidth=2, color='red')
                        plt.axhline(y=y_te.mean(), color='k', linestyle='--', alpha=0.5, label='Random')
                        plt.xlabel('Recall')
                        plt.ylabel('Precision')
                        if CH_FONT is not None:
                            plt.title('Precision-Recall 曲线', fontproperties=CH_FONT)
                        else:
                            plt.title('Precision-Recall Curve')
                        plt.legend()
                        plt.grid(True, alpha=0.3)
                        plt.tight_layout()
                        plt.show()
                        print("图表已显示！")
                    except Exception as e:
                        print(f"绘图失败: {e}")
                        pass
            else:
                print("染色体的非整倍体二分类：样本/类别不足。标签分布:")
                try:
                    print(cls_counts.to_string())
                except Exception:
                    pass
        else:
            print("染色体的非整倍体二分类：可用特征不足，跳过训练。")
    except Exception as e:
        print(f"染色体的非整倍体二分类步骤出错: {e}")

    # ====================== 基于BMI/GC/游离DNA的噪声清除建模 ======================
    # 识别目标列（游离DNA/胎儿分数/FF等）
    target_candidates = [c for c in raw.columns if any(k in str(c) for k in ["游离", "胎儿分数", "FF", "fetal", "Fetal", "胎儿比例", "胎儿DNA"])]
    if not target_candidates:
        # 回退：若无明确列，尝试使用 Y染色体浓度 作为代理
        if "Y染色体浓度" in raw.columns:
            target_col = "Y染色体浓度"
        else:
            print("未找到游离DNA/胎儿分数等目标列，跳过噪声建模环节。")
            return
    else:
        target_col = target_candidates[0]

    # 构建用于建模的数据集：BMI、GC、目标列及数值特征
    df_model = raw.copy()
    for c in df_model.columns:
        df_model[c] = pd.to_numeric(df_model[c], errors='coerce') if c != target_col else pd.to_numeric(df_model[c], errors='coerce')
    base_cols = [col for col in ["孕妇BMI", "GC含量", target_col] if col in df_model.columns]
    if len(base_cols) < 2:
        print("关键特征不足（需要至少包含BMI/GC/目标列中的两项），跳过噪声建模环节。")
        return

    df_model = df_model.dropna(subset=base_cols).reset_index(drop=True)
    if df_model.shape[0] < 50:
        print("样本量不足，噪声建模可能不稳定。")

    # 用线性回归残差定义“噪声”标签：目标 ~ BMI + GC（可用列）
    X_base_cols = [c for c in ["孕妇BMI", "GC含量"] if c in df_model.columns]
    X_base = df_model[X_base_cols].values
    y_target = df_model[target_col].values
    try:
        # 最小二乘拟合
        X_design = np.column_stack([np.ones(len(df_model))] + [df_model[c].values for c in X_base_cols])
        beta, *_ = np.linalg.lstsq(X_design, y_target, rcond=None)
        y_hat = X_design @ beta
        resid = y_target - y_hat
        # 以绝对残差分位数90%为噪声阈值（亦可改为IQR/MAD）
        thr = np.quantile(np.abs(resid), 0.90)
        df_model["is_noise"] = (np.abs(resid) >= thr).astype(int)
    except Exception as e:
        print(f"残差计算失败，改用分位数阈值直接定义噪声: {e}")
        z = (y_target - np.nanmedian(y_target)) / (np.nanstd(y_target) + 1e-8)
        df_model["is_noise"] = (np.abs(z) > 2.5).astype(int)

    # PCA筛选的高权重特征（与前面feat_df可能不完全对齐，这里从原表选择数值列）
    num_all = raw.select_dtypes(include=[np.number]).columns.tolist()
    # 结合载荷绝对值，选出贡献度高的前N=8特征（如存在）
    try:
        # 复用前面fit的pca_k与feat_df列对齐
        loadings_full = (pca_k.components_.T * np.sqrt(pca_k.explained_variance_))
        loading_df2 = pd.DataFrame(loadings_full, index=feat_df.columns, columns=[f"PC{i+1}" for i in range(pca_k.n_components_)])
        score_abs = loading_df2.abs().sum(axis=1).sort_values(ascending=False)
        top_feats = score_abs.head(8).index.tolist()
    except Exception:
        top_feats = []

    feature_cols_final = list(dict.fromkeys([*(c for c in ["孕妇BMI","GC含量"] if c in df_model.columns), *[f for f in top_feats if f in df_model.columns]]))
    if len(feature_cols_final) < 2:
        # 回退：至少使用BMI与GC或与目标的线性组合
        feature_cols_final = [c for c in ["孕妇BMI","GC含量"] if c in df_model.columns]

    # 训练/测试切分（70/30）
    X = df_model[feature_cols_final]
    y = df_model["is_noise"].astype(int)
    stratify = y if y.nunique() == 2 and min(y.value_counts()) >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42, stratify=stratify)

    # Logistic 回归（判断）
    logit_pipe = None
    try:
        logit_pipe = Pipeline(steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000))
        ])
        logit_pipe.fit(X_train, y_train)
        y_proba_logit = logit_pipe.predict_proba(X_test)[:, 1]
        y_pred_logit = (y_proba_logit >= 0.5).astype(int)
        print("\n[Logistic] 模型评估：")
        try:
            auc_lg = roc_auc_score(y_test, y_proba_logit)
        except Exception:
            auc_lg = float('nan')
        print(f"Accuracy={accuracy_score(y_test, y_pred_logit):.4f}, Precision={precision_score(y_test, y_pred_logit, zero_division=0):.4f}, Recall={recall_score(y_test, y_pred_logit, zero_division=0):.4f}, F1={f1_score(y_test, y_pred_logit, zero_division=0):.4f}, AUC={auc_lg:.4f}")
        print(classification_report(y_test, y_pred_logit, digits=4, zero_division=0))
    except Exception as e:
        print(f"Logistic 回归训练失败: {e}")

    # 随机森林（检验）
    rf_pipe = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(n_estimators=300, max_depth=None, random_state=42, n_jobs=-1, class_weight='balanced'))
    ])
    rf_pipe.fit(X_train, y_train)
    y_proba_rf = rf_pipe.predict_proba(X_test)[:, 1]
    y_pred_rf = (y_proba_rf >= 0.5).astype(int)
    print("\n[随机森林] 模型评估：")
    try:
        auc_rf = roc_auc_score(y_test, y_proba_rf)
    except Exception:
        auc_rf = float('nan')
    print(f"Accuracy={accuracy_score(y_test, y_pred_rf):.4f}, Precision={precision_score(y_test, y_pred_rf, zero_division=0):.4f}, Recall={recall_score(y_test, y_pred_rf, zero_division=0):.4f}, F1={f1_score(y_test, y_pred_rf, zero_division=0):.4f}, AUC={auc_rf:.4f}")
    print(classification_report(y_test, y_pred_rf, digits=4, zero_division=0))

    # ROC曲线
    plt.figure(figsize=(6,5))
    try:
        fpr_rf, tpr_rf, _ = roc_curve(y_test, y_proba_rf)
        plt.plot(fpr_rf, tpr_rf, label=f"RandomForest AUC={auc_rf:.3f}")
    except Exception:
        pass
    if logit_pipe is not None:
        try:
            fpr_lg, tpr_lg, _ = roc_curve(y_test, y_proba_logit)
            plt.plot(fpr_lg, tpr_lg, label=f"Logistic AUC={auc_lg:.3f}")
        except Exception:
            pass
    plt.plot([0,1],[0,1], 'k--', alpha=0.5)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC 曲线')
    plt.legend()
    plt.tight_layout()
    if SAVE_FIG:
        plt.savefig('pca_noise_roc.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred_rf)
    plt.figure(figsize=(4.2,3.6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('随机森林 混淆矩阵')
    plt.xlabel('预测')
    plt.ylabel('真实')
    plt.tight_layout()
    if SAVE_FIG:
        plt.savefig('pca_noise_cm_rf.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 特征重要性
    try:
        imp_values = rf_pipe.named_steps["clf"].feature_importances_
    except Exception:
        imp_values = np.zeros(len(feature_cols_final))
    imp = pd.Series(imp_values, index=feature_cols_final).sort_values(ascending=False)
    print("\n随机森林 特征重要性：")
    print(imp.round(4))
    plt.figure(figsize=(6, 0.35*len(imp)+2))
    sns.barplot(x=imp.values, y=imp.index, orient='h', color='#1f77b4')
    plt.title('随机森林 特征重要性')
    plt.xlabel('重要性')
    plt.tight_layout()
    if SAVE_FIG:
        plt.savefig('pca_noise_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    main()


