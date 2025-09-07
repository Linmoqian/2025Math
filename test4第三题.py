import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from test import build2,build
import importlib
from scipy.stats import chi2_contingency
from matplotlib import font_manager
from scipy.stats import norm

# 中文字体与负号显示
try:
    font_manager.fontManager.addfont(r"C:\Windows\Fonts\msyh.ttc")
except Exception:
    pass
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

# 可选：scikit-learn
_sk_spec = importlib.util.find_spec("sklearn")
if _sk_spec is not None:
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
else:
    SKLEARN_AVAILABLE = False

ALPHA = 1.0
BETA = 3.0
Y_THRESHOLD = 0.04
T_CAP = 28.0
# 非线性延误风险参数：12~28 凸增，≥28 为 1，12 附近有轻微起始
DELAY_EPSILON = 0.03
DELAY_RISK_GAMMA = 2.5
# 可行域阈值（logit 的“达标概率”下限，用于近似 Y≥0.04 约束）
FEASIBLE_PROB_MIN = 0.5
# 预测模式：'logit' 使用逻辑回归；'linear' 使用组内线性回归Y~T（第一问风格）
PREDICT_MODE = 'logit'

def delay_risk(t_candidates: np.ndarray, t0: float = 10.0, tcap: float = T_CAP, gamma: float = DELAY_RISK_GAMMA) -> np.ndarray:
    # 非线性延误风险：
    # - t < 10: 0
    # - 10 ≤ t ≤ 12: 从 0 平滑上升到 epsilon（smoothstep）
    # - 12 < t < 28: 从 epsilon 按 u^gamma 凸性增长至 1
    # - t ≥ 28: 1
    t = np.asarray(t_candidates, dtype=float)
    risk = np.zeros_like(t)
    # 10~12 平滑上升
    s = np.clip((t - 10.0) / 2.0, 0.0, 1.0)
    smooth = s * s * (3 - 2 * s)
    part1 = DELAY_EPSILON * smooth
    risk = np.where((t >= 10.0) & (t <= 12.0), part1, risk)
    # 12~tcap 凸增
    u = np.clip((t - 12.0) / max(tcap - 12.0, 1e-6), 0.0, 1.0)
    part2 = DELAY_EPSILON + (1.0 - DELAY_EPSILON) * np.power(u, gamma)
    risk = np.where((t > 12.0) & (t < tcap), part2, risk)
    # ≥ tcap
    risk = np.where(t >= tcap, 1.0, risk)
    return np.clip(risk, 0.0, 1.0)

def build_df():
    data = build()
    df = pd.DataFrame({
        "Y染色体浓度": [t.Y染色体浓度 for t in data],
        "孕周_数值": [t.孕周_数值 for t in data],
        "孕妇BMI": [t.孕妇BMI for t in data],
        "身高": [t.身高 for t in data],
        "体重": [t.体重 for t in data],
        "年龄": [getattr(t, "年龄", None) for t in data],
        "GC含量": [t.GC含量 for t in data],
        "胎儿是否健康": [t.胎儿是否健康 for t in data]
    })
    for c in ["Y染色体浓度","孕周_数值","孕妇BMI","身高","体重","年龄","GC含量"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Y染色体浓度","孕周_数值","孕妇BMI"]).reset_index(drop=True)
    df["检测阳性"] = (df["Y染色体浓度"] >= Y_THRESHOLD).astype(int)
    print(f"数据源: build()，样本数={len(df)}")
    return df

def train_logistic_model(df: pd.DataFrame):
    if not SKLEARN_AVAILABLE:
        print("未安装scikit-learn，无法训练逻辑回归。请先安装：pip install scikit-learn")
        return None
    feature_cols_all = ["孕周_数值","孕妇BMI","身高","体重","年龄","GC含量"]
    # 自动剔除全空特征，避免 Imputer 警告
    feature_cols = [c for c in feature_cols_all if df[c].notna().any()]
    X = df[feature_cols]
    # 优先使用标注的胎儿健康状态；若不存在/全空，退回到“检测阳性”
    if ("胎儿是否健康" in df.columns) and (df["胎儿是否健康"].notna().any()):
        y_raw = df["胎儿是否健康"].astype(str).str.strip()
        y = y_raw.map({"健康":1, "正常":1, "1":1, "不健康":0, "异常":0, "0":0}).fillna(np.nan)
        if y.notna().sum() < 3 or y.nunique(dropna=True) < 2:
            print("标注健康状态不足以训练，回退到‘检测阳性’。")
            y = df["检测阳性"].astype(int)
        else:
            y = y.astype(float).astype(int)
    else:
        y = df["检测阳性"].astype(int)

    # 若只有一个类别，跳过逻辑回归，改用线性近似路径
    if y.nunique() < 2:
        print("逻辑回归跳过：目标只有单一类别，自动回退为线性近似路径。")
        return None

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_transformer, feature_cols)]
    )
    clf = LogisticRegression(max_iter=1000, solver="lbfgs")
    model = Pipeline(steps=[("preprocess", preprocessor), ("clf", clf)])
    model.fit(X, y)
    return model, feature_cols

def optimize_best_time_by_bmi(df: pd.DataFrame, model, feature_cols, predict_mode: str = PREDICT_MODE):
    out = []
    # 五分位分组
    try:
        df["BMI_五分位组"] = pd.qcut(df["孕妇BMI"], q=5, labels=["P1","P2","P3","P4","P5"], duplicates="drop")
    except ValueError:
        df["BMI_五分位组"] = pd.qcut(df["孕妇BMI"], q=5, labels=None, duplicates="drop")

    for g in sorted(df["BMI_五分位组"].dropna().unique()):
        sub = df[df["BMI_五分位组"] == g]
        if len(sub) < 20:
            print(f"BMI组 {g}: 样本量不足（n={len(sub)}），跳过。")
            continue
        # 使用组内特征中位数作为代表（孕周除外）
        med = sub[feature_cols].median(numeric_only=True)
        t_min = max(10.0, float(sub["孕周_数值"].min()))
        t_max_obs = float(sub["孕周_数值"].max())
        t_cap = min(T_CAP, t_max_obs)
        if not np.isfinite(t_min) or not np.isfinite(t_cap) or t_cap <= t_min:
            print(f"BMI组 {g}: 孕周范围异常，跳过。")
            continue
        t_grid = np.arange(t_min, t_cap + 1e-9, 0.1)
        if (predict_mode == 'logit') and (model is not None):
            # 逻辑回归：用组内中位数+T网格预测达标概率
            X_grid = pd.DataFrame({
                "孕周_数值": t_grid,
                "孕妇BMI": np.full_like(t_grid, med.get("孕妇BMI", np.nan)),
                "身高": np.full_like(t_grid, med.get("身高", np.nan)),
                "体重": np.full_like(t_grid, med.get("体重", np.nan)),
                # 仅在特征存在于训练列时才提供列
                **({"年龄": np.full_like(t_grid, med.get("年龄", np.nan))} if "年龄" in feature_cols else {}),
                **({"GC含量": np.full_like(t_grid, med.get("GC含量", np.nan))} if "GC含量" in feature_cols else {}),
            })
            p_succ = model.predict_proba(X_grid)[:, 1]
            feasible = p_succ >= FEASIBLE_PROB_MIN
        else:
            # 线性回归（第一问风格）：组内拟合 Y ~ T，正态残差近似
            x = sub["孕周_数值"].astype(float).values
            y = sub["Y染色体浓度"].astype(float).values
            a, b = np.polyfit(x, y, 1)
            yhat = a * x + b
            resid = y - yhat
            dof = max(len(sub) - 2, 1)
            sigma = float(np.sqrt(np.sum(resid ** 2) / dof)) if dof > 0 else np.nan
            if (not np.isfinite(sigma)) or sigma <= 1e-6:
                sigma = max(float(np.std(y)) * 0.5, 1e-3)
            mu = a * t_grid + b
            miss_prob = norm.cdf((Y_THRESHOLD - mu) / max(sigma, 1e-6))
            p_succ = 1.0 - miss_prob
            feasible = mu >= Y_THRESHOLD
        risk = ALPHA * (1 - p_succ) + BETA * delay_risk(t_grid)
        # 仅在可行域内取最小风险
        if feasible.any():
            risk_masked = np.where(feasible, risk, np.inf)
            idx = int(np.argmin(risk_masked))
        else:
            idx = int(np.argmin(risk))  # 回退：若无可行点则取全域最小
        out.append({
            "BMI_group": str(g),
            "best_T": round(float(t_grid[idx]), 2),
            "risk_min": float(risk[idx]),
            "p_success": float(p_succ[idx]),
            "n": int(len(sub)),
            "T_min": float(t_min),
            "T_max": float(t_cap)
        })
    return pd.DataFrame(out)

def plot_risk_curves(df_res: pd.DataFrame, df: pd.DataFrame, model, feature_cols, predict_mode: str = PREDICT_MODE):
    if df_res.empty:
        return
    plt.figure(figsize=(7,5))
    for g in df_res["BMI_group"].values:
        sub = df[df["BMI_五分位组"] == g]
        med = sub[feature_cols].median(numeric_only=True)
        t_min = max(10.0, float(sub["孕周_数值"].min()))
        t_max_obs = float(sub["孕周_数值"].max())
        t_cap = min(T_CAP, t_max_obs)
        if not np.isfinite(t_min) or not np.isfinite(t_cap) or t_cap <= t_min:
            continue
        t_grid = np.linspace(t_min, t_cap, 400)
        if (predict_mode == 'logit') and (model is not None):
            X_grid = pd.DataFrame({
                "孕周_数值": t_grid,
                "孕妇BMI": np.full_like(t_grid, med.get("孕妇BMI", np.nan)),
                "身高": np.full_like(t_grid, med.get("身高", np.nan)),
                "体重": np.full_like(t_grid, med.get("体重", np.nan)),
                **({"年龄": np.full_like(t_grid, med.get("年龄", np.nan))} if "年龄" in feature_cols else {}),
                **({"GC含量": np.full_like(t_grid, med.get("GC含量", np.nan))} if "GC含量" in feature_cols else {}),
            })
            p_succ = model.predict_proba(X_grid)[:, 1]
            feasible = p_succ >= FEASIBLE_PROB_MIN
        else:
            x = sub["孕周_数值"].astype(float).values
            y = sub["Y染色体浓度"].astype(float).values
            a, b = np.polyfit(x, y, 1)
            resid = y - (a * x + b)
            dof = max(len(sub) - 2, 1)
            sigma = float(np.sqrt(np.sum(resid ** 2) / dof)) if dof > 0 else np.nan
            if (not np.isfinite(sigma)) or sigma <= 1e-6:
                sigma = max(float(np.std(y)) * 0.5, 1e-3)
            mu = a * t_grid + b
            miss_prob = norm.cdf((Y_THRESHOLD - mu) / max(sigma, 1e-6))
            p_succ = 1.0 - miss_prob
            feasible = mu >= Y_THRESHOLD
        risk = ALPHA * (1 - p_succ) + BETA * delay_risk(t_grid)
        if feasible.any():
            risk_masked = np.where(feasible, risk, np.nan)
        else:
            risk_masked = risk
        plt.plot(t_grid, risk_masked, label=f"组 {g}")
        best_T = float(df_res.loc[df_res["BMI_group"]==g, "best_T"].values[0])
        plt.axvline(best_T, color="gray", linestyle=":", linewidth=0.8)
    plt.title("基于逻辑回归的各BMI组风险曲线")
    plt.xlabel("孕周 T")
    plt.ylabel("Risk(T)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("logit_risk_curves_by_BMI.png", dpi=300, bbox_inches="tight")
    plt.show()

def plot_risk_curves_by_age_layer(df: pd.DataFrame, model, feature_cols, predict_mode: str = PREDICT_MODE):
    df_in = df.copy()
    # 年龄层：若可用则按两档，否则统一为All
    if ("年龄" in df_in.columns) and (df_in["年龄"].notna().sum() >= 3):
        bins = [0, 30, np.inf]
        labels = ["<30", ">=30"]
        df_in["年龄层"] = pd.cut(df_in["年龄"].astype(float), bins=bins, labels=labels, right=False, include_lowest=True)
    else:
        df_in["年龄层"] = "All"

    for age_lv in sorted(df_in["年龄层"].dropna().unique()):
        df_a = df_in[df_in["年龄层"] == age_lv].copy()
        if len(df_a) < 30:
            print(f"年龄层 {age_lv}: 样本量不足（n={len(df_a)}），跳过风险曲线绘制。")
            continue
        # 分层内 BMI 五分位
        try:
            df_a["BMI_五分位组"] = pd.qcut(df_a["孕妇BMI"], q=5, labels=["P1","P2","P3","P4","P5"], duplicates="drop")
        except Exception:
            df_a["BMI_五分位组"] = pd.qcut(df_a["孕妇BMI"], q=5, labels=None, duplicates="drop")

        plt.figure(figsize=(7,5))
        drew_any = False
        for g in sorted(df_a["BMI_五分位组"].dropna().unique()):
            sub = df_a[df_a["BMI_五分位组"] == g]
            if len(sub) < 20:
                continue
            t_min = max(10.0, float(sub["孕周_数值"].min()))
            t_max_obs = float(sub["孕周_数值"].max())
            t_cap = min(T_CAP, t_max_obs)
            if not np.isfinite(t_min) or not np.isfinite(t_cap) or t_cap <= t_min:
                continue
            t_grid = np.linspace(t_min, t_cap, 400)
            if (predict_mode == 'logit') and (model is not None):
                med = sub[feature_cols].median(numeric_only=True)
                X_grid = pd.DataFrame({
                    "孕周_数值": t_grid,
                    "孕妇BMI": np.full_like(t_grid, med.get("孕妇BMI", np.nan)),
                    "身高": np.full_like(t_grid, med.get("身高", np.nan)),
                    "体重": np.full_like(t_grid, med.get("体重", np.nan)),
                    **({"年龄": np.full_like(t_grid, med.get("年龄", np.nan))} if "年龄" in feature_cols else {}),
                    **({"GC含量": np.full_like(t_grid, med.get("GC含量", np.nan))} if "GC含量" in feature_cols else {}),
                })
                p_succ = model.predict_proba(X_grid)[:, 1]
                feasible = p_succ >= FEASIBLE_PROB_MIN
            else:
                x = sub["孕周_数值"].astype(float).values
                y = sub["Y染色体浓度"].astype(float).values
                a, b = np.polyfit(x, y, 1)
                resid = y - (a * x + b)
                dof = max(len(sub) - 2, 1)
                sigma = float(np.sqrt(np.sum(resid ** 2) / dof)) if dof > 0 else np.nan
                if (not np.isfinite(sigma)) or sigma <= 1e-6:
                    sigma = max(float(np.std(y)) * 0.5, 1e-3)
                mu = a * t_grid + b
                miss_prob = norm.cdf((Y_THRESHOLD - mu) / max(sigma, 1e-6))
                p_succ = 1.0 - miss_prob
                feasible = mu >= Y_THRESHOLD
            risk = ALPHA * (1 - p_succ) + BETA * delay_risk(t_grid)
            risk_plot = np.where(feasible, risk, np.nan) if feasible.any() else risk
            plt.plot(t_grid, risk_plot, label=f"组 {g}")
            # 标注可行域内最佳点
            if feasible.any():
                risk_masked = np.where(feasible, risk, np.inf)
                idx = int(np.argmin(risk_masked))
            else:
                idx = int(np.argmin(risk))
            plt.axvline(float(t_grid[idx]), color="gray", linestyle=":", linewidth=0.8)
            drew_any = True

        if drew_any:
            plt.title(f"年龄层 {age_lv} 的BMI组风险曲线")
            plt.xlabel("孕周 T")
            plt.ylabel("Risk(T)")
            plt.legend()
            plt.tight_layout()
            safe_age = str(age_lv).replace("<","lt").replace(">","ge").replace("=","eq")
            plt.savefig(f"risk_curves_by_BMI_age_{safe_age}.png", dpi=300, bbox_inches="tight")
            plt.show()
            # 同时输出每组的最优点摘要表
            try:
                rows = []
                for g in sorted(df_a["BMI_五分位组"].dropna().unique()):
                    sub = df_a[df_a["BMI_五分位组"] == g]
                    if len(sub) < 20:
                        continue
                    t_min = max(10.0, float(sub["孕周_数值"].min()))
                    t_max_obs = float(sub["孕周_数值"].max())
                    t_cap = min(T_CAP, t_max_obs)
                    if not np.isfinite(t_min) or not np.isfinite(t_cap) or t_cap <= t_min:
                        continue
                    t_grid = np.linspace(t_min, t_cap, 400)
                    x = sub["孕周_数值"].astype(float).values
                    y = sub["Y染色体浓度"].astype(float).values
                    a, b = np.polyfit(x, y, 1)
                    resid = y - (a * x + b)
                    dof = max(len(sub) - 2, 1)
                    sigma = float(np.sqrt(np.sum(resid ** 2) / dof)) if dof > 0 else np.nan
                    if (not np.isfinite(sigma)) or sigma <= 1e-6:
                        sigma = max(float(np.std(y)) * 0.5, 1e-3)
                    mu = a * t_grid + b
                    miss_prob = norm.cdf((Y_THRESHOLD - mu) / max(sigma, 1e-6))
                    p_succ = 1.0 - miss_prob
                    feasible = mu >= Y_THRESHOLD
                    risk = ALPHA * (1 - p_succ) + BETA * delay_risk(t_grid)
                    risk_masked = np.where(feasible, risk, np.inf) if feasible.any() else risk
                    idx = int(np.argmin(risk_masked))
                    rows.append({
                        "年龄层": str(age_lv),
                        "BMI_group": str(g),
                        "best_T": round(float(t_grid[idx]), 2),
                        "risk_min": round(float(risk[idx]), 4),
                        "miss_at_best": round(float(1 - p_succ[idx]), 4)
                    })
                if rows:
                    df_age_best = pd.DataFrame(rows)
                    outp = f"risk_best_points_by_age_{safe_age}.csv"
                    df_age_best.to_csv(outp, index=False, encoding="utf-8-sig")
                    print(f"年龄层 {age_lv} 的BMI组最优点摘要已导出: {outp}")
            except Exception as e:
                print(f"导出年龄层 {age_lv} 摘要失败: {e}")
def main():
    df = build_df()
    if df.empty:
        print("数据为空，无法训练模型。")
        return
    model_pack = train_logistic_model(df)
    if model_pack is None:
        model, feature_cols = None, ["孕周_数值","孕妇BMI","身高","体重","年龄","GC含量"]
    else:
        model, feature_cols = model_pack

    # 按BMI五分位优化
    df_res = optimize_best_time_by_bmi(df, model, feature_cols, predict_mode=PREDICT_MODE)
    if not df_res.empty:
        df_res.to_csv("logit_best_time_by_BMI.csv", index=False, encoding="utf-8-sig")
        print("最优时点结果已导出: logit_best_time_by_BMI.csv")
        plot_risk_curves(df_res, df, model, feature_cols, predict_mode=PREDICT_MODE)
        # 按年龄层分别绘制风险曲线
        plot_risk_curves_by_age_layer(df, model, feature_cols, predict_mode=PREDICT_MODE)
        print(df_res.to_string(index=False))

    # 若成功训练到逻辑回归模型，输出二分类评估信息
    if model is not None:
        try:
            from sklearn.metrics import classification_report, roc_auc_score
            X_eval = df[feature_cols]
            y_eval = (df["胎儿是否健康"].astype(str).str.strip().map({"健康":1, "正常":1, "1":1, "不健康":0, "异常":0, "0":0})
                      if ("胎儿是否健康" in df.columns) and (df["胎儿是否健康"].notna().any())
                      else df["检测阳性"]).astype(int)
            y_prob = model.predict_proba(X_eval)[:,1]
            y_pred = (y_prob >= 0.5).astype(int)
            print("\n[Logistic] 全量样本评估：")
            try:
                auc = roc_auc_score(y_eval, y_prob)
                print(f"AUC={auc:.4f}")
            except Exception:
                pass
            print(classification_report(y_eval, y_pred, digits=4, zero_division=0))
        except Exception as e:
            print(f"逻辑回归评估失败: {e}")
    else:
        print("无可导出结果。")

    # ---------------- KMeans 聚类 + 阈值扫描/比例差异检验 ----------------
    if not SKLEARN_AVAILABLE:
        print("未安装scikit-learn，跳过KMeans聚类与阈值扫描。")
        return

    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer

    cluster_feats = [c for c in ["身高","体重","年龄"] if df[c].notna().sum() >= 3]
    # 若特征不足，回退加入BMI
    if len(cluster_feats) < 2 and df["孕妇BMI"].notna().sum() >= 3:
        cluster_feats.append("孕妇BMI")
    if len(cluster_feats) < 2:
        print("可用于聚类的特征不足，跳过聚类阈值扫描。")
        return

    K = 3 if len(df) >= 60 else max(2, min(3, len(df)//20))
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    Xc = scaler.fit_transform(imputer.fit_transform(df[cluster_feats]))
    km = KMeans(n_clusters=K, random_state=42, n_init=10)
    labels = km.fit_predict(Xc)
    df_cls = df.copy()
    df_cls["ClusterHW"] = labels

    results = []
    candidates = [10,12,14,16,18,20,22,24,26,28]
    for c in sorted(df_cls["ClusterHW"].unique()):
        sub = df_cls.loc[df_cls["ClusterHW"] == c].copy()
        if len(sub) < 20:
            print(f"Cluster {c}: 样本量不足（n={len(sub)}），跳过阈值扫描。")
            continue
        best = None
        for k in candidates:
            grp = (sub["孕周_数值"] >= k).astype(int)
            tab = pd.crosstab(grp, sub["检测阳性"])  # 2x2
            if tab.shape == (2,2):
                chi2, p, _, _ = chi2_contingency(tab.values)
                rate_low = tab.loc[0,1] / tab.loc[0].sum() if tab.loc[0].sum() > 0 else np.nan
                rate_high = tab.loc[1,1] / tab.loc[1].sum() if tab.loc[1].sum() > 0 else np.nan
                diff = (rate_high - rate_low) * 100 if np.isfinite(rate_low) and np.isfinite(rate_high) else np.nan
                rec = (p, chi2, k, diff, int(tab.values.sum()), int(sub.shape[0]))
                if best is None or p < best[0]:
                    best = rec
        if best is not None:
            p, chi2, k, diff, n_tot, n_sub = best
            results.append({
                "Cluster": int(c),
                "best_K": int(k),
                "chi2": float(chi2),
                "p_value": float(p),
                "rate_diff_pct": float(diff) if np.isfinite(diff) else np.nan,
                "n_table": int(n_tot),
                "n_cluster": int(n_sub)
            })

    if results:
        df_out = pd.DataFrame(results).sort_values(by=["p_value","Cluster"]).reset_index(drop=True)
        df_out.to_csv("cluster_threshold_scan.csv", index=False, encoding="utf-8-sig")
        print("聚类阈值扫描结果已导出: cluster_threshold_scan.csv")
        print(df_out.to_string(index=False))

    # ---------------- 年龄分层 → BMI分组 → PCA成分分组 → 阈值扫描 ----------------
    if not SKLEARN_AVAILABLE:
        print("未安装scikit-learn，跳过 PCA 分层阈值扫描。")
        return

    def pca_stratified_threshold_scan(df_in: pd.DataFrame) -> pd.DataFrame:
        df_in = df_in.copy()
        # 年龄分层（若不可用，则统一为 All）
        if ("年龄" in df_in.columns) and (df_in["年龄"].notna().sum() >= 3):
            bins = [0, 30, np.inf]
            labels = ["<30", ">=30"]
            df_in["年龄层"] = pd.cut(df_in["年龄"].astype(float), bins=bins, labels=labels, right=False, include_lowest=True)
            # 打印年龄分层统计
            vc_age = df_in["年龄层"].value_counts(dropna=False).sort_index()
            print("年龄分层：区间 [0,30), [30,∞)")
            print(vc_age.to_string())
            # 可视化：年龄层计数
            try:
                plt.figure(figsize=(4.8, 3.2))
                vc_age.plot(kind="bar", color="#4e79a7")
                plt.title("年龄层样本量")
                plt.xlabel("年龄层")
                plt.ylabel("样本数")
                plt.tight_layout()
                plt.savefig("age_layer_counts.png", dpi=300, bbox_inches="tight")
                plt.show()
            except Exception as e:
                print(f"绘制年龄层计数图失败: {e}")
        else:
            df_in["年龄层"] = "All"

        out_rows = []
        best_rows = []  # 存放各(年龄层×BMI组×PC1组)的最佳时点与误差敏感性
        eq_rows = []  # 保存各年龄层的PC方程（标准化与原始变量）
        age_levels = sorted(df_in["年龄层"].dropna().unique())
        for age_lv in age_levels:
            df_a = df_in[df_in["年龄层"] == age_lv].copy()
            if len(df_a) < 30:
                print(f"年龄层 {age_lv}: 样本量不足（n={len(df_a)}），跳过。")
                continue
            # 分层内 BMI 五分位
            try:
                df_a["BMI_五分位组"] = pd.qcut(df_a["孕妇BMI"], q=5, labels=["P1","P2","P3","P4","P5"], duplicates="drop")
            except Exception:
                df_a["BMI_五分位组"] = pd.qcut(df_a["孕妇BMI"], q=5, labels=None, duplicates="drop")
            # 打印BMI五分位分组统计
            try:
                vc_bmi = df_a["BMI_五分位组"].value_counts().sort_index()
                print(f"年龄层 {age_lv} 的 BMI 五分位分组样本数：\n{vc_bmi.to_string()}")
                # 可视化：年龄层内 BMI 组计数
                plt.figure(figsize=(5.2, 3.2))
                vc_bmi.plot(kind="bar", color="#59a14f")
                plt.title(f"年龄层 {age_lv} 的BMI组样本量")
                plt.xlabel("BMI组")
                plt.ylabel("样本数")
                plt.tight_layout()
                safe_age = str(age_lv).replace("<","lt").replace(">","ge").replace("=","eq")
                plt.savefig(f"bmi_quintile_counts_age_{safe_age}.png", dpi=300, bbox_inches="tight")
                plt.show()
            except Exception:
                pass

            # PCA 特征选择与拟合
            pca_feats_all = [c for c in ["Y染色体浓度","孕妇BMI","身高","体重","GC含量"] if c in df_a.columns]
            pca_feats = [c for c in pca_feats_all if df_a[c].notna().sum() >= 5]
            if len(pca_feats) < 2:
                print(f"年龄层 {age_lv}: 可用于PCA的特征不足，跳过。")
                continue
            Xp = df_a[pca_feats].copy()
            Xp = Xp.fillna(Xp.median(numeric_only=True))
            scaler = StandardScaler()
            Xs = scaler.fit_transform(Xp.values)
            n_comp = min(2, len(pca_feats))
            pca = PCA(n_components=n_comp)
            scores = pca.fit_transform(Xs)
            df_a["PC1"] = scores[:, 0]
            if n_comp >= 2:
                df_a["PC2"] = scores[:, 1]

            # 打印并记录PC方程（标准化与原始变量形式）
            try:
                v1 = pca.components_[0]
                scales = np.array(scaler.scale_)
                means = np.array(scaler.mean_)
                # 原始变量系数与截距：PC1 ≈ c1 + Σ w_orig_j * x_j
                w_orig = v1 / np.where(np.abs(scales) < 1e-12, 1.0, scales)
                c1 = -float(np.sum(w_orig * means))
                # 组装打印文本
                std_terms = " + ".join([f"{v1[i]:+.4f}·Z[{pca_feats[i]}]" for i in range(len(pca_feats))])
                orig_terms = " + ".join([f"{w_orig[i]:+.6f}·{pca_feats[i]}" for i in range(len(pca_feats))])
                print(f"年龄层 {age_lv} 的PC1（标准化变量）: PC1 = {std_terms}")
                print(f"年龄层 {age_lv} 的PC1（原始变量近似）: PC1 ≈ {c1:+.6f} + {orig_terms}")
                eq_row = {"年龄层": str(age_lv), "component": "PC1", "intercept": round(c1, 6)}
                for i, feat in enumerate(pca_feats):
                    eq_row[f"coef[{feat}]"] = float(round(w_orig[i], 6))
                eq_rows.append(eq_row)
            except Exception as e:
                print(f"年龄层 {age_lv}: 打印PC1方程失败: {e}")

            # PC1 分位分组（3组）
            try:
                df_a["PC1_分组"] = pd.qcut(df_a["PC1"], q=3, labels=["C1","C2","C3"], duplicates="drop")
            except Exception:
                df_a["PC1_分组"] = pd.qcut(df_a["PC1"], q=3, labels=None, duplicates="drop")
            # 打印PC1分组统计
            try:
                vc_pc1 = df_a["PC1_分组"].value_counts().sort_index()
                print(f"年龄层 {age_lv} 的 PC1 三分位分组样本数：\n{vc_pc1.to_string()}")
                # 可视化：年龄层内 PC1 组计数
                plt.figure(figsize=(5.2, 3.2))
                vc_pc1.plot(kind="bar", color="#e15759")
                plt.title(f"年龄层 {age_lv} 的PC1组样本量")
                plt.xlabel("PC1组")
                plt.ylabel("样本数")
                plt.tight_layout()
                safe_age = str(age_lv).replace("<","lt").replace(">","ge").replace("=","eq")
                plt.savefig(f"pc1_tertile_counts_age_{safe_age}.png", dpi=300, bbox_inches="tight")
                plt.show()
            except Exception:
                pass

            candidates = [10,12,14,16,18,20,22,24,26,28]
            for bmi_g in sorted(df_a["BMI_五分位组"].dropna().unique()):
                for pc_g in sorted(df_a["PC1_分组"].dropna().unique()):
                    sub = df_a.loc[(df_a["BMI_五分位组"] == bmi_g) & (df_a["PC1_分组"] == pc_g), ["孕周_数值","Y染色体浓度","检测阳性"]].dropna()
                    n = len(sub)
                    if n < 20:
                        # 样本不足，跳过
                        continue
                    # ---- 预测最优时点（线性近似 y~t + 正态残差，带可行域 mu>=阈值）----
                    try:
                        x = sub["孕周_数值"].astype(float).values
                        y = sub["Y染色体浓度"].astype(float).values
                        a, b = np.polyfit(x, y, 1)
                        yhat = a * x + b
                        resid = y - yhat
                        dof = max(n - 2, 1)
                        sigma = float(np.sqrt(np.sum(resid ** 2) / dof)) if dof > 0 else np.nan
                        if (not np.isfinite(sigma)) or sigma <= 1e-6:
                            sigma = max(float(np.std(y)) * 0.5, 1e-3)
                        t_min = float(max(10.0, np.nanmin(x)))
                        t_max_obs = float(np.nanmax(x))
                        t_cap = float(min(T_CAP, t_max_obs))
                        if not np.isfinite(t_min) or not np.isfinite(t_cap) or t_cap <= t_min:
                            raise ValueError("孕周范围异常")
                        t_grid = np.linspace(t_min, t_cap, 400)
                        mu = a * t_grid + b
                        delay = delay_risk(t_grid)
                        # 基准阈值
                        z = (Y_THRESHOLD - mu) / max(sigma, 1e-6)
                        miss = norm.cdf(z)
                        risk = ALPHA * miss + BETA * delay
                        feasible = mu >= Y_THRESHOLD
                        if feasible.any():
                            risk_eff = np.where(feasible, risk, np.inf)
                        else:
                            risk_eff = risk
                        idx = int(np.argmin(risk_eff))
                        t_best = float(t_grid[idx])
                        risk_best = float(risk[idx])
                        miss_best = float(miss[idx])
                        delay_best = float(delay[idx])
                        # 阈值敏感性（±0.005）
                        def _best_with_threshold(thr: float) -> float:
                            z2 = (thr - mu) / max(sigma, 1e-6)
                            miss2 = norm.cdf(z2)
                            risk2 = ALPHA * miss2 + BETA * delay
                            feas2 = mu >= thr
                            risk2_eff = np.where(feas2, risk2, np.inf) if feas2.any() else risk2
                            return float(t_grid[int(np.argmin(risk2_eff))])
                        t_best_thr_low = _best_with_threshold(max(0.0, Y_THRESHOLD - 0.005))
                        t_best_thr_high = _best_with_threshold(Y_THRESHOLD + 0.005)
                        # 残差方差敏感性（±20%）
                        def _best_with_sigma(scale: float) -> float:
                            z3 = (Y_THRESHOLD - mu) / max(sigma * scale, 1e-6)
                            miss3 = norm.cdf(z3)
                            risk3 = ALPHA * miss3 + BETA * delay
                            feas3 = mu >= Y_THRESHOLD
                            risk3_eff = np.where(feas3, risk3, np.inf) if feas3.any() else risk3
                            return float(t_grid[int(np.argmin(risk3_eff))])
                        t_best_sig_low = _best_with_sigma(0.8)
                        t_best_sig_high = _best_with_sigma(1.2)
                        best_rows.append({
                            "年龄层": str(age_lv),
                            "BMI_group": str(bmi_g),
                            "PC1_group": str(pc_g),
                            "n": int(n),
                            "T_min": float(t_min),
                            "T_max": float(t_cap),
                            "best_T": round(t_best, 2),
                            "risk_min": round(risk_best, 4),
                            "miss_at_best": round(miss_best, 4),
                            "delay_at_best": round(delay_best, 4),
                            "best_T_thr_minus": round(t_best_thr_low, 2),
                            "best_T_thr_plus": round(t_best_thr_high, 2),
                            "best_T_sigma_minus": round(t_best_sig_low, 2),
                            "best_T_sigma_plus": round(t_best_sig_high, 2),
                        })
                    except Exception as e:
                        pass
                    best = None
                    for k in candidates:
                        grp = (sub["孕周_数值"] >= k).astype(int)
                        tab = pd.crosstab(grp, sub["检测阳性"])  # 2x2
                        if tab.shape == (2,2):
                            chi2, p, _, _ = chi2_contingency(tab.values)
                            rate_low = tab.loc[0,1] / tab.loc[0].sum() if tab.loc[0].sum() > 0 else np.nan
                            rate_high = tab.loc[1,1] / tab.loc[1].sum() if tab.loc[1].sum() > 0 else np.nan
                            diff = (rate_high - rate_low) * 100 if np.isfinite(rate_low) and np.isfinite(rate_high) else np.nan
                            rec = (p, chi2, k, diff, int(tab.values.sum()), int(n))
                            if best is None or p < best[0]:
                                best = rec
                    if best is not None:
                        p, chi2, k, diff, n_tot, n_sub = best
                        out_rows.append({
                            "年龄层": str(age_lv),
                            "BMI_group": str(bmi_g),
                            "PC1_group": str(pc_g),
                            "best_K": int(k),
                            "chi2": float(chi2),
                            "p_value": float(p),
                            "rate_diff_pct": float(diff) if np.isfinite(diff) else np.nan,
                            "n_table": int(n_tot),
                            "n_class": int(n_sub)
                        })
        # 导出PC方程
        if eq_rows:
            df_eq = pd.DataFrame(eq_rows)
            df_eq.to_csv("pca_equations_by_age.csv", index=False, encoding="utf-8-sig")
            print("各年龄层PC1方程已导出: pca_equations_by_age.csv")
            try:
                print(df_eq.to_string(index=False))
            except Exception:
                pass
        # 导出各组最佳时点与误差敏感性
        if best_rows:
            df_best = pd.DataFrame(best_rows).sort_values(["年龄层","BMI_group","PC1_group"]).reset_index(drop=True)
            df_best.to_csv("pca_stratified_best_time.csv", index=False, encoding="utf-8-sig")
            print("PCA分层各组最佳时点（含误差敏感性）已导出: pca_stratified_best_time.csv")
            try:
                print(df_best.head(20).to_string(index=False))
            except Exception:
                pass
        return pd.DataFrame(out_rows)

    df_pca_scan = pca_stratified_threshold_scan(df)
    if not df_pca_scan.empty:
        df_pca_scan = df_pca_scan.sort_values(["p_value","年龄层","BMI_group","PC1_group"]).reset_index(drop=True)
        df_pca_scan.to_csv("pca_stratified_threshold_scan.csv", index=False, encoding="utf-8-sig")
        print("PCA分层阈值扫描结果已导出: pca_stratified_threshold_scan.csv")
        print(df_pca_scan.head(20).to_string(index=False))

if __name__ == "__main__":
    main()