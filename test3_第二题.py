import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from test import build, build2
from scipy.stats import norm, chi2_contingency, fisher_exact
import importlib
_sk_spec = importlib.util.find_spec("sklearn")
if _sk_spec is not None:
    try:
        KMeans = importlib.import_module("sklearn.cluster").KMeans
        StandardScaler = importlib.import_module("sklearn.preprocessing").StandardScaler
        SKLEARN_AVAILABLE = True
    except Exception:
        SKLEARN_AVAILABLE = False
else:
    SKLEARN_AVAILABLE = False

#灵敏度设置
# Risk = α * 漏检概率 + β * 延误风险，要求 β > α，这里默认 α=1, β=3
ALPHA = 1.0
BETA = 1.1

data = build2()

Y_list=[t.Y染色体浓度 for t in data]
T_list = [t.孕周_数值 for t in data]
BMI_list=[t.孕妇BMI for t in data]

# 以四分位数与五分位数对 BMI 分组，保证组数合理且各组样本量尽量接近
df = pd.DataFrame({
    "Y染色体浓度": Y_list,
    "孕周_数值": T_list,
    "孕妇BMI": BMI_list
})

# ---- BMI极端值清洗与排序 ----
# 1) 仅保留非缺失BMI；2) IQR法剔除极端值（IQR=Q3-Q1，阈值=Q1-1.5IQR~Q3+1.5IQR）
# 若IQR过小/为0，则退化为1%~99%分位过滤；3) 按BMI升序排序并重建索引
bmi_series = pd.to_numeric(df["孕妇BMI"], errors="coerce")
valid_mask = bmi_series.notna()
df_valid = df.loc[valid_mask].copy()

q1 = df_valid["孕妇BMI"].quantile(0.25)
q3 = df_valid["孕妇BMI"].quantile(0.75)
iqr = q3 - q1

if pd.isna(iqr) or iqr <= 0:
    low = df_valid["孕妇BMI"].quantile(0.01)
    high = df_valid["孕妇BMI"].quantile(0.99)
else:
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr

df_clean = df_valid[(df_valid["孕妇BMI"] >= low) & (df_valid["孕妇BMI"] <= high)].copy()
df_clean = df_clean.sort_values(by="孕妇BMI", ascending=True).reset_index(drop=True)

removed_n = len(df_valid) - len(df_clean)
print(f"已按BMI升序排序，并清洗极端值 {removed_n} 条；阈值区间: [{low:.3f}, {high:.3f}]")

# 将清洗结果作为后续分析数据
df = df_clean

# 仅对非缺失 BMI 进行分组（此时应全为非缺失）
mask_bmi = df["孕妇BMI"].notna()

# 四分位分组（q=4），若存在大量重复值导致分箱不可行，则自动合并（duplicates='drop'）；
# 若因分箱减少导致标签长度不匹配，则回退为无标签的区间分类。
try:
    df.loc[mask_bmi, "BMI_四分位组"], bins4 = pd.qcut(
        df.loc[mask_bmi, "孕妇BMI"],
        q=4,
        labels=["Q1","Q2","Q3","Q4"],
        retbins=True,
        duplicates="drop"
    )
except ValueError:
    df.loc[mask_bmi, "BMI_四分位组"], bins4 = pd.qcut(
        df.loc[mask_bmi, "孕妇BMI"],
        q=4,
        labels=None,
        retbins=True,
        duplicates="drop"
    )

# 五分位分组（q=5）
try:
    df.loc[mask_bmi, "BMI_五分位组"], bins5 = pd.qcut(
        df.loc[mask_bmi, "孕妇BMI"],
        q=5,
        labels=["P1","P2","P3","P4","P5"],
        retbins=True,
        duplicates="drop"
    )
except ValueError:
    df.loc[mask_bmi, "BMI_五分位组"], bins5 = pd.qcut(
        df.loc[mask_bmi, "孕妇BMI"],
        q=5,
        labels=None,
        retbins=True,
        duplicates="drop"
    )

# 打印分组边界与样本量分布，检查是否基本均衡
def _fmt_bins(bins):
    return ", ".join([f"{b:.3f}" for b in bins])

counts4 = df.loc[mask_bmi, "BMI_四分位组"].value_counts().sort_index()
counts5 = df.loc[mask_bmi, "BMI_五分位组"].value_counts().sort_index()

print("\nBMI 四分位分组：")
print(f"分组数: 期望4，实际{len(bins4)-1}")
print(f"边界: [{_fmt_bins(bins4)}]")
print("各组样本量：")
print(counts4.to_string())
if len(bins4) - 1 < 4:
    print("注意：由于BMI存在较多重复值，四分位分组的有效组数少于4，已自动合并。")
total4 = int(counts4.sum()) if len(counts4) > 0 else 0
if total4 > 0:
    max_group4 = counts4.idxmax()
    max_n4 = int(counts4.max())
    max_pct4 = max_n4 / total4 * 100
    print(f"四分位最大占比: 组 {max_group4}, 占比={max_pct4:.1f}%, 样本量n={max_n4}/{total4}")

print("\nBMI 五分位分组：")
print(f"分组数: 期望5，实际{len(bins5)-1}")
print(f"边界: [{_fmt_bins(bins5)}]")
print("各组样本量：")
print(counts5.to_string())
if len(bins5) - 1 < 5:
    print("注意：由于BMI存在较多重复值，五分位分组的有效组数少于5，已自动合并。")
total5 = int(counts5.sum()) if len(counts5) > 0 else 0
if total5 > 0:
    max_group5 = counts5.idxmax()
    max_n5 = int(counts5.max())
    max_pct5 = max_n5 / total5 * 100
    print(f"五分位最大占比: 组 {max_group5}, 占比={max_pct5:.1f}%, 样本量n={max_n5}/{total5}")

# 如需将结果用于后续分析，可从 df["BMI_四分位组"], df["BMI_五分位组"] 读取

# ---------------------- BMI 分布直方图 ----------------------
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

if mask_bmi.sum() > 0:
    plt.figure(figsize=(6, 4))
    bmi_vals = df.loc[mask_bmi, "孕妇BMI"].astype(float).values
    plt.hist(bmi_vals, bins='auto', color="#69b3a2", edgecolor="white", alpha=0.85)
    plt.title("BMI分布直方图")
    plt.xlabel("孕妇BMI")
    plt.ylabel("频数")
    plt.tight_layout()
    plt.savefig("hist_BMI.png", dpi=300, bbox_inches="tight")
    plt.show()
else:
    print("BMI 全为缺失，无法绘制直方图。")

# ---------------------- BMI 分组饼图 ----------------------
if mask_bmi.sum() > 0:
    # 优先使用五分位分组，若类别数不足则回退四分位分组
    series_group = df.loc[mask_bmi, "BMI_五分位组"].dropna()
    title_suffix = "五分位"
    if series_group.nunique() < 2 and df.loc[mask_bmi, "BMI_四分位组"].notna().any():
        series_group = df.loc[mask_bmi, "BMI_四分位组"].dropna()
        title_suffix = "四分位"

    if series_group.nunique() >= 2:
        counts = series_group.value_counts().sort_index()
        labels = [str(idx) for idx in counts.index]
        sizes = counts.values

        plt.figure(figsize=(5.5, 5.5))
        plt.pie(
            sizes,
            labels=labels,
            autopct='%.1f%%',
            startangle=90,
            counterclock=False
        )
        plt.title(f"BMI分组占比（{title_suffix}）")
        plt.tight_layout()
        plt.savefig("pie_BMI_group.png", dpi=300, bbox_inches="tight")
        plt.show()
    else:
        print("BMI 分组类别不足，无法绘制饼图。")
else:
    print("BMI 全为缺失，无法绘制饼图。")

# ---------------------- 按BMI五分位分组的风险最优时点 ----------------------
# Risk = α * 漏检概率 + β * 延误风险，要求 β > α，这里默认 α=1, β=3
# 设定检测阈值（可按业务调整），参考前文示例取 0.04
Y_THRESHOLD = 0.04

# 非线性延误风险：在 [t0, tcap] 内按幂次增长；t < t0 视为风险=0
T_CAP = 28.0
DELAY_RISK_GAMMA = 2.0  # >1 为凸增长，越晚风险增长越快

def _delay_risk(t_candidates: np.ndarray, t0: float = 10.0, tcap: float = T_CAP, gamma: float = DELAY_RISK_GAMMA) -> np.ndarray:
    if tcap <= t0:
        return np.zeros_like(t_candidates, dtype=float)
    base = (t_candidates - t0) / max(tcap - t0, 1e-6)
    base = np.clip(base, 0.0, 1.0)
    risks = np.where(t_candidates < t0, 0.0, np.power(base, gamma))
    return np.clip(risks, 0.0, 1.0)

# 分析函数
def analyze_bmi_groups(df, group_col_name, output_suffix):
    results = []
    if group_col_name not in df.columns or df[group_col_name].dropna().nunique() < 2:
        print(f"{group_col_name} 分组不可用，跳过分析。")
        return
    
    groups = df[group_col_name].dropna().sort_values().unique()
    for g in groups:
        mask_g = df[group_col_name] == g
        sub = df.loc[mask_g & df["孕周_数值"].notna() & df["Y染色体浓度"].notna(), ["孕周_数值", "Y染色体浓度"]]
        n = len(sub)
        if n < 3:
            print(f"分组 {g}: 样本量不足（n={n}），跳过最优时点计算。")
            continue

        x = sub["孕周_数值"].astype(float).values
        y = sub["Y染色体浓度"].astype(float).values

        # 线性拟合 Y ~ T
        a, b = np.polyfit(x, y, 1)
        yhat = a * x + b
        resid = y - yhat
        # 残差标准差（无偏估计）
        dof = max(n - 2, 1)
        sigma = float(np.sqrt(np.sum(resid ** 2) / dof))
        if not np.isfinite(sigma) or sigma <= 1e-6:
            sigma = max(float(np.std(y)) * 0.5, 1e-3)

        # 在T≥10的范围内搜索最优时点
        t_min = float(max(10.0, np.nanmin(x)))
        t_max_obs = float(np.nanmax(x))
        t_cap = float(min(T_CAP, t_max_obs))
        if not np.isfinite(t_min) or not np.isfinite(t_cap) or t_cap <= t_min:
            print(f"分组 {g}: 孕周范围异常（[{t_min}, {t_cap}]），跳过。")
            continue

        t_grid = np.arange(t_min, t_cap + 1e-9, 0.1)
        mu = a * t_grid + b
        # 漏检概率：P(Y < 阈值) = Phi((Y* - mu)/sigma)
        z = (Y_THRESHOLD - mu) / max(sigma, 1e-6)
        miss_prob = norm.cdf(z)
        delay_prob = _delay_risk(t_grid, t0=10.0, tcap=t_cap, gamma=DELAY_RISK_GAMMA)
        risk = ALPHA * miss_prob + BETA * delay_prob

        idx = int(np.argmin(risk))
        t_best = float(t_grid[idx])
        results.append({
            "BMI_group": str(g),
            "best_T": round(t_best, 2),
            "risk_min": float(risk[idx]),
            "miss_prob_at_T": float(miss_prob[idx]),
            "delay_risk_at_T": float(delay_prob[idx]),
            "slope_a": float(a),
            "intercept_b": float(b),
            "sigma": float(sigma),
            "n": int(n),
            "T_grid_min": float(t_min),
            "T_grid_max": float(t_cap)
        })

    if results:
        df_res = pd.DataFrame(results)
        df_res["alpha"] = ALPHA
        df_res["beta"] = BETA
        df_res["Y_threshold"] = Y_THRESHOLD
        df_res = df_res.sort_values(by="BMI_group").reset_index(drop=True)
        print(f"\n按BMI{output_suffix}分组的最佳时点（T≥10）：")
        print(df_res.to_string(index=False))
        out_path = f"risk_opt_by_BMI_{output_suffix}.csv"
        df_res.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"结果已导出: {out_path}")
        return df_res, results
    return None, None

# 执行分析
if mask_bmi.sum() > 0 and df["孕周_数值"].notna().any() and df["Y染色体浓度"].notna().any():
    # 1. 四分位分析
    df_quartile, results_quartile = analyze_bmi_groups(df, "BMI_四分位组", "quartile")
    
    # 2. 五分位分析
    df_quintile, results_quintile = analyze_bmi_groups(df, "BMI_五分位组", "quintile")

    # 绘制风险函数曲线
    def plot_risk_curves(results, title_suffix, filename_suffix):
        if not results:
            return
        try:
            plt.figure(figsize=(7, 5))
            for r in results:
                g = r["BMI_group"]
                a = r["slope_a"]; b = r["intercept_b"]; sigma = max(r["sigma"], 1e-6)
                t0 = max(10.0, r["T_grid_min"])
                t1 = min(T_CAP, r["T_grid_max"]) if "T_grid_max" in r else T_CAP
                if not np.isfinite(t0) or not np.isfinite(t1) or t1 <= t0:
                    continue
                t_grid = np.linspace(t0, t1, 300)
                mu = a * t_grid + b
                z = (Y_THRESHOLD - mu) / sigma
                miss_prob = norm.cdf(z)
                delay_prob = _delay_risk(t_grid, t0=10.0, tcap=T_CAP, gamma=DELAY_RISK_GAMMA)
                risk = ALPHA * miss_prob + BETA * delay_prob
                plt.plot(t_grid, risk, label=f"组 {g}")
            plt.text(0.02, 0.02, f"Risk(T)=α·P(Y<阈值)+β·Delay(T)\nDelay: T<10→0; T∈[10,28]→((T-10)/(28-10))^{DELAY_RISK_GAMMA}", transform=plt.gca().transAxes, fontsize=8, va='bottom', ha='left')
            plt.title(f"各BMI{title_suffix}组风险函数曲线 Risk(T)")
            plt.xlabel("孕周 T")
            plt.ylabel("Risk(T)")
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"risk_curves_by_BMI_{filename_suffix}.png", dpi=300, bbox_inches="tight")
            plt.show()
        except Exception as e:
            print(f"绘制{title_suffix}风险曲线时发生错误: {e}")
    
    # 绘制四分位和五分位的风险曲线
    if results_quartile:
        plot_risk_curves(results_quartile, "四分位", "quartile")
    if results_quintile:
        plot_risk_curves(results_quintile, "五分位", "quintile")

        # ---- 绘制非线性延误风险曲线（仅展示形状） ----
        try:
            t_demo = np.linspace(0, T_CAP, 400)
            delay_demo = _delay_risk(t_demo, t0=10.0, tcap=T_CAP, gamma=DELAY_RISK_GAMMA)
            plt.figure(figsize=(6,4))
            plt.plot(t_demo, delay_demo, color="#d62728")
            plt.axvline(10.0, color="gray", linestyle="--", linewidth=1)
            plt.text(10.1, 0.05, "T=10", fontsize=9)
            plt.title(f"非线性延误风险（γ={DELAY_RISK_GAMMA}）\nT<10: 0; T∈[10,28]: ((T-10)/(28-10))^γ")
            plt.xlabel("孕周 T")
            plt.ylabel("延误风险")
            plt.tight_layout()
            plt.savefig("delay_risk_curve.png", dpi=300, bbox_inches="tight")
            plt.show()
        except Exception as e:
            print(f"绘制延误风险曲线时发生错误: {e}")

        # ---- 为样本量最大的组绘制 Risk / 漏检 / 延误 组件曲线 ----
        try:
            # 从四分位或五分位结果中选择样本量最大的组
            all_results = []
            if results_quartile:
                all_results.extend(results_quartile)
            if results_quintile:
                all_results.extend(results_quintile)
            
            if all_results:
                r_star = max(all_results, key=lambda r: r.get("n", 0))
                a = r_star["slope_a"]; b = r_star["intercept_b"]; sigma = max(r_star["sigma"], 1e-6)
                t0 = max(10.0, r_star["T_grid_min"])
                t1 = min(T_CAP, r_star["T_grid_max"]) if "T_grid_max" in r_star else T_CAP
                if np.isfinite(t0) and np.isfinite(t1) and t1 > t0:
                    t_grid = np.linspace(t0, t1, 400)
                    mu = a * t_grid + b
                    miss_prob = norm.cdf((Y_THRESHOLD - mu) / sigma)
                    delay_prob = _delay_risk(t_grid, t0=10.0, tcap=T_CAP, gamma=DELAY_RISK_GAMMA)
                    risk = ALPHA * miss_prob + BETA * delay_prob
                    plt.figure(figsize=(7,4.5))
                    plt.plot(t_grid, risk, label="Risk(T)", color="#1f77b4")
                    plt.plot(t_grid, miss_prob, label="漏检概率", color="#2ca02c")
                    plt.plot(t_grid, delay_prob, label="延误风险", color="#d62728")
                    plt.axhline(Y_THRESHOLD, color="gray", linestyle=":", linewidth=0.8)
                    plt.title(f"组件曲线（样本量最大组 {r_star['BMI_group']}）")
                    plt.xlabel("孕周 T")
                    plt.ylabel("值")
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig("risk_components_largest_group.png", dpi=300, bbox_inches="tight")
                    plt.show()
            else:
                print("没有可用的结果用于绘制组件曲线")
        except Exception as e:
            print(f"绘制组件曲线时发生错误: {e}")
    else:
        print("无可计算的分组结果，未导出CSV。")
else:
    print("数据不足以计算风险最优时点（BMI或孕周或Y缺失）。")

# ---------------------- KMeans 聚类与风险最优时点 ----------------------
try:
    # 选取用于聚类的特征：孕周、BMI、Y；仅保留非缺失
    df_k = df[["孕周_数值", "孕妇BMI", "Y染色体浓度"]].dropna().copy()
    n_k = len(df_k)
    if n_k >= 3:
        K = min(3, n_k)  # 默认K=3，若样本不足则自动下调
        # 兜底导入 StandardScaler / KMeans，避免名字未定义
        try:
            _StandardScaler = StandardScaler
        except NameError:
            try:
                _StandardScaler = importlib.import_module("sklearn.preprocessing").StandardScaler
            except Exception as e:
                print(f"KMeans 聚类步骤出错: StandardScaler导入失败: {e}")
                raise
        try:
            _KMeans = KMeans
        except NameError:
            try:
                _KMeans = importlib.import_module("sklearn.cluster").KMeans
            except Exception as e:
                print(f"KMeans 聚类步骤出错: KMeans导入失败: {e}")
                raise

        scaler = _StandardScaler()
        X = scaler.fit_transform(df_k[["孕周_数值", "孕妇BMI", "Y染色体浓度"]])
        kmeans = _KMeans(n_clusters=K, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        df_k["KMeans_Cluster"] = labels.astype(int)
        # 打印KMeans最大占比与样本量
        counts_km = df_k["KMeans_Cluster"].value_counts()
        total_km = int(counts_km.sum()) if len(counts_km) > 0 else 0
        if total_km > 0:
            max_cluster = int(counts_km.idxmax())
            max_nk = int(counts_km.max())
            max_pctk = max_nk / total_km * 100
            print(f"KMeans最大占比: Cluster {max_cluster}, 占比={max_pctk:.1f}%, 样本量n={max_nk}/{total_km}")

        # 将聚类标签并回主df（按索引对齐）
        df = df.join(df_k["KMeans_Cluster"], how="left")

        # 按聚类计算最优时点
        km_results = []
        for c in sorted(df_k["KMeans_Cluster"].unique()):
            sub = df_k[df_k["KMeans_Cluster"] == c]
            x = sub["孕周_数值"].astype(float).values
            y = sub["Y染色体浓度"].astype(float).values
            n = len(sub)
            if n < 3:
                print(f"KMeans 聚类 {c}: 样本量不足（n={n}），跳过。")
                continue
            a, b = np.polyfit(x, y, 1)
            yhat = a * x + b
            resid = y - yhat
            dof = max(n - 2, 1)
            sigma = float(np.sqrt(np.sum(resid ** 2) / dof))
            if not np.isfinite(sigma) or sigma <= 1e-6:
                sigma = max(float(np.std(y)) * 0.5, 1e-3)

            t_min = float(max(10.0, np.nanmin(x)))
            t_max_obs = float(np.nanmax(x))
            t_cap = float(min(T_CAP, t_max_obs))
            if not np.isfinite(t_min) or not np.isfinite(t_cap) or t_cap <= t_min:
                print(f"KMeans 聚类 {c}: 孕周范围异常（[{t_min}, {t_cap}]），跳过。")
                continue
            t_grid = np.arange(t_min, t_cap + 1e-9, 0.1)
            mu = a * t_grid + b
            miss_prob = norm.cdf((Y_THRESHOLD - mu) / max(sigma, 1e-6))
            delay_prob = _delay_risk(t_grid, t0=10.0, tcap=T_CAP, gamma=DELAY_RISK_GAMMA)
            risk = ALPHA * miss_prob + BETA * delay_prob
            idx = int(np.argmin(risk))
            t_best = float(t_grid[idx])

            km_results.append({
                "Cluster": int(c),
                "best_T": round(t_best, 2),
                "risk_min": float(risk[idx]),
                "miss_prob_at_T": float(miss_prob[idx]),
                "delay_risk_at_T": float(delay_prob[idx]),
                "slope_a": float(a),
                "intercept_b": float(b),
                "sigma": float(sigma),
                "n": int(n),
                "T_grid_min": float(t_min),
                "T_grid_max": float(t_cap)
            })

        if km_results:
            df_km = pd.DataFrame(km_results).sort_values(by="Cluster").reset_index(drop=True)
            df_km["alpha"] = ALPHA
            df_km["beta"] = BETA
            df_km["Y_threshold"] = Y_THRESHOLD
            df_km["gamma"] = DELAY_RISK_GAMMA
            print("\nKMeans 聚类最优时点（T≥10，T≤28）：")
            print(df_km.to_string(index=False))
            out_km = "risk_opt_by_kmeans.csv"
            df_km.to_csv(out_km, index=False, encoding="utf-8-sig")
            print(f"结果已导出: {out_km}")

            # TY散点（按聚类着色）
            plt.figure(figsize=(6.5, 5))
            colors = plt.cm.tab10(np.linspace(0, 1, max(10, K)))
            for c in sorted(df_k["KMeans_Cluster"].unique()):
                sub = df_k[df_k["KMeans_Cluster"] == c]
                plt.scatter(sub["孕周_数值"], sub["Y染色体浓度"], s=25, alpha=0.8, label=f"Cluster {c}")
            # 可选：在TY平面绘制聚类中心（将中心反标准化）
            centers = scaler.inverse_transform(kmeans.cluster_centers_)
            plt.scatter(centers[:, 0], centers[:, 2], marker="X", s=120, c="black", label="中心")
            plt.title("T–Y散点（KMeans聚类）")
            plt.xlabel("孕周 T")
            plt.ylabel("Y染色体浓度")
            plt.legend()
            plt.tight_layout()
            plt.savefig("scatter_TY_by_kmeans.png", dpi=300, bbox_inches="tight")
            plt.show()

            # 风险曲线（按聚类）
            plt.figure(figsize=(7, 5))
            for r in km_results:
                c = r["Cluster"]
                a = r["slope_a"]; b = r["intercept_b"]; sigma = max(r["sigma"], 1e-6)
                t0 = max(10.0, r["T_grid_min"])
                t1 = min(T_CAP, r["T_grid_max"]) if "T_grid_max" in r else T_CAP
                if not np.isfinite(t0) or not np.isfinite(t1) or t1 <= t0:
                    continue
                t_grid = np.linspace(t0, t1, 300)
                mu = a * t_grid + b
                miss_prob = norm.cdf((Y_THRESHOLD - mu) / sigma)
                delay_prob = _delay_risk(t_grid, t0=10.0, tcap=T_CAP, gamma=DELAY_RISK_GAMMA)
                risk = ALPHA * miss_prob + BETA * delay_prob
                plt.plot(t_grid, risk, label=f"Cluster {c}")
                # 标注最优时点
                plt.axvline(r["best_T"], color="gray", linestyle=":", linewidth=0.8)
            plt.text(0.02, 0.02, f"Risk(T)=α·P(Y<阈值)+β·Delay(T)\nDelay: T<10→0; T∈[10,28]→((T-10)/(28-10))^{DELAY_RISK_GAMMA}", transform=plt.gca().transAxes, fontsize=8, va='bottom', ha='left')
            plt.title("KMeans聚类的风险函数曲线")
            plt.xlabel("孕周 T")
            plt.ylabel("Risk(T)")
            plt.legend()
            plt.tight_layout()
            plt.savefig("risk_curves_by_kmeans.png", dpi=300, bbox_inches="tight")
            plt.show()
        else:
            print("KMeans：无可计算的聚类结果。")
    else:
        print("样本量不足（<3），跳过KMeans聚类。")
except Exception as e:
    print(f"KMeans 聚类步骤出错: {e}")

# ---------------------- 原始数据(data2)的二分类与卡方检验 ----------------------
try:
    df2 = pd.DataFrame({
        "Y染色体浓度": [t.Y染色体浓度 for t in data],
        "孕周_数值": [t.孕周_数值 for t in data],
        "孕妇BMI": [t.孕妇BMI for t in data]
    }).dropna()
    if len(df2) >= 4:
        df2 = df2.astype({"Y染色体浓度": float, "孕周_数值": float, "孕妇BMI": float})
        df2["检测阳性"] = (df2["Y染色体浓度"] >= Y_THRESHOLD).astype(int)

        # BMI 五分位分组
        try:
            df2["BMI_五分位组"] = pd.qcut(df2["孕妇BMI"], q=5, labels=["P1","P2","P3","P4","P5"], duplicates="drop")
        except ValueError:
            df2["BMI_五分位组"] = pd.qcut(df2["孕妇BMI"], q=5, labels=None, duplicates="drop")

        # 总体列联表与卡方
        tab_all = pd.crosstab(df2["BMI_五分位组"], df2["检测阳性"])  # 行: BMI组，列: 阴/阳
        if tab_all.shape[0] >= 2 and tab_all.shape[1] == 2:
            chi2, p, dof, exp = chi2_contingency(tab_all.values)
            pos_rate = (tab_all[1] / tab_all.sum(axis=1)).rename("阳性率")
            print("\n[原始数据] BMI五分位 × 检测(二分类) 列联表：")
            print(tab_all)
            print("各BMI组阳性率：")
            print((pos_rate * 100).round(1).astype(str) + "%")
            print(f"总体卡方: χ²={chi2:.4f}, df={dof}, p={p:.4g}")

            # 相邻组两两卡方（2×2）
            groups = list(tab_all.index)
            for i in range(len(groups) - 1):
                g1, g2 = groups[i], groups[i+1]
                a1 = int(tab_all.loc[g1, 1] if 1 in tab_all.columns else 0)
                b1 = int(tab_all.loc[g1].sum() - a1)
                a2 = int(tab_all.loc[g2, 1] if 1 in tab_all.columns else 0)
                b2 = int(tab_all.loc[g2].sum() - a2)
                table2 = np.array([[a1, b1], [a2, b2]])
                chi2_2, p2, _, _ = chi2_contingency(table2)
                rate1 = a1 / (a1 + b1) if (a1 + b1) > 0 else np.nan
                rate2 = a2 / (a2 + b2) if (a2 + b2) > 0 else np.nan
                diff = (rate2 - rate1) * 100 if np.isfinite(rate1) and np.isfinite(rate2) else np.nan
                print(f"相邻组比较 {g1} vs {g2}: χ²={chi2_2:.4f}, p={p2:.4g}, 阳性率差={diff:.1f}%")

        # 组内按孕周阈值的二分类卡方：寻找最显著阈值
        candidate_T = [10, 12, 14, 16, 18, 20, 22, 24, 26, 28]
        groups2 = df2["BMI_五分位组"].dropna().unique()
        for g in sorted(groups2):
            sub = df2.loc[df2["BMI_五分位组"] == g].copy()
            if len(sub) < 8:
                print(f"BMI组 {g}: 样本量不足（n={len(sub)}），跳过阈值扫描。")
                continue
            best = None
            for k in candidate_T:
                thresh_grp = (sub["孕周_数值"] >= k).astype(int)
                tab_k = pd.crosstab(thresh_grp, sub["检测阳性"])
                if tab_k.shape == (2,2):
                    chi2_k, p_k, _, _ = chi2_contingency(tab_k.values)
                    rate_low = tab_k.loc[0, 1] / tab_k.loc[0].sum() if tab_k.loc[0].sum() > 0 else np.nan
                    rate_high = tab_k.loc[1, 1] / tab_k.loc[1].sum() if tab_k.loc[1].sum() > 0 else np.nan
                    diff = (rate_high - rate_low) * 100 if np.isfinite(rate_low) and np.isfinite(rate_high) else np.nan
                    rec = (p_k, chi2_k, k, diff, int(tab_k.values.sum()))
                    if (best is None) or (p_k < best[0]):
                        best = rec
            if best is not None:
                p_k, chi2_k, k, diff, n_tot = best
                print(f"BMI组 {g}: 最显著孕周阈值K={k}周, χ²={chi2_k:.4f}, p={p_k:.4g}, 阳性率差={diff:.1f}%, n={n_tot}")
    else:
        print("数据data样本量不足，跳过卡方分析。")
except Exception as e:
    print(f"数据(data)卡方分析出错: {e}")

# ---------------------- 两比例z检验与稳健性分析（Step4/Step5） ----------------------
def two_proportion_z_test(x1: int, n1: int, x2: int, n2: int):
    # 合并比例
    if min(n1, n2) == 0:
        return np.nan, np.nan
    p_hat = (x1 + x2) / (n1 + n2)
    denom = np.sqrt(p_hat * (1 - p_hat) * (1 / n1 + 1 / n2))
    if denom <= 0:
        return np.nan, np.nan
    z = (x1 / n1 - x2 / n2) / denom
    p = 2 * (1 - norm.cdf(abs(z)))
    return float(z), float(p)

try:
    if len(df) > 0 and df["孕周_数值"].notna().any():
        # 准备“达标”二分类（c>0.04）
        tmp = df[["孕周_数值","孕妇BMI","Y染色体浓度"]].dropna().copy()
        tmp["达标"] = (tmp["Y染色体浓度"] > Y_THRESHOLD).astype(int)

        # 稳健性：不同分组数q & 阈值邻域
        q_list = [4, 5, 6]
        candidate_T = [10,12,14,16,18,20,22,24,26,28]
        rows = []
        for q in q_list:
            try:
                tmp[f"BMI_q{q}"] = pd.qcut(tmp["孕妇BMI"], q=q, labels=[f"G{i+1}" for i in range(q)], duplicates="drop")
            except ValueError:
                tmp[f"BMI_q{q}"] = pd.qcut(tmp["孕妇BMI"], q=q, labels=None, duplicates="drop")
            groups = tmp[f"BMI_q{q}"].dropna().unique()
            for g in sorted(groups):
                sub = tmp.loc[tmp[f"BMI_q{q}"] == g]
                if len(sub) < 20:
                    continue
                # 在候选阈值上扫描，找z检验最显著
                best = None
                for k in candidate_T:
                    grp = (sub["孕周_数值"] >= k).astype(int)
                    n1 = int((grp == 0).sum()); n2 = int((grp == 1).sum())
                    x1 = int(((grp == 0) & (sub["达标"] == 1)).sum())
                    x2 = int(((grp == 1) & (sub["达标"] == 1)).sum())
                    z, p = two_proportion_z_test(x1, n1, x2, n2)
                    if np.isfinite(p):
                        rec = (p, z, k, x1, n1, x2, n2)
                        if best is None or p < best[0]:
                            best = rec
                if best is None:
                    continue
                p_best, z_best, k_best, x1, n1, x2, n2 = best
                # 阈值稳健性：±2周内最近的候选点（若存在）
                neighbor_ks = [kk for kk in candidate_T if kk in [k_best-2, k_best-1, k_best+1, k_best+2]]
                for kk in ([k_best] + neighbor_ks):
                    grp = (sub["孕周_数值"] >= kk).astype(int)
                    n1k = int((grp == 0).sum()); n2k = int((grp == 1).sum())
                    x1k = int(((grp == 0) & (sub["达标"] == 1)).sum())
                    x2k = int(((grp == 1) & (sub["达标"] == 1)).sum())
                    z_k, p_k = two_proportion_z_test(x1k, n1k, x2k, n2k)
                    rows.append({
                        "q": int(q), "BMI_group": str(g), "K": int(kk),
                        "z": z_k if np.isfinite(z_k) else np.nan,
                        "p_value": p_k if np.isfinite(p_k) else np.nan,
                        "rate_low": (x1k/n1k if n1k>0 else np.nan),
                        "rate_high": (x2k/n2k if n2k>0 else np.nan),
                        "n_low": int(n1k), "n_high": int(n2k)
                    })
        if rows:
            df_z = pd.DataFrame(rows).sort_values(["q","BMI_group","p_value","K"]).reset_index(drop=True)
            df_z.to_csv("proportion_ztests_stability.csv", index=False, encoding="utf-8-sig")
            print("两比例z检验与稳健性结果已导出: proportion_ztests_stability.csv")
            # 打印每(q, 组)的最佳记录
            best_print = df_z.sort_values(["q","BMI_group","p_value"]).groupby(["q","BMI_group"], as_index=False).first()
            print(best_print.to_string(index=False))
        else:
            print("两比例z检验：无可用记录。")
except Exception as e:
    print(f"两比例z检验步骤出错: {e}")