import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager
from test import build, build2, build3
from scipy.stats import pearsonr, spearmanr, f, t as student_t
import numpy as np
import pandas as pd

font_manager.fontManager.addfont(r"C:\Windows\Fonts\msyh.ttc")
CH_FONT = font_manager.FontProperties(fname=r"C:\Windows\Fonts\msyh.ttc")
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

sns.set_theme(style="whitegrid", rc={
    'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'SimSun'],
    'axes.unicode_minus': False
})

def analyze_dataset(data, tag: str):
    Y_list = [t.Y染色体浓度 for t in data]
    T_list = [t.孕周_数值 for t in data]
    BMI_list = [t.孕妇BMI for t in data]
    GC_list = [getattr(t, 'GC含量', None) for t in data]
    X_list = [getattr(t, 'X染色体浓度', None) for t in data]
    H_list = [getattr(t, '身高', None) for t in data]
    W_list = [getattr(t, '体重', None) for t in data]
    N_list = [getattr(t, '检测抽血次数', None) for t in data]

    def _to_float_list(lst):
        out = []
        for v in lst:
            try:
                out.append(float(v))
            except Exception:
                out.append(np.nan)
        return out

    df_plot = pd.DataFrame({
        "Y染色体浓度": _to_float_list(Y_list),
        "孕周_数值": _to_float_list(T_list),
        "孕妇BMI": _to_float_list(BMI_list),
        "GC含量": _to_float_list(GC_list),
        "X染色体浓度": _to_float_list(X_list),
        "身高": _to_float_list(H_list),
        "体重": _to_float_list(W_list),
        "检测抽血次数": _to_float_list(N_list)
    }).dropna().round(4)

    if len(df_plot) == 0:
        raise ValueError(f"[{tag}] 可用于绘图的数据为空")

    x_t = df_plot["孕周_数值"].values
    y_t = df_plot["Y染色体浓度"].values
    a_t, b_t = np.polyfit(x_t, y_t, 1)
    r_t = np.corrcoef(x_t, y_t)[0, 1]
    eq_t = f"Y = {a_t:.6f}×孕周 + {b_t:.6f}  (R²={r_t**2:.4f})"
    print(f"[{tag}] 线性回归（Y ~ 孕周_数值）：", eq_t)

    x_b = df_plot["孕妇BMI"].values
    y_b = df_plot["Y染色体浓度"].values
    a_b, b_b = np.polyfit(x_b, y_b, 1)
    r_b = np.corrcoef(x_b, y_b)[0, 1]
    eq_b = f"Y = {a_b:.6f}×BMI + {b_b:.6f}  (R²={r_b**2:.4f})"
    print(f"[{tag}] 线性回归（Y ~ 孕妇BMI）：", eq_b)

    n_t = len(x_t)
    yhat_t = a_t * x_t + b_t
    sse_t = float(np.sum((y_t - yhat_t) ** 2))
    sst_t = float(np.sum((y_t - y_t.mean()) ** 2))
    ssm_t = sst_t - sse_t
    df1_t, df2_t = 1, n_t - 2
    F_t = (ssm_t / df1_t) / (sse_t / df2_t) if (df2_t > 0 and sse_t > 0) else np.nan
    pF_t = float(1 - f.cdf(F_t, df1_t, df2_t)) if (df2_t > 0 and not np.isnan(F_t)) else np.nan
    sigma2_t = (sse_t / df2_t) if df2_t > 0 else np.nan
    Sxx_t = float(np.sum((x_t - x_t.mean()) ** 2))
    se_a_t = np.sqrt(sigma2_t / Sxx_t) if (Sxx_t > 0 and not np.isnan(sigma2_t)) else np.nan
    t_a_t = (a_t / se_a_t) if (se_a_t and se_a_t > 0) else np.nan
    p_a_t = float(2 * (1 - student_t.cdf(abs(t_a_t), df2_t))) if (df2_t > 0 and not np.isnan(t_a_t)) else np.nan
    print(f"[{tag}] 方程显著性（Y ~ 孕周_数值）: F={F_t:.4f}, df=({df1_t},{df2_t}), p={pF_t:.4g}; 斜率t检验: t={t_a_t:.4f}, p={p_a_t:.4g}")

    n_b = len(x_b)
    yhat_b = a_b * x_b + b_b
    sse_b = float(np.sum((y_b - yhat_b) ** 2))
    sst_b = float(np.sum((y_b - y_b.mean()) ** 2))
    ssm_b = sst_b - sse_b
    df1_b, df2_b = 1, n_b - 2
    F_b = (ssm_b / df1_b) / (sse_b / df2_b) if (df2_b > 0 and sse_b > 0) else np.nan
    pF_b = float(1 - f.cdf(F_b, df1_b, df2_b)) if (df2_b > 0 and not np.isnan(F_b)) else np.nan
    sigma2_b = (sse_b / df2_b) if df2_b > 0 else np.nan
    Sxx_b = float(np.sum((x_b - x_b.mean()) ** 2))
    se_a_b = np.sqrt(sigma2_b / Sxx_b) if (Sxx_b > 0 and not np.isnan(sigma2_b)) else np.nan
    t_a_b = (a_b / se_a_b) if (se_a_b and se_a_b > 0) else np.nan
    p_a_b = float(2 * (1 - student_t.cdf(abs(t_a_b), df2_b))) if (df2_b > 0 and not np.isnan(t_a_b)) else np.nan
    print(f"[{tag}] 方程显著性（Y ~ 孕妇BMI）  : F={F_b:.4f}, df=({df1_b},{df2_b}), p={pF_b:.4g}; 斜率t检验: t={t_a_b:.4f}, p={p_a_b:.4g}")

    X = np.column_stack([
        np.ones(len(df_plot)),
        df_plot["孕周_数值"].values,
        df_plot["孕妇BMI"].values
    ])
    y_m = df_plot["Y染色体浓度"].values
    beta, residuals, rank, svals = np.linalg.lstsq(X, y_m, rcond=None)
    b0, b1, b2 = beta
    y_hat = X @ beta
    ss_res = float(np.sum((y_m - y_hat) ** 2))
    ss_tot = float(np.sum((y_m - y_m.mean()) ** 2))
    r2_m = (1 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    n_m, p_m = X.shape
    adj_r2_m = (1 - (1 - r2_m) * (n_m - 1) / (n_m - p_m)) if (n_m > p_m and not np.isnan(r2_m)) else np.nan
    eq_m = f"Y = {b1:.6f}×孕周 + {b2:.6f}×BMI + {b0:.6f}  (R²={r2_m:.4f}, 调整R²={adj_r2_m:.4f})"
    print(f"[{tag}] 多元线性回归（Y ~ 孕周_数值 + 孕妇BMI）：", eq_m)

    df1_m = (p_m - 1)
    df2_m = (n_m - p_m)
    ssm_m = ss_tot - ss_res
    F_m = (ssm_m / df1_m) / (ss_res / df2_m) if (df1_m > 0 and df2_m > 0 and ss_res > 0) else np.nan
    pF_m = float(1 - f.cdf(F_m, df1_m, df2_m)) if (df1_m > 0 and df2_m > 0 and not np.isnan(F_m)) else np.nan
    XtX_inv = np.linalg.inv(X.T @ X) if n_m >= p_m else np.full((p_m, p_m), np.nan)
    sigma2_m = (ss_res / df2_m) if df2_m > 0 else np.nan
    se_betas = np.sqrt(np.diag(XtX_inv) * sigma2_m) if (np.all(np.isfinite(XtX_inv)) and not np.isnan(sigma2_m)) else np.array([np.nan]*p_m)
    t_b0 = (b0 / se_betas[0]) if (se_betas[0] and se_betas[0] > 0) else np.nan
    t_b1 = (b1 / se_betas[1]) if (se_betas[1] and se_betas[1] > 0) else np.nan
    t_b2 = (b2 / se_betas[2]) if (se_betas[2] and se_betas[2] > 0) else np.nan
    p_b0 = float(2 * (1 - student_t.cdf(abs(t_b0), df2_m))) if (df2_m > 0 and not np.isnan(t_b0)) else np.nan
    p_b1 = float(2 * (1 - student_t.cdf(abs(t_b1), df2_m))) if (df2_m > 0 and not np.isnan(t_b1)) else np.nan
    p_b2 = float(2 * (1 - student_t.cdf(abs(t_b2), df2_m))) if (df2_m > 0 and not np.isnan(t_b2)) else np.nan
    print(f"[{tag}] 多元方程显著性: F={F_m:.4f}, df=({df1_m},{df2_m}), p={pF_m:.4g}")
    print(f"[{tag}] 系数t检验: 截距 b0={b0:.6f}, t={t_b0:.4f}, p={p_b0:.4g}; 孕周 b1={b1:.6f}, t={t_b1:.4f}, p={p_b1:.4g}; BMI b2={b2:.6f}, t={t_b2:.4f}, p={p_b2:.4g}")

    plt.figure(figsize=(6,4))
    sns.scatterplot(data=df_plot, x="孕周_数值", y="Y染色体浓度", s=25, alpha=0.7)
    sns.regplot(data=df_plot, x="孕周_数值", y="Y染色体浓度", scatter=False, ci=95, color="red", line_kws={"label": "线性拟合"})
    plt.title(f"[{tag}] Y染色体浓度 vs 孕周_数值", fontproperties=CH_FONT)
    plt.xlabel("孕周_数值", fontproperties=CH_FONT)
    plt.ylabel("Y染色体浓度", fontproperties=CH_FONT)
    plt.xticks(fontproperties=CH_FONT)
    plt.yticks(fontproperties=CH_FONT)
    ax = plt.gca()
    ax.text(0.03, 0.97, eq_t, transform=ax.transAxes, va='top', ha='left',
            fontproperties=CH_FONT, bbox={"facecolor": "white", "alpha": 0.6, "edgecolor": "none"})
    plt.legend(prop=CH_FONT)
    plt.tight_layout()
    plt.savefig(f"scatter_{tag}_Y_vs_T.png", dpi=300, bbox_inches="tight")
    plt.show()

    plt.figure(figsize=(6,4))
    sns.scatterplot(data=df_plot, x="孕妇BMI", y="Y染色体浓度", s=25, alpha=0.7, color="orange")
    sns.regplot(data=df_plot, x="孕妇BMI", y="Y染色体浓度", scatter=False, ci=95, color="red", line_kws={"label": "线性拟合"})
    plt.title(f"[{tag}] Y染色体浓度 vs 孕妇BMI", fontproperties=CH_FONT)
    plt.xlabel("孕妇BMI", fontproperties=CH_FONT)
    plt.ylabel("Y染色体浓度", fontproperties=CH_FONT)
    plt.xticks(fontproperties=CH_FONT)
    plt.yticks(fontproperties=CH_FONT)
    ax = plt.gca()
    ax.text(0.03, 0.97, eq_b, transform=ax.transAxes, va='top', ha='left',
            fontproperties=CH_FONT, bbox={"facecolor": "white", "alpha": 0.6, "edgecolor": "none"})
    plt.legend(prop=CH_FONT)
    plt.tight_layout()
    plt.savefig(f"scatter_{tag}_Y_vs_BMI.png", dpi=300, bbox_inches="tight")
    plt.show()

    def _poly_fit_and_report(x, y, deg: int, name: str):
        coefs = np.polyfit(x, y, deg)
        yhat = np.polyval(coefs, x)
        sse = float(np.sum((y - yhat) ** 2))
        sst = float(np.sum((y - y.mean()) ** 2))
        r2 = (1 - sse / sst) if sst > 0 else np.nan
        terms = [f"{coefs[i]:+.6f}×{name}^{deg-i}" for i in range(0, deg)]
        eq = f"Y = {' '.join(terms)} {coefs[-1]:+.6f}  (R²={r2:.4f})"
        return coefs, r2, eq

    coefs_t2, r2_t2, eq_t2 = _poly_fit_and_report(x_t, y_t, 2, "孕周")
    coefs_t3, r2_t3, eq_t3 = _poly_fit_and_report(x_t, y_t, 3, "孕周")
    print(f"[{tag}] 二次多项式（Y ~ 孕周）:", eq_t2)
    print(f"[{tag}] 三次多项式（Y ~ 孕周）:", eq_t3)

    coefs_b2, r2_b2, eq_b2 = _poly_fit_and_report(x_b, y_b, 2, "BMI")
    coefs_b3, r2_b3, eq_b3 = _poly_fit_and_report(x_b, y_b, 3, "BMI")
    print(f"[{tag}] 二次多项式（Y ~ BMI）:", eq_b2)
    print(f"[{tag}] 三次多项式（Y ~ BMI）:", eq_b3)

    grid_t = np.linspace(x_t.min(), x_t.max(), 400)
    plt.figure(figsize=(6,4))
    sns.scatterplot(x=x_t, y=y_t, s=25, alpha=0.7, label="数据")
    plt.plot(grid_t, np.polyval(coefs_t2, grid_t), color="red", label=f"二次: R²={r2_t2:.3f}")
    plt.plot(grid_t, np.polyval(coefs_t3, grid_t), color="green", linestyle="--", label=f"三次: R²={r2_t3:.3f}")
    plt.title(f"[{tag}] Y vs 孕周：多项式拟合对比", fontproperties=CH_FONT)
    plt.xlabel("孕周_数值", fontproperties=CH_FONT)
    plt.ylabel("Y染色体浓度", fontproperties=CH_FONT)
    plt.legend(prop=CH_FONT)
    plt.tight_layout()
    plt.savefig(f"scatter_{tag}_Y_vs_T_poly.png", dpi=300, bbox_inches="tight")
    plt.show()

    grid_b = np.linspace(x_b.min(), x_b.max(), 400)
    plt.figure(figsize=(6,4))
    sns.scatterplot(x=x_b, y=y_b, s=25, alpha=0.7, color="orange", label="数据")
    plt.plot(grid_b, np.polyval(coefs_b2, grid_b), color="red", label=f"二次: R²={r2_b2:.3f}")
    plt.plot(grid_b, np.polyval(coefs_b3, grid_b), color="green", linestyle="--", label=f"三次: R²={r2_b3:.3f}")
    plt.title(f"[{tag}] Y vs BMI：多项式拟合对比", fontproperties=CH_FONT)
    plt.xlabel("孕妇BMI", fontproperties=CH_FONT)
    plt.ylabel("Y染色体浓度", fontproperties=CH_FONT)
    plt.legend(prop=CH_FONT)
    plt.tight_layout()
    plt.savefig(f"scatter_{tag}_Y_vs_BMI_poly.png", dpi=300, bbox_inches="tight")
    plt.show()

    T_vals = df_plot["孕周_数值"].values
    BMI_vals = df_plot["孕妇BMI"].values
    X_inter = np.column_stack([
        np.ones(len(df_plot)),
        T_vals,
        BMI_vals,
        T_vals * BMI_vals,
        T_vals ** 2,
        BMI_vals ** 2
    ])
    y_vals = df_plot["Y染色体浓度"].values
    beta_inter, residuals_inter, rank_inter, svals_inter = np.linalg.lstsq(X_inter, y_vals, rcond=None)
    y_hat_inter = X_inter @ beta_inter
    ss_res_i = float(np.sum((y_vals - y_hat_inter) ** 2))
    ss_tot_i = float(np.sum((y_vals - y_vals.mean()) ** 2))
    r2_i = (1 - ss_res_i / ss_tot_i) if ss_tot_i > 0 else np.nan
    n_i, p_i = X_inter.shape
    adj_r2_i = (1 - (1 - r2_i) * (n_i - 1) / (n_i - p_i)) if (n_i > p_i and not np.isnan(r2_i)) else np.nan
    print(f"[{tag}] 带交互与二次项（Y ~ T + BMI + T*BMI + T^2 + BMI^2）: R²={r2_i:.4f}, 调整R²={adj_r2_i:.4f}")

    cols = ["Y染色体浓度","孕周_数值","孕妇BMI","GC含量"]
    corr = df_plot[cols].corr(method="spearman")
    plt.figure(figsize=(5,4))
    sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1, center=0, fmt=".2f")
    plt.title(f"[{tag}] 相关性矩阵 (Spearman)", fontproperties=CH_FONT)
    plt.xticks(fontproperties=CH_FONT)
    plt.yticks(fontproperties=CH_FONT)
    plt.tight_layout()
    plt.savefig(f"heatmap_{tag}_Y_T_BMI_GC.png", dpi=300, bbox_inches="tight")
    plt.show()

    Y_T_pairs = [(y, t) for y, t in zip(Y_list, T_list) if y is not None and t is not None]
    Y_B_pairs = [(y, b) for y, b in zip(Y_list, BMI_list) if y is not None and b is not None]
    if Y_T_pairs:
        y_arr, t_arr = map(np.array, zip(*Y_T_pairs))
        pr, pp = pearsonr(y_arr, t_arr)
        sr, sp = spearmanr(y_arr, t_arr)
        print(f"[{tag}] Y染色体浓度与孕周数: Pearson r={pr:.4f}, p={pp:.4g} | Spearman ρ={sr:.4f}, p={sp:.4g}")
    if Y_B_pairs:
        y, b = map(np.array, zip(*Y_B_pairs))
        pr, pp = pearsonr(y, b)
        sr, sp = spearmanr(y, b)
        print(f"[{tag}] Y染色体浓度与孕妇BMI: Pearson r={pr:.4f}, p={pp:.4g} | Spearman ρ={sr:.4f}, p={sp:.4g}")

    # ============== 可解释模型对比：OLS / WLS / ElasticNet（T, BMI, GC + 交互 + 二次） ==============
    def _build_design(df):
        Tv = df["孕周_数值"].values
        Bv = df["孕妇BMI"].values
        Gv = df["GC含量"].values
        feats = {
            "常数": np.ones(len(df)),
            "T": Tv,
            "BMI": Bv,
            "GC": Gv,
            "T*BMI": Tv * Bv,
            "T*GC": Tv * Gv,
            "BMI*GC": Bv * Gv,
            "T^2": Tv ** 2,
            "BMI^2": Bv ** 2,
            "GC^2": Gv ** 2,
        }
        names = list(feats.keys())
        X = np.column_stack([feats[k] for k in names])
        return X, names

    def _reg_metrics(y_true, y_pred, k_params):
        ss_res = float(np.sum((y_true - y_pred) ** 2))
        ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
        r2 = (1 - ss_res / ss_tot) if ss_tot > 0 else np.nan
        n = len(y_true)
        adj = (1 - (1 - r2) * (n - 1) / (n - k_params)) if (n > k_params and not np.isnan(r2)) else np.nan
        return r2, adj

    X_full, names_full = _build_design(df_plot)
    y_full = df_plot["Y染色体浓度"].values

    # OLS
    beta_ols, *_ = np.linalg.lstsq(X_full, y_full, rcond=None)
    yhat_ols = X_full @ beta_ols
    r2_ols, adj_ols = _reg_metrics(y_full, yhat_ols, k_params=X_full.shape[1])
    print(f"[{tag}] OLS（线性+交互+二次, 特征数={X_full.shape[1]}）: R²={r2_ols:.4f}, 调整R²={adj_ols:.4f}")
    coef_ols = {n: float(b) for n, b in zip(names_full, beta_ols)}
    print(f"[{tag}] OLS系数: {coef_ols}")

    # WLS（按GC权重，线性缩放至[0.1,1]防止权重过小）
    Gv = df_plot["GC含量"].values
    gmin, gmax = float(np.nanmin(Gv)), float(np.nanmax(Gv))
    if gmax > gmin:
        w = 0.1 + 0.9 * (Gv - gmin) / (gmax - gmin)
    else:
        w = np.ones_like(Gv)
    Wsqrt = np.sqrt(w)[:, None]
    X_w = X_full * Wsqrt
    y_w = y_full * Wsqrt[:, 0]
    beta_wls, *_ = np.linalg.lstsq(X_w, y_w, rcond=None)
    yhat_wls = X_full @ beta_wls
    r2_wls, adj_wls = _reg_metrics(y_full, yhat_wls, k_params=X_full.shape[1])
    print(f"[{tag}] WLS（GC加权）: R²={r2_wls:.4f}, 调整R²={adj_wls:.4f}")
    coef_wls = {n: float(b) for n, b in zip(names_full, beta_wls)}
    print(f"[{tag}] WLS系数: {coef_wls}")

    # ElasticNet（标准化特征，若不可用则跳过）
    try:
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import ElasticNetCV
        from sklearn.pipeline import make_pipeline
        import warnings as _warnings
        _warnings.filterwarnings("ignore")
        enet = make_pipeline(
            StandardScaler(with_mean=True, with_std=True),
            ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], cv=5, max_iter=10000, n_alphas=100, random_state=0)
        )
        enet.fit(X_full, y_full)
        yhat_en = enet.predict(X_full)
        r2_en, adj_en = _reg_metrics(y_full, yhat_en, k_params=X_full.shape[1])
        # 提取标准化下的系数（可解释为标准化系数）
        en_model = enet.named_steps['elasticnetcv']
        coefs_std = en_model.coef_
        intercept_std = en_model.intercept_
        coef_en = {n: float(c) for n, c in zip(names_full[1:], coefs_std)}
        coef_en.update({names_full[0]: float(intercept_std)})
        print(f"[{tag}] ElasticNet（标准化系数）: R²={r2_en:.4f}, 调整R²={adj_en:.4f}, 最优alpha={en_model.alpha_:.4g}, l1_ratio={en_model.l1_ratio_}")
        print(f"[{tag}] ElasticNet系数(标准化空间): {coef_en}")
    except Exception as _e:
        print(f"[{tag}] ElasticNet 不可用（未安装sklearn或拟合失败）：{_e}")

    # Stepwise OLS（K折CV，前进+后退）
    try:
        from sklearn.model_selection import KFold
        from sklearn.metrics import r2_score

        intercept_idx = 0
        all_feat_idx = list(range(1, X_full.shape[1]))
        selected = []  # 存放特征索引（不含常数）

        def cv_r2(feat_idx_list):
            idx = [intercept_idx] + feat_idx_list
            Xs = X_full[:, idx]
            kf = KFold(n_splits=5, shuffle=True, random_state=0)
            scores = []
            for tr, va in kf.split(Xs):
                beta, *_ = np.linalg.lstsq(Xs[tr], y_full[tr], rcond=None)
                ypred = Xs[va] @ beta
                scores.append(r2_score(y_full[va], ypred))
            return float(np.mean(scores)) if scores else np.nan

        best_score = cv_r2(selected)
        improved = True
        min_delta = 1e-4

        while improved:
            improved = False
            # forward step
            cand_best = None
            cand_best_score = best_score
            for fidx in all_feat_idx:
                if fidx in selected:
                    continue
                sc = cv_r2(selected + [fidx])
                if sc > cand_best_score + min_delta:
                    cand_best_score = sc
                    cand_best = fidx
            if cand_best is not None:
                selected.append(cand_best)
                best_score = cand_best_score
                improved = True

            # backward step
            removed_any = False
            for fidx in list(selected):
                trial = [x for x in selected if x != fidx]
                sc = cv_r2(trial)
                if sc > best_score + min_delta:
                    selected = trial
                    best_score = sc
                    removed_any = True
                    improved = True
            if removed_any:
                continue

        # 最终拟合与报告
        final_idx = [intercept_idx] + selected
        X_sel = X_full[:, final_idx]
        beta_sel, *_ = np.linalg.lstsq(X_sel, y_full, rcond=None)
        yhat_sel = X_sel @ beta_sel
        r2_sel, adj_sel = _reg_metrics(y_full, yhat_sel, k_params=X_sel.shape[1])
        sel_names = [names_full[i] for i in final_idx]
        coef_sel = {n: float(b) for n, b in zip(sel_names, beta_sel)}
        print(f"[{tag}] Stepwise-OLS（CV 5 折）: 选中特征={sel_names}, R²={r2_sel:.4f}, 调整R²={adj_sel:.4f}, CV_R²={best_score:.4f}")
        print(f"[{tag}] Stepwise-OLS系数: {coef_sel}")
    except Exception as _e:
        print(f"[{tag}] Stepwise-OLS 失败：{_e}")

    # 针对“筛选”数据：扩展特征集后再做一次逐步回归并给出方程
    if tag == "筛选":
        try:
            from sklearn.model_selection import KFold
            from sklearn.metrics import r2_score

            def _build_design_ext(df):
                Tv = df["孕周_数值"].values
                Bv = df["孕妇BMI"].values
                Gv = df["GC含量"].values
                Xv = df["X染色体浓度"].values
                Hv = df["身高"].values
                Wv = df["体重"].values
                Nv = df["检测抽血次数"].values
                feats = {
                    "常数": np.ones(len(df)),
                    "T": Tv,
                    "BMI": Bv,
                    "GC": Gv,
                    "X": Xv,
                    "H": Hv,
                    "W": Wv,
                    "N": Nv,
                    "T*BMI": Tv * Bv,
                    "T*GC": Tv * Gv,
                    "BMI*GC": Bv * Gv,
                    "T^2": Tv ** 2,
                    "BMI^2": Bv ** 2,
                    "GC^2": Gv ** 2,
                    "X*T": Xv * Tv,
                    "X*BMI": Xv * Bv,
                    "X*GC": Xv * Gv,
                    "H*T": Hv * Tv,
                    "H*BMI": Hv * Bv,
                    "H*GC": Hv * Gv,
                    "W*T": Wv * Tv,
                    "W*BMI": Wv * Bv,
                    "W*GC": Wv * Gv,
                    "N*T": Nv * Tv,
                    "N*BMI": Nv * Bv,
                    "N*GC": Nv * Gv,
                }
                names = list(feats.keys())
                X = np.column_stack([feats[k] for k in names])
                return X, names

            X_ext, names_ext = _build_design_ext(df_plot)
            intercept_idx = 0
            all_feat_idx = list(range(1, X_ext.shape[1]))
            selected = []

            def cv_r2_ext(feat_idx_list):
                idx = [intercept_idx] + feat_idx_list
                Xs = X_ext[:, idx]
                kf = KFold(n_splits=5, shuffle=True, random_state=0)
                scores = []
                for tr, va in kf.split(Xs):
                    beta, *_ = np.linalg.lstsq(Xs[tr], y_full[tr], rcond=None)
                    ypred = Xs[va] @ beta
                    scores.append(r2_score(y_full[va], ypred))
                return float(np.mean(scores)) if scores else np.nan

            best_score = cv_r2_ext(selected)
            improved = True
            min_delta = 1e-4
            while improved:
                improved = False
                cand_best = None
                cand_best_score = best_score
                for fidx in all_feat_idx:
                    if fidx in selected:
                        continue
                    sc = cv_r2_ext(selected + [fidx])
                    if sc > cand_best_score + min_delta:
                        cand_best_score = sc
                        cand_best = fidx
                if cand_best is not None:
                    selected.append(cand_best)
                    best_score = cand_best_score
                    improved = True

                removed_any = False
                for fidx in list(selected):
                    trial = [x for x in selected if x != fidx]
                    sc = cv_r2_ext(trial)
                    if sc > best_score + min_delta:
                        selected = trial
                        best_score = sc
                        removed_any = True
                        improved = True
                if removed_any:
                    continue

            final_idx = [intercept_idx] + selected
            X_sel = X_ext[:, final_idx]
            beta_sel, *_ = np.linalg.lstsq(X_sel, y_full, rcond=None)
            yhat_sel = X_sel @ beta_sel
            r2_sel, adj_sel = _reg_metrics(y_full, yhat_sel, k_params=X_sel.shape[1])
            sel_names = [names_ext[i] for i in final_idx]
            coef_sel = {n: float(b) for n, b in zip(sel_names, beta_sel)}
            # 生成可读方程
            terms = [f"{coef_sel[sel_names[0]]:.6f}"]
            for n, b in zip(sel_names[1:], beta_sel[1:]):
                terms.append(f"{b:+.6f}×{n}")
            eq = "Y = " + " ".join(terms)
            print(f"[筛选] Stepwise-OLS-扩展（CV 5 折）: 选中特征={sel_names}, R²={r2_sel:.4f}, 调整R²={adj_sel:.4f}, CV_R²={best_score:.4f}")
            print(f"[筛选] 最终方程: {eq}")

            # 真实Y vs 拟合Y 散点图
            y_true = y_full
            y_pred = yhat_sel
            lim_min = float(min(y_true.min(), y_pred.min()))
            lim_max = float(max(y_true.max(), y_pred.max()))
            plt.figure(figsize=(5.2,5.2))
            plt.scatter(y_true, y_pred, s=18, alpha=0.7)
            plt.plot([lim_min, lim_max], [lim_min, lim_max], color="red", linestyle="--", label="y=x")
            plt.title(f"[筛选-扩展] 真实Y vs 拟合Y (R²={r2_sel:.3f})", fontproperties=CH_FONT)
            plt.xlabel("真实Y", fontproperties=CH_FONT)
            plt.ylabel("拟合Y", fontproperties=CH_FONT)
            plt.legend(prop=CH_FONT)
            plt.tight_layout()
            plt.savefig("scatter_final_eq_pred_vs_true.png", dpi=300, bbox_inches="tight")
            plt.show()

            # 可解释图：按方程绘制主要项的边际/交互效应
            # 基线取各变量中位数
            q = df_plot.quantile([0.25, 0.5, 0.75])
            med = {"T": q.loc[0.5, "孕周_数值"], "BMI": q.loc[0.5, "孕妇BMI"], "GC": q.loc[0.5, "GC含量"],
                   "X": q.loc[0.5, "X染色体浓度"], "H": q.loc[0.5, "身高"], "W": q.loc[0.5, "体重"], "N": q.loc[0.5, "检测抽血次数"]}

            def predict_from_terms(T=None, BMI=None, GC=None, X=None, H=None, W=None, N=None):
                v = {
                    "T": med["T"] if T is None else T,
                    "BMI": med["BMI"] if BMI is None else BMI,
                    "GC": med["GC"] if GC is None else GC,
                    "X": med["X"] if X is None else X,
                    "H": med["H"] if H is None else H,
                    "W": med["W"] if W is None else W,
                    "N": med["N"] if N is None else N,
                }
                y = coef_sel.get("常数", 0.0)
                def term(name):
                    if name == "T": return v["T"]
                    if name == "BMI": return v["BMI"]
                    if name == "GC": return v["GC"]
                    if name == "X": return v["X"]
                    if name == "H": return v["H"]
                    if name == "W": return v["W"]
                    if name == "N": return v["N"]
                    if name == "T*BMI": return v["T"] * v["BMI"]
                    if name == "T*GC": return v["T"] * v["GC"]
                    if name == "BMI*GC": return v["BMI"] * v["GC"]
                    if name == "X*T": return v["X"] * v["T"]
                    if name == "X*BMI": return v["X"] * v["BMI"]
                    if name == "X*GC": return v["X"] * v["GC"]
                    if name == "H*T": return v["H"] * v["T"]
                    if name == "H*BMI": return v["H"] * v["BMI"]
                    if name == "H*GC": return v["H"] * v["GC"]
                    if name == "W*T": return v["W"] * v["T"]
                    if name == "W*BMI": return v["W"] * v["BMI"]
                    if name == "W*GC": return v["W"] * v["GC"]
                    if name == "N*T": return v["N"] * v["T"]
                    if name == "N*BMI": return v["N"] * v["BMI"]
                    if name == "N*GC": return v["N"] * v["GC"]
                    if name == "T^2": return v["T"] ** 2
                    if name == "BMI^2": return v["BMI"] ** 2
                    if name == "GC^2": return v["GC"] ** 2
                    return 0.0
                for n in sel_names[1:]:
                    y += coef_sel.get(n, 0.0) * term(n)
                return float(y)

            # 1) T的边际曲线（按X分层）
            T_grid = np.linspace(df_plot["孕周_数值"].min(), df_plot["孕周_数值"].max(), 200)
            X_levels = [q.loc[0.25, "X染色体浓度"], q.loc[0.5, "X染色体浓度"], q.loc[0.75, "X染色体浓度"]]
            plt.figure(figsize=(6,4))
            for xv, lb in zip(X_levels, ["X P25", "X P50", "X P75"]):
                y_curve = [predict_from_terms(T=t, X=xv) for t in T_grid]
                plt.plot(T_grid, y_curve, label=lb)
            plt.title("[筛选-扩展] Y vs T（分层X）", fontproperties=CH_FONT)
            plt.xlabel("孕周_数值", fontproperties=CH_FONT)
            plt.ylabel("Y染色体浓度(预测)", fontproperties=CH_FONT)
            plt.legend(prop=CH_FONT)
            plt.tight_layout()
            plt.savefig("effect_T_by_X.png", dpi=300, bbox_inches="tight")
            plt.show()

            # 2) T的边际曲线（按BMI分层）
            BMI_levels = [q.loc[0.25, "孕妇BMI"], q.loc[0.5, "孕妇BMI"], q.loc[0.75, "孕妇BMI"]]
            plt.figure(figsize=(6,4))
            for bv, lb in zip(BMI_levels, ["BMI P25", "BMI P50", "BMI P75"]):
                y_curve = [predict_from_terms(T=t, BMI=bv) for t in T_grid]
                plt.plot(T_grid, y_curve, label=lb)
            plt.title("[筛选-扩展] Y vs T（分层BMI）", fontproperties=CH_FONT)
            plt.xlabel("孕周_数值", fontproperties=CH_FONT)
            plt.ylabel("Y染色体浓度(预测)", fontproperties=CH_FONT)
            plt.legend(prop=CH_FONT)
            plt.tight_layout()
            plt.savefig("effect_T_by_BMI.png", dpi=300, bbox_inches="tight")
            plt.show()

            # 3) X×GC 交互热力图
            X_grid = np.linspace(df_plot["X染色体浓度"].min(), df_plot["X染色体浓度"].max(), 80)
            GC_grid = np.linspace(df_plot["GC含量"].min(), df_plot["GC含量"].max(), 80)
            XX, GG = np.meshgrid(X_grid, GC_grid)
            YY = np.array([[predict_from_terms(X=xx, GC=gg) for xx in X_grid] for gg in GC_grid])
            plt.figure(figsize=(5.5,4.5))
            sns.heatmap(YY, cmap="coolwarm", xticklabels=False, yticklabels=False)
            plt.title("[筛选-扩展] 交互热力图：X × GC", fontproperties=CH_FONT)
            plt.xlabel("X染色体浓度", fontproperties=CH_FONT)
            plt.ylabel("GC含量", fontproperties=CH_FONT)
            plt.tight_layout()
            plt.savefig("heat_X_by_GC.png", dpi=300, bbox_inches="tight")
            plt.show()

            # 4) N×GC 交互热力图
            N_grid = np.linspace(df_plot["检测抽血次数"].min(), df_plot["检测抽血次数"].max(), 50)
            GC_grid2 = np.linspace(df_plot["GC含量"].min(), df_plot["GC含量"].max(), 80)
            NN, GG2 = np.meshgrid(N_grid, GC_grid2)
            YY2 = np.array([[predict_from_terms(N=nn, GC=gg) for nn in N_grid] for gg in GC_grid2])
            plt.figure(figsize=(5.5,4.5))
            sns.heatmap(YY2, cmap="coolwarm", xticklabels=False, yticklabels=False)
            plt.title("[筛选-扩展] 交互热力图：N × GC", fontproperties=CH_FONT)
            plt.xlabel("检测抽血次数", fontproperties=CH_FONT)
            plt.ylabel("GC含量", fontproperties=CH_FONT)
            plt.tight_layout()
            plt.savefig("heat_N_by_GC.png", dpi=300, bbox_inches="tight")
            plt.show()

            # 5) H×BMI 交互热力图
            H_grid = np.linspace(df_plot["身高"].min(), df_plot["身高"].max(), 80)
            BMI_grid = np.linspace(df_plot["孕妇BMI"].min(), df_plot["孕妇BMI"].max(), 80)
            HH, BB = np.meshgrid(H_grid, BMI_grid)
            YY3 = np.array([[predict_from_terms(H=hh, BMI=bb) for hh in H_grid] for bb in BMI_grid])
            plt.figure(figsize=(5.5,4.5))
            sns.heatmap(YY3, cmap="coolwarm", xticklabels=False, yticklabels=False)
            plt.title("[筛选-扩展] 交互热力图：H × BMI", fontproperties=CH_FONT)
            plt.xlabel("身高", fontproperties=CH_FONT)
            plt.ylabel("孕妇BMI", fontproperties=CH_FONT)
            plt.tight_layout()
            plt.savefig("heat_H_by_BMI.png", dpi=300, bbox_inches="tight")
            plt.show()

            # 6) W 的边际曲线
            W_grid = np.linspace(df_plot["体重"].min(), df_plot["体重"].max(), 200)
            yW = [predict_from_terms(W=w) for w in W_grid]
            plt.figure(figsize=(6,4))
            plt.plot(W_grid, yW)
            plt.title("[筛选-扩展] Y vs 体重(W)", fontproperties=CH_FONT)
            plt.xlabel("体重", fontproperties=CH_FONT)
            plt.ylabel("Y染色体浓度(预测)", fontproperties=CH_FONT)
            plt.tight_layout()
            plt.savefig("effect_W.png", dpi=300, bbox_inches="tight")
            plt.show()
        except Exception as _e:
            print(f"[筛选] Stepwise-OLS-扩展 失败：{_e}")

analyze_dataset(build(), "筛选")
analyze_dataset(build2(), "全量")
analyze_dataset(build3(), "轻洗")
