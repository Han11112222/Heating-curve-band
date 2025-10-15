# app.py — HeatBand Insight (Simple v2025-10-14)
# 단위: 공급량(MJ), 변화율 dQ/dT(MJ/℃)

import os
from typing import Tuple, List
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.font_manager as fm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import streamlit as st

st.set_page_config(page_title="HeatBand Insight — Simple", layout="wide")

# ── Font ──
FONT_PATH = "NanumGothic-Regular.ttf"
if os.path.exists(FONT_PATH):
    try:
        fm.fontManager.addfont(FONT_PATH)
    except Exception:
        pass
PLOT_FONT = "NanumGothic, Arial, Noto Sans KR, sans-serif"

st.title("🔥 HeatBand Insight — 간단 요약보드")
st.caption("단위: 공급량 **MJ**, 변화율 **MJ/℃** · 3차다항식 · 구간별 Δ1℃ 증가량 · Heating Start(θ*) · Heating Slowdown(T_slow)")

# ── Utils ──
def to_num(x):
    if isinstance(x, str):
        x = x.replace(",", "")
    return pd.to_numeric(x, errors="coerce")

def fit_poly3(x: np.ndarray, y: np.ndarray):
    x = x.reshape(-1, 1)
    pf = PolynomialFeatures(degree=3, include_bias=True)
    Xp = pf.fit_transform(x)
    m = LinearRegression().fit(Xp, y)
    return m, pf, Xp

def poly3_coeffs(m: LinearRegression) -> Tuple[float, float, float, float]:
    b0 = float(m.intercept_)
    c = m.coef_
    b1 = float(c[1]) if len(c) > 1 else 0.0
    b2 = float(c[2]) if len(c) > 2 else 0.0
    b3 = float(c[3]) if len(c) > 3 else 0.0
    return b0, b1, b2, b3

def poly3_predict(m, pf, t):
    return m.predict(pf.transform(np.array(t).reshape(-1,1)))

def poly3_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1.0 - ss_res/ss_tot if ss_tot > 0 else np.nan

def poly3_conf_band(x_train, y_train, tgrid, m, pf, alpha=0.05):
    X = pf.transform(x_train.reshape(-1,1))
    yhat = m.predict(X)
    n, p = X.shape
    sigma2 = np.sum((y_train - yhat)**2) / max(1, (n - p))
    XtX_inv = np.linalg.pinv(X.T @ X)
    Tg = pf.transform(tgrid.reshape(-1,1))
    se = np.sqrt(np.sum(Tg @ XtX_inv * Tg, axis=1) * sigma2)
    z = 1.96
    ypred = m.predict(Tg)
    return ypred - z*se, ypred + z*se

def poly3_d1_at(m: LinearRegression, pf: PolynomialFeatures, t: float) -> float:
    b0, b1, b2, b3 = poly3_coeffs(m)
    return b1 + 2*b2*t + 3*b3*(t**2)

def hinge_base_temp(T: np.ndarray, Q: np.ndarray,
                    grid_min: float=0.0, grid_max: float=20.0, step: float=0.1) -> Tuple[float, float, float]:
    thetas = np.arange(grid_min, grid_max + 1e-9, step)
    best_th, best_a, best_b, best_rmse = np.nan, np.nan, np.nan, np.inf
    T = T.reshape(-1); Q = Q.reshape(-1)
    X1 = np.ones_like(T)
    for th in thetas:
        H = np.clip(th - T, 0, None)
        X = np.column_stack([X1, H])
        beta, *_ = np.linalg.lstsq(X, Q, rcond=None)
        pred = X @ beta
        rmse = np.sqrt(np.mean((Q - pred)**2))
        if rmse < best_rmse:
            best_rmse = rmse
            best_th, best_a, best_b = th, float(beta[0]), float(beta[1])
    return best_th, best_a, best_b

def nice_poly_string(a,b,c,d, digits=1):
    def term(v, s, lead=False):
        if abs(v) < 1e-12: return ""
        sign = "" if lead and v>=0 else (" - " if v<0 else " + ")
        return f"{sign}{abs(v):,.{digits}f}{s}"
    s = f"y = {a:,.{digits}f}"
    s += term(b, "·T")
    s += term(c, "·T²")
    s += term(d, "·T³")
    return s

def fmt_int(x):
    try:
        return f"{int(np.round(float(x))):,}"
    except Exception:
        return str(x)

# ── Excel Loader (safe cache) ──
@st.cache_data(show_spinner=False)
def read_excel_cached(path_or_buf) -> pd.DataFrame:
    try:
        import openpyxl  # noqa: F401
        engine = None
    except Exception:
        engine = "openpyxl"
    try:
        try:
            return pd.read_excel(path_or_buf, sheet_name="data", engine=engine)
        except Exception:
            xls = pd.ExcelFile(path_or_buf, engine=engine)
            return pd.read_excel(xls, sheet_name=xls.sheet_names[0])
    except Exception as e:
        st.error(f"엑셀 로딩 오류: {type(e).__name__} — {e}")
        st.stop()

# ── Data In ──
st.sidebar.header("① 데이터")
repo_file = "실적.xlsx"
uploaded = st.sidebar.file_uploader("엑셀(.xlsx) 업로드 (없으면 리포지토리 파일 사용)", type=["xlsx"])
if uploaded is not None:
    raw = read_excel_cached(uploaded)
elif os.path.exists(repo_file):
    st.sidebar.info("리포지토리의 '실적.xlsx' 자동 사용")
    raw = read_excel_cached(repo_file)
else:
    st.info("엑셀을 업로드하거나 리포지토리에 '실적.xlsx'를 넣어줘.")
    st.stop()

cols = raw.columns.tolist()
st.sidebar.header("② 컬럼 매핑")
def _pick(cands, default_idx=0):
    for k in cands:
        for c in cols:
            if k in str(c): return c
    return cols[default_idx]

date_col    = st.sidebar.selectbox("날짜", cols, index=cols.index(_pick(["날짜","date"])) if _pick(["날짜","date"]) in cols else 0)
temp_col    = st.sidebar.selectbox("평균기온(℃)", cols, index=cols.index(_pick(["평균기온","기온","temp"])) if _pick(["평균기온","기온","temp"]) in cols else 1)
q_total_col = st.sidebar.selectbox("전체 공급량(MJ)", cols, index=cols.index(_pick(["공급","공급량","total","MJ"])) if _pick(["공급","공급량","total","MJ"]) in cols else 2)

df = raw.copy()
df["date"] = pd.to_datetime(df[date_col])
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["temp"]  = df[temp_col].apply(to_num)
df["Q"]     = df[q_total_col].apply(to_num)
df = df.dropna(subset=["temp","Q"]).sort_values("date")

st.success(f"전체 행 {len(df):,} · 기간 {df['date'].min().date()} ~ {df['date'].max().date()}")

# 학습 연도
st.sidebar.header("③ 학습 연도")
years = sorted(df["year"].unique().tolist())
sel_years = st.sidebar.multiselect("연도 선택", years, default=years)
train = df[df["year"].isin(sel_years)].copy()
if train.empty:
    st.warning("선택 연도 데이터가 없어.")
    st.stop()

# 시각 범위
T = train["temp"].values
p1, p99 = np.percentile(T, 1), np.percentile(T, 99)
pad = 1.5
xmin_vis = float(np.floor(p1 - pad))
xmax_vis = float(np.ceil(min(25.0, p99 + pad)))

# ── 1) 3차 다항식 + 산점도(개선 디자인) ──
st.subheader("① 기온–공급량 3차 다항식(상관)")
m_all, pf_all, Xp_all = fit_poly3(train["temp"].values, train["Q"].values)
yhat = m_all.predict(Xp_all)
r2 = poly3_r2(train["Q"].values, yhat)
tgrid = np.linspace(xmin_vis, xmax_vis, 500)
y_pred = poly3_predict(m_all, pf_all, tgrid)
ci_lo, ci_hi = poly3_conf_band(train["temp"].values, train["Q"].values, tgrid, m_all, pf_all)
a,b,c,d = poly3_coeffs(m_all)
eq_str = nice_poly_string(a,b,c,d, digits=1)

fig_corr = go.Figure()
# confidence band (더 진한 투명도)
fig_corr.add_trace(go.Scatter(
    x=np.r_[tgrid, tgrid[::-1]],
    y=np.r_[ci_hi, ci_lo[::-1]],
    fill="toself", name="95% 신뢰구간",
    line=dict(color="rgba(0,0,0,0)"),
    fillcolor="rgba(100,149,237,0.25)", hoverinfo="skip"
))
# regression line
fig_corr.add_trace(go.Scatter(x=tgrid, y=y_pred, mode="lines", name="Poly-3 추정",
                              line=dict(width=4)))
# prettier markers
fig_corr.add_trace(go.Scatter(
    x=train["temp"], y=train["Q"],
    mode="markers", name="샘플",
    marker=dict(size=9, line=dict(width=0.5, color="white")),
    opacity=0.9,
    hovertemplate="T=%{x:.2f}℃<br>Q=%{y:,.0f} MJ<extra></extra>"
))
fig_corr.update_layout(template="simple_white", font=dict(family=PLOT_FONT, size=14),
                       margin=dict(l=40,r=20,t=60,b=40),
                       xaxis=dict(title="기온(℃)", range=[xmin_vis, xmax_vis]),
                       yaxis=dict(title="공급량(MJ)", tickformat=","),
                       title=f"기온–공급량 상관 (R²={r2:.3f})")
fig_corr.add_annotation(xref="paper", yref="paper", x=0.01, y=0.02,
                        text=eq_str, showarrow=False,
                        bgcolor="rgba(255,255,255,0.9)", bordercolor="#888",
                        borderwidth=1, font=dict(size=12))
st.plotly_chart(fig_corr, use_container_width=True, config={"displaylogo": False})

# ── 2) 구간별 Δ1℃ 증가량(−5~0, 0~5, 5~10) ──
def band_mean(model, pf, temps: List[int]) -> float:
    vals = [max(0.0, -poly3_d1_at(model, pf, float(t0))) for t0 in temps]
    return float(np.mean(vals)) if vals else np.nan

mean_m5_0 = band_mean(m_all, pf_all, [-5,-4,-3,-2,-1,0])
mean_0_5  = band_mean(m_all, pf_all, [0,1,2,3,4,5])
mean_5_10 = band_mean(m_all, pf_all, [5,6,7,8,9,10])

c1, c2, c3 = st.columns(3)
c1.metric("구간 평균 −5→0℃", f"{fmt_int(mean_m5_0)} MJ/℃")
c2.metric("구간 평균 0→5℃",   f"{fmt_int(mean_0_5)} MJ/℃")
c3.metric("구간 평균 5→10℃",  f"{fmt_int(mean_5_10)} MJ/℃")

bar_fig = go.Figure()
bar_fig.add_trace(go.Bar(
    x=["−5~0℃","0~5℃","5~10℃"],
    y=[mean_m5_0, mean_0_5, mean_5_10],
    text=[fmt_int(mean_m5_0), fmt_int(mean_0_5), fmt_int(mean_5_10)],
    textposition="outside"
))
bar_fig.update_layout(template="simple_white", font=dict(family=PLOT_FONT, size=14),
                      margin=dict(l=40,r=20,t=40,b=40),
                      yaxis=dict(title="Δ1℃ 증가량 (MJ/℃)", tickformat=","))
st.plotly_chart(bar_fig, use_container_width=True, config={"displaylogo": False})

# ── 3) dQ/dT 곡선 + Heating Start/Slowdown ──
st.subheader("② 변화율(dQ/dT)과 난방구간")
theta_star, a_hat, b_hat = hinge_base_temp(train["temp"].values, train["Q"].values, 0.0, 20.0, 0.1)
tgrid_d = np.linspace(xmin_vis, xmax_vis, 800)
d1 = np.array([poly3_d1_at(m_all, pf_all, t) for t in tgrid_d])
T_slow = float(tgrid_d[int(np.argmin(d1))])

fig_d = go.Figure()
fig_d.add_trace(go.Scatter(x=tgrid_d, y=d1, mode="lines", name="dQ/dT (MJ/℃)",
                           line=dict(width=4),
                           hovertemplate="T=%{x:.2f}℃<br>dQ/dT=%{y:,.0f} MJ/℃<extra></extra>"))
# zones
fig_d.add_vline(x=T_slow, line_dash="dash", line_color="crimson",
                annotation_text=f"Slowdown {T_slow:.2f}℃", annotation_position="top left")
fig_d.add_vline(x=theta_star, line_dash="dash", line_color="royalblue",
                annotation_text=f"Start θ*={theta_star:.2f}℃", annotation_position="top right")
fig_d.add_vrect(x0=xmin_vis, x1=T_slow, fillcolor="LightCoral", opacity=0.15, line_width=0,
                annotation_text="Heating Slowdown Zone", annotation_position="top left")
fig_d.add_vrect(x0=T_slow, x1=theta_star, fillcolor="LightSkyBlue", opacity=0.15, line_width=0,
                annotation_text="Heating Start Zone", annotation_position="top right")
fig_d.update_layout(template="simple_white", font=dict(family=PLOT_FONT, size=14),
                    margin=dict(l=40,r=20,t=50,b=40),
                    xaxis=dict(title="기온(℃)", range=[xmin_vis, xmax_vis]),
                    yaxis=dict(title="dQ/dT (MJ/℃)", tickformat=","),
                    title="Rate of Change vs Temperature")
st.plotly_chart(fig_d, use_container_width=True, config={"displaylogo": False})

# ── 4) 간단 표(최종 요약) ──
st.subheader("③ 구간별 Δ1℃ 증가량 요약표")
summary_df = pd.DataFrame({
    "구간": ["−5~0℃", "0~5℃", "5~10℃"],
    "Δ1℃ 증가량 [MJ/℃]": [mean_m5_0, mean_0_5, mean_5_10]
})
st.dataframe(summary_df.assign(**{"Δ1℃ 증가량 [MJ/℃]": summary_df["Δ1℃ 증가량 [MJ/℃]"].map(fmt_int)}))
