# app.py — HeatBand Insight (simplified deck for execs · restore A/B · pretty charts · bottom matplotlib)
# 단위: 공급량(MJ), 변화율 dQ/dT(MJ/℃)

import os
from typing import Tuple, List
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import streamlit as st

st.set_page_config(page_title="HeatBand Insight", layout="wide")

# ── Korean font ──────────────────────────────────────────────
FONT_PATH = "NanumGothic-Regular.ttf"
if os.path.exists(FONT_PATH):
    try:
        fm.fontManager.addfont(FONT_PATH)
        plt.rcParams["font.family"] = "NanumGothic"
    except Exception:
        pass
PLOT_FONT = "NanumGothic, Arial, Noto Sans KR, sans-serif"

st.title("🔥 HeatBand Insight — 난방구간·민감도 분석")
st.caption("단위: 공급량 **MJ**, 변화율 **MJ/℃** · Heating Start(θ*) · Heating Slowdown · Δ1°C Impact")

# ── Utils ───────────────────────────────────────────────────
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

def poly3_conf_band(x_train, y_train, tgrid, m, pf):
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

def df_commas(df, except_cols=None):
    except_cols = set(except_cols or [])
    out = df.copy()
    for c in out.columns:
        if c in except_cols: 
            continue
        if pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].apply(fmt_int)
    return out

# ── Excel Loader (safe) ─────────────────────────────────────
@st.cache_data(show_spinner=False)
def read_excel_cached(path_or_buf) -> pd.DataFrame:
    """'data' 시트 우선, 없으면 첫 시트."""
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
        st.error(f"엑셀을 읽는 중 문제: {type(e).__name__} — {e}")
        st.stop()

# ── Data in ──────────────────────────────────────────────────
st.sidebar.header("① 데이터")
repo_file = "실적.xlsx"
uploaded = st.sidebar.file_uploader("엑셀(.xlsx) 업로드 (없으면 리포지토리 파일 사용)", type=["xlsx"])
if uploaded is not None:
    raw = read_excel_cached(uploaded)
elif os.path.exists(repo_file):
    st.sidebar.info("리포지토리의 '실적.xlsx' 자동 사용 중")
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
q_total_col = st.sidebar.selectbox("전체 공급량(MJ)", cols, index=cols.index(_pick(["공급량","총","total","MJ"])) if _pick(["공급량","총","total","MJ"]) in cols else 2)
q_res_col_options = ["(없음)"] + cols
q_res_col  = st.sidebar.selectbox("가정용 공급량(MJ) (선택)", q_res_col_options, index=0)

df_all = raw.copy()
df_all["date"] = pd.to_datetime(df_all[date_col])
df_all["year"] = df_all["date"].dt.year
df_all["month"] = df_all["date"].dt.month
df_all["temp"]  = df_all[temp_col].apply(to_num)
df_all["Q_total"] = df_all[q_total_col].apply(to_num)
if q_res_col != "(없음)":
    df_all["Q_res"] = df_all[q_res_col].apply(to_num)
df_all = df_all.dropna(subset=["temp","Q_total"]).sort_values("date")

st.success(f"전체 행 {len(df_all):,} · 기간 {df_all['date'].min().date()} ~ {df_all['date'].max().date()}")

# ── Target & years ──────────────────────────────────────────
st.sidebar.header("③ 분석 대상")
targets = ["전체(MJ)"]
if "Q_res" in df_all.columns: targets.append("가정용(MJ)")
target_choice = st.sidebar.radio("대상 선택", targets, horizontal=True)
target_col = "Q_total" if target_choice.startswith("전체") else "Q_res"

st.sidebar.header("④ 학습 데이터 연도 선택")
years = sorted(df_all["year"].unique().tolist())
sel_years = st.sidebar.multiselect("연도 선택", years, default=years)
df_train = df_all[df_all["year"].isin(sel_years)].copy().dropna(subset=[target_col])
if df_train.empty:
    st.warning("선택된 학습 연도에 데이터가 없습니다.")
    st.stop()

# ── θ* 탐색 ─────────────────────────────────────────────────
th_min, th_max, th_step = 0.0, 20.0, 0.1

# 시각 구간
T_train = df_train["temp"].values
p1, p99 = np.percentile(T_train, 1), np.percentile(T_train, 99)
pad = 1.5
xmin_vis = float(np.floor(p1 - pad))
xmax_vis = float(np.ceil(min(25.0, p99 + pad)))

# ── A0: Poly-3 상관(업그레이드 스타일) ─────────────────────
st.subheader(f"A0. 기온–공급량 상관(Poly-3) — 대상: {target_choice}")
x_tr = df_train["temp"].values
y_tr = df_train[target_col].values
m_all, pf_all, Xp_all = fit_poly3(x_tr, y_tr)
yhat_tr = m_all.predict(Xp_all)
r2 = poly3_r2(y_tr, yhat_tr)
tgrid0 = np.linspace(xmin_vis, xmax_vis, 500)
y_pred0 = poly3_predict(m_all, pf_all, tgrid0)
ci_lo, ci_hi = poly3_conf_band(x_tr, y_tr, tgrid0, m_all, pf_all)

a,b,c,d = poly3_coeffs(m_all)
eq_str = nice_poly_string(a,b,c,d, digits=1)

fig_corr = go.Figure()
fig_corr.add_trace(go.Scatter(
    x=df_train["temp"], y=df_train[target_col], mode="markers", name="학습 샘플",
    marker=dict(
        size=9,
        color=df_train["temp"],
        colorscale="Blues",
        line=dict(width=0.6, color="rgba(0,0,0,0.15)"),
        opacity=0.9,
        symbol="circle"
    ),
    hovertemplate="T=%{x:.2f}℃<br>Q=%{y:,.0f} MJ<extra></extra>"
))
fig_corr.add_traces([
    go.Scatter(x=np.r_[tgrid0, tgrid0[::-1]],
               y=np.r_[ci_hi, ci_lo[::-1]],
               fill="toself", name="95% 신뢰구간",
               line=dict(color="rgba(31,119,180,0)"),
               fillcolor="rgba(31,119,180,0.18)", hoverinfo="skip")
])
fig_corr.add_trace(go.Scatter(x=tgrid0, y=y_pred0, mode="lines", name="Poly-3",
                              line=dict(width=3)))
fig_corr.update_layout(
    template="simple_white", font=dict(family=PLOT_FONT, size=14),
    margin=dict(l=40,r=20,t=50,b=40),
    xaxis=dict(title="기온(℃)", range=[xmin_vis, xmax_vis], zeroline=True, showgrid=True),
    yaxis=dict(title="공급량(MJ)", tickformat=",", showgrid=True),
    title=f"기온–공급량 상관(Train, R²={r2:.3f})"
)
fig_corr.add_annotation(xref="paper", yref="paper", x=0.01, y=0.02,
                        text=eq_str, showarrow=False,
                        bgcolor="rgba(255,255,255,0.85)", bordercolor="rgba(0,0,0,0.3)",
                        borderwidth=1, font=dict(size=12))
st.plotly_chart(fig_corr, use_container_width=True, config={"displaylogo": False})

# ── A: Heating Start(복원) ───────────────────────────────────
st.subheader(f"A. Heating Start Zone — 베이스온도(θ*) · 대상: {target_choice}")
theta_star, a_hat, b_hat = hinge_base_temp(df_train["temp"].values, df_train[target_col].values,
                                           th_min, th_max, th_step)
st.metric("베이스온도 θ*", f"{theta_star:.2f} ℃")

def make_start_figure(df_all_in, df_train_in, theta_star, a_hat, b_hat, xmin_vis, xmax_vis, y_title="공급량(MJ)") -> go.Figure:
    tline = np.linspace(xmin_vis, xmax_vis, 320)
    H = np.clip(theta_star - tline, 0, None)
    qhat = a_hat + b_hat*H
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_all_in["temp"], y=df_all_in["Q"], mode="markers", name="전체 실측(참고)",
                             marker=dict(size=6, color="rgba(0,0,0,0.15)"), opacity=0.35,
                             hovertemplate="T=%{x:.2f}℃<br>Q=%{y:,.0f} MJ<extra></extra>"))
    fig.add_trace(go.Scatter(x=df_train_in["temp"], y=df_train_in["Q"], mode="markers", name="학습 실측",
                             marker=dict(size=8, color="rgba(31,119,180,0.9)",
                                         line=dict(width=0.5, color="rgba(0,0,0,0.2)")),
                             hovertemplate="T=%{x:.2f}℃<br>Q=%{y:,.0f} MJ<extra></extra>"))
    fig.add_trace(go.Scatter(x=tline, y=qhat, mode="lines", name="힌지 적합(훈련)", line=dict(width=3)))
    fig.add_vline(x=theta_star, line_dash="dash",
                  annotation_text=f"θ* = {theta_star:.2f}℃", annotation_position="top right")
    fig.add_vrect(x0=xmin_vis, x1=theta_star, fillcolor="LightSkyBlue", opacity=0.18, line_width=0,
                  annotation_text="Heating Start Zone", annotation_position="top left")
    fig.update_layout(template="simple_white", font=dict(family=PLOT_FONT, size=14),
                      margin=dict(l=40,r=20,t=50,b=40), hovermode="x unified",
                      xaxis=dict(title="기온(℃)", range=[xmin_vis, xmax_vis]),
                      yaxis=dict(title=y_title, tickformat=","), title="힌지 적합과 Heating Start Zone")
    return fig

df_all_plot   = df_all[["temp", target_col]].rename(columns={target_col:"Q"})
df_train_plot = df_train[["temp", target_col]].rename(columns={target_col:"Q"})
st.plotly_chart(
    make_start_figure(df_all_plot, df_train_plot, theta_star, a_hat, b_hat, xmin_vis, xmax_vis),
    use_container_width=True, config={"displaylogo": False}
)

# ── B: Heating Slowdown & dQ/dT(복원) ────────────────────────
st.subheader("B. Heating Slowdown Zone & dQ/dT (Poly-3)")
m_poly, pf_poly, _ = fit_poly3(df_train["temp"].values, df_train[target_col].values)
tgrid = np.linspace(xmin_vis, xmax_vis, 800)
d1 = np.array([poly3_d1_at(m_poly, pf_poly, t) for t in tgrid])
T_slow = float(tgrid[int(np.argmin(d1))])  # 최대 음의 기울기 위치

st.metric("Slowdown 경계 T_slow", f"{T_slow:.2f} ℃")

def make_derivative_figure(tgrid, d1, theta_star, T_slow, xmin_vis, xmax_vis,
                           y_title="변화율 dQ/dT (MJ/℃)") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=tgrid, y=d1, mode="lines", name="dQ/dT (MJ/℃)",
                             line=dict(width=3),
                             hovertemplate="T=%{x:.2f}℃<br>dQ/dT=%{y:,.0f} MJ/℃<extra></extra>"))
    fig.add_vline(x=T_slow, line_dash="dash", line_color="red",
                  annotation_text=f"Slowdown {T_slow:.2f}℃", annotation_position="top left")
    fig.add_vline(x=theta_star, line_dash="dash", line_color="steelblue",
                  annotation_text=f"Start θ*={theta_star:.2f}℃", annotation_position="top right")
    fig.add_vrect(x0=xmin_vis, x1=T_slow, fillcolor="LightCoral", opacity=0.14, line_width=0,
                  annotation_text="Heating Slowdown Zone", annotation_position="top left")
    fig.add_vrect(x0=T_slow, x1=theta_star, fillcolor="LightSkyBlue", opacity=0.14, line_width=0,
                  annotation_text="Heating Start Zone", annotation_position="top right")
    fig.update_layout(template="simple_white", font=dict(family=PLOT_FONT, size=14),
                      margin=dict(l=40,r=20,t=50,b=40), hovermode="x unified",
                      xaxis=dict(title="기온(℃)", range=[xmin_vis, xmax_vis]),
                      yaxis=dict(title=y_title, tickformat=","), title="Rate of Change vs Temperature")
    return fig

st.plotly_chart(
    make_derivative_figure(tgrid, d1, theta_star, T_slow, xmin_vis, xmax_vis),
    use_container_width=True, config={"displaylogo": False}
)

# ── C: 구간별 Δ1℃ 증가량(−5~0 / 0~5 / 5~10) — 깔끔 막대 ──────
def band_mean_from_model(temp_list: List[float], model, pf) -> float:
    vals = [max(0.0, -poly3_d1_at(model, pf, float(t0))) for t0 in temp_list]
    return float(np.mean(vals)) if vals else np.nan

mean_m5_0 = band_mean_from_model([-5,-4,-3,-2,-1,0], m_poly, pf_poly)
mean_0_5  = band_mean_from_model([0,1,2,3,4,5], m_poly, pf_poly)
mean_5_10 = band_mean_from_model([5,6,7,8,9,10], m_poly, pf_poly)

st.subheader("C. 기온 구간별 Δ1℃ 증가량 요약 [MJ/℃]")
bar = go.Figure()
cats = ["−5~0℃", "0~5℃", "5~10℃"]
vals = [mean_m5_0, mean_0_5, mean_5_10]
bar.add_trace(go.Bar(x=cats, y=vals, text=[fmt_int(v) for v in vals],
                     textposition="outside", marker=dict(line=dict(width=0), opacity=0.9)))
bar.update_layout(template="simple_white", font=dict(family=PLOT_FONT, size=14),
                  margin=dict(l=40,r=20,t=50,b=40),
                  yaxis=dict(title="Δ1℃ 증가량 (MJ/℃)", tickformat=","),
                  xaxis=dict(title="기온 구간"),
                  showlegend=False)
st.plotly_chart(bar, use_container_width=True, config={"displaylogo": False})

# ── D: (하단) Refined Gas Supply Rate of Change — Matplotlib ─
st.subheader("D. Refined Gas Supply Rate of Change (Matplotlib)")
fig, ax = plt.subplots(figsize=(9.5, 4.6), dpi=150)
ax.plot(tgrid, d1, linewidth=2.5)  # 기본 색상(규정: 색 지정하지 않음)
# 영역 채움
ax.axvspan(xmin=xmin_vis, xmax=T_slow, alpha=0.12)   # slowdown zone
ax.axvspan(xmin=T_slow, xmax=theta_star, alpha=0.08) # start zone
# 라인
ax.axvline(T_slow, linestyle="--", linewidth=1.8)
ax.axvline(theta_star, linestyle="--", linewidth=1.8)
ax.set_xlabel("Temperature (°C)")
ax.set_ylabel("Rate of Change in Gas Supply (MJ/°C)")
ax.set_title("Refined Gas Supply Rate of Change vs Temperature (Recalculated)")
ax.grid(True, linewidth=0.3)
st.pyplot(fig, clear_figure=True)
