# app.py — HeatBand Insight (요약 UI + 동적 그래프 + XLSX Export)
# 단위: 공급량 Q(MJ), 민감도/증가량 Δ1°C(MJ/℃ = −dQ/dT)

import os, io
from typing import Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.font_manager as fm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import streamlit as st

st.set_page_config(page_title="HeatBand Insight", layout="wide")

# ── Korean font (optional) ───────────────────────────────────
FONT_PATH = "NanumGothic-Regular.ttf"
if os.path.exists(FONT_PATH):
    try:
        fm.fontManager.addfont(FONT_PATH)
    except Exception:
        pass
PLOT_FONT = "NanumGothic, Arial, Noto Sans KR, sans-serif"

st.title("🔥 HeatBand Insight — 난방구간·민감도 분석")

# ── Utils ───────────────────────────────────────────────────
def to_num(x):
    if isinstance(x, str): x = x.replace(",", "")
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

def conf_band_y(x_train, y_train, tgrid, m, pf, z=1.96):
    X = pf.transform(x_train.reshape(-1,1))
    yhat = m.predict(X)
    n, p = X.shape
    sigma2 = np.sum((y_train - yhat)**2) / max(1, (n - p))
    XtX_inv = np.linalg.pinv(X.T @ X)
    Tg = pf.transform(tgrid.reshape(-1,1))
    se = np.sqrt(np.sum(Tg @ XtX_inv * Tg, axis=1) * sigma2)
    ypred = m.predict(Tg)
    return ypred - z*se, ypred + z*se, ypred, sigma2, XtX_inv

def d1_at(m: LinearRegression, t: float) -> float:
    a,b,c,d = poly3_coeffs(m)
    return b + 2*c*t + 3*d*(t**2)   # dQ/dT

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
    try: return f"{int(np.round(float(x))):,}"
    except Exception: return str(x)

# ── Excel Loader ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def read_excel_cached(path_or_buf) -> pd.DataFrame:
    try:
        import openpyxl  # noqa
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
        st.error(f"엑셀 로딩 문제: {type(e).__name__} — {e}")
        st.stop()

# ── 데이터 입력 ──────────────────────────────────────────────
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

date_col = st.sidebar.selectbox("날짜", cols, index=cols.index(_pick(["날짜","date"])) if _pick(["날짜","date"]) in cols else 0)
temp_col = st.sidebar.selectbox("평균기온(℃)", cols, index=cols.index(_pick(["평균기온","기온","temp"])) if _pick(["평균기온","기온","temp"]) in cols else 1)
q_col    = st.sidebar.selectbox("공급량(MJ)", cols, index=cols.index(_pick(["공급량","총","total","MJ"])) if _pick(["공급량","총","total","MJ"]) in cols else 2)

df = raw.copy()
df["date"] = pd.to_datetime(df[date_col])
df["year"] = df["date"].dt.year
df["month"]= df["date"].dt.month
df["temp"] = df[temp_col].apply(to_num)
df["Q"]    = df[q_col].apply(to_num)
df = df.dropna(subset=["temp","Q"]).sort_values("date")

st.success(f"행 {len(df):,} · 기간 {df['date'].min().date()} ~ {df['date'].max().date()}")

# 학습 연도
st.sidebar.header("③ 학습 연도")
years = sorted(df["year"].unique().tolist())
sel_years = st.sidebar.multiselect("연도 선택", years, default=years)
train = df[df["year"].isin(sel_years)].copy()
if train.empty:
    st.warning("선택된 연도에 데이터가 없습니다."); st.stop()

# ── 시각 범위 ────────────────────────────────────────────────
T = train["temp"].values
p1, p99 = np.percentile(T, 1), np.percentile(T, 99)
xmin_vis = float(np.floor(min(-5, p1 - 1.5))))
xmax_vis = float(np.ceil(max(25, p99 + 1.5)))

# ── Poly-3 적합(전체) ────────────────────────────────────────
m_all, pf_all, Xp = fit_poly3(train["temp"].values, train["Q"].values)
yhat = m_all.predict(Xp)
r2   = poly3_r2(train["Q"].values, yhat)
a,b,c,d = poly3_coeffs(m_all)
eq_str  = nice_poly_string(a,b,c,d, digits=1)

# 촘촘한 그리드
tgrid = np.linspace(xmin_vis, xmax_vis, 1201)
ci_lo_95, ci_hi_95, y_pred, sigma2, XtX_inv = conf_band_y(
    train["temp"].values, train["Q"].values, tgrid, m_all, pf_all, z=1.96
)

# ── 도함수(민감도) ───────────────────────────────────────────
J = np.vstack([np.ones_like(tgrid)*0, np.ones_like(tgrid), 2*tgrid, 3*(tgrid**2)]).T
deriv_mean = np.array([d1_at(m_all, t) for t in tgrid])       # dQ/dT (음수 가능)
deriv_se   = np.sqrt(np.sum(J @ XtX_inv * J, axis=1) * sigma2)
z90 = 1.645
d_lo = deriv_mean - z90*deriv_se
d_hi = deriv_mean + z90*deriv_se

# 증가량(양수화)
base_inc   = np.maximum(0.0, -deriv_mean)
base_lo    = np.maximum(0.0, -d_hi)   # 하한(증가량 관점)
base_hi    = np.maximum(0.0, -d_lo)   # 상한

# ── 저온 완화(Attenuation) 옵션 ─────────────────────────────
st.sidebar.header("④ 시뮬레이션 옵션")
auto_zoom = st.sidebar.toggle("밴드 자동 Y축 줌(곡률 강조)", value=True)
use_cold  = st.sidebar.toggle("저온 완화 적용(아주 낮은 온도에서 증가량 둔화)", value=True)
T_cold    = st.sidebar.slider("저온 완화 시작온도 T_cold(℃)", -10.0, 5.0, -2.0, 0.1)
tau       = st.sidebar.slider("완화 전이폭 τ(℃, 클수록 완만)", 0.5, 5.0, 1.5, 0.1)
# 수요곡선 곡률 강조
curve_k   = st.sidebar.slider("수요곡선 곡률 강조(×)", 1.0, 4.0, 2.0, 0.1)

def sigmoid(x): return 1/(1+np.exp(-x))
cold_factor = sigmoid((tgrid - T_cold)/tau) if use_cold else np.ones_like(tgrid)

inc      = base_inc * cold_factor
inc_lo   = base_lo  * cold_factor
inc_hi   = base_hi  * cold_factor

# ── 난방 시작/둔화/포화 ─────────────────────────────────────
def hinge_base_temp(T: np.ndarray, Q: np.ndarray,
                    grid_min: float=0.0, grid_max: float=20.0, step: float=0.1):
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

theta_star, a_hat, b_hat = hinge_base_temp(train["temp"].values, train["Q"].values, 0.0, 20.0, 0.1)
T_slow   = float(tgrid[int(np.argmin(deriv_mean))])             # 최저 기울기 지점
max_neg  = float(np.max(inc))
T_cap    = float(tgrid[np.argmax(inc <= 0.02*max_neg)]) if max_neg>0 else np.nan

# ========== 곡선 힌지(큐빅) 적합 (섹션 B) ==========
def fit_hinge_cubic(T: np.ndarray, Q: np.ndarray, theta: float) -> Tuple[float,float,float,float]:
    """
    Q ~ a + b*H + c*H^2 + d*H^3, H=max(theta - T, 0)
    """
    H = np.clip(theta - T, 0, None)
    X = np.column_stack([np.ones_like(H), H, H**2, H**3])
    beta, *_ = np.linalg.lstsq(X, Q, rcond=None)
    a_c, b_c, c_c, d_c = map(float, beta)
    return a_c, b_c, c_c, d_c

a_c, b_c, c_c, d_c = fit_hinge_cubic(train["temp"].values, train["Q"].values, theta_star)

def qhat_cubic(t: np.ndarray, theta: float, a_c: float, b_c: float, c_c: float, d_c: float, k: float) -> np.ndarray:
    H = np.clip(theta - t, 0, None)
    # 곡률 강조: 2·3차항에 k를 곱해 눈에 보이는 휨을 강화
    return a_c + b_c*H + (k*c_c)*(H**2) + (k*d_c)*(H**3)

# ── (A) 상관 그래프 ─────────────────────────────────────────
st.subheader("🧮 A. 기온–공급량 상관 (Poly-3, 95% CI)")
figA = go.Figure()
figA.add_trace(go.Scatter(
    x=train["temp"], y=train["Q"], mode="markers", name="샘플",
    marker=dict(size=8, opacity=0.75, line=dict(width=0.5), symbol="circle"),
    hovertemplate="T=%{x:.2f}℃<br>Q=%{y:,.0f} MJ<extra></extra>"
))
figA.add_trace(go.Scatter(
    x=np.r_[tgrid, tgrid[::-1]], y=np.r_[ci_hi_95, ci_lo_95[::-1]],
    fill="toself", name="95% CI",
    line=dict(color="rgba(0,0,0,0)"),
    fillcolor="rgba(0,123,255,0.18)", hoverinfo="skip"
))
figA.add_trace(go.Scatter(
    x=tgrid, y=y_pred, mode="lines", name="Poly-3", line=dict(width=3),
    hovertemplate="T=%{x:.2f}℃<br>예측=%{y:,.0f} MJ<extra></extra>"
))
figA.update_layout(template="simple_white", font=dict(family=PLOT_FONT, size=14),
                   margin=dict(l=40,r=20,t=40,b=40),
                   xaxis=dict(title="기온(℃)", range=[xmin_vis, xmax_vis]),
                   yaxis=dict(title="공급량(MJ)", tickformat=","),
                   title=f"R²={r2:.3f} · 식: {eq_str}")
st.plotly_chart(figA, use_container_width=True, config={"displaylogo": False})

# ── (B) 수요곡선 — 곡선 힌지 + 라벨 겹침 해결 ──────────────
st.subheader("🧊 B. Heating Start / Slowdown — 수요곡선")
tline = np.linspace(xmin_vis, xmax_vis, 600)
qhat_curve = qhat_cubic(tline, theta_star, a_c, b_c, c_c, d_c, curve_k)  # 곡선

figB = go.Figure()
figB.add_trace(go.Scatter(x=df["temp"], y=df["Q"], mode="markers", name="전체(참고)",
                          marker=dict(size=6, color="lightgray"), opacity=0.45))
figB.add_trace(go.Scatter(x=train["temp"], y=train["Q"], mode="markers", name="학습",
                          marker=dict(size=7), marker_color="orange"))
figB.add_trace(go.Scatter(
    x=tline, y=qhat_curve, mode="lines", name="힌지(곡선) 적합",
    line=dict(width=3, shape="spline", smoothing=1.1)
))

figB.add_vrect(x0=xmin_vis, x1=theta_star, fillcolor="LightSkyBlue", opacity=0.18, line_width=0, layer="below")
figB.add_annotation(x=(xmin_vis+theta_star)/2, y=1.12, xref="x", yref="paper",
                    text="Heating Start Zone", showarrow=False,
                    font=dict(size=12), bgcolor="rgba(255,255,255,0.7)", bordercolor="rgba(0,0,0,0.1)")
figB.add_vrect(x0=xmin_vis, x1=T_slow, fillcolor="LightCoral", opacity=0.14, line_width=0, layer="below")
figB.add_annotation(x=(xmin_vis+T_slow)/2, y=1.12, xref="x", yref="paper",
                    text=f"Heating Slowdown Zone (≤ {T_slow:.2f}℃)", showarrow=False,
                    font=dict(size=12), bgcolor="rgba(255,255,255,0.7)", bordercolor="rgba(0,0,0,0.1)")
figB.add_vline(x=theta_star, line_dash="dash")
figB.add_annotation(x=theta_star, y=1.14, xref="x", yref="paper",
                    text=f"Start θ* = {theta_star:.2f}℃", showarrow=False, font=dict(size=12),
                    bgcolor="rgba(255,255,255,0.7)", bordercolor="rgba(0,0,0,0.1)")
if np.isfinite(T_cap):
    figB.add_vline(x=T_cap, line_dash="dot")
    figB.add_annotation(x=T_cap, y=1.10, xref="x", yref="paper",
                        text=f"Saturation {T_cap:.2f}℃", showarrow=False, font=dict(size=12),
                        bgcolor="rgba(255,255,255,0.7)", bordercolor="rgba(0,0,0,0.1)")

# 범례 하단으로 이동(겹침 방지)
figB.update_layout(template="simple_white", font=dict(family=PLOT_FONT, size=14),
                   margin=dict(l=40,r=20,t=60,b=70),
                   xaxis=dict(title="기온(℃)", range=[xmin_vis, xmax_vis]),
                   yaxis=dict(title="공급량(MJ)", tickformat=","),
                   legend=dict(orientation="h", yanchor="top", y=-0.18, x=0.01))
st.plotly_chart(figB, use_container_width=True, config={"displaylogo": False})

# ── (C) 기온별 공급량 변화량 요약 ───────────────────────────
st.subheader("🌡️ C. 기온별 공급량 변화량 요약")

# 아래 함수가 섹션 D와 '정의'를 일치시킴(저온 완화 옵션 반영)
def band_mean(temp_array, apply_cold=True):
    temps = np.array(temp_array, dtype=float)
    base = np.maximum(0.0, -np.array([d1_at(m_all, t) for t in temps]))  # Raw(Poly-3) 증가량
    if apply_cold and use_cold:
        cf = 1.0 / (1.0 + np.exp(-(temps - T_cold) / tau))               # cold_factor
        base = base * cf
    return float(np.mean(base))

band = {"−5~0℃": np.arange(-5, 0.001, 0.1),
        "0~5℃" : np.arange(0, 5.001, 0.1),
        "5~10℃": np.arange(5,10.001,0.1)}

# 그래프(D/E)와 동일 정의(저온 완화 반영)
avg_m5_0  = band_mean(band["−5~0℃"], apply_cold=True)
avg_0_5   = band_mean(band["0~5℃"],  apply_cold=True)
avg_5_10  = band_mean(band["5~10℃"], apply_cold=True)

# 비교 참고: 순수 Poly-3
raw_m5_0  = band_mean(band["−5~0℃"], apply_cold=False)
raw_0_5   = band_mean(band["0~5℃"],  apply_cold=False)
raw_5_10  = band_mean(band["5~10℃"], apply_cold=False)

st.markdown(
f"""
**Polynomial Regression (degree 3)**  
**{eq_str}**  

- **Supply ↑ per −1°C** from **10→5℃**: **{fmt_int(avg_5_10)} MJ/℃**  
- **Supply ↑ per −1°C** from **5→0℃** : **{fmt_int(avg_0_5)} MJ/℃**  
- **Supply ↑ per −1°C** from **0→−5℃**: **{fmt_int(avg_m5_0)} MJ/℃**

<sub>Raw(Poly-3) 참고 → 10→5:{fmt_int(raw_5_10)}, 5→0:{fmt_int(raw_0_5)}, 0→−5:{fmt_int(raw_m5_0)} (MJ/℃)</sub>
"""
)

# ── (D) 구간별 동적 그래프 ───────────────────────────────────
st.subheader("📈 D. 기온 구간별 동적 그래프 (−dQ/dT = 1℃ 하락 시 증가량, 5% CI ≈ 90%)")
tab1, tab2, tab3 = st.tabs(["−5~0℃", "0~5℃", "5~10℃"])

def band_plot(ax, loT, hiT, label):
    mask = (tgrid>=loT) & (tgrid<=hiT)
    x = tgrid[mask]
    y_mid = inc[mask]
    y_lo  = inc_lo[mask]
    y_hi  = inc_hi[mask]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.r_[x, x[::-1]],
        y=np.r_[y_hi, y_lo[::-1]],
        fill="toself", name="5% CI (±)", line=dict(color="rgba(0,0,0,0)"),
        fillcolor="rgba(0,123,255,0.15)", hoverinfo="skip"
    ))
    fig.add_trace(go.Scatter(
        x=x, y=y_mid, mode="lines", name="증가량(MJ/℃)",
        line=dict(width=3, shape="spline", smoothing=0.9),
        hovertemplate="T=%{x:.2f}℃<br>증가량=%{y:,.0f} MJ/℃<extra></extra>"
    ))
    avg = float(np.mean(y_mid))
    fig.add_annotation(x=(loT+hiT)/2, y=np.max(y_hi),
                       text=f"Band Avg = {fmt_int(avg)} MJ/℃", showarrow=False, yshift=18)
    if auto_zoom:
        y_min, y_max = float(np.min(y_mid)), float(np.max(y_mid))
        pad = 0.08 * (y_max - y_min if y_max>y_min else max(1.0, y_max))
        fig.update_yaxes(range=[y_min - pad, y_max + pad])
    fig.update_layout(template="simple_white", font=dict(family=PLOT_FONT, size=14),
                      margin=dict(l=40,r=20,t=40,b=40),
                      xaxis=dict(title="기온(℃)", range=[loT, hiT]),
                      yaxis=dict(title="Δ1℃ 증가량(MJ/℃)", tickformat=","),
                      title=f"Band {label} Response")
    ax.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

with tab1: band_plot(st, -5, 0, "−5~0℃")
with tab2: band_plot(st, 0, 5, "0~5℃")
with tab3: band_plot(st, 5, 10, "5~10℃")

# ── (E) 전체 곡선(저온 완화 포함) ───────────────────────────
st.subheader("🧭 E. Refined Gas Supply Rate of Change (Dynamic)")
figE = go.Figure()
figE.add_trace(go.Scatter(
    x=tgrid, y=inc, mode="lines", name="증가량(MJ/℃)",
    line=dict(width=3, shape="spline", smoothing=1.2),
    hovertemplate="T=%{x:.2f}℃<br>증가량=%{y:,.0f} MJ/℃<extra></extra>"
))

# 영역 음영
figE.add_vrect(x0=xmin_vis, x1=T_slow, fillcolor="LightCoral", opacity=0.12, line_width=0, layer="below")
figE.add_vrect(x0=T_slow, x1=theta_star, fillcolor="LightSkyBlue", opacity=0.12, line_width=0, layer="below")

# 상단 영역 라벨(플롯 밖 y=paper에 고정, 얇은 배경으로 가독성 ↑)
def top_note(x, text, y=1.12):
    figE.add_annotation(
        x=x, y=y, xref="x", yref="paper", showarrow=False, text=text,
        font=dict(size=12),
        bgcolor="rgba(255,255,255,0.75)", bordercolor="rgba(0,0,0,0.12)", borderwidth=1
    )

top_note((xmin_vis+T_slow)/2,  f"Heating Slowdown (≤ {T_slow:.2f}℃)")
top_note((T_slow+theta_star)/2, f"Heating Start ({T_slow:.2f}~{theta_star:.2f}℃)")

# 기준선 표기
figE.add_vline(x=theta_star, line_dash="dash", line_color="black")
figE.add_annotation(x=theta_star, y=1.14, xref="x", yref="paper",
                    text=f"Start θ* {theta_star:.2f}℃", showarrow=False,
                    font=dict(size=12), bgcolor="rgba(255,255,255,0.75)",
                    bordercolor="rgba(0,0,0,0.12)", borderwidth=1)

# 저온 완화 라벨의 겹침 회피 로직
if use_cold:
    figE.add_vline(x=T_cold, line_dash="dot", line_color="gray")
    y_tcold = 1.10
    xshift  = 0
    if T_cold <= (xmin_vis + 0.35*(T_slow - xmin_vis)):
        y_tcold = 1.18
    if T_cold <= xmin_vis + 0.8:
        y_tcold = 1.06
        xshift  = 28
    figE.add_annotation(
        x=T_cold, y=y_tcold, xref="x", yref="paper", xshift=xshift,
        text=f"저온 완화시작 T_cold={T_cold:.1f}℃", showarrow=False,
        font=dict(size=12), bgcolor="rgba(255,255,255,0.85)",
        bordercolor="rgba(0,0,0,0.12)", borderwidth=1
    )

figE.update_layout(
    template="simple_white", font=dict(family=PLOT_FONT, size=14),
    margin=dict(l=40,r=20,t=40,b=40),
    xaxis=dict(title="Temperature (℃)", range=[xmin_vis, xmax_vis]),
    yaxis=dict(title="Rate of Change (MJ/℃, +가 증가)", tickformat=","),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0)
)
st.plotly_chart(figE, use_container_width=True, config={"displaylogo": False})

# ── (F) XLSX 다운로드 ───────────────────────────────────────
st.subheader("📥 F. 결과 다운로드")
@st.cache_data(show_spinner=False)
def build_xlsx_bytes():
    try:
        import xlsxwriter  # noqa
        engine = "xlsxwriter"
    except Exception:
        engine = "openpyxl"
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine=engine) as wr:
        summary = pd.DataFrame({
            "항목":["식(Poly-3)","R²","Start θ*","Slowdown","Saturation(추정)",
                   "곡선힌지 곡률강조 배율(k)"],
            "값":[eq_str, r2, theta_star, T_slow, T_cap, curve_k]
        })
        summary.to_excel(wr, index=False, sheet_name="Summary")
        pd.DataFrame({"a0":[a], "b1":[b], "c2":[c], "d3":[d]}).to_excel(wr, index=False, sheet_name="Coefficients")
        pd.DataFrame({
            "Band":["−5~0℃","0~5℃","5~10℃"],
            "Δ1℃ 증가량(MJ/℃)":[avg_m5_0, avg_0_5, avg_5_10]
        }).to_excel(wr, index=False, sheet_name="Band_Average")
        pd.DataFrame({"T(℃)":tgrid,
                      "Δ1℃ 증가량(MJ/℃)":inc,
                      "CI_lo(5%)":inc_lo, "CI_hi(5%)":inc_hi}).to_excel(wr, index=False, sheet_name="Curve")
    buf.seek(0)
    return buf.getvalue()

xlsx_bytes = build_xlsx_bytes()
st.download_button(
    "📥 민감도 요약·곡선 XLSX 다운로드",
    data=xlsx_bytes, file_name="HeatBand_Insight.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

st.caption("섹션 B는 Q=a+b·H+c·H²+d·H³(곡률강조 ×k)로 적합, 섹션 E는 주석 자동 위치조정으로 겹침을 방지합니다.")
