# app.py — HeatBand Insight (요약 UI + 동적 그래프 + XLSX Export)
# 단위: 공급량 Q(MJ), 민감도/증가량 Δ1°C(MJ/℃ = −dQ/dT)

import os, io
from typing import Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
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

def conf_band_y(x_train, y_train, tgrid, m, pf, z=1.645):  # 90% CI
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
    try:
        return f"{int(np.round(float(x))):,}"
    except Exception:
        return str(x)

# ── Loader ─────────────────────────────────────────────
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

@st.cache_data(show_spinner=False)
def load_google_sheet_csv(url: str) -> pd.DataFrame:
    try:
        return pd.read_csv(url)
    except Exception as e:
        st.error(f"구글 시트 연동 오류: {type(e).__name__} — {e}")
        st.stop()

# ── 데이터 입력 (수정됨: 구글 시트 연동) ──────────────────────────────────────────────
st.sidebar.header("① 데이터")

# 구글 시트 URL을 CSV 다운로드 링크로 변환
SHEET_ID = "13HrIz6OytYDykXeXzXJ02I6XbaKin1YaKBoO2kBd6Bs"
GID = "0"
SHEET_CSV_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={GID}"

uploaded = st.sidebar.file_uploader("엑셀 직접 업로드 (선택사항)", type=["xlsx"])
if uploaded is not None:
    raw = read_excel_cached(uploaded)
    st.sidebar.success("업로드된 파일을 사용합니다.")
else:
    # 파일 업로드가 없으면 기본적으로 지정한 구글 시트(일일실적)를 불러옵니다.
    raw = load_google_sheet_csv(SHEET_CSV_URL)
    st.sidebar.info("구글 시트(일일실적) 데이터 연동 완료")

cols = raw.columns.tolist()

# ── ★추가됨: 기온 데이터 유형 선택 ──────────────────────────────────────────────────
# 일일평균기온(원본) vs 월평균기온(월별 집계) 중 선택하여 분석에 사용
st.sidebar.header("② 기온 데이터 유형")
temp_mode = st.sidebar.radio(
    "분석에 사용할 기온 유형",
    options=["일일평균기온", "월평균기온"],
    index=0,
    help=(
        "• 일일평균기온: 원본 일별 데이터 그대로 사용 (분산 큼, 극단 기온 반영)\n"
        "• 월평균기온: 월별로 기온·공급량을 평균 집계 후 사용 (평활화, 계절 패턴 강조)"
    )
)
# ── ★추가됨 끝 ───────────────────────────────────────────────────────────────────

# ③ 컬럼 매핑 — UI 숨김, 자동 매핑 로직만 유지
def _pick(cands, default_idx=0):
    for k in cands:
        for c in cols:
            if k in str(c): return c
    return cols[default_idx] if len(cols) > default_idx else cols[-1]

date_col = _pick(["일자","날짜","date"])
temp_col = _pick(["평균기온","기온","temp"])
q_col    = _pick(["공급량(MJ)","공급량","총","total","MJ"])

df = raw.copy()
df["date"] = pd.to_datetime(df[date_col])
df["year"] = df["date"].dt.year
df["month"]= df["date"].dt.month
df["temp"] = df[temp_col].apply(to_num)
df["Q"]    = df[q_col].apply(to_num)
df = df.dropna(subset=["temp","Q"]).sort_values("date")

# ── ★추가됨: 기온 유형에 따라 분석용 데이터프레임(df_model) 분기 ─────────────────────
# df_model 은 이후 train 생성, 그래프 A·B·산점도 학습 데이터로만 사용
# G섹션(히트맵) 등 원본 일일 데이터를 쓰는 곳은 df / raw 그대로 유지
if temp_mode == "월평균기온":
    # 월별 집계: 기온은 평균, 공급량은 합산
    # → 월 단위 분석이므로 공급량은 월합계가 물리적으로 맞음
    df_monthly = (
        df.groupby(["year", "month"])
        .agg(temp=("temp", "mean"), Q=("Q", "sum"))
        .reset_index()
    )
    # 대표 날짜(월 1일)를 date 컬럼으로 부여 (year/month 컬럼 유지)
    df_monthly["date"] = pd.to_datetime(
        df_monthly[["year","month"]].assign(day=1)
    )
    df_model = df_monthly.sort_values("date").reset_index(drop=True)
    mode_label = "월평균기온 기반 (월합산 공급량)"
else:
    df_model = df.copy()
    mode_label = "일일평균기온 기반 (일별 공급량)"

st.info(f"📊 현재 분석 모드: **{mode_label}** · 분석 행 수: {len(df_model):,}")
# ── ★추가됨 끝 ───────────────────────────────────────────────────────────────────

st.success(f"행 {len(df):,} · 기간 {df['date'].min().date()} ~ {df['date'].max().date()}")

# 학습 연도
st.sidebar.header("④ 학습 연도")
years = sorted(df["year"].unique().tolist())
sel_years = st.sidebar.multiselect("연도 선택", years, default=years)

# ── ★수정됨: train을 df_model 기준으로 생성 (기존: df 기준) ──────────────────────────
train = df_model[df_model["year"].isin(sel_years)].copy()
# ── ★수정됨 끝 ───────────────────────────────────────────────────────────────────

if train.empty:
    st.warning("선택된 연도에 데이터가 없습니다."); st.stop()

# ── 시각 범위 ────────────────────────────────────────────────
T = train["temp"].values
p1, p99 = np.percentile(T, 1), np.percentile(T, 99)
xmin_vis = float(np.floor(min(-5, p1 - 1.5)))
xmax_vis = float(np.ceil(max(25, p99 + 1.5)))

# ── Poly-3 적합(전체) ────────────────────────────────────────
m_all, pf_all, Xp = fit_poly3(train["temp"].values, train["Q"].values)
yhat = m_all.predict(Xp)
r2   = poly3_r2(train["Q"].values, yhat)
a,b,c,d = poly3_coeffs(m_all)
eq_str  = nice_poly_string(a,b,c,d, digits=1)

# 촘촘한 그리드 + 90% CI
tgrid = np.linspace(xmin_vis, xmax_vis, 1201)
ci_lo_90, ci_hi_90, y_pred, sigma2, XtX_inv = conf_band_y(
    train["temp"].values, train["Q"].values, tgrid, m_all, pf_all, z=1.645
)

# ── 도함수(민감도) ───────────────────────────────────────────
J = np.vstack([np.ones_like(tgrid)*0, np.ones_like(tgrid), 2*tgrid, 3*(tgrid**2)]).T
deriv_mean = np.array([d1_at(m_all, t) for t in tgrid])        # dQ/dT
deriv_se   = np.sqrt(np.sum(J @ XtX_inv * J, axis=1) * sigma2)
z90 = 1.645
d_lo = deriv_mean - z90*deriv_se
d_hi = deriv_mean + z90*deriv_se

# ── 부드러운 ReLU로 양수화(0 부근 꺾임 제거) ────────────────
def smooth_relu(x, eps):
    return 0.5 * (x + np.sqrt(x*x + eps*eps))
eps_rel = 0.015 * max(1.0, float(np.nanmax(np.abs(deriv_mean))))

base_inc = smooth_relu(-deriv_mean, eps_rel)   # Raw 증가량
base_lo  = smooth_relu(-d_hi, eps_rel)
base_hi  = smooth_relu(-d_lo, eps_rel)

# ── 저온 완화(시나리오 토글만, 기본 OFF) ─────────────────────
st.sidebar.header("⑤ 시뮬레이션 옵션")
auto_zoom = st.sidebar.toggle("밴드 자동 Y축 줌(곡률 강조)", value=True)
use_cold  = st.sidebar.toggle("저온 완화 시나리오(극저온에서 증가량 둔화)", value=False)

# 고정 파라미터(문서화 목적)
T_COLD_FIXED = -2.0   # ℃
TAU_FIXED    = 1.5    # ℃

def sigmoid(x): return 1/(1+np.exp(-x))
def smoothstep(x, w=1.2, c=0.0): return 0.5 * (1 + np.tanh((x - c) / w))

if use_cold:
    cf_raw = sigmoid((tgrid - T_COLD_FIXED)/TAU_FIXED)
    blend  = smoothstep(tgrid, w=1.2, c=0.0)
    cold_factor = cf_raw*(1.0 - blend) + 1.0*blend
else:
    cold_factor = np.ones_like(tgrid)

inc    = base_inc * cold_factor
inc_lo = base_lo  * cold_factor
inc_hi = base_hi  * cold_factor

# ── 열량 입력(환산) ──────────────────────────────────────────
st.sidebar.header("⑥ 열량(환산 단위)")
calorific = st.sidebar.number_input(
    "열량 (MJ/Nm³)", min_value=30.000, max_value=55.000, value=42.563, step=0.001, format="%.3f"
)
def to_m3_per_deg(mj_per_deg: float, cv: float) -> float:
    if cv is None or cv <= 0:
        return np.nan
    return mj_per_deg / cv

# ── 난방 시작/둔화/포화(참고) ────────────────────────────────
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
T_slow   = float(tgrid[int(np.argmin(deriv_mean))])
max_neg  = float(np.max(inc))
T_cap    = float(tgrid[np.argmax(inc <= 0.02*max_neg)]) if max_neg>0 else np.nan

def fit_hinge_cubic(T: np.ndarray, Q: np.ndarray, theta: float) -> Tuple[float,float,float,float]:
    H = np.clip(theta - T, 0, None)
    X = np.column_stack([np.ones_like(H), H, H**2, H**3])
    beta, *_ = np.linalg.lstsq(X, Q, rcond=None)
    a_c, b_c, c_c, d_c = map(float, beta)
    return a_c, b_c, c_c, d_c
a_c, b_c, c_c, d_c = fit_hinge_cubic(train["temp"].values, train["Q"].values, theta_star)

def qhat_cubic(t: np.ndarray, theta: float, a_c: float, b_c: float, c_c: float, d_c: float, k: float) -> np.ndarray:
    H = np.clip(theta - t, 0, None)
    return a_c + b_c*H + (k*c_c)*(H**2) + (k*d_c)*(H**3)

# ── (A) 상관 그래프 (수정됨: 일일 데이터 시각화를 위해 점 크기 축소) ────────────────────────
st.subheader("🧮 A. 기온–공급량 상관 (Poly-3, 90% CI)")
figA = go.Figure()
figA.add_trace(go.Scatter(
    x=train["temp"], y=train["Q"], mode="markers", name="샘플",
    marker=dict(size=4, opacity=0.5, line=dict(width=0.2), symbol="circle"),
    hovertemplate="T=%{x:.2f}℃<br>Q=%{y:,.0f} MJ<extra></extra>"
))
figA.add_trace(go.Scatter(
    x=np.r_[tgrid, tgrid[::-1]], y=np.r_[ci_hi_90, ci_lo_90[::-1]],
    fill="toself", name="90% CI",
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

# ── (B) 수요곡선 — 힌지 시각화 (수정됨: 일일 데이터 시각화를 위해 점 크기 축소) ───────────────
st.subheader("🧊 B. Heating Start / Slowdown — 수요곡선")
tline = np.linspace(xmin_vis, xmax_vis, 600)
qhat_curve = qhat_cubic(tline, theta_star, a_c, b_c, c_c, d_c, 1.0) # ★수정됨: 2.0에서 1.0으로 변경하여 곡선이 중앙을 관통하도록 보완

figB = go.Figure()
figB.add_trace(go.Scatter(x=df_model["temp"], y=df_model["Q"], mode="markers", name="전체(참고)",
                          marker=dict(size=3, color="lightgray"), opacity=0.3))
figB.add_trace(go.Scatter(x=train["temp"], y=train["Q"], mode="markers", name="학습",
                          marker=dict(size=4), marker_color="orange", opacity=0.6))
figB.add_trace(go.Scatter(
    x=tline, y=qhat_curve, mode="lines", name="힌지(곡선) 적합",
    line=dict(width=3, shape="spline", smoothing=1.1)
))

figB.add_vrect(x0=xmin_vis, x1=theta_star, fillcolor="LightSkyBlue", opacity=0.18, line_width=0, layer="below")
figB.add_annotation(x=(xmin_vis+theta_star)/2, y=1.12, xref="x", yref="paper",
                    text="난방 시작 구간 (Heating Start)", showarrow=False,
                    font=dict(size=12), bgcolor="rgba(255,255,255,0.7)", bordercolor="rgba(0,0,0,0.1)")
figB.add_vrect(x0=xmin_vis, x1=T_slow, fillcolor="LightCoral", opacity=0.14, line_width=0, layer="below")
figB.add_annotation(x=(xmin_vis+T_slow)/2, y=1.12, xref="x", yref="paper",
                    text=f"난방 포화 구간 (총 사용량 증가, 증가폭은 둔화 ≤ {T_slow:.2f}℃)", showarrow=False,
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

figB.update_layout(template="simple_white", font=dict(family=PLOT_FONT, size=14),
                   margin=dict(l=40,r=20,t=60,b=70),
                   xaxis=dict(title="기온(℃)", range=[xmin_vis, xmax_vis]),
                   yaxis=dict(title="공급량(MJ)", tickformat=","),
                   legend=dict(orientation="h", yanchor="top", y=-0.18, x=0.01))
st.plotly_chart(figB, use_container_width=True, config={"displaylogo": False})

# ── (C) 기온별 공급량 변화량 요약 ───────────────────────────
st.subheader("🌡️ C. 기온별 공급량 변화량 요약")

def band_mean(temp_array, apply_cold=True):
    temps = np.array(temp_array, dtype=float)
    base = smooth_relu(-np.array([d1_at(m_all, t) for t in temps]), eps_rel)
    if apply_cold and use_cold:
        cf_raw = 1/(1+np.exp(-(temps - T_COLD_FIXED)/TAU_FIXED))
        blend  = 0.5*(1 + np.tanh((temps - 0.0)/1.2))
        cf = cf_raw*(1.0 - blend) + 1.0*blend
        base = base * cf
    return float(np.mean(base))

total_mean_temp = train["temp"].mean()
avg_total = band_mean([total_mean_temp], apply_cold=True)
avg_total_nm3 = to_m3_per_deg(avg_total, calorific)

band = {"−5~0℃": np.arange(-5, 0.001, 0.1),
        "0~5℃" : np.arange(0, 5.001, 0.1),
        "5~10℃": np.arange(5,10.001,0.1)}

avg_m5_0  = band_mean(band["−5~0℃"], apply_cold=True)
avg_0_5   = band_mean(band["0~5℃"],  apply_cold=True)
avg_5_10  = band_mean(band["5~10℃"], apply_cold=True)

avg_m5_0_nm3 = to_m3_per_deg(avg_m5_0, calorific)
avg_0_5_nm3  = to_m3_per_deg(avg_0_5,  calorific)
avg_5_10_nm3 = to_m3_per_deg(avg_5_10, calorific)

st.markdown(
f"""
**Polynomial Regression (degree 3)** **{eq_str}**

- **Supply ↑ per −1°C (Total Avg @ {total_mean_temp:.1f}℃)** : **{fmt_int(avg_total)} MJ/℃, {fmt_int(avg_total_nm3)} Nm³/℃** (단위열량 {calorific:.3f} MJ/Nm³ 적용)
- **Supply ↑ per −1°C (−5~0℃)** : **{fmt_int(avg_m5_0)} MJ/℃, {fmt_int(avg_m5_0_nm3)} Nm³/℃** (단위열량 {calorific:.3f} MJ/Nm³ 적용)
- **Supply ↑ per −1°C (0~5℃)**  : **{fmt_int(avg_0_5)} MJ/℃, {fmt_int(avg_0_5_nm3)} Nm³/℃** (단위열량 {calorific:.3f} MJ/Nm³ 적용)
- **Supply ↑ per −1°C (5~10℃)** : **{fmt_int(avg_5_10)} MJ/℃, {fmt_int(avg_5_10_nm3)} Nm³/℃** (단위열량 {calorific:.3f} MJ/Nm³ 적용)
"""
)

# ── (D) 구간별 동적 그래프 ───────────────────────────────────
st.subheader("📈 D. 기온 구간별 동적 그래프 (−dQ/dT = 1℃ 하락 시 증가량, 90% CI)")
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
        fill="toself", name="90% CI (±)", line=dict(color="rgba(0,0,0,0)"),
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

# ── (E) 전체 곡선 ───────────────────────────────────────────
st.subheader("🧭 E. Refined Gas Supply Rate of Change (Dynamic)")
figE = go.Figure()
figE.add_trace(go.Scatter(
    x=tgrid, y=inc, mode="lines", name="증가량(MJ/℃)",
    line=dict(width=3, shape="spline", smoothing=1.2),
    hovertemplate="T=%{x:.2f}℃<br>증가량=%{y:,.0f} MJ/℃<extra></extra>"
))
figE.add_vrect(x0=xmin_vis, x1=T_slow, fillcolor="LightCoral", opacity=0.12, line_width=0, layer="below")
figE.add_vrect(x0=T_slow, x1=theta_star, fillcolor="LightSkyBlue", opacity=0.12, line_width=0, layer="below")

def top_note(x, text, y=1.12):
    figE.add_annotation(x=x, y=y, xref="x", yref="paper", showarrow=False, text=text,
                        font=dict(size=12), bgcolor="rgba(255,255,255,0.75)",
                        bordercolor="rgba(0,0,0,0.12)", borderwidth=1)
                        
top_note((xmin_vis+T_slow)/2,  f"난방 포화 구간 (증가폭 둔화, ≤ {T_slow:.2f}℃)")
top_note((T_slow+theta_star)/2, f"난방 시작 구간 ({T_slow:.2f}~{theta_star:.2f}℃)")

figE.add_vline(x=theta_star, line_dash="dash", line_color="black")
figE.add_annotation(x=theta_star, y=1.14, xref="x", yref="paper",
                    text=f"Start θ* {theta_star:.2f}℃", showarrow=False,
                    font=dict(size=12), bgcolor="rgba(255,255,255,0.75)",
                    bordercolor="rgba(0,0,0,0.12)", borderwidth=1)

figE.update_layout(
    template="simple_white", font=dict(family=PLOT_FONT, size=14),
    margin=dict(l=40,r=20,t=40,b=80),
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
                  "T_cold(℃)","τ(℃)","시나리오 사용여부","열량(MJ/Nm³)"],
            "값":[eq_str, r2, theta_star, T_slow, T_cap, T_COLD_FIXED, TAU_FIXED, use_cold, calorific]
        })
        summary.to_excel(wr, index=False, sheet_name="Summary")
        pd.DataFrame({"a0":[a], "b1":[b], "c2":[c], "d3":[d]}).to_excel(wr, index=False, sheet_name="Coefficients")
        pd.DataFrame({
            "Band":["−5~0℃","0~5℃","5~10℃"],
            "Δ1℃ 증가량(MJ/℃)":[float(avg_m5_0), float(avg_0_5), float(avg_5_10)],
            "Δ1℃ 증가량(Nm³/℃)":[float(avg_m5_0_nm3), float(avg_0_5_nm3), float(avg_5_10_nm3)],
            "열량(MJ/Nm³)":[float(calorific), float(calorific), float(calorific)]
        }).to_excel(wr, index=False, sheet_name="Band_Average")
        pd.DataFrame({"T(℃)":tgrid,
                      "Δ1℃ 증가량(MJ/℃)":inc,
                      "CI_lo(90%)":inc_lo, "CI_hi(90%)":inc_hi}).to_excel(wr, index=False, sheet_name="Curve")
    buf.seek(0)
    return buf.getvalue()

xlsx_bytes = build_xlsx_bytes()
st.download_button(
    "📥 민감도 요약·곡선 XLSX 다운로드",
    data=xlsx_bytes, file_name="HeatBand_Insight.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

st.caption("본 화면의 기본 수치는 Raw(Poly-3 직접 민감도)이며, '저온 완화'는 별도 시나리오로만 적용됩니다.")

# ============================================================
# G. 기온분석 — 선택 월 히트맵(일자×연도) (수정됨: 상단 구글시트 데이터 공용사용)
# ============================================================
PLOT_FONT = "Noto Sans KR"

st.subheader("🧊 G. 기온분석 — 일일 평균기온 히트맵")

# 파일 업로드(옵션)가 없으면 위에서 읽어들인 구글시트(raw)를 그대로 사용
u = st.file_uploader("히트맵용 파일 별도 업로드 (선택사항)", type=["xlsx"], key="g_daily_uploader")
if u is not None:
    dt_raw = pd.read_excel(u)
else:
    dt_raw = raw.copy() # 상단 구글 시트 데이터를 히트맵에도 동일하게 적용

if dt_raw.empty:
    st.warning("데이터가 없습니다.")
    st.stop()

def _guess_g(df: pd.DataFrame, keys, default=None):
    for k in keys:
        for c in df.columns:
            if k in str(c):
                return c
    return default

date_c  = _guess_g(dt_raw, ["일자","날짜","Date","date"], dt_raw.columns[0])
tmean_c = _guess_g(dt_raw, ["평균기온","기온","Tmean","avg"], dt_raw.columns[1])

dt = dt_raw.copy()
dt["date"]  = pd.to_datetime(dt[date_c], errors="coerce")
dt["tmean"] = pd.to_numeric(dt[tmean_c], errors="coerce")
dt = dt.dropna(subset=["date","tmean"]).sort_values("date").reset_index(drop=True)
dt["year"]  = dt["date"].dt.year
dt["month"] = dt["date"].dt.month
dt["day"]   = dt["date"].dt.day

years_all = sorted(dt["year"].unique().tolist())
y_min, y_max = int(min(years_all)), int(max(years_all))
months_all = list(range(1,13))
month_names = {1:"January",2:"February",3:"March",4:"April",5:"May",6:"June",
               7:"July",8:"August",9:"September",10:"October",11:"November",12:"December"}

def get_current_selection(dt: pd.DataFrame):
    ys = sorted(dt["year"].unique().tolist())
    y_min_l, y_max_l = int(min(ys)), int(max(ys))
    sel_range = st.session_state.get("g_year_range", (y_min_l, y_max_l))
    sel_month = st.session_state.get("g_month", int(dt["month"].iloc[-1]))
    sel_years = [y for y in ys if sel_range[0] <= y <= sel_range[1]]
    dsel = dt[(dt["year"].isin(sel_years)) & (dt["month"] == sel_month)].copy()
    if dsel.empty:
        return dsel, sel_years, sel_month, 0
    if "day" not in dsel.columns:
        dsel["day"] = dsel["date"].dt.day
    try:
        last_day = int(np.nanmax(dsel["day"].to_numpy()))
    except Exception:
        last_day = int(dsel["day"].max()) if not dsel.empty else 0
    return dsel, sel_years, sel_month, last_day

c1, c2 = st.columns([2,1])
with c1:
    st.slider("연도 범위", min_value=y_min, max_value=y_max,
              value=(y_min, y_max), step=1, key="g_year_range")
with c2:
    default_month = int(dt["month"].iloc[-1])
    st.selectbox("월 선택", options=months_all,
                 index=months_all.index(default_month),
                 format_func=lambda m: f"{m:02d} ({month_names[m]})",
                 key="g_month")

dsel, sel_years, sel_month, last_day = get_current_selection(dt)
if dsel.empty or last_day == 0:
    st.info("선택한 연·월에 데이터가 없습니다.")
    st.stop()

pivot = (dsel.pivot_table(index="day", columns="year", values="tmean", aggfunc="mean")
                .reindex(range(1, last_day+1)))

avg_row = pivot.mean(axis=0, skipna=True)
pivot_with_avg = pd.concat([pivot, pd.DataFrame([avg_row], index=["평균"])])

# y축 라벨: MM-DD + 마지막 "평균"
y_labels = [f"{sel_month:02d}-{int(d):02d}" for d in pivot.index]
y_labels.append("평균")
pivot_with_avg.index = y_labels

# x축 연도 문자열
pivot_with_avg.columns = [str(c) for c in pivot_with_avg.columns]

# 행 수 기반 높이 (참고 코드와 동일 방식)
n_rows = len(pivot_with_avg)
height = max(800, n_rows * 28 + 120)

# px.imshow — 참고 코드와 동일한 렌더링 방식
heat = px.imshow(
    pivot_with_avg,
    labels=dict(x="연도", y="일", color="평균기온(℃)"),
    x=pivot_with_avg.columns.tolist(),
    y=pivot_with_avg.index.tolist(),
    color_continuous_scale="RdBu_r",
    aspect="auto",
    text_auto=".1f",
)

heat.update_layout(
    height=height,
    margin=dict(l=40, r=40, t=80, b=40),
    title=f"{sel_month:02d}월 일일 평균기온 히트맵 (선택연도 {len(pivot_with_avg.columns)}개)",
    font=dict(family=PLOT_FONT, size=13),
    yaxis=dict(
        type="category",
        tickmode="linear",
        dtick=1,
        autorange="reversed",
    ),
    xaxis=dict(
        type="category",
        tickmode="linear",
        dtick=1,
        side="top",       # x축(연도) 상단
    ),
)
heat.update_traces(textfont=dict(size=14))

st.plotly_chart(heat, use_container_width=True, config={"displaylogo": False})
