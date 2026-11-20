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
    # ★수정됨: T³ → T² → T → 절편 순 (고차항 우선)
    s = term(d, "·T³", lead=True)
    s += term(c, "·T²")
    s += term(b, "·T")
    s += term(a, "")
    return "y = " + s

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

@st.cache_data(ttl=3600, show_spinner=False)   # ★수정됨: 1시간마다 시트를 다시 읽음
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

if st.sidebar.button("🔄 구글 시트 새로고침", help="시트에 새로 입력한 값(강수량·적설량 등)을 바로 반영합니다. 기본은 1시간마다 자동 갱신됩니다."):
    st.cache_data.clear()
    (getattr(st, "rerun", None) or st.experimental_rerun)()

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

# ── ★추가됨: 습도·최저/최고·계획 열 (없으면 None) ──
def _find(cands):
    for k in cands:
        for c in cols:
            if k in str(c): return c
    return None
hum_col  = _find(["습도", "humid"])
tmin_col = _find(["최저"])
tmax_col = _find(["최고"])
plan_col = _find(["계획 공급량(MJ)", "계획공급량(MJ)"])
snow_col = _find(["적설"])    # L열: 적설량(cm)
rain_col = _find(["강수"])    # M열: 강수량(mm)

df = raw.copy()
df["date"] = pd.to_datetime(df[date_col])
df["year"] = df["date"].dt.year
df["month"]= df["date"].dt.month
df["temp"] = df[temp_col].apply(to_num)
df["Q"]    = df[q_col].apply(to_num)
df["hum"]  = df[hum_col].apply(to_num)  if hum_col  else np.nan   # ★추가됨
df["tmin"] = df[tmin_col].apply(to_num) if tmin_col else np.nan   # ★추가됨
df["tmax"] = df[tmax_col].apply(to_num) if tmax_col else np.nan   # ★추가됨
df["plan"] = df[plan_col].apply(to_num) if plan_col else np.nan   # ★추가됨
df["snow"] = df[snow_col].apply(to_num) if snow_col else np.nan
df["rain"] = df[rain_col].apply(to_num) if rain_col else np.nan
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
DEFAULT_START_YEAR = 2016   # 습도 데이터가 있는 첫 해 (기본 학습 연도 시작)
default_years = [y for y in years if y >= DEFAULT_START_YEAR] or years
sel_years = st.sidebar.multiselect("연도 선택", years, default=default_years)

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
st.plotly_chart(figA, use_container_width=True, config={"displaylogo": False, "scrollZoom": True, "modeBarButtonsToAdd": ["resetScale2d"]})

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
st.plotly_chart(figB, use_container_width=True, config={"displaylogo": False, "scrollZoom": True, "modeBarButtonsToAdd": ["resetScale2d"]})

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
                      xaxis=dict(title="기온(℃)", range=[loT, hiT], fixedrange=True),
                      yaxis=dict(title="Δ1℃ 증가량(MJ/℃)", tickformat=",", fixedrange=False),
                      title=f"Band {label} Response",
                      dragmode="zoom")
    ax.plotly_chart(fig, use_container_width=True, config={
        "displaylogo": False,
        "scrollZoom": True,
        "modeBarButtonsToAdd": ["resetScale2d"],
    })

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


H_SEL_YEARS = list(sel_years)   # G(히트맵)가 sel_years를 덮어쓰므로, 습도 분석용 연도를 미리 보관

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
heat.update_traces(textfont=dict(size=12))

st.plotly_chart(heat, use_container_width=True, config={"displaylogo": False})

# ============================================================
# I. Humidity Lens — 습도와 공급량의 관계 (과거 실적 해석용, 예측에는 사용하지 않음)  ★ 맨 아래로 이동·간소화
# ============================================================
H_FONT = "NanumGothic, Arial, Noto Sans KR, sans-serif"   # G(히트맵)가 PLOT_FONT를 바꾸므로 이 구역은 따로 지정

st.markdown("---")
st.subheader("💧 Humidity Lens — 습도와 공급량의 관계")
st.caption("과거 실적을 해석하기 위한 분석이며 예측에는 사용하지 않습니다. 월~금만 사용하고, 기온이 같은 날끼리 비교한 결과입니다.")

HUM_SEASONS = [("동절기 (12~2월)", [12, 1, 2]),
               ("간절기·봄 (3~4월)", [3, 4]),
               ("간절기·가을 (10~11월)", [10, 11])]
RAIN_MM = 1.0   # 비 온 날 기준(mm). 이보다 적게 온 날(미량)은 비교에서 제외

if hum_col is None or int(df["hum"].notna().sum()) < 100:
    st.info("시트에서 '습도' 열을 찾지 못했거나 습도 데이터가 부족합니다.")
else:
    dd = df[df["year"].isin(H_SEL_YEARS)].dropna(subset=["temp", "Q", "hum"]).copy()
    dd = dd[dd["date"].dt.dayofweek < 5].copy()          # ★ 월~금만 사용

    if len(dd) < 200:
        st.info("선택한 학습 연도에 습도 데이터가 충분하지 않습니다.")
    else:
        st.caption(f"분석에 사용한 데이터: {dd['date'].min().date()} ~ {dd['date'].max().date()} ({len(dd):,}일, 월~금). "
                   "공휴일은 따로 제외하지 않았습니다. 습도가 비어 있는 연도는 자동으로 제외됩니다.")

        # 난방기 = 난방 시작 기온(θ*) 미만인 날
        theta_h, _, _ = hinge_base_temp(dd["temp"].values, dd["Q"].values, 0.0, 20.0, 0.1)

        # 기온 효과 제거: Poly-3 추정 대비 실적 차이(잔차, %) + 연도별 공급 수준 보정
        m_h, pf_h, Xh = fit_poly3(dd["temp"].values, dd["Q"].values)
        dd["Qhat"] = m_h.predict(Xh)
        dd = dd[dd["Qhat"] > 0].copy()
        dd["res_pct"] = (dd["Q"] - dd["Qhat"]) / dd["Qhat"] * 100.0
        yr_level = dd[dd["temp"] < theta_h].groupby("year")["res_pct"].mean()
        dd["res_pct"] = dd["res_pct"] - dd["year"].map(yr_level).fillna(0.0)

        # 난방기 데이터 + 같은 달 평균 대비 습도(%p)
        heat_d = dd[dd["temp"] < theta_h].copy()
        heat_d["hum_rel"] = heat_d["hum"] - heat_d.groupby("month")["hum"].transform("mean")
        heat_d = heat_d[heat_d.groupby("month")["hum"].transform("size") >= 15].copy()

        def ols_fit(Xm, y):
            beta_, *_ = np.linalg.lstsq(Xm, y, rcond=None)
            resid_ = y - Xm @ beta_
            n_, p_ = Xm.shape
            s2_ = float(resid_ @ resid_) / max(1, n_ - p_)
            cov_ = s2_ * np.linalg.pinv(Xm.T @ Xm)
            return beta_, np.sqrt(np.clip(np.diag(cov_), 0.0, None))

        def _fmt(v, f):
            return "-" if (v is None or not np.isfinite(v)) else format(v, f)

        has_pr = bool(rain_col) and bool(snow_col)
        pr = heat_d.dropna(subset=["rain", "snow"]).copy() if has_pr else pd.DataFrame()
        use_pr = len(pr) >= 100
        if use_pr:
            pr["pgroup"] = np.where((pr["snow"] > 0) | (pr["rain"] >= RAIN_MM), "비·눈",
                             np.where(pr["rain"] > 0, "미량", "무강수"))
            pr["grp"] = pr["pgroup"].where(pr["pgroup"] == "비·눈")
            dry_mask = pr["pgroup"] == "무강수"
            pr.loc[dry_mask, "hq"] = pr[dry_mask].groupby("month")["hum"].rank(pct=True, method="first")
            pr.loc[dry_mask & (pr["hq"] <= 1 / 3), "grp"] = "건조"
            pr.loc[dry_mask & (pr["hq"] > 1 / 3) & (pr["hq"] <= 2 / 3), "grp"] = "보통"
            pr.loc[dry_mask & (pr["hq"] > 2 / 3), "grp"] = "습함"
        else:
            st.info("강수량·적설량이 입력된 난방기 일수가 부족해(시트 L·M열) 비·눈 비교는 생략하고 습도만으로 계산했습니다. "
                    "입력이 끝난 뒤 사이드바의 '🔄 구글 시트 새로고침'을 눌러 주세요.")

        GROUPS = ["건조", "보통", "습함", "비·눈"]
        GROUP_DESC = {"건조": "건조 (비·눈 없음, 습도 하위 1/3)", "보통": "보통 (비·눈 없음, 습도 중간 1/3) ← 기준",
                      "습함": "습함 (비·눈 없음, 습도 상위 1/3)", "비·눈": f"비·눈 (강수 {RAIN_MM:g}mm 이상 또는 적설 있음)"}

        def _sig(e, c):
            return np.isfinite(e) and ((e - c > 0) or (e + c < 0))

        # ── 색상(차분한 블루 계열) ─────────────────────────────
        C_NAVY, C_BLUE, C_SKY, C_PALE = "#16427A", "#2F74C0", "#8DBBE8", "#CFE3F7"
        C_ERR, C_TXT, C_GRID = "#6C8FB8", "#35506F", "#E8F0FA"
        SEASON_COLOR = {"동절기 (12~2월)": C_NAVY, "간절기·봄 (3~4월)": "#3E86D1", "간절기·가을 (10~11월)": C_SKY}

        # ── 계산 ① 날씨 유형별: '보통' 날 대비 공급량 차이 (같은 기온 기준) ─────────
        eff_rows = []
        if use_pr:
            for sname, months in HUM_SEASONS:
                s = pr[pr["month"].isin(months) & pr["grp"].notna()]
                if len(s) < 60 or int((s["grp"] == "보통").sum()) < 8:
                    continue
                present = [g for g in GROUPS if g != "보통" and int((s["grp"] == g).sum()) >= 8]
                Xs = np.column_stack([np.ones(len(s)), s["temp"].values] +
                                     [(s["grp"] == g).astype(float).values for g in present])
                b_, se_ = ols_fit(Xs, s["res_pct"].values)
                qmean = float(s["Q"].mean())
                for g in GROUPS:
                    sg = s[s["grp"] == g]
                    if len(sg) == 0:
                        continue
                    if g == "보통":
                        e, c = 0.0, 0.0
                    elif g in present:
                        j = 2 + present.index(g)
                        e, c = float(b_[j]), 1.645 * float(se_[j])
                    else:
                        e, c = np.nan, np.nan
                    eff_rows.append(dict(season=sname, grp=g, n=len(sg), hum=float(sg["hum"].mean()),
                                         temp=float(sg["temp"].mean()), eff=e, ci=c,
                                         mj=e / 100.0 * qmean if np.isfinite(e) else np.nan))

        # ── 계산 ② 계절별 요약: 습도 +10%p당 영향 + 평균습도 ─────────
        sum_rows, sum_eff = [], []
        for sname, months in HUM_SEASONS:
            s_heat = heat_d[heat_d["month"].isin(months)]
            s_all = dd[dd["month"].isin(months)]
            hum_all = float(s_all["hum"].mean()) if len(s_all) else np.nan
            row = {"계절": sname, "난방기 일수": len(s_heat), "평균습도(%)": _fmt(hum_all, ".1f")}
            hum_dry = hum_wet = wet_share = np.nan
            if use_pr:
                s_p = pr[pr["month"].isin(months)]
                wet_m = (s_p["snow"] > 0) | (s_p["rain"] >= RAIN_MM)
                dry_m = (s_p["snow"] <= 0) & (s_p["rain"] <= 0)
                if int(wet_m.sum()) >= 10 and int(dry_m.sum()) >= 10:
                    hum_dry = float(s_p.loc[dry_m, "hum"].mean())
                    hum_wet = float(s_p.loc[wet_m, "hum"].mean())
                    wet_share = float(wet_m.mean() * 100.0)
                base_sub = s_p[dry_m]            # 비·눈이 전혀 없던 날끼리만 비교
                basis = "비·눈 없던 날끼리"
            else:
                base_sub = s_heat
                basis = "전체 난방기"
            row["비·눈 안 온 날 평균습도(%)"] = _fmt(hum_dry, ".1f")
            row["비·눈 온 날 평균습도(%)"] = _fmt(hum_wet, ".1f")
            row["비·눈 온 날 비율(%)"] = _fmt(wet_share, ".0f")
            eff = ci = np.nan
            if len(base_sub) >= 50:
                Xc = np.column_stack([np.ones(len(base_sub)), base_sub["temp"].values, base_sub["temp"].values ** 2,
                                      base_sub["hum_rel"].values / 10.0])
                b_, se_ = ols_fit(Xc, base_sub["res_pct"].values)
                eff, ci = float(b_[-1]), 1.645 * float(se_[-1])
            row[f"습도 +10%p → 공급량(%) · {basis}"] = _fmt(eff, "+.2f")
            row["90% 신뢰구간(%)"] = "-" if not np.isfinite(eff) else f"{eff - ci:+.2f} ~ {eff + ci:+.2f}"
            row["판정"] = "유의미" if _sig(eff, ci) else ("확인 안 됨" if np.isfinite(eff) else "표본 부족")
            sum_rows.append(row)
            sum_eff.append(dict(season=sname, eff=eff, ci=ci, hum=hum_all))

        # ============================================================
        # 표시 1) 습도 +10%p당 공급량 변화 (평균습도 함께 표시) — 가장 먼저
        # ============================================================
        st.markdown("##### 1) 습도가 10%p 높을 때 공급량은 얼마나 달라지나")
        ok_rows = [r for r in sum_eff if np.isfinite(r["eff"])]
        if ok_rows:
            fig_s = go.Figure(go.Bar(
                x=[f"{r['season']}<br><span style='font-size:12px'>평균습도 {r['hum']:.0f}%</span>" for r in ok_rows],
                y=[r["eff"] for r in ok_rows],
                error_y=dict(type="data", array=[r["ci"] for r in ok_rows], visible=True, color=C_ERR, thickness=1.5, width=6),
                marker_color=[C_BLUE if _sig(r["eff"], r["ci"]) else C_PALE for r in ok_rows],
                marker_line_width=0,
                text=[f"{r['eff']:+.2f}%" for r in ok_rows], textposition="outside", textfont=dict(color=C_TXT),
                customdata=[[r["hum"]] for r in ok_rows],
                hovertemplate="%{x}<br>습도 +10%p당 %{y:+.2f}%<extra></extra>"))
            fig_s.add_hline(y=0, line_dash="dash", line_color=C_ERR)
            fig_s.update_layout(template="simple_white", bargap=0.5, font=dict(family=H_FONT, size=13, color=C_TXT),
                                margin=dict(l=40, r=20, t=30, b=40),
                                yaxis=dict(title="습도 +10%p당 공급량 변화(%)", showgrid=True,
                                           gridcolor=C_GRID, zeroline=False))
            st.plotly_chart(fig_s, use_container_width=True, config={"displaylogo": False})
            st.caption("진한 파랑 = 통계적으로 유의미, 연한 파랑 = 확인 안 됨(신뢰구간이 0을 포함). 막대 위아래의 가는 세로선은 90% 신뢰구간(오차 범위)입니다. "
                       "x축 아래 '평균습도'는 그 계절 월~금의 평균습도입니다. "
                       + ("비·눈이 전혀 없던 날끼리 비교한 값입니다." if use_pr else "강수 자료가 없어 전체 난방기 기준입니다."))
        else:
            st.info("계절별 습도 효과를 계산할 표본이 부족합니다.")

        # 월별 평균 습도 (바로 아래)
        hm = df[df["year"].isin(H_SEL_YEARS)].dropna(subset=["hum"]).groupby("month")["hum"].mean().reindex(range(1, 13))
        ANALYZED = {12, 1, 2, 3, 4, 10, 11}
        figM = go.Figure(go.Bar(
            x=[f"{m}월" for m in hm.index], y=hm.values, marker_line_width=0,
            marker_color=[C_BLUE if m in ANALYZED else C_PALE for m in hm.index],
            text=[("" if not np.isfinite(v) else f"{v:.0f}") for v in hm.values], textposition="outside", textfont=dict(color=C_TXT),
            hovertemplate="%{x}<br>평균 습도=%{y:.1f}%<extra></extra>"))
        figM.update_layout(template="simple_white", bargap=0.3, font=dict(family=H_FONT, size=13, color=C_TXT),
                           margin=dict(l=40, r=20, t=20, b=40),
                           yaxis=dict(title="평균습도(%)", range=[0, 100], gridcolor=C_GRID))
        st.markdown("**월별 평균 습도**")
        st.plotly_chart(figM, use_container_width=True, config={"displaylogo": False})
        st.caption("진한 파랑 = 위 분석에 쓰인 달(12~2월, 3~4월, 10~11월), 연한 파랑 = 분석에 쓰이지 않은 달입니다.")

        # ============================================================
        # 표시 2) 비·눈이 온 날은 공급량이 얼마나 다른가 ('보통' 기준 0%, 보통 막대는 그리지 않음)
        # ============================================================
        st.markdown("##### 2) 비·눈이 온 날은 공급량이 얼마나 다른가")
        if not eff_rows:
            if use_pr:
                st.info("계절별로 비교할 표본이 아직 부족합니다. 시트 입력이 더 진행된 뒤 새로고침해 주세요.")
        else:
            for sname, _ in HUM_SEASONS:
                rs = {r["grp"]: r for r in eff_rows if r["season"] == sname}
                if not rs:
                    continue
                parts = []
                for g, label in [("비·눈", "비·눈 온 날"), ("습함", "비·눈 없이 습한 날"), ("건조", "건조한 날")]:
                    r = rs.get(g)
                    if r is None or not np.isfinite(r["eff"]):
                        continue
                    parts.append(f"{label} {r['eff']:+.1f}%" + ("" if _sig(r["eff"], r["ci"]) else " (차이 확인 안 됨)"))
                if parts:
                    st.markdown(f"- **{sname}**: " + " / ".join(parts))
            st.caption("같은 기온에서 '보통' 날(비·눈이 없고 습도가 중간인 날)과 비교한 공급량 차이입니다. '보통' 날이 0%의 기준선이라 막대로는 그리지 않았습니다.")

            SHOW = [g for g in GROUPS if g != "보통"]
            fig_h2 = go.Figure()
            for sname, _ in HUM_SEASONS:
                rr = [r for r in eff_rows if r["season"] == sname and r["grp"] in SHOW and np.isfinite(r["eff"])]
                if not rr:
                    continue
                fig_h2.add_trace(go.Bar(
                    name=sname, x=[r["grp"] for r in rr], y=[r["eff"] for r in rr],
                    error_y=dict(type="data", array=[r["ci"] for r in rr], visible=True, color=C_ERR, thickness=1.5, width=5),
                    marker_color=SEASON_COLOR[sname], marker_line_width=0,
                    text=[f"{r['eff']:+.1f}%" for r in rr], textposition="outside", textfont=dict(color=C_TXT),
                    customdata=[[r["n"]] for r in rr],
                    hovertemplate=sname + "<br>%{x}: %{y:+.2f}% (n=%{customdata[0]}일)<extra></extra>"))
            fig_h2.add_hline(y=0, line_dash="dash", line_color=C_ERR)
            fig_h2.update_layout(template="simple_white", barmode="group", bargap=0.3, font=dict(family=H_FONT, size=13, color=C_TXT),
                                 margin=dict(l=40, r=20, t=30, b=40), legend=dict(orientation="h", y=1.1),
                                 xaxis=dict(categoryorder="array", categoryarray=SHOW),
                                 yaxis=dict(title="'보통' 날 대비 공급량 차이(%)", showgrid=True, gridcolor=C_GRID, zeroline=False))
            st.plotly_chart(fig_h2, use_container_width=True, config={"displaylogo": False})
            st.caption("막대 위아래의 가는 세로선은 90% 신뢰구간(오차 범위)입니다. 선이 0을 포함하면 차이가 확인되지 않은 것입니다. "
                       "'건조·습함'은 비·눈이 없던 날을 같은 달 안에서 습도 순위로 3등분한 것이고, 비·눈 온 날은 따로 구분했습니다.")

            with st.expander("상세 수치 보기 (일수·평균습도·신뢰구간)"):
                t2 = []
                for r in eff_rows:
                    t2.append({
                        "계절": r["season"], "구분": GROUP_DESC[r["grp"]], "일수": r["n"],
                        "평균습도(%)": f"{r['hum']:.1f}", "평균기온(℃)": f"{r['temp']:.1f}",
                        "보통 대비 공급량(%)": "기준" if r["grp"] == "보통" else _fmt(r["eff"], "+.2f"),
                        "90% 신뢰구간(%)": "-" if (r["grp"] == "보통" or not np.isfinite(r["eff"])) else f"{r['eff'] - r['ci']:+.2f} ~ {r['eff'] + r['ci']:+.2f}",
                        "일 환산(MJ/일)": "-" if (r["grp"] == "보통" or not np.isfinite(r["mj"])) else f"{r['mj']:+,.0f}",
                        "판정": "기준" if r["grp"] == "보통" else ("유의미" if _sig(r["eff"], r["ci"]) else ("확인 안 됨" if np.isfinite(r["eff"]) else "표본 부족")),
                    })
                st.dataframe(pd.DataFrame(t2).set_index(["계절", "구분"]), use_container_width=True)
                st.caption("'유의미'는 90% 신뢰구간이 0을 포함하지 않아 우연으로 보기 어려운 차이라는 뜻입니다(차이가 크다는 뜻은 아닙니다).")

        # ============================================================
        # 표시 3) 계절별 요약표
        # ============================================================
        st.markdown("##### 3) 계절별 요약")
        st.dataframe(pd.DataFrame(sum_rows).set_index("계절"), use_container_width=True)
        st.caption("습도 +10%p는 '같은 달 평균보다 습도가 10%p 높은 날'을 뜻합니다. 비·눈이 있었던 날을 빼고(강수 자료가 있을 때) "
                   "기온이 같은 월~금끼리 비교했기 때문에, 비·눈 효과가 섞이지 않은 습도 차이만 남습니다. "
                   "'유의미'는 90% 신뢰구간이 0을 포함하지 않는다는 뜻이며 차이가 크다는 뜻은 아닙니다.")

        with st.expander("연도별 사용 일수 확인 (습도·강수 데이터가 어느 해부터 쓰였는지)"):
            base_yr = df[df["year"].isin(H_SEL_YEARS)].groupby("year").size()
            hum_yr = df[df["year"].isin(H_SEL_YEARS)].dropna(subset=["hum"]).groupby("year").size()
            heat_yr = heat_d.groupby("year").size()
            yt_d = {"공급량·기온 있는 날": base_yr, "습도도 있는 날": hum_yr, "난방기(분석 사용, 월~금)": heat_yr}
            if has_pr:
                yt_d["난방기 중 강수·적설도 있는 날"] = heat_d.dropna(subset=["rain", "snow"]).groupby("year").size()
            yt = pd.DataFrame(yt_d).fillna(0).astype(int)
            yt["연도별 공급 수준(%)"] = yr_level.reindex(yt.index).round(1)
            yt.index.name = "연도"
            st.dataframe(yt, use_container_width=True)
            st.caption("'습도도 있는 날'이 0이면 그 해는 습도 분석에 쓰이지 않습니다. '연도별 공급 수준'은 같은 기온 기준으로 "
                       "전체 평균보다 그 해 공급량이 몇 % 높았는지를 나타내며, 분석에서는 이 차이를 제거했습니다.")

        st.caption(f"방법: 난방기(평균기온 {theta_h:.1f}℃ 미만)만 사용 · 기온 효과는 Poly-3로 제거하고 연도별 공급 수준 차이도 보정한 뒤 남은 차이(%)를 비교했습니다. "
                   "비·눈 온 날은 강수량 1mm 이상이거나 적설이 있는 날이고, 그보다 적게 온 날(미량)은 제외했습니다. "
                   "구름양·일교차·공휴일은 통제하지 못했으므로, 남은 값에는 이들의 영향이 일부 섞일 수 있습니다"
                   "(예: 봄에 습한 날은 포근한 날과 겹쳐 공급량이 낮게 나올 수 있음). 계절마다 부호가 다르면 습도 자체의 효과로 읽지 마세요.")
