# app.py — Heating-curve-band | HeatBand Insight
# 단위: 공급량(MJ), 변화율 dQ/dT(MJ/℃)
# 기능:
#  - 리포지토리의 '실적.xlsx' 자동 로드(없으면 업로드)
#  - 컬럼 매핑, 대상 토글(전체/가정용), 학습 연도 선택
#  - [A0 추가] Poly-3 산점도+R2+95%CI 밴드+식
#  - 힌지모형 θ*, Poly-3 dQ/dT 슬로다운
#  - Δ1℃(= 기온 1℃ 하락 시 증가량) 표: 표준기온 0/5/10℃ & 대표기온(월 중앙값)
#  - [C 강화] 선택 월 Poly-3 식 명시 / 0~5℃ 구간 Δ1℃ 증가량(정수 온도)
#  - 월×표준기온 히트맵(값+표본수, robust 색상, 수동 스케일 조정)
#  - 표/툴팁 천단위 콤마

import os
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

# ── Korean font (Plotly는 시스템 폰트 사용, Matplotlib 백업 등록)
FONT_PATH = "NanumGothic-Regular.ttf"
if os.path.exists(FONT_PATH):
    try:
        fm.fontManager.addfont(FONT_PATH)
    except Exception:
        pass
PLOT_FONT = "NanumGothic, Arial, Noto Sans KR, sans-serif"

st.title("🔥 HeatBand Insight — 난방구간·민감도 분석")
st.caption("단위: 공급량 **MJ**, 변화율 **MJ/℃** · Heating Start(θ*) · Heating Slowdown · Δ1°C Impact")

# ─────────────────────────────────────────────────────────────
# Utils
# ─────────────────────────────────────────────────────────────
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

def poly3_predict(m: LinearRegression, pf: PolynomialFeatures, t: np.ndarray) -> np.ndarray:
    return m.predict(pf.transform(t.reshape(-1,1)))

def poly3_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1.0 - ss_res/ss_tot if ss_tot > 0 else np.nan

def poly3_conf_band(x_train: np.ndarray, y_train: np.ndarray,
                    tgrid: np.ndarray, m: LinearRegression, pf: PolynomialFeatures,
                    alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """예측 평균의 95% 신뢰구간(=회귀선의 CI)."""
    X = pf.transform(x_train.reshape(-1,1))
    yhat = m.predict(X)
    n, p = X.shape
    # 잔차분산
    sigma2 = np.sum((y_train - yhat)**2) / (n - p)
    # (X'X)^-1
    XtX_inv = np.linalg.inv(X.T @ X)
    Tg = pf.transform(tgrid.reshape(-1,1))
    se = np.sqrt(np.sum(Tg @ XtX_inv * Tg, axis=1) * sigma2)
    # 정규근사로 1.96 사용
    z = 1.96
    ypred = m.predict(Tg)
    upper = ypred + z*se
    lower = ypred - z*se
    return lower, upper

def poly3_d1_at(m: LinearRegression, pf: PolynomialFeatures, t: float) -> float:
    b0, b1, b2, b3 = poly3_coeffs(m)
    return b1 + 2*b2*t + 3*b3*(t**2)

def hinge_base_temp(T: np.ndarray, Q: np.ndarray,
                    grid_min: float=0.0, grid_max: float=20.0, step: float=0.1) -> Tuple[float, float, float]:
    """Q ≈ a + b * max(0, θ - T). θ∈[min,max]에서 RMSE 최소 θ* 탐색."""
    thetas = np.arange(grid_min, grid_max + 1e-9, step)
    best_th, best_a, best_b, best_rmse = np.nan, np.nan, np.nan, np.inf
    T = T.reshape(-1); Q = Q.reshape(-1)
    X1 = np.ones_like(T)
    for th in thetas:
        H = np.clip(th - T, 0, None)
        X = np.column_stack([X1, H])
        beta, *_ = np.linalg.lstsq(X, Q, rcond=None)  # a,b
        pred = X @ beta
        rmse = np.sqrt(np.mean((Q - pred)**2))
        if rmse < best_rmse:
            best_rmse = rmse
            best_th, best_a, best_b = th, float(beta[0]), float(beta[1])
    return best_th, best_a, best_b

@st.cache_data
def load_excel(path_or_buf) -> pd.DataFrame:
    import openpyxl
    try:
        return pd.read_excel(path_or_buf, sheet_name="data")
    except Exception:
        xls = pd.ExcelFile(path_or_buf)
        return pd.read_excel(xls, sheet_name=xls.sheet_names[0])

def nice_poly_string_abcd(a,b,c,d, digits=3):
    def term(v, s, sign_first=True):
        if abs(v) < 1e-12: return ""
        sign = " + " if v >= 0 else " - "
        mag = abs(v)
        return (("" if sign_first and v>=0 else "- ") if sign_first and v<0 else ("" if sign_first else sign)) + f"{mag:,.{digits}f}{s}"
    s = f"y = {a:,.{digits}f}"
    s += term(b, "·T", False)
    s += term(c, "·T²")
    s += term(d, "·T³")
    return s

def fmt_int(x):
    try:
        return f"{int(np.round(float(x))):,}"
    except Exception:
        return str(x)

def df_commas(df, except_cols=None):
    """숫자열을 천단위 콤마 문자열로 변환(표시용)."""
    except_cols = set(except_cols or [])
    out = df.copy()
    for c in out.columns:
        if c in except_cols:
            continue
        if pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].apply(fmt_int)
    return out

# ─────────────────────────────────────────────────────────────
# Plot helpers
# ─────────────────────────────────────────────────────────────
def make_start_figure(df_all, df_train, theta_star, a_hat, b_hat, xmin_vis, xmax_vis, y_title="공급량(MJ)") -> go.Figure:
    tline = np.linspace(xmin_vis, xmax_vis, 320)
    H = np.clip(theta_star - tline, 0, None)
    qhat = a_hat + b_hat*H
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_all["temp"], y=df_all["Q"], mode="markers", name="전체 실측(참고)",
                             marker=dict(size=7, color="lightgray"), opacity=0.45,
                             hovertemplate="기온: %{x:.2f}℃<br>공급량: %{y:,.0f} MJ<extra></extra>"))
    fig.add_trace(go.Scatter(x=df_train["temp"], y=df_train["Q"], mode="markers", name="학습 실측",
                             marker=dict(size=8), opacity=0.95,
                             hovertemplate="기온: %{x:.2f}℃<br>공급량: %{y:,.0f} MJ<extra></extra>"))
    fig.add_trace(go.Scatter(x=tline, y=qhat, mode="lines", name="힌지 적합(훈련)",
                             hovertemplate="기온: %{x:.2f}℃<br>예측: %{y:,.0f} MJ<extra></extra>"))
    fig.add_vline(x=theta_star, line_dash="dash",
                  annotation_text=f"θ* = {theta_star:.2f}℃", annotation_position="top right")
    fig.add_vrect(x0=xmin_vis, x1=theta_star, fillcolor="LightSkyBlue", opacity=0.18, line_width=0,
                  annotation_text="Heating Start Zone", annotation_position="top left")
    fig.update_layout(template="simple_white", font=dict(family=PLOT_FONT, size=14),
                      margin=dict(l=40,r=20,t=50,b=40), hovermode="x unified",
                      xaxis=dict(title="기온(℃)", range=[xmin_vis, xmax_vis]),
                      yaxis=dict(title=y_title, tickformat=","), title="힌지 적합과 Heating Start Zone (학습 연도 기준)")
    return fig

def make_derivative_figure(tgrid, d1, theta_star, T_slow, xmin_vis, xmax_vis, y_title="변화율 dQ/dT (MJ/℃)") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=tgrid, y=d1, mode="lines", name="dQ/dT (MJ/℃)",
                             hovertemplate="기온: %{x:.2f}℃<br>dQ/dT: %{y:,.0f} MJ/℃<extra></extra>"))
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
                      yaxis=dict(title=y_title, tickformat=","), title="Rate of Change vs Temperature — HeatBand")
    return fig

# ─────────────────────────────────────────────────────────────
# Data in
# ─────────────────────────────────────────────────────────────
st.sidebar.header("① 데이터")
repo_file = "실적.xlsx"
uploaded = st.sidebar.file_uploader("엑셀(.xlsx) 업로드 (없으면 리포지토리 파일 사용)", type=["xlsx"])
if uploaded is not None:
    raw = load_excel(uploaded)
elif os.path.exists(repo_file):
    st.sidebar.info("리포지토리의 '실적.xlsx' 자동 사용 중")
    raw = load_excel(repo_file)
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

# ─────────────────────────────────────────────────────────────
# Target & years
# ─────────────────────────────────────────────────────────────
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

# ─────────────────────────────────────────────────────────────
# Options
# ─────────────────────────────────────────────────────────────
st.sidebar.header("⑤ 분석 옵션")
th_min  = st.sidebar.number_input("θ* 탐색 최소(℃)", value=0.0, step=0.5)
th_max  = st.sidebar.number_input("θ* 탐색 최대(℃) (≤20 권장)", value=20.0, step=0.5)
th_step = st.sidebar.number_input("θ* 탐색 간격", value=0.1, step=0.1)

# 표시 범위: 학습 데이터 1~99p ±1.5℃, 상한 25℃
T_train = df_train["temp"].values
p1, p99 = np.percentile(T_train, 1), np.percentile(T_train, 99)
pad = 1.5
auto_min = float(np.floor(p1 - pad))
auto_max = float(np.ceil(min(25.0, p99 + pad)))

st.sidebar.markdown("**표시 온도 범위(℃)**")
mode = st.sidebar.radio("범위 모드", ["자동(권장)", "수동"], index=0, horizontal=True)
if mode == "수동":
    xmin_vis, xmax_vis = st.sidebar.slider(
        "x축 범위(℃)",
        min_value=float(np.floor(df_all["temp"].min()-5)),
        max_value=float(np.ceil(max(25.0, df_all["temp"].max()+5))),
        value=(auto_min, auto_max), step=0.5
    )
else:
    xmin_vis, xmax_vis = auto_min, auto_max

# ─────────────────────────────────────────────────────────────
# A0. 기온–공급량 상관(Poly-3, R², 95% CI)  ← [신규 추가]
# ─────────────────────────────────────────────────────────────
st.subheader(f"A0. 기온–공급량 상관(Poly-3) — 대상: {target_choice}")
x_tr = df_train["temp"].values
y_tr = df_train[target_col].values
m_poly_all, pf_all, Xp_all = fit_poly3(x_tr, y_tr)
yhat_tr = m_poly_all.predict(Xp_all)
r2 = poly3_r2(y_tr, yhat_tr)

tgrid0 = np.linspace(xmin_vis, xmax_vis, 400)
y_pred0 = poly3_predict(m_poly_all, pf_all, tgrid0)
ci_lo, ci_hi = poly3_conf_band(x_tr, y_tr, tgrid0, m_poly_all, pf_all, alpha=0.05)

a,b,c,d = poly3_coeffs(m_poly_all)
eq_str = nice_poly_string_abcd(a,b,c,d, digits=1)  # 화면 라벨은 소수1자리로 간결

fig_corr = go.Figure()
fig_corr.add_trace(go.Scatter(x=df_train["temp"], y=df_train[target_col],
                              mode="markers", name="학습 샘플",
                              marker=dict(size=8),
                              hovertemplate="T=%{x:.2f}℃<br>Q=%{y:,.0f} MJ<extra></extra>"))

# 95% 신뢰구간 밴드
fig_corr.add_traces([
    go.Scatter(x=np.r_[tgrid0, tgrid0[::-1]],
               y=np.r_[ci_hi, ci_lo[::-1]],
               fill="toself", name="95% 신뢰구간",
               line=dict(color="rgba(255,165,0,0)"),
               fillcolor="rgba(255,165,0,0.25)",
               hoverinfo="skip")
])

# 회귀선
fig_corr.add_trace(go.Scatter(x=tgrid0, y=y_pred0, mode="lines", name="Poly-3",
                              line=dict(width=3),
                              hovertemplate="T=%{x:.2f}℃<br>예측=%{y:,.0f} MJ<extra></extra>"))

fig_corr.update_layout(template="simple_white",
                       font=dict(family=PLOT_FONT, size=14),
                       margin=dict(l=40,r=20,t=50,b=40),
                       xaxis=dict(title="기온(℃)", range=[xmin_vis, xmax_vis]),
                       yaxis=dict(title="공급량 (MJ)", tickformat=","),
                       legend=dict(bgcolor="rgba(255,255,255,0.6)"),
                       title=f"기온–공급량 상관(Train, R²={r2:.3f})")
fig_corr.add_annotation(xref="paper", yref="paper", x=0.01, y=0.02,
                        text=eq_str, showarrow=False,
                        bgcolor="rgba(255,255,255,0.85)", bordercolor="black",
                        borderwidth=1, font=dict(size=12))
st.plotly_chart(fig_corr, use_container_width=True, config={"displaylogo": False})

# ─────────────────────────────────────────────────────────────
# A. Heating Start (θ*)
# ─────────────────────────────────────────────────────────────
st.subheader(f"A. Heating Start Zone — 베이스온도(θ*) · 대상: {target_choice}")
theta_star, a_hat, b_hat = hinge_base_temp(df_train["temp"].values, df_train[target_col].values,
                                           th_min, th_max, th_step)
st.metric("베이스온도 θ*", f"{theta_star:.2f} ℃")

df_all_plot   = df_all[["temp", target_col]].rename(columns={target_col:"Q"})
df_train_plot = df_train[["temp", target_col]].rename(columns={target_col:"Q"})
fig_start = make_start_figure(df_all_plot, df_train_plot, theta_star, a_hat, b_hat,
                              xmin_vis, xmax_vis, y_title="공급량(MJ)")
st.plotly_chart(fig_start, use_container_width=True, config={"displaylogo": False})

# ─────────────────────────────────────────────────────────────
# B. Slowdown & dQ/dT (Poly-3)
# ─────────────────────────────────────────────────────────────
st.subheader("B. Heating Slowdown Zone & dQ/dT (Poly-3)")
m_poly, pf_poly, _ = fit_poly3(df_train["temp"].values, df_train[target_col].values)
tgrid = np.linspace(xmin_vis, xmax_vis, 600)
d1 = np.array([poly3_d1_at(m_poly, pf_poly, t) for t in tgrid])
T_slow = float(tgrid[int(np.argmin(d1))])
st.metric("Slowdown 경계 T_slow", f"{T_slow:.2f} ℃")
fig_d1 = make_derivative_figure(tgrid, d1, theta_star, T_slow, xmin_vis, xmax_vis,
                                y_title="변화율 dQ/dT (MJ/℃)")
st.plotly_chart(fig_d1, use_container_width=True, config={"displaylogo": False})

# ─────────────────────────────────────────────────────────────
# C. Δ1℃: 표준기온 & 대표기온 + Poly-3 계수/식 (강화)
# ─────────────────────────────────────────────────────────────
st.subheader("C. Δ1°C Impact — 동절기 같은 월 & 표준기온(0/5/10℃) (Poly-3)")
winter_months = st.multiselect("동절기 월", [12,1,2,3,11,4], default=[12,1,2,3], key="winter_sel")

# 추가: 식을 보고 싶은 '월 선택'
sel_month_for_equation = st.multiselect("식/세부표를 보고 싶은 월(선택)", [1,2,3,4,5,6,7,8,9,10,11,12], default=winter_months, key="eq_months")

rows_std = []; rows_med = []; poly_rows = []; eq_rows = []; inc05_rows = []
for m in sorted(set(winter_months)):
    dm = df_train[df_train["month"] == m]
    if len(dm) < 6:
        continue
    Tm, Qm = dm["temp"].values, dm[target_col].values
    model, pf, _ = fit_poly3(Tm, Qm)
    a0,b1,c2,d3 = poly3_coeffs(model)
    poly_rows.append({"월": m, "식": nice_poly_string_abcd(a0,b1,c2,d3, digits=3),
                      "β0": a0, "β1": b1, "β2": c2, "β3": d3, "표본수": len(dm)})

    # 선택 월에 대해 0~5℃ 구간 Δ1℃
    if m in sel_month_for_equation:
        for t0 in [0,1,2,3,4,5]:
            dqdT = b1 + 2*c2*t0 + 3*d3*(t0**2)
            inc05_rows.append({"월": m, "T(℃)": t0, "Δ1℃ 증가량(MJ)": -dqdT})

    for t0 in [0.0, 5.0, 10.0]:
        dqdT = b1 + 2*c2*t0 + 3*d3*(t0**2)
        rows_std.append({
            "월": m, "표준기온(℃)": t0,
            "Δ1℃ 증가량(MJ)": -dqdT,        # 1℃ 하락 시 증가량
            "dQ/dT(MJ/℃)": dqdT,            # 참고
            "난방구간?": "예" if t0 <= theta_star else "아니오",
            "표본수": len(dm)
        })

    Trep = float(np.median(Tm))
    dqdT_med = b1 + 2*c2*Trep + 3*d3*(Trep**2)
    rows_med.append({"월": m, "대표기온(℃)": round(Trep,2),
                     "Δ1℃ 증가량(MJ)": -dqdT_med, "dQ/dT(MJ/℃)": dqdT_med, "표본수": len(dm)})

# Poly-3 계수/식
if poly_rows:
    st.markdown("**월별 3차 다항식(학습 연도, 대상: "+target_choice+")**")
    pdf = pd.DataFrame(poly_rows).sort_values("월").set_index("월")
    st.dataframe(df_commas(pdf[["식","β0","β1","β2","β3","표본수"]], except_cols=["식"]))

# 선택 월의 0~5℃ 구간 Δ1℃
if inc05_rows:
    inc05 = pd.DataFrame(inc05_rows)
    # 행: 월, 열: T(0~5), 값: Δ1℃
    inc_piv = inc05.pivot(index="월", columns="T(℃)", values="Δ1℃ 증가량(MJ)").sort_index()
    st.markdown("**0℃~5℃ 구간: 1℃ 하락 시 증가량 [MJ]**")
    st.dataframe(df_commas(inc_piv.reset_index()).set_index("월"))
    st.download_button("0~5℃ Δ1℃ CSV 다운로드",
                       data=inc_piv.reset_index().to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"delta1c_0to5_{target_col}.csv", mime="text/csv")

# 표준기온: Δ1℃ 증가량(메인)
if rows_std:
    std_df = pd.DataFrame(rows_std)
    pivot_inc = std_df.pivot_table(index="월", columns="표준기온(℃)",
                                   values="Δ1℃ 증가량(MJ)", aggfunc="mean").sort_index()
    st.markdown("**표준기온 Δ1℃ 증가량(= 기온 1℃ 하락 시 공급 증가) [단위: MJ]**")
    st.dataframe(df_commas(pivot_inc.reset_index()).set_index("월"))
    st.download_button("표준기온 Δ1℃ CSV 다운로드",
                       data=pivot_inc.reset_index().to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"delta1c_standard_{target_col}.csv", mime="text/csv")

    with st.expander("상세 해석 & 원자료(dQ/dT) 보기"):
        pivot_dq = std_df.pivot_table(index="월", columns="표준기온(℃)",
                                      values="dQ/dT(MJ/℃)", aggfunc="mean").sort_index()
        st.markdown("- `dQ/dT(MJ/℃)`는 **기온 상승 시** 변화율(부호 포함).")
        st.markdown("- **Δ1℃ 증가량 = −dQ/dT** → 실제 운영 해석은 위 메인 표(증가량)를 사용.")
        st.dataframe(df_commas(pivot_dq.reset_index()).set_index("월"))
        info = std_df.drop_duplicates(subset=["표준기온(℃)"])[["표준기온(℃)"]].copy()
        info["난방구간 여부(θ* 기준)"] = info["표준기온(℃)"].apply(lambda t0: "예(난방)" if t0 <= theta_star else "아니오(비난방/전이)")
        st.table(info.set_index("표준기온(℃)"))

# 대표기온(월 중앙값) 기준 Δ1℃
if rows_med:
    med = pd.DataFrame(rows_med).sort_values("월").set_index("월")
    st.markdown("**동절기 같은 월 — 대표기온(월 중앙값) 기준 Δ1℃ 증가량 [MJ]**")
    st.dataframe(df_commas(med[["대표기온(℃)","Δ1℃ 증가량(MJ)","표본수"]], except_cols=["대표기온(℃)"]))
    st.download_button("동절기(대표기온) Δ1℃ CSV 다운로드",
                       data=med.reset_index().to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"delta1c_winter_median_{target_col}.csv", mime="text/csv")

# ─────────────────────────────────────────────────────────────
# D. 월별 탄력성 히트맵(값+표본수, robust 스케일)
# ─────────────────────────────────────────────────────────────
st.subheader("D. 월별 탄력성 히트맵 — −dQ/dT@표준기온 (MJ/℃)")

heat_rows = []
for m in range(1,13):
    dm = df_train[df_train["month"] == m]
    n = len(dm)
    if n < 6:
        continue
    model, pf, _ = fit_poly3(dm["temp"].values, dm[target_col].values)
    for t0 in [0.0, 5.0, 10.0]:
        val = -poly3_d1_at(model, pf, t0)  # 1℃ 하락 시 증가량
        heat_rows.append({"월": m, "표준기온(℃)": t0, "증가량(MJ/℃)": val, "표본수": n})

if heat_rows:
    H = pd.DataFrame(heat_rows)
    # robust 색상 스케일(5~95 분위, 0 중심 대칭)
    vmin_p = float(np.percentile(H["증가량(MJ/℃)"], 5))
    vmax_p = float(np.percentile(H["증가량(MJ/℃)"], 95))
    vmax = max(abs(vmin_p), abs(vmax_p))
    vmin = -vmax

    piv = H.pivot(index="월", columns="표준기온(℃)", values="증가량(MJ/℃)").sort_index()
    n_piv = H.pivot(index="월", columns="표준기온(℃)", values="표본수").sort_index().astype(int)

    # 셀 텍스트: 값(콤마) + n
    text = np.empty_like(piv.values).astype(object)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            val = piv.values[i, j]
            n   = n_piv.values[i, j]
            text[i, j] = f"{fmt_int(val)}\n(n={n})" if pd.notna(val) else ""

    heat = go.Heatmap(
        z=piv.values,
        x=piv.columns.astype(str),
        y=piv.index.astype(str),
        colorscale="RdBu_r",
        zmin=vmin, zmax=vmax,
        colorbar=dict(title="−dQ/dT (MJ/℃)"),
        text=text,
        hovertemplate="월=%{y} / 표준기온=%{x}℃<br>증가량=%{z:,} MJ/℃<br>%{text}<extra></extra>",
        showscale=True
    )
    fig_hm = go.Figure(data=[heat])
    fig_hm.update_layout(template="simple_white", font=dict(family=PLOT_FONT, size=14),
                         margin=dict(l=40,r=20,t=40,b=40),
                         title="월×기온 탄력성(기온 1℃ 하락 시 증가량) 히트맵")

    with st.expander("색상 스케일 조정(선택)"):
        vmax_user = st.slider("대칭 vmax (MJ/℃)",
                              min_value=1_000_000.0,
                              max_value=max(50_000_000_000.0, float(vmax)),
                              value=float(vmax), step=1_000_000.0)
        fig_hm.update_traces(zmin=-vmax_user, zmax=vmax_user)

    st.plotly_chart(fig_hm, use_container_width=True, config={"displaylogo": False})
    st.download_button("히트맵 데이터 CSV",
                       data=piv.reset_index().to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"elasticity_heatmap_{target_col}.csv", mime="text/csv")

    st.markdown("""
**히트맵 해석(쉽게)**
- 값은 **−dQ/dT(MJ/℃)** = ‘**기온이 1℃ 하락**할 때 늘어나는 공급량(증가량/℃)’.
- 각 셀의 `n=`은 학습에 사용된 **표본 수**. n이 작을수록 변동성↑ → 색이 과장될 수 있음.
- 색이 진할수록 **기온 하락에 더 민감**. 보통 **동절기(12–3월)**와 **낮은 기온(0℃ 인근)**에서 진하게 나타남.
- 표준기온이 **θ***보다 높으면(비난방/전이구간) 값이 작거나 0에 가까운 것이 자연스러움.
""")
else:
    st.info("표본이 부족해 히트맵을 만들 수 없는 달이 있습니다.")
