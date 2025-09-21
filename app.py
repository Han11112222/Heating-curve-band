# app.py — Heating-curve-band | HeatBand Insight (Plotly + NanumGothic)
# - 시각 범위 자동 클리핑(1~99퍼센타일 ± 패딩) + 수동 슬라이더
# - 외삽에 의한 -100℃ 같은 비현실 축 제거
import os
from typing import Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.font_manager as fm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import streamlit as st

st.set_page_config(page_title="HeatBand Insight", layout="wide")

FONT_PATH = "NanumGothic-Regular.ttf"
if os.path.exists(FONT_PATH):
    try: fm.fontManager.addfont(FONT_PATH)
    except Exception: pass
PLOT_FONT = "NanumGothic, Arial, Noto Sans KR, sans-serif"

st.title("🔥 HeatBand Insight — 난방구간·민감도 분석")
st.caption("Heating Start(θ*) · Heating Slowdown · Δ1°C Impact (월별/동절기)")

# ── Utils ─────────────────────────────────────
def to_num(x):
    if isinstance(x, str): x = x.replace(",", "")
    return pd.to_numeric(x, errors="coerce")

def fit_poly3(x: np.ndarray, y: np.ndarray):
    x = x.reshape(-1, 1)
    pf = PolynomialFeatures(degree=3, include_bias=True)
    Xp = pf.fit_transform(x)
    m = LinearRegression().fit(Xp, y)
    return m, pf

def poly3_d1_at(m, pf, t: float) -> float:
    coef = m.coef_; b1 = coef[1] if len(coef)>1 else 0.0
    b2 = coef[2] if len(coef)>2 else 0.0
    b3 = coef[3] if len(coef)>3 else 0.0
    return b1 + 2*b2*t + 3*b3*(t**2)

def hinge_base_temp(T, Q, grid_min=0.0, grid_max=20.0, step=0.1) -> Tuple[float,float,float]:
    thetas = np.arange(grid_min, grid_max + 1e-9, step)
    best_th, best_a, best_b, best_rmse = np.nan, np.nan, np.nan, np.inf
    T = T.reshape(-1); Q = Q.reshape(-1); X1 = np.ones_like(T)
    for th in thetas:
        H = np.clip(th - T, 0, None)
        X = np.column_stack([X1, H])
        beta, *_ = np.linalg.lstsq(X, Q, rcond=None)
        pred = X @ beta
        rmse = np.sqrt(np.mean((Q - pred)**2))
        if rmse < best_rmse:
            best_rmse = rmse; best_th, best_a, best_b = th, float(beta[0]), float(beta[1])
    return best_th, best_a, best_b

def find_slowdown_threshold(T, Q):
    m, pf = fit_poly3(T, Q)
    # tgrid는 가시 범위에서 만들도록 본문에서 주입(★)
    return m, pf

@st.cache_data
def load_excel(path_or_buf) -> pd.DataFrame:
    import openpyxl
    try:    return pd.read_excel(path_or_buf, sheet_name="data")
    except: 
        xls = pd.ExcelFile(path_or_buf)
        return pd.read_excel(xls, sheet_name=xls.sheet_names[0])

def make_start_figure(df, theta_star, a_hat, b_hat, xmin_vis, xmax_vis) -> go.Figure:
    T = df["temp"].values; Q = df["Q"].values
    tline = np.linspace(xmin_vis, xmax_vis, 320)
    H = np.clip(theta_star - tline, 0, None)
    qhat = a_hat + b_hat*H
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=T, y=Q, mode="markers", name="실측",
        hovertemplate="기온: %{x:.2f}℃<br>공급량: %{y:,.0f}㎥<extra></extra>"))
    fig.add_trace(go.Scatter(x=tline, y=qhat, mode="lines", name="힌지 적합",
        hovertemplate="기온: %{x:.2f}℃<br>예측: %{y:,.0f}㎥<extra></extra>"))
    fig.add_vline(x=theta_star, line_dash="dash",
                  annotation_text=f"θ* = {theta_star:.2f}℃", annotation_position="top right")
    fig.add_vrect(x0=xmin_vis, x1=theta_star, fillcolor="LightSkyBlue", opacity=0.18, line_width=0,
                  annotation_text="Heating Start Zone", annotation_position="top left")
    fig.update_layout(template="simple_white", font=dict(family=PLOT_FONT, size=14),
        margin=dict(l=40,r=20,t=50,b=40), hovermode="x unified",
        xaxis=dict(title="기온(℃)", range=[xmin_vis, xmax_vis]),
        yaxis=dict(title="공급량(㎥)", tickformat=","), title="힌지 적합과 Heating Start Zone")
    return fig

def make_derivative_figure(tgrid, d1, theta_star, T_slow, xmin_vis, xmax_vis) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=tgrid, y=d1, mode="lines", name="dQ/dT (㎥/℃)",
        hovertemplate="기온: %{x:.2f}℃<br>dQ/dT: %{y:,.0f} ㎥/℃<extra></extra>"))
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
        yaxis=dict(title="변화율 dQ/dT (㎥/℃)", tickformat=","), 
        title="Rate of Change vs Temperature — HeatBand")
    return fig

# ── Data in ───────────────────────────────────
st.sidebar.header("① 데이터")
repo_file = "실적.xlsx"
uploaded = st.sidebar.file_uploader("엑셀(.xlsx) 업로드 (없으면 리포지토리 파일 사용)", type=["xlsx"])
if uploaded is not None: raw = load_excel(uploaded)
elif os.path.exists(repo_file): st.sidebar.info("리포지토리의 '실적.xlsx' 자동 사용 중"); raw = load_excel(repo_file)
else:
    st.info("엑셀을 업로드하거나 리포지토리에 '실적.xlsx'를 넣어줘."); st.stop()

cols = raw.columns.tolist()
st.sidebar.header("② 컬럼 매핑")
def _pick(cands, default_idx=0):
    for k in cands:
        for c in cols:
            if k in str(c): return c
    return cols[default_idx]
date_col = st.sidebar.selectbox("날짜", cols, index=cols.index(_pick(["날짜","date"])) if _pick(["날짜","date"]) in cols else 0)
temp_col = st.sidebar.selectbox("평균기온(℃)", cols, index=cols.index(_pick(["평균기온","기온","temp"])) if _pick(["평균기온","기온","temp"]) in cols else 1)
q_col    = st.sidebar.selectbox("공급량(㎥)", cols, index=cols.index(_pick(["공급량","M3","m3"])) if _pick(["공급량","M3","m3"]) in cols else 2)

df = raw.copy()
df["date"] = pd.to_datetime(df[date_col]); df["month"] = df["date"].dt.month
df["temp"] = df[temp_col].apply(to_num); df["Q"] = df[q_col].apply(to_num)
df = df.dropna(subset=["temp","Q"]).sort_values("date")

st.success(f"행 {len(df):,} · 기간 {df['date'].min().date()} ~ {df['date'].max().date()}")

# ── Options ───────────────────────────────────
st.sidebar.header("③ 옵션")
win_months = st.sidebar.multiselect("동절기 월", [12,1,2,3,11,4], default=[12,1,2,3])
th_min = st.sidebar.number_input("θ* 탐색 최소(℃)", value=0.0, step=0.5)
th_max = st.sidebar.number_input("θ* 탐색 최대(℃) (≤20 권장)", value=20.0, step=0.5)
th_step = st.sidebar.number_input("θ* 탐색 간격", value=0.1, step=0.1)

# ★ 가시범위 자동 산정: 1~99퍼센타일 ± 패딩
T = df["temp"].values; Qv = df["Q"].values
p1, p99 = np.percentile(T, 1), np.percentile(T, 99)
pad = 1.5
auto_min = float(np.floor(p1 - pad))
auto_max = float(np.ceil(p99 + pad))
# 운영 요구사항: 상한은 20℃ 이내 권장
auto_max = min(auto_max, 25.0)

st.sidebar.markdown("**표시 온도 범위(℃)**")
mode = st.sidebar.radio("범위 모드", ["자동(권장)", "수동"])
if mode == "수동":
    xmin_vis, xmax_vis = st.sidebar.slider(
        "x축 범위(℃)", min_value=float(np.floor(T.min()-5)), max_value=float(np.ceil(max(25.0, T.max()+5))),
        value=(auto_min, auto_max), step=0.5
    )
else:
    xmin_vis, xmax_vis = auto_min, auto_max

# ── A. Heating Start Zone ─────────────────────
st.subheader("A. Heating Start Zone — 베이스온도(θ*)")
theta_star, a_hat, b_hat = hinge_base_temp(T, Qv, th_min, th_max, th_step)
st.metric("베이스온도 θ*", f"{theta_star:.2f} ℃")
fig_start = make_start_figure(df, theta_star, a_hat, b_hat, xmin_vis, xmax_vis)
st.plotly_chart(fig_start, use_container_width=True, config={"displaylogo": False})

# ── B. Slowdown & dQ/dT ───────────────────────
st.subheader("B. Heating Slowdown Zone & dQ/dT (Poly-3)")
m_poly, pf_poly = find_slowdown_threshold(T, Qv)
# dQ/dT는 가시범위에서만 계산(★)
tgrid = np.linspace(xmin_vis, xmax_vis, 600)
d1 = np.array([poly3_d1_at(m_poly, pf_poly, t) for t in tgrid])
T_slow = float(tgrid[int(np.argmin(d1))])  # 가시범위 내 최소점(★)
st.metric("Slowdown 경계 T_slow", f"{T_slow:.2f} ℃")
fig_d1 = make_derivative_figure(tgrid, d1, theta_star, T_slow, xmin_vis, xmax_vis)
st.plotly_chart(fig_d1, use_container_width=True, config={"displaylogo": False})

# ── C. Δ1℃ Impact (winter) ────────────────────
st.subheader("C. 동절기 같은 월 Δ1°C Impact (Poly-3)")
rows = []
for m in sorted(set(win_months)):
    dm = df[df["month"] == m]
    if len(dm) < 6: continue
    Tm, Qm = dm["temp"].values, dm["Q"].values
    model, pf = fit_poly3(Tm, Qm)
    Trep = float(np.median(Tm))
    dqdT = poly3_d1_at(model, pf, Trep)
    rows.append({"월": m, "표본수": len(dm), "대표기온(℃)": round(Trep,2),
                 "dQ/dT(㎥/℃)": round(dqdT,2), "1℃ 하락 시 증가(㎥)": round(-dqdT,2)})
if rows:
    out = pd.DataFrame(rows).sort_values("월").set_index("월")
    st.dataframe(out)
    st.download_button("동절기 민감도 CSV 다운로드",
        data=out.reset_index().to_csv(index=False).encode("utf-8-sig"),
        file_name="winter_delta1c.csv", mime="text/csv")
else:
    st.info("선택한 월의 표본이 부족하면 표가 비어 있을 수 있어.")

with st.expander("해석 가이드"):
    st.markdown("""
- **자동 축 범위**: 실측 온도 1~99퍼센타일에 ±1.5℃ 패딩을 주어 외삽 왜곡을 제거.
- **Heating Start Zone**: 힌지모형 기준온도 θ* 이하 구간.
- **Heating Slowdown Zone**: 가시 범위 내 dQ/dT 최소점 **T_slow**를 경계로 설정.
- **Δ1℃ Impact**: 해당 월 중앙기온에서 1℃ 하락 시 증가량 = −dQ/dT.
""")
