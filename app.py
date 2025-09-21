# app.py — Heating-curve-band | HeatBand Insight
# 단위: 공급량(MJ), 변화율 dQ/dT(MJ/℃)
# 추가: 표준기온(0/5/10℃) 병렬 Δ1℃, 용도별(가정용/전체) 토글, 월별 탄력성 히트맵

import os
from typing import Tuple, Optional, Dict, List
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.font_manager as fm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import streamlit as st

st.set_page_config(page_title="HeatBand Insight", layout="wide")

# ── Korean font fallback
FONT_PATH = "NanumGothic-Regular.ttf"
if os.path.exists(FONT_PATH):
    try: fm.fontManager.addfont(FONT_PATH)
    except Exception: pass
PLOT_FONT = "NanumGothic, Arial, Noto Sans KR, sans-serif"

st.title("🔥 HeatBand Insight — 난방구간·민감도 분석")
st.caption("단위: 공급량 **MJ**, 변화율 **MJ/℃** · Heating Start(θ*) · Heating Slowdown · Δ1°C Impact")

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
    coef = m.coef_
    b1 = coef[1] if len(coef)>1 else 0.0
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
        beta, *_ = np.linalg.lstsq(X, Q, rcond=None)  # a,b
        pred = X @ beta
        rmse = np.sqrt(np.mean((Q - pred)**2))
        if rmse < best_rmse:
            best_rmse = rmse; best_th, best_a, best_b = th, float(beta[0]), float(beta[1])
    return best_th, best_a, best_b

@st.cache_data
def load_excel(path_or_buf) -> pd.DataFrame:
    import openpyxl
    try:    return pd.read_excel(path_or_buf, sheet_name="data")
    except:
        xls = pd.ExcelFile(path_or_buf)
        return pd.read_excel(xls, sheet_name=xls.sheet_names[0])

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
q_total_col = st.sidebar.selectbox("전체 공급량(MJ)", cols, index=cols.index(_pick(["공급량","총","total","MJ"])) if _pick(["공급량","총","total","MJ"]) in cols else 2)
# 가정용이 없을 수도 있으니 '없음' 옵션 허용
q_res_col_options = ["(없음)"] + cols
q_res_col = st.sidebar.selectbox("가정용 공급량(MJ) (선택)", q_res_col_options, index=0)

df_all = raw.copy()
df_all["date"] = pd.to_datetime(df_all[date_col])
df_all["year"] = df_all["date"].dt.year
df_all["month"] = df_all["date"].dt.month
df_all["temp"] = df_all[temp_col].apply(to_num)
df_all["Q_total"] = df_all[q_total_col].apply(to_num)
if q_res_col != "(없음)":
    df_all["Q_res"] = df_all[q_res_col].apply(to_num)
df_all = df_all.dropna(subset=["temp","Q_total"]).sort_values("date")

st.success(f"전체 행 {len(df_all):,} · 기간 {df_all['date'].min().date()} ~ {df_all['date'].max().date()}")

# ── Target toggle ─────────────────────────────
st.sidebar.header("③ 분석 대상")
targets = ["전체(MJ)"]
if "Q_res" in df_all.columns: targets.append("가정용(MJ)")
target_choice = st.sidebar.radio("대상 선택", targets, horizontal=True)
target_col = "Q_total" if target_choice.startswith("전체") else "Q_res"

# ── Train year selection ──────────────────────
st.sidebar.header("④ 학습 데이터 연도 선택")
years = sorted(df_all["year"].unique().tolist())
sel_years = st.sidebar.multiselect("연도 선택", years, default=years)
df_train = df_all[df_all["year"].isin(sel_years)].copy()
df_train = df_train.dropna(subset=[target_col])
if df_train.empty:
    st.warning("선택된 학습 연도에 데이터가 없습니다."); st.stop()

# ── Options ───────────────────────────────────
st.sidebar.header("⑤ 분석 옵션")
th_min = st.sidebar.number_input("θ* 탐색 최소(℃)", value=0.0, step=0.5)
th_max = st.sidebar.number_input("θ* 탐색 최대(℃) (≤20 권장)", value=20.0, step=0.5)
th_step = st.sidebar.number_input("θ* 탐색 간격", value=0.1, step=0.1)

# 표시 범위: 학습 데이터 1~99p ± 1.5℃
T_train = df_train["temp"].values
p1, p99 = np.percentile(T_train, 1), np.percentile(T_train, 99)
pad = 1.5
auto_min = float(np.floor(p1 - pad)); auto_max = float(np.ceil(p99 + pad))
auto_max = min(auto_max, 25.0)

st.sidebar.markdown("**표시 온도 범위(℃)**")
mode = st.sidebar.radio("범위 모드", ["자동(권장)", "수동"], index=0, horizontal=True)
if mode == "수동":
    xmin_vis, xmax_vis = st.sidebar.slider(
        "x축 범위(℃)", min_value=float(np.floor(df_all["temp"].min()-5)),
        max_value=float(np.ceil(max(25.0, df_all["temp"].max()+5))),
        value=(auto_min, auto_max), step=0.5
    )
else:
    xmin_vis, xmax_vis = auto_min, auto_max

# ── A. Heating Start Zone (θ*) ─────────────────
st.subheader(f"A. Heating Start Zone — 베이스온도(θ*) · 대상: {target_choice}")
theta_star, a_hat, b_hat = hinge_base_temp(df_train["temp"].values, df_train[target_col].values, th_min, th_max, th_step)
st.metric("베이스온도 θ*", f"{theta_star:.2f} ℃")

df_all_plot = df_all[["temp", target_col]].rename(columns={target_col:"Q"})
df_train_plot = df_train[["temp", target_col]].rename(columns={target_col:"Q"})
fig_start = make_start_figure(df_all_plot, df_train_plot, theta_star, a_hat, b_hat, xmin_vis, xmax_vis, y_title="공급량(MJ)")
st.plotly_chart(fig_start, use_container_width=True, config={"displaylogo": False})

# ── B. Slowdown & dQ/dT (Poly-3) ───────────────
st.subheader("B. Heating Slowdown Zone & dQ/dT (Poly-3)")
m_poly, pf_poly = fit_poly3(df_train["temp"].values, df_train[target_col].values)
tgrid = np.linspace(xmin_vis, xmax_vis, 600)
d1 = np.array([poly3_d1_at(m_poly, pf_poly, t) for t in tgrid])
T_slow = float(tgrid[int(np.argmin(d1))])
st.metric("Slowdown 경계 T_slow", f"{T_slow:.2f} ℃")
fig_d1 = make_derivative_figure(tgrid, d1, theta_star, T_slow, xmin_vis, xmax_vis, y_title="변화율 dQ/dT (MJ/℃)")
st.plotly_chart(fig_d1, use_container_width=True, config={"displaylogo": False})

# ── C. Δ1℃ Impact (겨울·같은 월 + 표준기온 병렬) ──
st.subheader("C. Δ1°C Impact — 동절기 같은 월 & 표준기온(0/5/10℃) 병렬 (Poly-3)")
winter_months = st.multiselect("동절기 월", [12,1,2,3,11,4], default=[12,1,2,3], key="winter_sel")

rows_std = []
rows_med = []
for m in sorted(set(winter_months)):
    dm = df_train[df_train["month"] == m]
    if len(dm) < 6: 
        continue
    Tm, Qm = dm["temp"].values, dm[target_col].values
    model, pf = fit_poly3(Tm, Qm)
    # 표준기온 0/5/10℃ 병렬
    for t0 in [0.0, 5.0, 10.0]:
        dqdT = poly3_d1_at(model, pf, t0)
        rows_std.append({"월": m, "기온(℃)": t0, "dQ/dT(MJ/℃)": dqdT, "1℃ 하락 시 증가(MJ)": -dqdT})
    # 대표기온(중앙값)
    Trep = float(np.median(Tm))
    dqdT_med = poly3_d1_at(model, pf, Trep)
    rows_med.append({"월": m, "대표기온(℃)": round(Trep,2), "dQ/dT(MJ/℃)": dqdT_med, "1℃ 하락 시 증가(MJ)": -dqdT_med})

if rows_std:
    df_std = pd.DataFrame(rows_std)
    pivot_std = df_std.pivot_table(index="월", columns="기온(℃)", values="1℃ 하락 시 증가(MJ)", aggfunc="mean").sort_index()
    st.markdown("**표준 기온(0/5/10℃) 기준 Δ1℃(= −dQ/dT) [단위: MJ]**")
    st.dataframe(pivot_std.round(0))
    st.download_button("표준기온 Δ1℃ CSV 다운로드",
                       data=pivot_std.reset_index().to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"delta1c_standard_{target_col}.csv", mime="text/csv")

if rows_med:
    df_med = pd.DataFrame(rows_med).sort_values("월").set_index("월")
    st.markdown("**동절기 같은 월 — 대표기온(월 중앙값) 기준 Δ1℃(= −dQ/dT) [단위: MJ]**")
    st.dataframe(df_med[["대표기온(℃)","1℃ 하락 시 증가(MJ)"]].round(0))
    st.download_button("동절기(대표기온) Δ1℃ CSV 다운로드",
                       data=df_med.reset_index().to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"delta1c_winter_median_{target_col}.csv", mime="text/csv")

# ── D. 월별 탄력성 히트맵 (Elasticity Map) ──────
st.subheader("D. 월별 탄력성 히트맵 — −dQ/dT@표준기온 (MJ/℃)")
# 모든 월(1–12)에 대해 표준기온 0/5/10℃에서 −dQ/dT 계산
heat_rows = []
for m in range(1,13):
    dm = df_train[df_train["month"] == m]
    if len(dm) < 6: 
        continue
    model, pf = fit_poly3(dm["temp"].values, dm[target_col].values)
    for t0 in [0.0, 5.0, 10.0]:
        val = -poly3_d1_at(model, pf, t0)  # 탄력성의 '증가량' 관점(양수 기대)
        heat_rows.append({"월": m, "기온(℃)": t0, "−dQ/dT (MJ/℃)": val})

if heat_rows:
    heat_df = pd.DataFrame(heat_rows)
    heat_pivot = heat_df.pivot(index="월", columns="기온(℃)", values="−dQ/dT (MJ/℃)").sort_index()
    fig_hm = px.imshow(
        heat_pivot.values,
        labels=dict(x="기온(℃)", y="월", color="−dQ/dT (MJ/℃)"),
        x=heat_pivot.columns.astype(str), y=heat_pivot.index.astype(str),
        color_continuous_scale="RdBu_r", origin="lower"
    )
    fig_hm.update_layout(template="simple_white", font=dict(family=PLOT_FONT, size=14),
                         margin=dict(l=40,r=20,t=40,b=40), title="월×기온 탄력성(증가량) 히트맵")
    st.plotly_chart(fig_hm, use_container_width=True, config={"displaylogo": False})
    st.download_button("월별 탄력성 히트맵 데이터 CSV",
                       data=heat_pivot.reset_index().to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"elasticity_heatmap_{target_col}.csv", mime="text/csv")
else:
    st.info("표본이 부족해 히트맵을 만들 수 없는 월이 있습니다.")

# ── Guide ─────────────────────────────────────
with st.expander("해석 가이드"):
    st.markdown("""
- **Δ1℃(= −dQ/dT)**: 기온이 **1℃ 하락**할 때 공급량이 **얼마나 증가**하는지의 절대 증가량(MJ).
- **표준기온 비교(0/5/10℃)**: 동일 기온에서의 민감도를 월별로 공정 비교.
- **탄력성 히트맵**: 색이 진할수록(양의 큰 값) **기온 하락에 민감**한 월/구간.
- **용도 토글**: 사이드바에서 **전체/가정용** 전환 분석(가정용 컬럼이 있을 때 활성화).
""")
