# app.py — Heating-curve-band | HeatBand Insight
# Repo에 있는 '실적.xlsx' 자동로드(없으면 업로드), 히트밴드(θ*·Slowdown) + 동절기 Δ1°C 민감도
import os
from typing import Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import streamlit as st

st.set_page_config(page_title="HeatBand Insight", layout="wide")
st.title("🔥 HeatBand Insight — 난방구간·민감도 분석")
st.caption("Heating Start(θ*) · Heating Slowdown · Δ1°C Impact (월별/동절기)")

# ------------------------------
# 유틸
# ------------------------------
def to_num(x):
    if isinstance(x, str):
        x = x.replace(",", "")
    return pd.to_numeric(x, errors="coerce")

def fit_poly3(x: np.ndarray, y: np.ndarray):
    x = x.reshape(-1, 1)
    pf = PolynomialFeatures(degree=3, include_bias=True)
    Xp = pf.fit_transform(x)
    m = LinearRegression().fit(Xp, y)
    return m, pf

def poly3_predict(m: LinearRegression, pf: PolynomialFeatures, t: np.ndarray) -> np.ndarray:
    t = t.reshape(-1, 1)
    return m.predict(pf.transform(t))

def poly3_d1_at(m: LinearRegression, pf: PolynomialFeatures, t: float) -> float:
    coef = m.coef_
    b1 = coef[1] if len(coef) > 1 else 0.0
    b2 = coef[2] if len(coef) > 2 else 0.0
    b3 = coef[3] if len(coef) > 3 else 0.0
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
        beta, *_ = np.linalg.lstsq(X, Q, rcond=None)  # a,b
        pred = X @ beta
        rmse = np.sqrt(np.mean((Q - pred)**2))
        if rmse < best_rmse:
            best_rmse = rmse
            best_th, best_a, best_b = th, float(beta[0]), float(beta[1])
    return best_th, best_a, best_b

def find_slowdown_threshold(T: np.ndarray, Q: np.ndarray):
    """Poly-3 dQ/dT가 가장 음수(감소율 최대)인 온도 T_slow를 둔화 경계로 정의"""
    m, pf = fit_poly3(T, Q)
    tgrid = np.linspace(min(T)-2, max(T)+2, 400)
    d1 = np.array([poly3_d1_at(m, pf, t) for t in tgrid])
    T_slow = float(tgrid[int(np.argmin(d1))])
    return T_slow, m, pf

@st.cache_data
def load_excel(path: str) -> pd.DataFrame:
    import openpyxl  # ensure installed
    try:
        df = pd.read_excel(path, sheet_name="data")
    except Exception:
        xls = pd.ExcelFile(path)
        df = pd.read_excel(xls, sheet_name=xls.sheet_names[0])
    return df

# ------------------------------
# 데이터 입력: 자동로드 + 업로더
# ------------------------------
st.sidebar.header("① 데이터")
repo_file = "실적.xlsx"
use_repo = os.path.exists(repo_file)
uploaded = st.sidebar.file_uploader("엑셀(.xlsx) 업로드 (없으면 리포지토리 파일 자동 사용)", type=["xlsx"])

if uploaded is not None:
    raw = load_excel(uploaded)
elif use_repo:
    st.sidebar.info("리포지토리의 '실적.xlsx'을 자동 사용 중")
    raw = load_excel(repo_file)
else:
    st.info("엑셀을 업로드하거나 리포지토리에 '실적.xlsx'를 넣어줘.")
    st.stop()

cols = raw.columns.tolist()

st.sidebar.header("② 컬럼 매핑")
def _find_col(cands):
    for k in cands:
        for c in cols:
            if k in str(c):
                return c
    return cols[0]

date_col = st.sidebar.selectbox("날짜", cols, index=cols.index(_find_col(["날짜", "date"])) if _find_col(["날짜","date"]) in cols else 0)
temp_col = st.sidebar.selectbox("평균기온(℃)", cols, index=cols.index(_find_col(["평균기온","기온","temp"])) if _find_col(["평균기온","기온","temp"]) in cols else 1)
q_col = st.sidebar.selectbox("공급량(㎥)", cols, index=cols.index(_find_col(["공급량","M3","m3"])) if _find_col(["공급량","M3","m3"]) in cols else 2)

df = raw.copy()
df["date"] = pd.to_datetime(df[date_col])
df["month"] = df["date"].dt.month
df["temp"] = df[temp_col].apply(to_num)
df["Q"] = df[q_col].apply(to_num)
df = df.dropna(subset=["temp", "Q"]).sort_values("date")

st.success(f"행 {len(df):,} · 기간 {df['date'].min().date()} ~ {df['date'].max().date()}")

# ------------------------------
# 옵션
# ------------------------------
st.sidebar.header("③ 옵션")
win_months = st.sidebar.multiselect("동절기 월", [12,1,2,3,4,11], default=[12,1,2,3])
th_min = st.sidebar.number_input("θ* 탐색 최소(℃)", value=0.0, step=0.5)
th_max = st.sidebar.number_input("θ* 탐색 최대(℃) (≤20 권장)", value=20.0, step=0.5)
th_step = st.sidebar.number_input("θ* 탐색 간격", value=0.1, step=0.1)
rep_method = st.sidebar.selectbox("Δ1°C 대표기온", ["월별 실측 중앙값", "사용자 지정"])
rep_user = st.sidebar.number_input("대표기온 직접 입력(℃)", value=0.0, step=0.5)

T = df["temp"].values
Qv = df["Q"].values

# ------------------------------
# A. Heating Start Zone — θ*
# ------------------------------
st.subheader("A. Heating Start Zone — 베이스온도(θ*)")
theta_star, a_hat, b_hat = hinge_base_temp(T, Qv, th_min, th_max, th_step)
st.metric("베이스온도 θ*", f"{theta_star:.2f} ℃")

tline = np.linspace(min(T)-2, max(T)+2, 320)
H = np.clip(theta_star - tline, 0, None)
q_hat = a_hat + b_hat*H

fig1 = plt.figure(figsize=(7.2, 4.2))
plt.scatter(T, Qv, alpha=0.5, s=14, label="실측")
plt.plot(tline, q_hat, linewidth=2, label="힌지 적합")
plt.axvline(theta_star, linestyle="--", label=f"θ* = {theta_star:.2f}℃")
plt.fill_betweenx([min(Qv), max(Qv)], -100, theta_star, alpha=0.08, label="Heating Start Zone")
plt.xlabel("기온(℃)"); plt.ylabel("공급량(㎥)"); plt.title("힌지 적합과 Heating Start Zone"); plt.legend()
st.pyplot(fig1, clear_figure=True)

# ------------------------------
# B. Heating Slowdown Zone & dQ/dT
# ------------------------------
st.subheader("B. Heating Slowdown Zone & dQ/dT (Poly-3)")
T_slow, m_poly, pf_poly = find_slowdown_threshold(T, Qv)
st.metric("Slowdown 경계 T_slow", f"{T_slow:.2f} ℃")

tgrid = np.linspace(min(T)-5, max(T)+5, 400)
d1 = np.array([poly3_d1_at(m_poly, pf_poly, t) for t in tgrid])

fig2 = plt.figure(figsize=(10, 4.4))
plt.plot(tgrid, d1, linewidth=2, label="dQ/dT (㎥/℃)")
plt.axvline(theta_star, linestyle="--", label=f"Start θ*={theta_star:.2f}℃")
plt.axvline(T_slow, linestyle="--", color="red", label=f"Slowdown {T_slow:.2f}℃")
ymin, ymax = np.min(d1), np.max(d1)
plt.fill_betweenx([ymin, ymax], -100, T_slow, alpha=0.10, color="red", label="Heating Slowdown Zone")
plt.fill_betweenx([ymin, ymax], T_slow, theta_star, alpha=0.10, label="Heating Start Zone")
plt.xlabel("기온(℃)"); plt.ylabel("변화율 dQ/dT (㎥/℃)"); plt.title("Rate of Change vs Temperature — HeatBand"); plt.legend()
st.pyplot(fig2, clear_figure=True)

# ------------------------------
# C. 동절기 '같은 월' Δ1°C Impact
# ------------------------------
st.subheader("C. 동절기 같은 월 Δ1°C Impact (Poly-3)")
rows = []
for m in sorted(set(win_months)):
    dm = df[df["month"] == m]
    if len(dm) < 6:
        continue
    Tm, Qm = dm["temp"].values, dm["Q"].values
    model, pf = fit_poly3(Tm, Qm)
    Trep = float(np.median(Tm)) if rep_method == "월별 실측 중앙값" else float(rep_user)
    dqdT = poly3_d1_at(model, pf, Trep)
    impact = -dqdT  # 1℃ 하락 시 증가량
    rows.append({"월": m, "표본수": len(dm), "대표기온(℃)": round(Trep,2),
                 "dQ/dT(㎥/℃)": round(dqdT,2), "1℃ 하락 시 증가(㎥)": round(impact,2)})

if rows:
    out = pd.DataFrame(rows).sort_values("월").set_index("월")
    st.dataframe(out)
    st.download_button("동절기 민감도 CSV 다운로드",
                       data=out.reset_index().to_csv(index=False).encode("utf-8-sig"),
                       file_name="winter_delta1c.csv", mime="text/csv")
else:
    st.info("선택한 월의 표본이 부족하면 표가 비게 돼.")

# ------------------------------
# 해석 가이드
# ------------------------------
with st.expander("해석 가이드"):
    st.markdown("""
- **Heating Start Zone**: 힌지모형 기준온도 θ* 이하 영역. 이 온도부터 수요가 선형적으로 증가.
- **Heating Slowdown Zone**: Poly-3의 dQ/dT가 가장 음수인 온도 T_slow를 기준으로 그보다 낮은 영역. 매우 추운 구간에서 증가율이 완만해지는 현상.
- **Δ1°C Impact**: 특정 월의 대표기온에서 **1℃ 하락 시 증가량 = −dQ/dT**.
""")
