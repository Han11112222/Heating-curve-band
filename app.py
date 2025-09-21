# app.py — GasPlan Hub | HeatBand Insight
# 기능:
#  1) 엑셀(data 시트) 업로드 → 컬럼 매핑
#  2) 힌지모형으로 Heating Start Zone의 기준온도 θ* 도출(탐색범위≤20℃)
#  3) Poly-3 회귀 → dQ/dT(미분) 곡선, Heating Slowdown Zone 탐지(미분 최소점)
#  4) 동절기(12~3월) 같은 월끼리 Poly-3 → 1℃ 민감도 표 & CSV
#  5) 도표/지표 표시 및 다운로드

import io
from typing import Tuple, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import streamlit as st

st.set_page_config(page_title="HeatBand Insight", layout="wide")
st.title("🔥 HeatBand Insight — 난방구간·민감도 분석")
st.caption("베이스온도(θ*)·난방개시/둔화 구간 + 동절기 1℃ 민감도(같은 월 비교)")

# =========================
# 유틸
# =========================
def to_num(x):
    if isinstance(x, str):
        x = x.replace(",", "")
    return pd.to_numeric(x, errors="coerce")

def fit_poly3(x: np.ndarray, y: np.ndarray) -> Tuple[LinearRegression, PolynomialFeatures]:
    x = x.reshape(-1, 1)
    pf = PolynomialFeatures(degree=3, include_bias=True)
    Xp = pf.fit_transform(x)
    m = LinearRegression().fit(Xp, y)
    return m, pf

def poly3_predict(m: LinearRegression, pf: PolynomialFeatures, t: np.ndarray) -> np.ndarray:
    t = t.reshape(-1, 1)
    return m.predict(pf.transform(t))

def poly3_d1_at(m: LinearRegression, pf: PolynomialFeatures, t: float) -> float:
    # pf features: [1, T, T^2, T^3]
    # m(x) = b0 + b1*T + b2*T^2 + b3*T^3
    coef = m.coef_
    b1 = coef[1] if len(coef) > 1 else 0.0
    b2 = coef[2] if len(coef) > 2 else 0.0
    b3 = coef[3] if len(coef) > 3 else 0.0
    return b1 + 2*b2*t + 3*b3*(t**2)

def hinge_base_temp(T: np.ndarray, Q: np.ndarray,
                    grid_min: float=0.0, grid_max: float=20.0, step: float=0.1) -> Tuple[float, float, float]:
    """
    힌지 모형: Q ≈ a + b * max(0, θ - T)
    θ in [grid_min, grid_max]에서 RMSE 최소 θ* 반환.
    반환: (θ*, a, b)
    """
    thetas = np.arange(grid_min, grid_max + 1e-9, step)
    best_th, best_a, best_b, best_rmse = np.nan, np.nan, np.nan, np.inf
    T = T.reshape(-1); Q = Q.reshape(-1)
    X1 = np.ones_like(T)
    for th in thetas:
        H = np.clip(th - T, 0, None)
        X = np.column_stack([X1, H])
        beta, *_ = np.linalg.lstsq(X, Q, rcond=None)  # [a, b]
        pred = X @ beta
        rmse = np.sqrt(np.mean((Q - pred)**2))
        if rmse < best_rmse:
            best_rmse = rmse
            best_th, best_a, best_b = th, float(beta[0]), float(beta[1])
    return best_th, best_a, best_b

def find_slowdown_threshold(T: np.ndarray, Q: np.ndarray) -> Tuple[float, LinearRegression, PolynomialFeatures]:
    """
    Poly-3 적합 후 dQ/dT 곡선의 '가장 음(neg.)'인 지점을 T_slow로 정의.
    (그 이하 영역은 더 추워져도 증가율이 둔화되는 Slowdown Zone로 해석)
    """
    m, pf = fit_poly3(T, Q)
    tgrid = np.linspace(min(T)-2, max(T)+2, 400)
    d1 = np.array([poly3_d1_at(m, pf, t) for t in tgrid])
    idx = int(np.argmin(d1))  # 가장 음수(감소율 최대) 지점
    T_slow = float(tgrid[idx])
    return T_slow, m, pf

# =========================
# 사이드바: 데이터 업로드
# =========================
st.sidebar.header("① 데이터 업로드")
file = st.sidebar.file_uploader("엑셀(.xlsx) 업로드 — 시트명: data", type=["xlsx"])

if not file:
    st.info("엑셀을 올려줘. 예: 열 이름이 '날짜', '평균기온', '공급량' 또는 '공급량(M3)'")
    st.stop()

# 시트 로드
try:
    df = pd.read_excel(file, sheet_name="data")
except Exception:
    # 첫 시트 사용
    xls = pd.ExcelFile(file)
    df = pd.read_excel(xls, sheet_name=xls.sheet_names[0])

cols = df.columns.tolist()

st.sidebar.header("② 컬럼 매핑")
col_date = st.sidebar.selectbox("날짜 컬럼", cols, index=0)
# 기온
guess_temp = "평균기온" if "평균기온" in cols else cols[1]
col_temp = st.sidebar.selectbox("기온(℃) 컬럼", cols, index=cols.index(guess_temp) if guess_temp in cols else 1)
# 공급량
candidates_q = [c for c in cols if "공급" in c or "M3" in c.upper() or "m3" in c]
col_q = st.sidebar.selectbox("공급량 컬럼", cols, index=cols.index(candidates_q[0]) if candidates_q else 2)

# 전처리
d = df.copy()
d["date"] = pd.to_datetime(d[col_date])
d["month"] = d["date"].dt.month
d["temp"] = d[col_temp].apply(to_num)
d["Q"] = d[col_q].apply(to_num)
d = d.dropna(subset=["temp", "Q"])
d = d.sort_values("date")

st.success(f"로우: {len(d):,} · 기간: {d['date'].min().date()} ~ {d['date'].max().date()}")

# 옵션
st.sidebar.header("③ 분석 옵션")
win_months_default = [12, 1, 2, 3]
win_months = st.sidebar.multiselect("동절기 월 선택(같은 월 비교)", list(range(1,13)), default=win_months_default)
th_min = st.sidebar.number_input("θ* 탐색 최소(℃)", value=0.0, step=0.5)
th_max = st.sidebar.number_input("θ* 탐색 최대(℃)", value=20.0, step=0.5)
th_step = st.sidebar.number_input("θ* 탐색 간격", value=0.1, step=0.1)
rep_method = st.sidebar.selectbox("민감도 대표기온", ["월별 실측 중앙값", "사용자 지정"])
rep_user = st.sidebar.number_input("대표기온 직접 입력(℃)", value=0.0, step=0.5)

work = d.copy()

# =========================
# 1) Heating Start Zone (θ*)
# =========================
st.subheader("A. Heating Start Zone — 베이스온도(θ*)")
T = work["temp"].values
Qv = work["Q"].values

theta_star, a_hat, b_hat = hinge_base_temp(T, Qv, grid_min=th_min, grid_max=th_max, step=th_step)
st.metric("베이스온도 θ*", f"{theta_star:.2f} ℃", help="이 온도 이하에서 난방수요가 선형 증가한다고 보는 기준")

# 곡선
tline = np.linspace(min(T)-2, max(T)+2, 300)
H = np.clip(theta_star - tline, 0, None)
q_hat = a_hat + b_hat*H

fig1 = plt.figure(figsize=(7.2, 4.2))
plt.scatter(T, Qv, alpha=0.5, s=14, label="실측")
plt.plot(tline, q_hat, linewidth=2, label="힌지 적합")
plt.axvline(theta_star, linestyle="--", label=f"θ* = {theta_star:.2f}℃")
plt.fill_betweenx([min(Qv), max(Qv)], -100, theta_star, alpha=0.08, label="Heating Start Zone")
plt.xlabel("기온(℃)"); plt.ylabel("공급량 (단위)")
plt.title("힌지 적합과 Heating Start Zone")
plt.legend()
st.pyplot(fig1, clear_figure=True)

# =========================
# 2) Heating Slowdown Zone & dQ/dT 곡선
# =========================
st.subheader("B. Heating Slowdown Zone & dQ/dT 곡선 (Poly-3)")

T_slow, m_poly, pf_poly = find_slowdown_threshold(T, Qv)
st.metric("Slowdown 경계 T_slow", f"{T_slow:.2f} ℃", help="dQ/dT(민감도)가 가장 음수인 지점; 그 이하 온도는 증가율 둔화 구간으로 해석")

tgrid = np.linspace(min(T)-5, max(T)+5, 400)
d1 = np.array([poly3_d1_at(m_poly, pf_poly, t) for t in tgrid])

fig2 = plt.figure(figsize=(10, 4.4))
plt.plot(tgrid, d1, linewidth=2, label="dQ/dT (단위/℃)")
plt.axvline(theta_star, linestyle="--", label=f"Heating Start (θ*={theta_star:.2f}℃)")
plt.axvline(T_slow, linestyle="--", color="red", label=f"Demand Slows ({T_slow:.2f}℃)")
# 음영: Slowdown Zone(왼쪽), Start Zone(왼쪽~θ*)
ymin, ymax = np.min(d1), np.max(d1)
plt.fill_betweenx([ymin, ymax], -100, T_slow, alpha=0.10, color="red", label="Heating Slowdown Zone")
plt.fill_betweenx([ymin, ymax], T_slow, theta_star, alpha=0.10, label="Heating Start Zone")
plt.xlabel("기온(℃)"); plt.ylabel("변화율 dQ/dT (단위/℃)")
plt.title("Rate of Change vs Temperature — HeatBand")
plt.legend()
st.pyplot(fig2, clear_figure=True)

# =========================
# 3) 동절기 '같은 월' 1℃ 민감도 (Poly-3)
# =========================
st.subheader("C. 동절기 같은 월 1℃ 민감도 (Δ1°C Impact, Poly-3)")

rows = []
for m in win_months:
    dm = d[d["month"] == m]
    if len(dm) < 6:
        continue
    Tm = dm["temp"].values
    Qm = dm["Q"].values
    model, pf = fit_poly3(Tm, Qm)
    if rep_method == "월별 실측 중앙값":
        Trep = float(np.median(Tm))
    else:
        Trep = float(rep_user)
    dqdT = poly3_d1_at(model, pf, Trep)
    impact_down1 = -dqdT  # 1℃ 하락 시 증가량
    rows.append({
        "월": m,
        "표본수": int(len(dm)),
        "대표기온(℃)": round(Trep, 2),
        "dQ/dT(단위/℃)": round(dqdT, 2),
        "1℃ 하락 시 증가(단위)": round(impact_down1, 2)
    })

if rows:
    out = pd.DataFrame(rows).sort_values("월").set_index("월")
    st.dataframe(out)
    st.download_button(
        "동절기 민감도 CSV 다운로드",
        data=out.reset_index().to_csv(index=False).encode("utf-8-sig"),
        file_name="winter_months_delta1c.csv",
        mime="text/csv"
    )
else:
    st.info("선택한 월에 대한 표본이 부족하면 표가 비어 있을 수 있어.")

# =========================
# 해석 가이드
# =========================
with st.expander("해석 가이드"):
    st.markdown("""
- **Heating Start Zone**: 힌지모형의 기준온도 θ* 이하 구간. θ*는 [설정한 범위 ≤ 20℃]에서 RMSE 최소가 되는 값.
- **Heating Slowdown Zone**: Poly-3의 **dQ/dT**(변화율)가 가장 음수인 온도 T_slow를 기준으로 그보다 더 낮은 온도 영역.
  즉, 아주 추운 영역에서는 추가 하강에 따른 증가율이 완만해지는 현상을 시사.
- **Δ1°C Impact**: 특정 월 표본의 대표기온(중앙값/사용자지정)에서 **1℃ 하락 시 증가량 = −dQ/dT**.
  (동절기 12–3월 위주로 같은 월끼리 비교하면 운영설득에 효과적)
""")
