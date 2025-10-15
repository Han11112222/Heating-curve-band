# app.py — HeatBand Insight (Slim v2: 핵심3개 + 5% CI + Excel DL 안전)
# 단위: 공급량(MJ), 변화율 dQ/dT(MJ/℃)

import os, io
from typing import Tuple, List
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.font_manager as fm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import streamlit as st

st.set_page_config(page_title="HeatBand Insight", layout="wide")

# ── Korean font ──────────────────────────────────────────────
FONT_PATH = "NanumGothic-Regular.ttf"
if os.path.exists(FONT_PATH):
    try: fm.fontManager.addfont(FONT_PATH)
    except: pass
PLOT_FONT = "NanumGothic, Arial, Noto Sans KR, sans-serif"

st.title("🔥 HeatBand Insight — 난방구간·민감도 핵심 보드")
st.caption("단위: 공급량 **MJ**, 변화율 **MJ/℃** · Heating Start(θ*) · Heating Slowdown · Δ1°C Impact")

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
    b0 = float(m.intercept_); c = m.coef_
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

def z_from_alpha(alpha: float) -> float:
    # two-sided CI: z = Φ^{-1}(1 - alpha/2) ; 간단 매핑(외부 lib 없이)
    table = {0.95: 0.0627, 0.2:1.2816, 0.1:1.6449, 0.05:1.96}
    return table.get(round(alpha,2), 1.96)

def poly3_conf_band(x_train, y_train, tgrid, m, pf, alpha=0.95):
    # alpha=0.95 → "5% CI" (두쪽 합계 5%만 남기는 아주 좁은 밴드)
    X = pf.transform(x_train.reshape(-1,1))
    yhat = m.predict(X)
    n, p = X.shape
    sigma2 = np.sum((y_train - yhat)**2) / max(1, (n - p))
    XtX_inv = np.linalg.pinv(X.T @ X)
    Tg = pf.transform(tgrid.reshape(-1,1))
    se = np.sqrt(np.sum(Tg @ XtX_inv * Tg, axis=1) * sigma2)
    z = z_from_alpha(alpha)
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
            best_rmse = rmse; best_th, best_a, best_b = th, float(beta[0]), float(beta[1])
    return best_th, best_a, best_b

def nice_poly_string(a,b,c,d, digits=2):
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
    except: return str(x)

# 안전평가: 월별 모델 평가 시 t0가 월별 샘플 범위 밖이면 글로벌 모델 사용
def safe_delta1c(t0: float, month_df: pd.DataFrame, month_model, month_pf,
                 global_model, global_pf) -> float:
    if len(month_df) < 6:
        dqdT = poly3_d1_at(global_model, global_pf, t0)
    else:
        tmin, tmax = float(month_df["temp"].min()), float(month_df["temp"].max())
        if (t0 < tmin) or (t0 > tmax):
            dqdT = poly3_d1_at(global_model, global_pf, t0)
        else:
            dqdT = poly3_d1_at(month_model, month_pf, t0)
    return max(0.0, -dqdT)  # 1℃ 하락 시 증가량(해석용; 음수 하한 0)

# ── Excel Loader / Export (openpyxl 우선, 미존재 시 CSV 폴백) ──
@st.cache_data(show_spinner=False)
def read_excel_cached(path_or_buf) -> pd.DataFrame:
    try:
        import openpyxl  # noqa
        engine = None
    except:
        engine = "openpyxl"
    try:
        try:
            return pd.read_excel(path_or_buf, sheet_name="data", engine=engine)
        except:
            xls = pd.ExcelFile(path_or_buf, engine=engine)
            return pd.read_excel(xls, sheet_name=xls.sheet_names[0])
    except Exception as e:
        st.error(f"엑셀 로딩 문제: {type(e).__name__} — {e}")
        st.stop()

def build_excel_bytes(sheets: dict) -> Tuple[bytes, str]:
    # sheets = {"SheetName": DataFrame or str}
    try:
        import openpyxl  # ensure available
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as wr:
            for name, obj in sheets.items():
                if isinstance(obj, pd.DataFrame):
                    obj.to_excel(wr, index=False, sheet_name=name)
                else:
                    # text를 단일 셀에 기록
                    pd.DataFrame({"note":[obj]}).to_excel(wr, index=False, sheet_name=name)
        buf.seek(0)
        return buf.read(), "xlsx"
    except Exception:
        # CSV zip으로 폴백
        import zipfile
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for name, obj in sheets.items():
                if isinstance(obj, pd.DataFrame):
                    z.writestr(f"{name}.csv", obj.to_csv(index=False, encoding="utf-8-sig"))
                else:
                    z.writestr(f"{name}.txt", str(obj))
        buf.seek(0)
        return buf.read(), "zip"

# ── Data In ──────────────────────────────────────────────────
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
if q_res_col != "(없음)": df_all["Q_res"] = df_all[q_res_col].apply(to_num)
df_all = df_all.dropna(subset=["temp","Q_total"]).sort_values("date")

st.success(f"전체 행 {len(df_all):,} · 기간 {df_all['date'].min().date()} ~ {df_all['date'].max().date()}")

# ── Target & Train years ────────────────────────────────────
st.sidebar.header("③ 분석 대상")
targets = ["전체(MJ)"] + (["가정용(MJ)"] if "Q_res" in df_all.columns else [])
target_choice = st.sidebar.radio("대상 선택", targets, horizontal=True)
target_col = "Q_total" if target_choice.startswith("전체") else "Q_res"

st.sidebar.header("④ 학습 연도")
years = sorted(df_all["year"].unique().tolist())
sel_years = st.sidebar.multiselect("연도 선택", years, default=years)
df_train = df_all[df_all["year"].isin(sel_years)].copy().dropna(subset=[target_col])
if df_train.empty:
    st.warning("선택된 학습 연도에 데이터가 없습니다."); st.stop()

# 시각 구간
T_train = df_train["temp"].values
p1, p99 = np.percentile(T_train, 1), np.percentile(T_train, 99)
pad = 1.5
xmin_vis = float(np.floor(p1 - pad))
xmax_vis = float(np.ceil(min(25.0, p99 + pad)))

# ── 1) Poly-3 식 + 상관 그래프(5% CI) ────────────────────────
st.subheader(f"A. 기온–공급량 상관(Poly-3) — {target_choice} 〔5% CI〕")
x_tr = df_train["temp"].values
y_tr = df_train[target_col].values
m_all, pf_all, Xp_all = fit_poly3(x_tr, y_tr)
yhat_tr = m_all.predict(Xp_all)
r2 = poly3_r2(y_tr, yhat_tr)
tgrid0 = np.linspace(xmin_vis, xmax_vis, 500)
y_pred0 = poly3_predict(m_all, pf_all, tgrid0)
ci_lo, ci_hi = poly3_conf_band(x_tr, y_tr, tgrid0, m_all, pf_all, alpha=0.95)  # 5% CI

a,b,c,d = poly3_coeffs(m_all)
eq_str = nice_poly_string(a,b,c,d, digits=1)

fig_corr = go.Figure()
fig_corr.add_trace(go.Scatter(x=df_train["temp"], y=df_train[target_col],
    mode="markers", name="샘플",
    marker=dict(size=9, line=dict(width=0.5, color="white")),
    hovertemplate="T=%{x:.2f}℃<br>Q=%{y:,.0f} MJ<extra></extra>"))
fig_corr.add_trace(go.Scatter(x=tgrid0, y=y_pred0, mode="lines",
    name="Poly-3", line=dict(width=3)))
fig_corr.add_trace(go.Scatter(x=np.r_[tgrid0, tgrid0[::-1]],
    y=np.r_[ci_hi, ci_lo[::-1]], fill="toself", name="5% CI",
    line=dict(color="rgba(30,144,255,0)"), fillcolor="rgba(30,144,255,0.18)",
    hoverinfo="skip"))
fig_corr.update_layout(template="simple_white", font=dict(family=PLOT_FONT, size=14),
    margin=dict(l=40,r=20,t=50,b=40), xaxis=dict(title="기온(℃)", range=[xmin_vis, xmax_vis]),
    yaxis=dict(title="공급량(MJ)", tickformat=","), title=f"R²={r2:.3f}")
fig_corr.add_annotation(xref="paper", yref="paper", x=0.01, y=0.02,
    text=eq_str, showarrow=False, bgcolor="rgba(255,255,255,0.85)",
    bordercolor="black", borderwidth=1, font=dict(size=12))
st.plotly_chart(fig_corr, use_container_width=True, config={"displaylogo": False})

# ── 2) 구간별 Δ1℃ 증가량(동적 탭 + 포물선/5% CI) ────────────────
st.subheader("B. 기온 구간별 Δ1℃ 증가량 (−dQ/dT > 0로 표기)")
m_poly, pf_poly, _ = fit_poly3(df_train["temp"].values, df_train[target_col].values)

def band_plot(title, t_from, t_to):
    tg = np.linspace(t_from, t_to, 121)
    d1 = np.array([poly3_d1_at(m_poly, pf_poly, t) for t in tg])
    incr = np.maximum(0.0, -d1)  # 증가량(+)
    # 5% CI를 파생(예측 CI를 수치미분)
    eps = 0.1
    y_lo, y_hi = poly3_conf_band(x_tr, y_tr, tg, m_poly, pf_poly, alpha=0.95)
    y_lo_l = poly3_predict(m_poly, pf_poly, tg-eps)
    y_lo_u = poly3_predict(m_poly, pf_poly, tg+eps)
    y_hi_l = poly3_predict(m_poly, pf_poly, tg-eps)
    y_hi_u = poly3_predict(m_poly, pf_poly, tg+eps)
    # 수치 미분으로 상/하 밴드 (보수적: hi/lo 모두에서 최대 증가량 사용)
    d_lo = (y_lo_u - y_lo_l)/(2*eps)
    d_hi = (y_hi_u - y_hi_l)/(2*eps)
    ci_top = np.maximum(0.0, -np.minimum(d_lo, d_hi))
    ci_bot = np.maximum(0.0, -np.maximum(d_lo, d_hi))

    avg = float(np.mean(incr))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=tg, y=incr, mode="lines+markers",
        name="증가량(MJ/℃)", marker=dict(size=5), line=dict(width=3),
        hovertemplate="T=%{x:.2f}℃<br>Δ1℃=%{y:,.0f} MJ<extra></extra>"))
    fig.add_trace(go.Scatter(x=np.r_[tg, tg[::-1]], y=np.r_[ci_top, ci_bot[::-1]],
        name="5% CI", fill="toself", line=dict(color="rgba(135,206,250,0)"),
        fillcolor="rgba(135,206,250,0.20)", hoverinfo="skip"))
    fig.add_annotation(x=(t_from+t_to)/2, y=avg, text=f"Band Avg = {fmt_int(avg)} MJ/℃",
        showarrow=False, font=dict(size=13), bgcolor="rgba(255,255,255,0.7)")
    fig.update_layout(template="simple_white", font=dict(family=PLOT_FONT, size=14),
        margin=dict(l=40,r=20,t=40,b=40), xaxis=dict(title="기온(℃)"),
        yaxis=dict(title="Δ1℃ 증가량 (MJ/℃)", tickformat=","),
        title=title)
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
    return pd.DataFrame({"T(℃)": tg, "Δ1℃ 증가량(MJ/℃)": incr})

tabs = st.tabs(["−5~0℃", "0~5℃", "5~10℃"])
with tabs[0]: df_band1 = band_plot("Band −5~0℃ Response", -5, 0)
with tabs[1]: df_band2 = band_plot("Band 0~5℃ Response", 0, 5)
with tabs[2]: df_band3 = band_plot("Band 5~10℃ Response", 5, 10)

# ── 3) Heating Start / Slowdown / Refined dQ/dT (하단) ─────────
st.subheader("C. Refined Gas Supply Rate of Change (Dynamic) — Start/Slowdown")
theta_star, a_hat, b_hat = hinge_base_temp(df_train["temp"].values, df_train[target_col].values,
                                           0.0, 20.0, 0.1)
tgrid = np.linspace(xmin_vis, xmax_vis, 700)
d1 = np.array([poly3_d1_at(m_poly, pf_poly, t) for t in tgrid])
incr = np.maximum(0.0, -d1)  # 양수로 표기
T_slow = float(tgrid[int(np.argmax(incr))])  # 증가량 최대점(민감 최고)
fig_ref = go.Figure()
fig_ref.add_trace(go.Scatter(x=tgrid, y=incr, mode="lines",
    name="증가량(MJ/℃)", line=dict(width=3),
    hovertemplate="T=%{x:.2f}℃<br>Δ1℃=%{y:,.0f} MJ<extra></extra>"))
# Zone 음영
fig_ref.add_vrect(x0=xmin_vis, x1=T_slow, fillcolor="LightCoral", opacity=0.12, line_width=0,
                  annotation_text=f"Heating Slowdown (≤ {T_slow:.2f}℃)", annotation_position="top left")
fig_ref.add_vrect(x0=T_slow, x1=theta_star, fillcolor="LightSkyBlue", opacity=0.12, line_width=0,
                  annotation_text=f"Heating Start ({T_slow:.2f}~{theta_star:.2f}℃)", annotation_position="top left")
fig_ref.add_vline(x=T_slow, line_dash="dash", line_color="red",
                  annotation_text=f"Slowdown {T_slow:.2f}℃", annotation_position="bottom left")
fig_ref.add_vline(x=theta_star, line_dash="dot", line_color="steelblue",
                  annotation_text=f"Start θ* {theta_star:.2f}℃", annotation_position="bottom right")
fig_ref.update_layout(template="simple_white", font=dict(family=PLOT_FONT, size=14),
    margin=dict(l=40,r=20,t=50,b=40), xaxis=dict(title="Temperature (℃)", range=[xmin_vis, xmax_vis]),
    yaxis=dict(title="Δ1℃ 증가량 (MJ/℃)", tickformat=","), title="Refined dQ/dT → Δ1℃ 증가량(+)")
st.plotly_chart(fig_ref, use_container_width=True, config={"displaylogo": False})

# ── 요약(보고서용 문구) + 다운로드 ────────────────────────────
a,b,c,d = poly3_coeffs(m_poly)
eq_text = (
    "◇ 3차 다항식에 근거한 1℃ 변화에 따른 월별 공급량 변화 요약\n"
    f"Polynomial Regression Equation (degree 3):\n  {nice_poly_string(a,b,c,d, digits=2)}\n"
    "- Supply Increase (in MJ) when Temperature Decreases by 1 Degree from 10°C to 5°C: "
    f"{fmt_int(np.mean(np.maximum(0.0, -np.array([poly3_d1_at(m_poly, pf_poly, t) for t in np.linspace(5,10,51)]))))}\n"
    "- Supply Increase (in MJ) when Temperature Decreases by 1 Degree from 5°C to 0°C: "
    f"{fmt_int(np.mean(np.maximum(0.0, -np.array([poly3_d1_at(m_poly, pf_poly, t) for t in np.linspace(0,5,51)]))))}\n"
    "- Supply Increase (in MJ) when Temperature Decreases by 1 Degree from 0°C to −5°C: "
    f"{fmt_int(np.mean(np.maximum(0.0, -np.array([poly3_d1_at(m_poly, pf_poly, t) for t in np.linspace(-5,0,51)]))))}\n"
    f"※ Heating Slowdown ~ Start 구간: {T_slow:.2f}℃ ~ {theta_star:.2f}℃"
)

st.markdown("### D. 보고서용 요약 텍스트")
st.code(eq_text)

st.subheader("E. 결과 다운로드")
sheets = {
    "Summary": pd.DataFrame({"summary":[eq_text]}),
    "Band_-5_0": df_band1,
    "Band_0_5": df_band2,
    "Band_5_10": df_band3
}
bytes_blob, ext = build_excel_bytes(sheets)
fname = f"HeatBand_Results.{ext}"
st.download_button(f"결과 내려받기 ({ext.upper()})", data=bytes_blob, file_name=fname,
                   mime="application/octet-stream")
