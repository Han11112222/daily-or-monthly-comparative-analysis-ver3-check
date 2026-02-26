import calendar
from io import BytesIO
from pathlib import Path
import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from openpyxl.styles import Alignment, Border, Side
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# ─────────────────────────────────────────────
# 1. 기본 설정 및 단위 환산
# ─────────────────────────────────────────────
MJ_PER_NM3 = 42.563
MJ_TO_GJ = 1.0 / 1000.0

def mj_to_gj(x):
    try: return x * MJ_TO_GJ
    except: return np.nan

st.set_page_config(page_title="도시가스 일별 공급량 재분배 (Ver.4 Hybrid)", layout="wide")

if 'rec_active' not in st.session_state: st.session_state['rec_active'] = False
if 'cal_start' not in st.session_state: st.session_state['cal_start'] = None
if 'cal_end' not in st.session_state: st.session_state['cal_end'] = None
if 'fix_start' not in st.session_state: st.session_state['fix_start'] = None
if 'fix_end' not in st.session_state: st.session_state['fix_end'] = None
if 'rec_rate' not in st.session_state: st.session_state['rec_rate'] = 0.0

# ─────────────────────────────────────────────
# 2. 데이터 로드
# ─────────────────────────────────────────────
@st.cache_data
def load_daily_data():
    excel_path = Path(__file__).parent / "공급량(일일실적).xlsx"
    if not excel_path.exists(): return pd.DataFrame()
    df = pd.read_excel(excel_path)
    df["일자"] = pd.to_datetime(df["일자"])
    df["연도"] = df["일자"].dt.year
    df["월"] = df["일자"].dt.month
    df["일"] = df["일자"].dt.day
    df["weekday_idx"] = df["일자"].dt.weekday
    if '공급량(MJ)' in df.columns and df['공급량(MJ)'].dtype == object:
        df['공급량(MJ)'] = df['공급량(MJ)'].astype(str).str.replace(',', '').astype(float)
    return df

@st.cache_data
def load_monthly_plan():
    excel_path = Path(__file__).parent / "공급량(계획_실적).xlsx"
    if not excel_path.exists(): return pd.DataFrame()
    return pd.read_excel(excel_path, sheet_name="월별계획_실적")

@st.cache_data
def load_effective_calendar():
    excel_path = Path(__file__).parent / "effective_days_calendar.xlsx"
    if not excel_path.exists(): return None
    df = pd.read_excel(excel_path)
    df["일자"] = pd.to_datetime(df["날짜"].astype(str), format="%Y%m%d", errors="coerce")
    for col in ["공휴일여부", "명절여부"]:
        if col not in df.columns: df[col] = False
        df[col] = df[col].fillna(False).astype(bool)
    return df[["일자", "공휴일여부", "명절여부"]].copy()

# ─────────────────────────────────────────────
# 3. 모델링 (Ver.3 기온 예측 로직)
# ─────────────────────────────────────────────
@st.cache_resource
def train_models(df):
    # 1. 최저/최고 -> 평균기온 모델
    df_t = df.dropna(subset=['최저기온(℃)', '최고기온(℃)', '평균기온(℃)'])
    model_temp = LinearRegression()
    if not df_t.empty:
        model_temp.fit(df_t[['최저기온(℃)', '최고기온(℃)']], df_t['평균기온(℃)'])
    
    # 2. 평균기온 -> 공급량 다항회귀 모델
    df_s = df.dropna(subset=['평균기온(℃)', '공급량(MJ)'])
    df_s = df_s[df_s['공급량(MJ)'] > 0]
    model_supply = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
    if not df_s.empty:
        model_supply.fit(df_s[['평균기온(℃)']], df_s['공급량(MJ)'])
        
    return model_temp, model_supply

def get_past_stats(df_raw, target_month):
    stats_dict = {} 
    df_past = df_raw[df_raw['월'] == target_month].copy()
    if df_past.empty: return stats_dict
    max_year = df_past['연도'].max()
    target_years = [max_year-1, max_year-2, max_year-3]
    df_past = df_past[df_past['연도'].isin(target_years)]
    grp = df_past.groupby('일')[['최저기온(℃)', '최고기온(℃)']].mean()
    for day, row in grp.iterrows():
        stats_dict[day] = (row['최저기온(℃)'], row['최고기온(℃)'])
    return stats_dict

# ─────────────────────────────────────────────
# 4. 하이브리드 분배 로직 (Ver.2 + Ver.3)
# ─────────────────────────────────────────────
def make_hybrid_daily_plan(df_daily, df_plan, cal_df, target_year, target_month, recent_window, temp_weight):
    all_years = sorted(df_daily["연도"].unique())
    candidate_years = [y for y in range(target_year - recent_window, target_year) if y in all_years]
    
    # 1. 월간 목표 공급량 가져오기
    plan_col = "계획(사업계획제출_MJ)" if "계획(사업계획제출_MJ)" in df_plan.columns else [c for c in df_plan.columns if "계획" in c][0]
    row_plan = df_plan[(df_plan["연"] == target_year) & (df_plan["월"] == target_month)]
    monthly_plan_total = float(row_plan[plan_col].iloc[0]) if not row_plan.empty else 0

    # 2. 타겟 일자 생성
    last_day = calendar.monthrange(target_year, target_month)[1]
    dates = pd.date_range(f"{target_year}-{target_month:02d}-01", periods=last_day, freq="D")
    df_target = pd.DataFrame({"일자": dates, "일": dates.day, "weekday_idx": dates.weekday})
    
    # 공휴일 조인
    if cal_df is not None:
        df_target = df_target.merge(cal_df, on="일자", how="left").fillna(False)
    else:
        df_target["공휴일여부"] = False; df_target["명절여부"] = False
    
    df_target["is_weekend"] = (df_target["weekday_idx"] >= 5) | df_target["공휴일여부"] | df_target["명절여부"]
    df_target["구분"] = np.where(df_target["is_weekend"], "주말/공휴일", np.where(df_target["weekday_idx"].isin([0,4]), "평일1(월,금)", "평일2(화,수,목)"))

    # ---------------------------------------------------------
    # [Ratio A] 패턴 기반 비율 계산 (Ver.2)
    # ---------------------------------------------------------
    df_pool = df_daily[(df_daily["연도"].isin(candidate_years)) & (df_daily["월"] == target_month)].copy()
    if not df_pool.empty:
        df_pool["month_total"] = df_pool.groupby("연도")["공급량(MJ)"].transform("sum")
        df_pool["ratio"] = df_pool["공급량(MJ)"] / df_pool["month_total"]
        df_pool["is_weekend"] = (df_pool["weekday_idx"] >= 5)
        df_pool["구분"] = np.where(df_pool["is_weekend"], "주말/공휴일", np.where(df_pool["weekday_idx"].isin([0,4]), "평일1(월,금)", "평일2(화,수,목)"))
        
        ratio_map = df_pool.groupby("구분")["ratio"].mean().to_dict()
        df_target["pattern_ratio"] = df_target["구분"].map(ratio_map)
        df_target["pattern_ratio"] = df_target["pattern_ratio"] / df_target["pattern_ratio"].sum()
    else:
        df_target["pattern_ratio"] = 1.0 / last_day

    # ---------------------------------------------------------
    # [Ratio B] 기온 예측 기반 비율 계산 (Ver.3)
    # ---------------------------------------------------------
    model_temp, model_supply = train_models(df_daily)
    stats_map = get_past_stats(df_daily, target_month)
    
    # st.data_editor로 화면에서 받은 기온 데이터를 적용하기 위해 일단 기본값 세팅
    df_target["최저기온(℃)"] = df_target["일"].map(lambda d: stats_map.get(d, (0,0))[0])
    df_target["최고기온(℃)"] = df_target["일"].map(lambda d: stats_map.get(d, (0,0))[1])
    
    return df_target, monthly_plan_total, model_temp, model_supply

# ─────────────────────────────────────────────
# 5. UI 및 메인 로직
# ─────────────────────────────────────────────
def main():
    st.title("🔥 도시가스 공급량 재분배 (Pattern + Temp Hybrid)")
    
    df_daily = load_daily_data()
    df_plan = load_monthly_plan()
    cal_df = load_effective_calendar()
    
    if df_daily.empty or df_plan.empty:
        st.error("엑셀 파일들을 불러오지 못했습니다. 경로를 확인해주세요.")
        return

    # 사이드바 설정
    with st.sidebar:
        st.header("📅 분석 설정")
        years_plan = sorted(df_plan["연"].dropna().unique())
        target_year = int(st.selectbox("목표 연도", years_plan, index=len(years_plan)-1 if years_plan else 0))
        target_month = st.selectbox("목표 월", range(1, 13), index=0)
        recent_window = st.slider("최근 N년 패턴 참조", 1, 5, 3)
        
        st.markdown("---")
        st.subheader("⚖️ 하이브리드 가중치")
        temp_weight = st.slider("기온 예측치 반영 비율(%)", 0, 100, 50, step=10) / 100.0
        st.caption("0%: 100% 요일패턴 분배\n100%: 100% 기온예측 분배")

    # 기본 데이터 프레임 뼈대 생성
    df_target, monthly_plan_total, model_temp, model_supply = make_hybrid_daily_plan(
        df_daily, df_plan, cal_df, target_year, target_month, recent_window, temp_weight
    )

    st.markdown(f"### 🌡️ 1. {target_month}월 기온 예보 입력 (Ver.3 로직)")
    st.info("과거 3년 평균 기온이 기본 입력되어 있습니다. 기상청 예보에 맞춰 기온을 수정하면 즉시 분배량에 반영됩니다.")
    
    edited_temps = st.data_editor(
        df_target[["일자", "구분", "최저기온(℃)", "최고기온(℃)"]],
        hide_index=True,
        use_container_width=True,
        column_config={"일자": st.column_config.DateColumn(format="MM-DD", disabled=True)}
    )
    
    # 기온에 따른 예상 공급량 계산
    df_target["최저기온(℃)"] = edited_temps["최저기온(℃)"]
    df_target["최고기온(℃)"] = edited_temps["최고기온(℃)"]
    
    pred_avg = model_temp.predict(df_target[["최저기온(℃)", "최고기온(℃)"]])
    df_target["평균기온(℃)"] = pred_avg
    pred_supply = model_supply.predict(df_target[["평균기온(℃)"]])
    
    # 음수 방지 및 비율 산출
    pred_supply = np.where(pred_supply < 0, 0, pred_supply)
    sum_pred = pred_supply.sum()
    df_target["temp_ratio"] = pred_supply / sum_pred if sum_pred > 0 else (1.0 / len(df_target))
    
    # ---------------------------------------------------------
    # [최종 Hybrid 분배]
    # ---------------------------------------------------------
    df_target["최종_일별비율"] = (df_target["pattern_ratio"] * (1 - temp_weight)) + (df_target["temp_ratio"] * temp_weight)
    df_target["최종_일별비율"] = df_target["최종_일별비율"] / df_target["최종_일별비율"].sum() # 정규화
    
    df_target["예상공급량(MJ)"] = (df_target["최종_일별비율"] * monthly_plan_total).round(0)
    df_target["예상공급량(GJ)"] = df_target["예상공급량(MJ)"].apply(mj_to_gj)
    df_target["보정_예상공급량(GJ)"] = df_target["예상공급량(GJ)"].copy()
    
    # 이상치(Outlier) 바운드 계산 (Ver.2 로직)
    df_target["WeekNum"] = df_target["일자"].dt.isocalendar().week
    df_target["Group_Mean"] = df_target.groupby(["WeekNum", "is_weekend"])["예상공급량(MJ)"].transform("mean")
    df_target["Bound_Upper"] = (df_target["Group_Mean"] * 1.10).apply(mj_to_gj)
    df_target["Bound_Lower"] = (df_target["Group_Mean"] * 0.90).apply(mj_to_gj)
    df_target["is_outlier"] = (df_target["예상공급량(GJ)"] > df_target["Bound_Upper"]) | (df_target["예상공급량(GJ)"] < df_target["Bound_Lower"])

    # ---------------------------------------------------------
    # 결과 시각화
    # ---------------------------------------------------------
    st.divider()
    st.markdown(f"### 📊 2. {target_year}년 {target_month}월 일별 공급량 분배 결과 (가중치 적용)")
    col1, col2 = st.columns(2)
    col1.metric(f"월간 목표 총 공급량", f"{mj_to_gj(monthly_plan_total):,.0f} GJ")
    col2.metric(f"적용된 기온 가중치", f"{temp_weight*100:.0f}%")
    
    fig = go.Figure()
    # 요일별 컬러
    colors = {"평일1(월,금)": "#1F77B4", "평일2(화,수,목)": "#87CEFA", "주말/공휴일": "#D62728"}
    for cat in colors.keys():
        sub = df_target[df_target["구분"] == cat]
        fig.add_trace(go.Bar(x=sub["일"], y=sub["예상공급량(GJ)"], name=cat, marker_color=colors[cat]))
        
    fig.add_trace(go.Scatter(x=df_target["일"], y=df_target["Bound_Upper"], mode='lines', line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=df_target["일"], y=df_target["Bound_Lower"], mode='lines', fill='tonexty', fillcolor='rgba(100,100,100,0.3)', name='±10% 범위'))
    fig.add_trace(go.Scatter(x=df_target["일"], y=df_target["평균기온(℃)"], name='추정 평균기온', mode='lines+markers', line=dict(color='red', dash='dot'), yaxis='y2'))
    
    outliers = df_target[df_target["is_outlier"]]
    if not outliers.empty:
        fig.add_trace(go.Scatter(x=outliers["일"], y=outliers["예상공급량(GJ)"], mode='markers', marker=dict(color='black', symbol='x', size=10), name='이상치(Outlier)'))

    fig.update_layout(
        xaxis_title="일", yaxis=dict(title="공급량(GJ)"), yaxis2=dict(title="기온(℃)", overlaying="y", side="right"),
        barmode="overlay", legend=dict(orientation="h", y=1.1)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # 다운로드
    st.markdown("### 💾 3. 최종 데이터 다운로드")
    buffer = BytesIO()
    dl_df = df_target[["일자", "구분", "최저기온(℃)", "최고기온(℃)", "평균기온(℃)", "pattern_ratio", "temp_ratio", "최종_일별비율", "예상공급량(MJ)", "예상공급량(GJ)"]].copy()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        dl_df.to_excel(writer, index=False, sheet_name="Hybrid_일별계획")
    st.download_button("📥 하이브리드 일별계획 다운로드", data=buffer.getvalue(), file_name=f"{target_year}년_{target_month}월_하이브리드_공급계획.xlsx", mime="application/vnd.ms-excel")

if __name__ == "__main__":
    main()
