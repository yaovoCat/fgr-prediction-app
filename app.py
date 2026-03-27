import streamlit as st
import joblib
import numpy as np
from datetime import datetime

st.set_page_config(page_title="FGR不良结局预测系统", page_icon="🏥", layout="wide")

st.markdown("""
<style>
    .main { background-color: #f5f7fa; }
    .title-box { background: #0052D9; padding: 20px; border-radius: 10px; color: white; margin-bottom: 20px; }
    .card { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return joblib.load("best_lightgbm_model.pkl")

model = load_model()

st.markdown("<div class='title-box'><h1>🏥 胎儿生长受限（FGR）不良妊娠结局预测系统</h1></div>", unsafe_allow_html=True)

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📋 患者基本信息")

    today = st.date_input("今日日期", datetime.today(), key="today")
    birthday = st.date_input("出生日期", datetime(1990, 1, 1), key="birthday")
    age = today.year - birthday.year - ((today.month, today.day) < (birthday.month, birthday.day))
    st.info(f"✅ 自动计算年龄：**{age} 岁**")

    parity = st.number_input("产次", min_value=0, max_value=10, value=1, key="parity")

    st.markdown("**既往胎盘相关妊娠并发症史**")
    st.caption("包括：FGR、子痫前期、死产或胎盘早剥等病史")
    history_placenta = st.selectbox("选择", [0, 1], format_func=lambda x: "无" if x == 0 else "有", key="history_placenta")

    st.markdown("**FGR 高危因素**")
    st.caption("包括：慢性高血压、慢性肾病、自身免疫病、妊娠前糖尿病等")
    fgr_risk = st.selectbox("选择", [0, 1], format_func=lambda x: "有" if x == 0 else "无", key="fgr_risk")

    st.markdown("#### ⚖️ 身高体重（自动算BMI）")
    height_cm = st.number_input("身高 (cm)", 120, 200, 160, key="height")
    weight_kg = st.number_input("体重 (kg)", 30, 150, 60, key="weight")
    bmi = weight_kg / ((height_cm / 100) ** 2)
    st.info(f"✅ BMI：**{bmi:.2f}**")

    st.subheader("📊 超声检查指标")

    st.markdown("#### 📅 FGR 确诊孕周")
    weeks = st.number_input("周", 20, 40, 30, key="weeks")
    days = st.number_input("天", 0, 6, 0, key="days")
    ga_fgr = weeks + days / 7
    st.info(f"✅ 换算孕周：**{ga_fgr:.2f} 周**")

    efw_percent = st.number_input("初诊EFW百分位（估计胎儿体重）", 0.0, 100.0, 50.0, key="efw")
    st.caption("💡 可利用微信小程序「孕算」辅助计算百分位数")

    st.markdown("**MCV-PI（大脑中动脉搏动指数，th%）**")
    mcv_pi = st.number_input("数值", 0.0, 10.0, 1.0, key="mcv_pi")

    st.markdown("**MCV-PSV（大脑中动脉收缩期峰值流速,MoM）**")
    mcv_psv = st.number_input("数值", 0.0, 10.0, 1.0, key="mcv_psv")

    st.markdown("**UA-EDF（脐血流舒张末期血流,th%）**")
    st.caption("1=正向 2=间歇性缺失 3=持续性缺失 4=倒置")
    ua_edf = st.selectbox("类型", [1,2,3,4], key="ua_edf")

    st.markdown("**UA-PI（脐动脉搏动指数,th%）**")
    ua_pi = st.number_input("数值", 0.0, 10.0, 1.0, key="ua_pi")

    st.markdown("**超声遗传标志物**")
    st.caption("包括：鼻骨缺失、NT增厚、肾盂扩张、肠回声增强、脉络膜囊肿等")
    echo_marker = st.selectbox("是否存在", [0,1], format_func=lambda x:"无"if x==0 else"有", key="echo_marker")

    st.markdown("**UtA-PI（子宫动脉搏动指数,th%）**")
    uta_pi = st.number_input("数值", 0.0, 10.0, 1.0, key="uta_pi")

    st.markdown("**子宫动脉切迹**")
    notch = st.selectbox("类型", [0,1,2], format_func=lambda x:{0:"无",1:"单侧",2:"双侧"}[x], key="notch")

    st.markdown("**羊水DVP（羊水最大深度，cm）**")
    amniotic = st.number_input("深度", 0.0, 20.0, 5.0, key="amniotic")

    predict_btn = st.button("🔍 开始预测", type="primary", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    if predict_btn:
        X = np.array([[
            age, parity, history_placenta, fgr_risk, bmi,
            ga_fgr, efw_percent, mcv_pi, mcv_psv, ua_edf,
            ua_pi, echo_marker, uta_pi, notch, amniotic
        ]])

        prob = model.predict_proba(X)[0, 1]
        pred = model.predict(X)[0]

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("📈 预测结果")
        st.metric("不良结局发生概率", f"{prob:.2%}")

        if pred == 1:
            st.error("⚠️ 高风险 → 建议加强监护、密切随访、及时干预")
        else:
            st.success("✅ 低风险 → 常规产检，定期监测胎儿生长即可")

        st.caption("⚠️ 本模型为机器学习辅助预测工具，仅供临床参考，不替代医师专业诊断")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("👈 请在左侧完整填写患者信息后，点击「开始预测」")