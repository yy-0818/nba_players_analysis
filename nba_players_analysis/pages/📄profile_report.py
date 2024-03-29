import streamlit as st
import pandas as pd
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report


st.set_page_config(
    layout="wide",
    page_title='NBA player data analysis',
    page_icon = '🏀',
)
# Streamlit 应用界面
st.title('Data Analysis with ydata_profiling')

# 用户上传 CSV 文件
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # 读取 CSV 文件为 DataFrame
    df = pd.read_csv(uploaded_file)
    
    # 生成数据分析报告
    profile = ProfileReport(df, title='Pandas Profiling Report', explorative=True)
    
    # 将报告显示在 Streamlit 应用中
    st_profile_report(profile)

else:
    st.write("Please upload a CSV file to generate the report.")