import streamlit as st
import pandas as pd
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import os


st.set_page_config(
    layout="wide",
    page_title='NBA player data analysis',
    page_icon = '🏀',
)
# Streamlit 应用界面
st.title('使用ydata_profiling进行数据分析')


@st.cache_data
def read_default_data() -> pd.DataFrame:
    current_script_path = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_script_path,"..","data", "all_seasons.csv")
    df = pd.read_csv(file_path)
    return df

def main():
    st.sidebar.title('')
    # 用户上传 CSV 文件
    uploaded_file = st.sidebar.file_uploader("上传 CSV 文件", type="csv")

    if uploaded_file is not None:
        # 如果用户上传了文件，则使用该文件创建渲染器
        df = pd.read_csv(uploaded_file)
        st.sidebar.success("文件上传成功.")
        # 生成数据分析报告
        profile = ProfileReport(df, explorative=True)
        # 将报告显示在 Streamlit 应用中
        st_profile_report(profile)
    else:
        # 如果用户没有上传文件，则读取默认数据文件
        df = read_default_data()
        profile = ProfileReport(df, explorative=True)
        st_profile_report(profile)
        
        
if __name__ == "__main__":
    main()

