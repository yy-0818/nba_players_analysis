import pandas as pd
import streamlit as st
import logging
import os
from pygwalker.api.streamlit import StreamlitRenderer, init_streamlit_comm

st.set_page_config(
    layout="wide",
    page_title='NBA player data analysis',
    page_icon = '🏀',
)
 
st.title("在 Streamlit 中使用 Pygwalker 分析NBA球员数据")

# 初始化Streamlit与Pygwalker的通信
init_streamlit_comm()


@st.cache_data
def read_default_data() -> pd.DataFrame:
    current_script_path = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_script_path,"..","data", "all_seasons.csv")
    df = pd.read_csv(file_path)
    return df

# 使用st.cache_resource装饰器来缓存渲染器创建函数
@st.cache_resource
def get_pyg_renderer(df) -> "StreamlitRenderer":
    return StreamlitRenderer(df, spec="./gw_config.json", debug=False)

def main():
    st.sidebar.title('上传部分')
    uploaded_file = st.sidebar.file_uploader("上传你的数据文件", type="csv")

    if uploaded_file is not None:
        # 如果用户上传了文件，则使用该文件创建渲染器
        df = pd.read_csv(uploaded_file)
        st.sidebar.success("文件上传成功.")
    else:
        # 如果用户没有上传文件，则读取默认数据文件
        df = read_default_data()

    # 创建渲染器并渲染数据
    renderer = get_pyg_renderer(df)
    renderer.render_explore()
    
if __name__ == '__main__':
    logging.basicConfig(level=logging.CRITICAL)
    main()