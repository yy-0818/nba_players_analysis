import logging
import os
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt


sns.set(style='whitegrid', font='SimHei')

st.set_page_config(
    layout="wide",
    page_title='NBA player data analysis',
    page_icon = '🏀',
)

def main():
    with st.sidebar:
        st.title('')
        st.info('该项目可以帮助你理解NBA球员数据，并提供一些分析结果。')
    st.markdown("<h1 style='text-align: center; color: black;'>NBA球员数据分析📋</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: black;'>NBA Players Analysis📋</h4>", unsafe_allow_html=True)

    current_dir = os.path.dirname(os.path.abspath(__file__))

    data_file = os.path.join(current_dir, 'data', 'all_seasons.csv')

    df = pd.read_csv(data_file)

    see_data = st.expander('查看原始数据 \ View the raw data 👉')
    with see_data:
        st.dataframe(data=df.reset_index(drop=True))

    st.markdown(
    """
    
    <br><br/>
    NBA球员数据分析项目旨在帮助用户更好地理解NBA球员数据，并提供一些分析结果。
    <br><br/>
    
    ### 数据来源
    
    - 数据来自1996年到2022年所有赛季的NBA球员数据。
    - 数据来源：https://www.basketball-reference.com/
    <br><br/>
    
    ### 分析结果
    
    - 统计分析：对数据进行统计分析，包括平均值、中位数、最大值、最小值等。
    - 相关性分析：分析不同指标之间的相关性。
    - 预测分析：使用机器学习算法（如线性回归、支持向量机、随机森林等）对数据进行预测。
    <br><br/>
    
    ### 项目目标
    
    - 帮助用户更好地理解NBA球员数据。
    - 提供一些分析结果，帮助用户做出决策。
    <br><br/>
    
    ### 项目特色
    
    - 数据来源广泛，覆盖多个赛季。
    - 采用多种统计分析和机器学习算法，提供全面的数据分析和预测结果。
    <br><br/>

    """
    , unsafe_allow_html=True)




if __name__ == '__main__':

    logging.basicConfig(level=logging.CRITICAL)

    main()




