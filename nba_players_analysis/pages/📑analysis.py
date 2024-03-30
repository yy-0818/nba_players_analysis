import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sea
import plotly.express as px
import plotly.graph_objs as go
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import os

st.title('NBA 最佳得分手分析')
st.sidebar.info('该项目可以帮助你理解NBA球员数据，并提供一些分析结果。')
# 假设原始数据CSV文件位于同一目录下
@st.cache_data
def read_default_data():
    current_script_path = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_script_path,"..","data", "new_seasons.csv")
    df = pd.read_csv(file_path)
    return df


df = read_default_data()

NumTopScorers = 10
top_scorers_names = df.groupby(['name'])['pts'].sum().sort_values(ascending=False).head(NumTopScorers).index
topp = df[df['name'].isin(top_scorers_names)][['name', 'team', 'age', 'draft_year', 'year', 'total_pts', 'total_ast', 'total_reb']].copy()

# 用于收集每个球员缺失年份数据的列表
all_missing_years = []

# 填充缺失的数据
for p in top_scorers_names:
    player_data = topp[topp['name'] == p]
    first_year = player_data['year'].min()
    last_row = player_data[player_data['year'] == first_year].to_dict('records')[0]

    for y in range(1996, 2022 + 1):
        if y < first_year:
            continue
        year_data = player_data[player_data['year'] == y]
        if year_data.empty:
            row = last_row.copy()
            row['year'] = y
            all_missing_years.append(row)
        else:
            last_row = year_data.to_dict('records')[0]

# 拼接缺失年份的数据至topp DataFrame
missing_years_df = pd.DataFrame(all_missing_years)
combined_data = pd.concat([topp, missing_years_df]).drop_duplicates().reset_index(drop=True)
combined_data.sort_values(['draft_year', 'name', 'year'], inplace=True)


def bubbleplot(dataset, x_column, y_column, bubble_column, z_column=None,
               time_column=None, size_column=None, color_column=None,   
               x_logscale=False, y_logscale=False, z_logscale=False, 
               x_range=None, y_range=None, z_range=None, 
               x_title=None, y_title=None, z_title=None, title=None, colorbar_title=None,
               scale_bubble=1, colorscale=None, marker_opacity=None, marker_border_width=None, 
               show_slider=True, show_button=True, show_colorbar=True, show_legend=None,
               width=None, height=None):
    ''' Makes the animated and interactive bubble charts from a given dataset.'''
    
    # Set category_column as None and update it as color_column only in case
    # color_column is not None and categorical, in which case set color_column as None
    category_column = None
    if color_column: # Can be numerical or categorical
        if dataset[color_column].dtype.name in ['category', 'object', 'bool']:
            category_column = color_column
            color_column = None
        
    # Set the variables for making the grid
    if time_column:
        years = dataset[time_column].unique()
    else:
        years = None
        show_slider = False
        show_button = False
        
    column_names = [x_column, y_column]
    
    if z_column:
        column_names.append(z_column)
        axes3D = True
    else:
        axes3D = False
    
    column_names.append(bubble_column)
    
    if size_column:
        column_names.append(size_column)
    
    if color_column:
        column_names.append(color_column)
        
    # Make the grid
    if category_column:
        categories = dataset[category_column].unique()
        col_name_template = '{}+{}+{}_grid'
        grid = make_grid_with_categories(dataset, column_names, time_column, category_column, years, categories)
        if show_legend is None:
            showlegend = True
        else: 
            showlegend = show_legend
    else:
        col_name_template = '{}+{}_grid'
        grid = make_grid(dataset, column_names, time_column, years)
        if show_legend is None:
            showlegend = False
        else: 
            showlegend = show_legend
        
    # Set the layout
    if show_slider:
        slider_scale = years
    else:
        slider_scale = None
                
    figure, sliders_dict = set_layout(x_title, y_title, z_title, title, 
                                x_logscale, y_logscale, z_logscale, axes3D,
                                show_slider, slider_scale, show_button, showlegend, width, height)
    
    if size_column:
        sizeref = 2.*max(dataset[size_column])/(scale_bubble*80**2) # Set the reference size for the bubbles
    else:
        sizeref = None
        
    # Add the frames
    if category_column:
        # Add the base frame
        for category in categories:
            if time_column:
                year = min(years) # The earliest year for the base frame
                col_name_template_year = col_name_template.format(year, {}, {})
            else:
                col_name_template_year = '{}+{}_grid'
            trace = get_trace(grid, col_name_template_year, x_column, y_column, 
                              bubble_column, z_column, size_column, 
                              sizeref, scale_bubble, marker_opacity, marker_border_width,
                              category=category)
            if z_column:
                trace['type'] = 'scatter3d'
            figure['data'].append(trace)
           
        # Add time frames
        if time_column: # Only if time_column is not None
            for year in years:
                frame = {'data': [], 'name': str(year)}
                for category in categories:
                    col_name_template_year = col_name_template.format(year, {}, {})
                    trace = get_trace(grid, col_name_template_year, x_column, y_column, 
                                      bubble_column, z_column, size_column, 
                                      sizeref, scale_bubble, marker_opacity, marker_border_width,
                                      category=category)
                    if z_column:
                        trace['type'] = 'scatter3d'
                    frame['data'].append(trace)

                figure['frames'].append(frame) 

                if show_slider:
                    add_slider_steps(sliders_dict, year)
                
    else:
        # Add the base frame
        if time_column:
            year = min(years) # The earliest year for the base frame
            col_name_template_year = col_name_template.format(year, {})
        else:
            col_name_template_year = '{}_grid'
        trace = get_trace(grid, col_name_template_year, x_column, y_column, 
                          bubble_column, z_column, size_column, 
                          sizeref, scale_bubble, marker_opacity, marker_border_width,
                          color_column, colorscale, show_colorbar, colorbar_title)
        if z_column:
                trace['type'] = 'scatter3d'
        figure['data'].append(trace)
        
        # Add time frames
        if time_column: # Only if time_column is not None
            for year in years:
                col_name_template_year = col_name_template.format(year, {})
                frame = {'data': [], 'name': str(year)}
                trace = get_trace(grid, col_name_template_year, x_column, y_column, 
                                  bubble_column, z_column, size_column, 
                                  sizeref, scale_bubble, marker_opacity, marker_border_width,
                                  color_column, colorscale, show_colorbar, colorbar_title)
                if z_column:
                    trace['type'] = 'scatter3d'
                frame['data'].append(trace)
                figure['frames'].append(frame) 
                if show_slider:
                    add_slider_steps(sliders_dict, year) 
    
    # Set ranges for the axes
    if x_range is None:
        x_range = set_range(dataset[x_column], x_logscale) 
    
    if y_range is None:
        y_range = set_range(dataset[y_column], y_logscale)
    
    if axes3D:
        if z_range is None:
            z_range = set_range(dataset[z_column], z_logscale)
        figure['layout']['scene']['xaxis']['range'] = x_range
        figure['layout']['scene']['yaxis']['range'] = y_range
        figure['layout']['scene']['zaxis']['range'] = z_range
    else:
        figure['layout']['xaxis']['range'] = x_range
        figure['layout']['yaxis']['range'] = y_range
        
    if show_slider:
        figure['layout']['sliders'] = [sliders_dict]
     
    return figure

def make_grid(dataset, column_names, time_column, years=None):
    '''Makes the grid for the plot as a pandas DataFrame by-passing the use of `plotly.grid_objs`
    that is unavailable in the offline mode for `plotly`. The grids are designed using the `col_name_template`
    from the `column_names` of the `dataset`.'''
    
    grid = pd.DataFrame()
    if time_column:
        col_name_template = '{}+{}_grid'
        if years is None:
            years = dataset[time_column].unique()

        for year in years:
            dataset_by_year = dataset[(dataset[time_column] == int(year))]
            for col_name in column_names:
                # Each column name is unique
                temp = col_name_template.format(year, col_name)
                if dataset_by_year[col_name].size != 0:
                    # grid = grid.append({'value': list(dataset_by_year[col_name]), 'key': temp}, ignore_index=True)
                    grid = pd.concat([grid, pd.DataFrame([{'value': list(dataset_by_year[col_name]), 'key': temp}])], ignore_index=True)
    else:
        # Check if this can be simplified
        for col_name in column_names:
            # Each column name is unique
            # grid = grid.append({'value': list(dataset[col_name]), 'key': col_name + '_grid'}, ignore_index=True)
             grid = pd.concat([grid, pd.DataFrame([{'value': list(dataset[col_name]), 'key': col_name + '_grid'}])], ignore_index=True)
        
    return grid

def make_grid_with_categories(dataset, column_names, time_column, category_column, years=None, categories=None):
    '''Makes the grid for the plot as a pandas DataFrame by-passing the use of plotly.grid_objs
    that is unavailable in the offline mode for plotly. The grids are designed using the `col_name_template`
    from the `column_names` of the `dataset` using the `category_column` for catergories.'''
    
    grid = pd.DataFrame()
    if categories is None:
        categories = dataset[category_column].unique()
    if time_column:
        col_name_template = '{}+{}+{}_grid'
        if years is None:
            years = dataset[time_column].unique()
            
        for year in years:
            for category in categories:
                dataset_by_year_and_cat = dataset[(dataset[time_column] == int(year)) & (dataset[category_column] == category)]
                for col_name in column_names:
                    # Each column name is unique
                    temp = col_name_template.format(year, col_name, category)
                    if dataset_by_year_and_cat[col_name].size != 0:
                        # grid = grid.append({'value': list(dataset_by_year_and_cat[col_name]), 'key': temp}, ignore_index=True)
                        grid = pd.concat([grid, pd.DataFrame([{'value': list(dataset_by_year_and_cat[col_name]), 'key': temp}])], ignore_index=True) 
    else:
        col_name_template = '{}+{}_grid'
        for category in categories:
            dataset_by_cat = dataset[(dataset[category_column] == category)]
            for col_name in column_names:
                # Each column name is unique
                temp = col_name_template.format(col_name, category)
                if dataset_by_cat[col_name].size != 0:
                        # grid = grid.append({'value': list(dataset_by_cat[col_name]), 'key': temp}, ignore_index=True)
                        grid = pd.concat([grid, pd.DataFrame([{'value': list(dataset_by_cat[col_name]), 'key': temp}])], ignore_index=True) 
        
    return grid

 
def set_layout(x_title=None, y_title=None, z_title=None, title=None,
            x_logscale=False, y_logscale=False, z_logscale=False, axes3D=False, 
            show_slider=True, slider_scale=None, show_button=True, show_legend=False,
            width=None, height=None):
    '''Sets the layout for the figure.'''
    
    # Define the figure object as a dictionary
    figure = {
        'data': [],
        'layout': {},
        'frames': []
    }
    
    # Start with filling the layout first
    if axes3D:
        figure = set_3Daxes(figure, x_title, y_title, z_title, 
            x_logscale, y_logscale, z_logscale)
    else:
        figure = set_2Daxes(figure, x_title, y_title, x_logscale, y_logscale)
        
    figure['layout']['title'] = title    
    figure['layout']['hovermode'] = 'closest'
    figure['layout']['showlegend'] = show_legend
    figure['layout']['margin'] = dict(b=50, t=50, pad=5)
    # 添加平滑曲线
    figure['layout']['transition'] = {
        'duration': 500,
        'easing': 'cubic-in-out'
    }
    if width:
        figure['layout']['width'] = width
    if height:
        figure['layout']['height'] = height
    
    # Add slider for the time scale
    if show_slider: 
        sliders_dict = add_slider(figure, slider_scale)
    else:
        sliders_dict = {}
    
    # Add a pause-play button
    if show_button:
        add_button(figure)
        
    # Return the figure object
    return figure, sliders_dict

def set_2Daxes(figure, x_title=None, y_title=None, x_logscale=False, y_logscale=False):
    '''Sets 2D axes'''
    
    figure['layout']['xaxis'] = {'title': x_title, 'autorange': False}
    figure['layout']['yaxis'] = {'title': y_title, 'autorange': False} 

    if x_logscale:
        figure['layout']['xaxis']['type'] = 'log'
    if y_logscale:
        figure['layout']['yaxis']['type'] = 'log'
        
    return figure
        
def set_3Daxes(figure, x_title=None, y_title=None, z_title=None, 
            x_logscale=False, y_logscale=False, z_logscale=False):
    '''Sets 3D axes'''
    
    figure['layout']['scene'] = {}
    figure['layout']['scene']['xaxis'] = {'title': x_title, 'autorange': False}
    figure['layout']['scene']['yaxis'] = {'title': y_title, 'autorange': False} 
    figure['layout']['scene']['zaxis'] = {'title': z_title, 'autorange': False} 

    if x_logscale:
        figure['layout']['scene']['xaxis']['type'] = 'log'
    if y_logscale:
        figure['layout']['scene']['yaxis']['type'] = 'log'
    if z_logscale:
        figure['layout']['scene']['zaxis']['type'] = 'log'
        
    return figure
        
def add_slider(figure, slider_scale):
    '''Adds slider for animation'''
    
    figure['layout']['sliders'] = {
        'args': [
            'slider.value', {
                'duration': 400,
                'ease': 'cubic-in-out'
            }
        ],
        'initialValue': min(slider_scale),
        'plotlycommand': 'animate',
        'values': slider_scale,
        'visible': True
    }
    
    sliders_dict = {
        'active': 0,
        'yanchor': 'top',
        'xanchor': 'left',
        'currentvalue': {
            'font': {'size': 20},
            'prefix': 'Year:',
            'visible': True,
            'xanchor': 'right'
        },
        'transition': {'duration': 300, 'easing': 'cubic-in-out'},
        'pad': {'b': 10, 't': 50},
        'len': 0.9,
        'x': 0.1,
        'y': 0,
        'steps': []
    }
    
    return sliders_dict

def add_slider_steps(sliders_dict, year):
    '''Adds the slider steps.'''
    # 修改每帧间隔时间
    slider_step = {'args': [
        [year],
        {
            'frame': {'duration': 500, 'redraw': True},
            'mode': 'immediate',
            'transition': {'duration': 500, 'easing': 'cubic-in-out'}
        }
    ],
    'label': str(year),
    'method': 'animate'}
    sliders_dict['steps'].append(slider_step)
    
def add_button(figure):
    '''Adds the pause-play button for animation'''
    
    figure['layout']['updatemenus'] = [
        {
            'buttons': [
                {
                    'args': [None, {'frame': {'duration': 500, 'redraw': False},
                             'fromcurrent': True, 'transition': {'duration': 300, 'easing': 'quadratic-in-out'}}],
                    'label': 'Play',
                    'method': 'animate'
                },
                {
                    'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate',
                    'transition': {'duration': 0}}],
                    'label': 'Pause',
                    'method': 'animate'
                }
            ],
            'direction': 'left',
            'pad': {'r': 10, 't': 87},
            'showactive': False,
            'type': 'buttons',
            'x': 0.1,
            'xanchor': 'right',
            'y': 0,
            'yanchor': 'top'
        }
    ]
    
def set_range(values, logscale=False): 
    ''' Finds the axis range for the figure.'''
    
    if logscale:
        rmin = min(np.log10(values))*0.97
        rmax = max(np.log10(values))*1.04
    else:
        rmin = min(values)*0.7
        rmax = max(values)*1.4
        
    return [rmin, rmax] 

def get_trace(grid, col_name_template, x_column, y_column, bubble_column, z_column=None, size_column=None, 
            sizeref=200000, scale_bubble=1, marker_opacity=None, marker_border_width=None,
            color_column=None, colorscale=None, show_colorbar=True, colorbar_title=None, category=None):
    ''' Makes the trace for the data as a dictionary object that can be added to the figure or time frames.'''
    
    trace = {
        'x': grid.loc[grid['key']==col_name_template.format(x_column, category), 'value'].values[0],
        'y': grid.loc[grid['key']==col_name_template.format(y_column, category), 'value'].values[0],
        'text': grid.loc[grid['key']==col_name_template.format(bubble_column, category), 'value'].values[0],
        'mode': 'markers'
        }
    
    if z_column:
        trace['z'] = grid.loc[grid['key']==col_name_template.format(z_column, category), 'value'].values[0]
        
    if size_column:
        trace['marker'] = {
            'sizemode': 'area',
            'sizeref': sizeref,
            'size': grid.loc[grid['key']==col_name_template.format(size_column, category), 'value'].values[0],
        }
    else:
        trace['marker'] = {
            'size': 10*scale_bubble,
        }
    if marker_opacity:
        trace['marker']['opacity'] = marker_opacity
    if marker_border_width:
        trace['marker']['line'] = {'width': marker_border_width}
    if color_column:
            trace['marker']['color'] = grid.loc[grid['key']==col_name_template.format(color_column), 'value'].values[0]
            trace['marker']['colorbar'] = {'title': colorbar_title}
            trace['marker']['colorscale'] = colorscale
    if category:
        trace['name'] = category
    return trace

x_axis = 'total_ast'
y_axis = 'total_reb'
bubble_column = 'name'
size_column = 'total_pts'
time_column = 'year'
title = 'NBA 最佳得分手：总助攻数与总篮板数'

# 调用 bubbleplot 函数
figure = bubbleplot(dataset=topp, x_column=x_axis, y_column=y_axis, bubble_column=bubble_column, 
                    time_column=time_column, size_column=size_column, title=title,
                    width=800, height=600)

st.plotly_chart(figure)




fig = go.Figure()
# 对每一年的数据进行处理
for year in topp['year'].unique():
    # 提取这一年的数据
    df_year = topp[topp['year'] == year]
    # 添加一个气泡图层
    fig.add_trace(go.Scatter(
        x=df_year[x_axis],  # x轴数据
        y=df_year[y_axis],  # y轴数据
        mode='markers',     # 设置为气泡图
        marker=dict(        # 设置气泡的属性
            size=df_year['total_pts'],  # 气泡的大小由总得分决定
            sizemode='area',   # 气泡的大小以面积方式展示
            sizeref=2.*max(df_year['total_pts'])/(40.**2),  # 设置气泡大小的参考值，使得气泡的大小在一个合适的范围内
            sizemin=4,  # 设置气泡的最小大小
        ),
        name=str(year),   # 设置图层的名称，这将作为图例的标签
        text=df_year['name'],  # 设置气泡的标签，这将在鼠标悬停时显示
    ))

# 设置标题和轴标签
fig.update_layout(
    title=f'NBA 最佳得分手： {min(topp["year"])} - {max(topp["year"])}',
    xaxis=dict(title=x_axis),
    yaxis=dict(title=y_axis),
    height=800,
)

st.plotly_chart(fig, use_container_width=True)


plt.rcParams['font.sans-serif'] = ['SimHei']  # 'SimHei'是一种支持中文的字体
plt.rcParams['axes.unicode_minus'] = False

# 
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
plt.suptitle("身高体重分布")

ax1.hist(df["height"], color='#9E96CF')
ax1.axvline(df["height"].mean(), color='#EF3D58', label="Mean Height")
ax1.set_xlabel("Height")
ax1.set_ylabel("Count")
ax2.hist(df["weight"], color='#9E96CF')
ax2.axvline(df["weight"].mean(), color='#EF3D58', label="Mean Weight")
ax2.set_xlabel("Weight")
ax2.set_ylabel("Count")
ax1.legend()
ax2.legend()

st.pyplot(fig)



fig0, ax = plt.subplots()
sea.regplot(data=df, x='height', y='weight', color='#2E0D49', ax=ax)
ax.set_xlabel("Height")
ax.set_ylabel("Weight")
# Plot 1: 每场比赛年龄线图
fig1 = plt.figure()
sea.lineplot(data=df, x='age', y='gp', errorbar=("se", 0.5))
plt.title("每场比赛的年龄分布")
plt.xlabel("Age")
plt.ylabel("Game Play")

col0, col1 = st.columns(2)
with col0:
    st.pyplot(fig0)
with col1:
    st.pyplot(fig1)

# Plot 2: 分组数据和重绘图
season_group = df.groupby(df['season']).agg({'age': 'mean', 'height': 'mean', 'weight': 'mean'})
season_group.reset_index(inplace=True)
fig2, ax = plt.subplots(1, 2, figsize=(12, 5))
sea.regplot(data=season_group, x='age', y='height', line_kws={'color': '#e07a5f'}, color='#0d3b66', ax=ax[0])
sea.regplot(data=season_group, x='age', y='weight', line_kws={'color': '#e07a5f'}, color='#0d3b66', ax=ax[1])
st.pyplot(fig2)
# Plot 3: 年龄和高度随季节变化的线性图
fig3, ax = plt.subplots(figsize=(8, 6))
ax.plot(season_group['season'], season_group['age'], color='#ed6a5a')
ax.set_xlabel('Season', labelpad=30, fontsize=15, fontname='Arial')
ax.set_ylabel('Age', labelpad=20, fontsize=13, fontname='Arial')
ax_right = ax.twinx()
ax_right.plot(season_group['season'], season_group['height'], color='#9bc1bc', label='Height per season')
ax_right.set_ylabel('Height', fontsize=13, labelpad=20)
ax_right.legend()
ax.set_xticklabels(season_group['season'], rotation='vertical')

# Plot 4: 年龄和体重随季节变化的线性图
fig4, ax = plt.subplots(figsize=(8, 6))
ax.plot(season_group['season'], season_group['age'], color='#0d3b66', label='Age per season')
ax.set_xlabel('Season', labelpad=30, fontsize=15, fontname='Arial')
ax.set_ylabel('Age', labelpad=20, fontsize=13, fontname='Arial')
ax.legend()
ax_right = ax.twinx()
ax_right.plot(season_group['season'], season_group['weight'], color='#f4d35e', label='Weight per season')
ax_right.set_ylabel('Weight', fontsize=13, labelpad=20)
ax_right.legend()
ax.set_xticklabels(season_group['season'], rotation='vertical')




col3, col4 = st.columns(2)
with col3:
    st.pyplot(fig3)
with col4:
    st.pyplot(fig4)


# 创建直方图
def create_histogram(df):
    plt.figure(figsize=(8, 8))
    ax = plt.axes()
    ax.hist(df.height, bins=25)
    ax.set(xlabel='球员身高 (cm)',
           ylabel='频率',
           title='球员身高分布')
    return ax

histogram_ax = create_histogram(df)


heatmap_data = df[['age', 'height', 'weight', 'gp', 'pts', 'reb', 'ast',
                         'net_rating', 'oreb_pct', 'dreb_pct', 'usg_pct', 'ts_pct', 'ast_pct']]
fig_dims = (10, 10)
fig, ax = plt.subplots(figsize=fig_dims)
sea.heatmap(heatmap_data.corr(), ax=ax)

col8, col9 = st.columns(2)
with col8:
    st.pyplot(histogram_ax.figure)
    
with col9:
    st.pyplot(fig)



# 分组计算平均值
teamGroup = df.groupby("team").aggregate({'age': 'mean', 'height': 'mean', 'weight': 'mean'})
teamGroup.reset_index(inplace=True)

# 分组计算平均值
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(19, 7))

# 主题
sea.set_theme(style='white')

# 年龄
sea.histplot(teamGroup, x='age', kde=True, stat="probability", ax=ax1, color='#3E001F')
ax1.set_xlabel('Age', fontsize=17, labelpad=20)
ax1.set_ylabel('Probability', fontsize=17, labelpad=20)

# 球员身高
sea.histplot(teamGroup, x='height', stat="probability", kde=True, ax=ax2, color='#557C55')
ax2.set_xlabel('Height', fontsize=17, labelpad=20)
ax2.set_ylabel('Probability', fontsize=17, labelpad=5)

# 球员体重
sea.histplot(teamGroup, x='weight', stat="probability", kde=True, ax=ax3, color='#7D0A0A')
ax3.set_xlabel('Weight', fontsize=17, labelpad=20)
ax3.set_ylabel('Probability', fontsize=17, labelpad=5)

st.pyplot(fig)


plt.rcParams['font.sans-serif'] = ['SimHei']
datx =df.drop(['oreb_pct','age','name','team','college','country','draft_year','draft_round','draft_number','season','gp','pts','reb','ast','net_rating','usg_pct','ts_pct','ast_pct'], axis=1)
x = datx['height']
y = datx['weight']
z = datx['dreb_pct']

fig = plt.figure(figsize=(8, 6), dpi=80)  # DPI设置为80以适应Streamlit的显示
ax = fig.add_subplot(111, projection='3d')

# 散点图，调整点大小和透明度
scatter = ax.scatter(x, y, z, c=z, cmap='viridis', s=50, edgecolors='w', alpha=0.7)

# 添加颜色条
cbar = plt.colorbar(scatter, shrink=0.5, aspect=10, label='防守篮板率(dreb_pct)')

# 拟合平面
A = np.vstack([x, y, np.ones_like(x)]).T
plane_coef, _, _, _ = np.linalg.lstsq(A, z, rcond=None)

# 为了创建网格，需要使用范围更广的x和y
x_surf = np.linspace(x.min(), x.max(), 50)
y_surf = np.linspace(y.min(), y.max(), 50)
x_surf, y_surf = np.meshgrid(x_surf, y_surf)
z_surf = plane_coef[0] * x_surf + plane_coef[1] * y_surf + plane_coef[2]

# 绘制拟合平面，调整透明度和颜色映射
ax.plot_surface(x_surf, y_surf, z_surf, color='m', alpha=0.3, edgecolor='none')

# 设置轴标签和标题
ax.set_xlabel('height', fontsize=12)
ax.set_ylabel('weight', fontsize=12)
# ax.set_zlabel('防守篮板率 (dreb_pct)', fontsize=10)
ax.set_title('身高、体重和防守篮板率的多元线性回归', fontsize=14)

# 使用 Streamlit 显示图表
st.pyplot(fig)