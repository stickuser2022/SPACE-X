#在本实验中，我们将执行一些探索性数据分析 (EDA) 以查找数据中的一些模式并确定训练监督模型的标签。
#在数据集中，有几种不同的助推器没有成功着陆的情况。有时尝试着陆但由于事故而失败；例如，True Ocean 表示任务结果成功降落到海洋的特定区域，
#而 False Ocean 表示任务结果未成功降落到海洋的特定区域。 True RTLS 表示任务结果成功降落到地面垫 False RTLS 表示任务结果未成功降落到地面垫。
#True ASDS 表示任务结果成功降落在无人机船上 False ASDS 表示任务结果未成功降落一艘无人机船。
#在本实验中，我们主要将这些结果转换为训练标签，1 表示助推器成功降落，0 表示失败。

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
# 允许我们将数据集分为训练集和测试集
from sklearn.model_selection import train_test_split
# 允许我们测试分类算法的参数并找到最佳算法
from sklearn.model_selection import GridSearchCV
# 逻辑回归
from sklearn.linear_model import LogisticRegression
# 支持向量机分类算法
from sklearn.svm import SVC
# 决策树算法
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
import plotly.express as px

#获取数据并查看前十
df=pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_1.csv")
df.head(10)

#识别并计算每个行列中缺失值的百分比
df.isnull().sum()/df.count()*100

#识别确定哪一类是数字型哪一列是类
df.dtypes
# 在LaunchSite列应用value_counts() 函数，以表现各个发射点发射的次数
df['LaunchSite'].value_counts()
#使用方法 .value_counts() 确定列 Orbit 中每个轨道的数量和出现次数
df['Orbit'].value_counts()

# 统计Outcome 列，该列为实际着陆成功与否的统计列，具体结果请看开头注释
landing_outcomes = df['Outcome'].value_counts()

landing_outcomes
#将每一种着陆结果遍历并指定一个索引
for i,outcome in enumerate(landing_outcomes.keys()):
    print(i,outcome)
#将着陆结果为失败的（不管是海上还是陆地）定义为bad_outcomes
bad_outcomes=set(landing_outcomes.keys()[[1,3,5,6,7]])
bad_outcomes

# 将成功着陆和失败着陆分别制定为1，0并赋予landing_class中
landing_class = []
for key,value in df['Outcome'].items():
    if value in bad_outcomes:
        landing_class.append(0)
    else:
        landing_class.append(1)
#将landing_class加入到数据集中成为新的1列，这样就可以直观的在数据集中直接看到是否着陆成功
df['Class']=landing_class
df[['Class']].head(20)

#计算均值来确定成功着陆的概率
df["Class"].mean()
#保存修改过的数据集
df.to_csv("dataset_part_2.csv", index=False)

'''''''SECTION2 '''''''''
'''''''可视化部分''''''''
#首先，我们可以绘制出 FlightNumber 与 PayloadMass 的关系，
#并叠加发射结果。我们看到，随着航班数量的增加，
#第一级成功降落的可能性更大。有效载荷质量也很重要；似乎有效载荷越大，第一级返回的可能性就越小
sns.catplot(y="PayloadMass", x="FlightNumber", hue="Class", data=df, aspect = 5)
plt.xlabel("Flight Number",fontsize=20)
plt.ylabel("Pay load Mass (kg)",fontsize=20)
plt.show()


#使用散点图，探索有效荷载VS发射地点的关系
sns.set_theme(color_codes=True)
plt.figure(figsize=(14,8))
sns.scatterplot(x="PayloadMass", y="LaunchSite", hue="Class", data = df)
plt.xlabel("Pay Load Mass (Kg)",fontsize=20)
plt.ylabel("Launch Site",fontsize=20)
plt.show()

#使用柱状图查看每个轨道的成功率
xh = df.groupby('Orbit')['Class'].mean()
ax = xh.plot(kind='bar', figsize=(14, 7), color='#86bf91', zorder=2, width=0.8)
ax.set_xlabel("Orbit", labelpad=20, weight='bold', size=17)
ax.set_ylabel("Sucess rate of each orbit", labelpad=20, weight='bold', size=17)

#使用折现图查看成功率和年份之间的关系
# 首先使用函数将年份从DATE列中提取出来并赋予year
year=[]
def Extract_year(date):
    for i in df["Date"]:
        year.append(i.split("-")[0])
    return year
# 创建折线图，X轴为year，Y轴为成功率
df_groupby_year=df.groupby("year",as_index=False)["Class"].mean()
sns.set(rc={'figure.figsize':(12,6)})
sns.lineplot(data=df_groupby_year, x="year", y="Class" )
plt.xlabel("Year",fontsize=15)
plt.title('Space X Rocket Success Rates', fontsize = 20)
plt.ylabel("Success Rate",fontsize=15)
plt.show()


#接下来我们将选取一些对着陆成功率有影响的列并赋予Features
features = df[['FlightNumber', 'PayloadMass', 'Orbit',
'LaunchSite', 'Flights', 'GridFins', 'Reused', 'Legs', 'LandingPad', 'Block', 'ReusedCount', 'Serial']]
features.head()

# 使用 get_dummies() 函数
features_one_hot = features

features_one_hot = pd.concat([features_one_hot,pd.get_dummies(df['Orbit'])],axis=1)
features_one_hot.drop(['Orbit'], axis = 1, inplace = True)

features_one_hot = pd.concat([features_one_hot,pd.get_dummies(df['LaunchSite'])],axis=1)
features_one_hot.drop(['LaunchSite'], axis = 1, inplace = True)

features_one_hot = pd.concat([features_one_hot,pd.get_dummies(df['LandingPad'])],axis=1)
features_one_hot.drop(['LandingPad'], axis = 1, inplace = True)

features_one_hot = pd.concat([features_one_hot,pd.get_dummies(df['Serial'])],axis=1)
features_one_hot.drop(['Serial'], axis = 1, inplace = True)

features_one_hot.head()

## 再使用 astype 函数将其转换为小数数据
features_one_hot = features_one_hot.astype(float)
features_one_hot
#最后保存修改的数据集
features_one_hot.to_csv('dataset_part_3.csv', index=False)


'''''''SECTION3 '''''''''
'''''''机器学习预测''''''''
data =features_one_hot
data.head()
#从 data 中的 Class 列创建一个 NumPy 数组，通过应用方法 to_numpy() 然后将其分配给变量 Y，确保输出是一个 Pandas 系列（只有一个括号 df['name of column']）
Y = data['Class'].to_numpy()

#标准化 X 中的数据，然后使用下面提供的转换将其重新分配给变量 X。
transform = preprocessing.StandardScaler()

x = transform.fit_transform(X)

#我们使用函数 train_test_split 将数据拆分为训练数据和测试数据。训练数据分为验证数据，第二组用于训练数据；然后使用函数 GridSearchCV 训练模型并选择超参数
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=2)
#我们可以看到我们只有18个测试样本
Y_test.shape

#创建一个逻辑回归对象，然后创建一个 网格搜索GridSearchCV 的对象 logreg_cv，cv = 10。拟合该对象以从dict参数中找到最佳参数
parameters ={"C":[0.01,0.1,1],'penalty':['l2'], 'solver':['lbfgs']}# l1 lasso l2 ridge
lr=LogisticRegression()

gsvc= GridSearchCV(lr,parameters, scoring = 'accuracy',cv=10)
logreg_cv = gsvc.fit(X_train, Y_train)

#我们输出用于逻辑回归的 GridSearchCV 对象。
#我们使用best_params_ 显示最佳参数，使用 best_score_ 显示验证数据的准确性。
print("tuned hpyerparameters :(best parameters) ",logreg_cv.best_params_)
print("accuracy :",logreg_cv.best_score_)

#使用方法得分计算测试数据的准确性
print("accuracy :",logreg_cv.best_score_)


'''''''SECTION4 '''''''''
'''''''仪表盘创建''''''''

spacex_df = pd.read_csv("spacex_launch_dash.csv")
max_payload = spacex_df['Payload Mass (kg)'].max()
min_payload = spacex_df['Payload Mass (kg)'].min()


app = dash.Dash(__name__)
server = app.server

uniquelaunchsites = spacex_df['Launch Site'].unique().tolist()
lsites = []
lsites.append({'label': 'All Sites', 'value': 'All Sites'})
for site in uniquelaunchsites:
 lsites.append({'label': site, 'value': site})



app.layout = html.Div(children=[html.H1('SpaceX Launch Records Dashboard',
                                        style={'textAlign': 'center', 'color': '#503D36',
                                               'font-size': 40}),




                                dcc.Dropdown(id='site_dropdown',options=lsites,placeholder='Select a Launch Site here', searchable = True , value = 'All Sites'),
                                html.Br(),


                                html.Div(dcc.Graph(id='success-pie-chart')),
                                html.Br(),

                                html.P("Payload range (Kg):"),

                                dcc.RangeSlider(
                                    id='payload_slider',
                                    min=0,
                                    max=10000,
                                    step=1000,
                                    marks = {
                                            0: '0 kg',
                                            1000: '1000 kg',
                                            2000: '2000 kg',
                                            3000: '3000 kg',
                                            4000: '4000 kg',
                                            5000: '5000 kg',
                                            6000: '6000 kg',
                                            7000: '7000 kg',
                                            8000: '8000 kg',
                                            9000: '9000 kg',
                                            10000: '10000 kg'
                                    },

                                    value=[min_payload,max_payload]
                                ),

                                html.Div(dcc.Graph(id='success-payload-scatter-chart')),

                                ])

@app.callback(
     Output(component_id='success-pie-chart',component_property='figure'),
     [Input(component_id='site_dropdown',component_property='value')]
)
def update_graph(site_dropdown):
    if (site_dropdown == 'All Sites'):
        df  = spacex_df[spacex_df['class'] == 1]
        fig = px.pie(df, names = 'Launch Site',hole=.3,title = 'Total Success Launches By all sites')
    else:
        df  = spacex_df.loc[spacex_df['Launch Site'] == site_dropdown]
        fig = px.pie(df, names = 'class',hole=.3,title = 'Total Success Launches for site '+site_dropdown)
    return fig

@app.callback(
     Output(component_id='success-payload-scatter-chart',component_property='figure'),
     [Input(component_id='site_dropdown',component_property='value'),Input(component_id="payload_slider", component_property="value")]
)
def update_scattergraph(site_dropdown,payload_slider):
    if site_dropdown == 'All Sites':
        low, high = payload_slider
        df  = spacex_df
        mask = (df['Payload Mass (kg)'] > low) & (df['Payload Mass (kg)'] < high)
        fig = px.scatter(
            df[mask], x="Payload Mass (kg)", y="class",
            color="Booster Version",
            size='Payload Mass (kg)',
            hover_data=['Payload Mass (kg)'])
    else:
        low, high = payload_slider
        df  = spacex_df.loc[spacex_df['Launch Site'] == site_dropdown]
        mask = (df['Payload Mass (kg)'] > low) & (df['Payload Mass (kg)'] < high)
        fig = px.scatter(
            df[mask], x="Payload Mass (kg)", y="class",
            color="Booster Version",
            size='Payload Mass (kg)',
            hover_data=['Payload Mass (kg)'])
    return fig



if __name__ == '__main__':
    app.run_server(debug=False)
