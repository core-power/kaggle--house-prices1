import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import sklearn
import numpy as np
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score,roc_auc_score
from time import time
warnings.filterwarnings('ignore')
train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')
# 挑选特征值
selected_features = ['LotArea','Foundation', 'Heating', 'Electrical', 'SaleType', 'SaleCondition', 'GarageArea','YearRemodAdd','YearBuilt','1stFlrSF', '2ndFlrSF', 'TotalBsmtSF', 'BsmtUnfSF', 'CentralAir']
X_train = train[selected_features]
X_test = test[selected_features]
y_train = train['SalePrice']
print(X_train.describe())
# 显示数据
#print(y_train['SalePrice'].describe())
print(y_train)
fig, ax = plt.subplots()
ax.scatter(x = train['GrLivArea'], y = train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
#plt.show()
# 补充特征缺失值
print(X_train['Electrical'].unique())
X_train["Electrical"] = train["Electrical"].fillna('SBrkr')
#loc通过索引获取数据
train.loc[train["Electrical"]=="SBrkr","Electrical"] = 0
train.loc[train["Electrical"]=="FuseF","Electrical"] = 1
train.loc[train["Electrical"]=="FuseA","Electrical"] = 2
train.loc[train["Electrical"]=="FuseP","Electrical"] = 3
train.loc[train["Electrical"]=="Mix","Electrical"] = 4
print(X_train['SaleType'].unique())
X_train["SaleType"] = train["SaleType"].fillna('WD', inplace=True)
#loc通过索引获取数据
train.loc[train["SaleType"]=="WD","SaleType"] = 0
train.loc[train["SaleType"]=="New","SaleType"] = 1
train.loc[train["SaleType"]=="COD","SaleType"] = 2
train.loc[train["SaleType"]=="ConLD","SaleType"] = 3
train.loc[train["SaleType"]=="ConLI","SaleType"] = 4
train.loc[train["SaleType"]=="CWD","SaleType"] = 5
train.loc[train["SaleType"]=="ConLw","SaleType"] = 6
train.loc[train["SaleType"]=="Con","SaleType"] = 7
train.loc[train["SaleType"]=='Oth',"SaleType"] = 8
X_train['LotArea'].fillna(X_train['LotArea'].mean(), inplace=True)
X_train['GarageArea'].fillna(X_train['GarageArea'].mean(), inplace=True)
X_train['TotalBsmtSF'].fillna(X_train['TotalBsmtSF'].mean(), inplace=True)
X_train['BsmtUnfSF'].fillna(X_train['BsmtUnfSF'].mean(), inplace=True)
X_test['Electrical'].fillna('SBrkr', inplace=True)
X_test['SaleType'].fillna('WD', inplace=True)
X_test['GarageArea'].fillna(X_test['GarageArea'].mean(), inplace=True)
X_test['TotalBsmtSF'].fillna(X_test['TotalBsmtSF'].mean(), inplace=True)
X_test['BsmtUnfSF'].fillna(X_test['BsmtUnfSF'].mean(), inplace=True)

print(X_train.info())
print(X_test.info())

# 采用DictVectorizer进行特征向量化
from sklearn.feature_extraction import DictVectorizer
dict_vec = DictVectorizer(sparse=False)

X_train = dict_vec.fit_transform(X_train.to_dict(orient='record'))
X_test = dict_vec.transform(X_test.to_dict(orient='record'))
'''# 使用随机森林回归模型进行 回归预测
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
rfr = RandomForestRegressor(random_state=20,n_estimators=7000,min_samples_split=6,min_samples_leaf=2)
#rfr2= GradientBoostingRegressor(n_estimators=1000,learning_rate=0.5,subsample=0.8)
rfr=rfr.fit(X_train, y_train)
#rfr2=rfr2.fit(X_train, y_train)
#rfr=(rfr1+rfr2)/2
rfr_y_predict= rfr.predict(X_test)
print('-------------------------------------------')
'''
print('开始训练模型，启动')
print('*************************')
#使用xgboost模型进行预测
import xgboost as xgb
from sklearn.metrics import make_scorer,mean_squared_error
RMSE = make_scorer(mean_squared_error, greater_is_better=False)
xgbreg = xgb.XGBRegressor(seed=0)
param_grid = {
    'base_score':[0.5,0.6,0.7,0.8],
    'colsample_bylevel':[0.7,0.75,0.8,0.85,0.9],
    'n_estimators': [1000],
    'learning_rate': [ 0.05],
    'min_child_weight':[4],
    'max_delta_step':[0],
    'missing':[None],
    'nthread':[-1],
    'reg_alpha':[100],
    'reg_lambda':[0.8],
    'scale_pos_weight':[1],
    'silent':[True],
    'gamma':[0],
    'max_depth': [6],
    'subsample': [ 0.8],
   'colsample_bytree': [0.75,0.8,0.85,0.9,0.95,1.0],
    }
rfr = GridSearchCV(estimator=xgbreg, param_grid=param_grid, n_jobs=1, cv=10,scoring=RMSE )
rfr.fit(X_train, y_train)
rfr_y_predict= rfr.predict(X_test)

#进行交叉验证
#kf = sklearn.cross_validation.KFold(train.shape[0],n_folds=3,random_state=1)
#scores = sklearn.cross_validation.cross_val_score(rfr,train[selected_features],train["SalePrice"],cv=kf)
#print(scores.mean())
# 输出结果

rfr_submission = pd.DataFrame({'Id': test['Id'], 'SalePrice': rfr_y_predict})
rfr_submission.to_csv('HP_submission.csv', index=False)
print('模型训练完成，over')