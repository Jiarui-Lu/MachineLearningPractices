import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf


class Read_fund(object):
    def getNavRt(self):
        fundNav = pd.read_excel('净值表.xls', index_col=0).dropna(axis=1)
        return fundNav.pct_change().dropna()

    def getFactorRt(self):
        factor = pd.read_excel('因子数据.xls', index_col=0)
        return factor.pct_change().dropna()


class Fama_French_reg(object):
    def __init__(self, NavRt, FactorRt):
        self.NavRt = NavRt
        self.FactorRt = FactorRt

    # 获得无风险利率日度收益率
    def getRisklessRt(self, Nav):
        r_one_year = 0.015  # 1年期银行定存利率
        r_one_day = r_one_year / 360
        riskless = pd.DataFrame(r_one_day, index=Nav.index, columns=['r_one_day'])
        return riskless

    # 利用指数计算市场因子、规模因子和价值因子
    def CalFactor(self, tmp):
        tmp['HML'] = tmp['399371.SZ'] - tmp['399370.SZ']
        tmp['SMB'] = tmp['399316.SZ'] - tmp['399314.SZ']
        tmp['Market'] = (tmp['000001.SH'] + tmp['399001.SZ']) / 2
        return tmp

    # Fama-French三因子回归
    def Linear_reg_fit(self):
        Factor = self.CalFactor(self.FactorRt)
        Nav = self.NavRt
        Riskless = self.getRisklessRt(self.NavRt)
        dependent = pd.DataFrame()
        for col in Nav.columns:
            dependent = pd.concat([dependent, Nav[col] - Riskless['r_one_day']], axis=1)
        dependent.columns = Nav.columns
        independent = pd.DataFrame(index=Factor.index)
        independent['Market'] = Factor['Market'] - Riskless['r_one_day']
        independent['HML'] = Factor['HML']
        independent['SMB'] = Factor['SMB']
        # print(independent)
        # print(dependent)
        coef = pd.DataFrame()
        for col in dependent.columns:
            dependent_tmp = dependent[col].T
            lin_reg = LinearRegression()
            lin_reg.fit(independent, dependent_tmp)
            # print(lin_reg.coef_,lin_reg.intercept_)
            coef = pd.concat([coef, pd.DataFrame(lin_reg.coef_)], axis=1)
        coef.index = independent.columns
        coef.columns = dependent.columns
        return coef


class ML_process(object):
    def __init__(self, HML, SMB):
        self.HML = HML
        self.SMB = SMB

    # 归一化处理
    def scaler_01(self, array):
        array = array.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        x_minmax = scaler.fit_transform(array)
        return x_minmax

    # 聚类运算+聚类图像展示
    def Cluster(self, a_array, b_array):
        a_array = self.scaler_01(a_array)
        b_array = self.scaler_01(b_array)
        c_array = np.array([a_array, b_array]).reshape(-1, 2)
        km = KMeans(n_clusters=9)  # 九宫格聚类
        result = km.fit(c_array)
        labels = result.labels_  # 获取聚类标签
        return labels

    def Cluster_plot(self, labels, a_array, b_array):
        a_array = self.scaler_01(a_array)
        b_array = self.scaler_01(b_array)
        c_array = np.array([a_array, b_array]).reshape(-1, 2)
        color = ['red', 'black', 'green', 'blue', 'yellow', 'orange', 'aqua', 'purple', 'indigo']
        X = c_array.T[0]
        Y = c_array.T[1]
        # plt.scatter(X,Y)
        label_iter = 0  # 计数器
        for label in labels:
            plt.scatter(X[label_iter], Y[label_iter], color=color[label])
            label_iter = label_iter + 1
        plt.xlabel('HML')
        plt.ylabel('SMB')
        plt.show()


# MLP神经网络识别基金风格
class MLP_NN(object):
    def __init__(self, category, dim):
        self.category = category  # y有几种标签
        self.dim = dim  # x的维度数据

    def train_test_process(self, x, y, test_size):
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
        y_train_one_hot = tf.keras.utils.to_categorical(y_train, num_classes=self.category)
        y_test_one_hot = tf.keras.utils.to_categorical(y_test, num_classes=self.category)
        return (X_train, X_test, y_train_one_hot, y_test_one_hot)

    def MLP_model(self, x_train, y_train, x_test, y_test):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.relu, input_dim=self.dim))
        model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.relu))
        model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.relu))
        model.add(tf.keras.layers.Dense(units=self.category, activation=tf.nn.softmax))
        # 编译
        model.compile(optimizer='adam', loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])
        # 训练
        model.fit(x=x_train, y=y_train, epochs=200, batch_size=128)
        score = model.evaluate(x_test, y_test, batch_size=128)
        predict = model.predict(x_test)
        return (score, predict)


if __name__ == '__main__':
    x = Read_fund()
    NavRt = x.getNavRt()
    FactorRt = x.getFactorRt()
    reg = Fama_French_reg(NavRt, FactorRt)
    coef = reg.Linear_reg_fit()
    HML = np.array(coef.loc['HML', :])
    SMB = np.array(coef.loc['SMB', :])
    ml = ML_process(HML, SMB)
    HML = ml.scaler_01(HML)
    SMB = ml.scaler_01(SMB)
    X = np.array([HML, SMB]).reshape(-1, 2)
    y = ml.Cluster(HML, SMB)
    # print(X_train,X_test,y_train,y_test)
    MLP = MLP_NN(category=9, dim=2)  # 9种标签，2个维度
    X_train, X_test, y_train, y_test = MLP.train_test_process(X, y, 0.2)
    score, predict = MLP.MLP_model(X_train, y_train, X_test, y_test)
    print(score, predict)
    ml.Cluster_plot(y, HML, SMB)
