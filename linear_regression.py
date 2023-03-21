import numpy as np
class LinearRegression:
    def __init__(self,data,labels,polynomial_degree=0,sinusoid_degree=0,normalize_data=True)
        self.data=data
        self.labels=labels
        self.polynomial_degree=polynomial_degree
        self.sinusoid_degree=sinusoid_degree
        self.normalize_data=normalize_data
        num_features=self.data.shape[1]
        self.theta=np.zeros((num_features))

    def train(self,alpha,num_iteratios=500):
        cost_history=self.gradient_descent(alpha,num_iteratios)
        return self.theta,cost_history
    
    def gradient_descent(self,alpha,num_iterations):
        cost_history=[]
        for i in range(num_iterations):
            self.gradient_step(alpha)
            cost_history.append(self.cost_function(self.data,self.lables))
        return cost_history
    
    def gradient_step(self,alpha):
        '''梯度下降参数更新计算方法，矩阵运算'''
        num_examples=self.data.shape[0]
        prediction=LinearRegression.hypothesis(self.data,self.theta)
        delta=prediction-self.labels
        theta=self.theta
        theta=theta-alpha/num_examples*(np.dot(delta.T,self.data)).T
        self.theta=theta
    
    def hypothesis(data,theta):
        predictions=np.dot(data,theta)
        return predictions
    
    def cost_function(self,data,labels):
        num_examples=data.shape[0]
        delta=LinearRegression.hypothesis(self.data,self.theta)-labels
        cost=(1/2)*np.dot(delta.T,delta) 
        return cost[0][0]
    
    def predict(self,data):
        '''预测回归值结果'''
        predictions=LinearRegression.hypothesis(data,self.theta)
        return predictions