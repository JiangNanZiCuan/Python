import numpy as np #导入numpy库并重命名为np
import pandas as pd #导入pandas库并重命名为pd
df = pd.read_csv('train.csv') 
# 数据清洗
def clean_data(df):
    # 创建副本
    df_clean = df.copy()
    
    # 删除不需要的列
    columns_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin']
    df_clean = df_clean.drop(columns=columns_to_drop)
    
    # 处理缺失值
    # Age用中位数填充
    age_median = df_clean['Age'].median()
    df_clean['Age'] = df_clean['Age'].fillna(age_median)
    
    # Embarked用众数填充
    embarked_mode = df_clean['Embarked'].mode()[0]
    df_clean['Embarked'] = df_clean['Embarked'].fillna(embarked_mode)
    
    # 编码分类变量
    df_clean['Sex'] = df_clean['Sex'].map({'male': 0, 'female': 1})
    df_clean['Embarked'] = df_clean['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    
    return df_clean
df_clean = clean_data(df)
train_size = 600
train_data = df_clean.iloc[:train_size]  # 前600行
test_data = df_clean.iloc[train_size:]   # 剩余的行
print(f"\n=== 数据集划分 ===")
print(f"训练集大小: {len(train_data)} 条")
print(f"测试集大小: {len(test_data)} 条")
X_train = train_data.drop('Survived', axis=1)
y_train = train_data['Survived']
X_test = test_data.drop('Survived', axis=1)
y_test = test_data['Survived']
# 输出清洗之后的特征数量
print(f"\n=== 特征数量 ===")
print(f"训练集特征数量: {X_train.shape[1]}")
print(f"测试集特征数量: {X_test.shape[1]}")
# Logistic Regression模型
def LogisticRegression(X, y, learning_rate=0.01, num_iterations=15000):
    # 初始化参数
    m, n = X.shape
    theta = np.zeros(n+1)
    # 添加截距项
    X = np.hstack((np.ones((m, 1)), X))
    # 梯度下降
    for i in range(num_iterations):
        z = np.dot(X, theta)
        h = 1 / (1 + np.exp(-z))
        # 添加一个小的常数epsilon避免log(0)
        epsilon = 1e-15
        # 将h限制在[epsilon, 1-epsilon]范围内
        h = np.clip(h, epsilon, 1 - epsilon)  
        gradient = np.dot(X.T, (h - y)) / m
        theta -= learning_rate * gradient
        # 调整学习率
        if i % 1000 == 0:
            learning_rate *= 0.7
        #输出训练过程中的损失值
        if i % 100 == 0:
            loss = -np.mean(y * np.log(h) + (1 - y) * np.log(1 - h))
            print(f"迭代次数: {i}, 损失值: {loss:.4f}")
    return theta

# Softmax Regression模型训练
def softmaxRegression(X, y, learning_rate=0.01, num_iterations=15000):
    # 初始化参数
    m, n = X.shape
    theta = np.zeros((n+1, 2))
    # 添加截距项
    X = np.hstack((np.ones((m, 1)), X))
    # 梯度下降
    for i in range(num_iterations):
        z = np.dot(X, theta)
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        h = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        # 创建one-hot编码的y
        y_one_hot = np.zeros((m, 2))
        y_one_hot[np.arange(m), y] = 1
        gradient = np.dot(X.T, (h - y_one_hot)) / m
        theta -= learning_rate * gradient
        # 调整学习率
        if i % 1000 == 0:
            learning_rate *= 0.7
        #输出训练过程中的损失值
        if i % 100 == 0:
            loss = -np.mean(np.sum(y_one_hot * np.log(h + 1e-15), axis=1))
            print(f"迭代次数: {i}, 损失值: {loss:.4f}")
    return theta
# 训练模型
print(f"\n=== 训练模型 ===")
print(f"\n=== Logistic Regression模型训练 ===")
theta1 = LogisticRegression(X_train, y_train)
print(f"\n=== Softmax Regression模型训练 ===")
theta2 = softmaxRegression(X_train, y_train)
# 预测
def predictL(X, theta):
    m = X.shape[0]
    X = np.hstack((np.ones((m, 1)), X))
    z = np.dot(X, theta)
    h = 1 / (1 + np.exp(-z))
    return (h >= 0.5).astype(int)
def predictS(X, theta):
    m = X.shape[0]
    X = np.hstack((np.ones((m, 1)), X))
    z = np.dot(X, theta)
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    h = exp_z / np.sum(exp_z, axis=1, keepdims=True)
    return np.argmax(h, axis=1)
#使用模型进行预测
#Logistic Regression模型预测
y_pred = predictL(X_test, theta1)
print(f"\n=== Logistic Regression模型预测 ===")
# 计算准确率
accuracy = np.mean(y_pred == y_test.values)
print(f"准确率: {accuracy:.2f}")
#Softmax Regression模型预测
y_pred = predictS(X_test, theta2)
print(f"\n=== Softmax Regression模型预测 ===")
# 计算准确率
accuracy = np.mean(y_pred == y_test.values)
print(f"准确率: {accuracy:.2f}")