import math
import random

def mean(values):
    """计算平均值"""
    return sum(values) / len(values)

def standardize_features(X_train, X_test):
    """标准化特征数据：对每个特征进行标准化 (减去均值，除以标准差)"""
    # 计算训练集的均值和标准差
    means = [mean([x[i] for x in X_train]) for i in range(len(X_train[0]))]
    stds = [math.sqrt(mean([(x[i] - means[i]) ** 2 for x in X_train])) for i in range(len(X_train[0]))]
    
    # 避免除以零
    stds = [s if s != 0 else 1 for s in stds]
    
    # 标准化训练集
    X_train_standardized = []
    for x in X_train:
        standardized_x = [(x[i] - means[i]) / stds[i] for i in range(len(x))]
        X_train_standardized.append(standardized_x)
    
    # 使用训练集的均值和标准差标准化测试集
    X_test_standardized = []
    for x in X_test:
        standardized_x = [(x[i] - means[i]) / stds[i] for i in range(len(x))]
        X_test_standardized.append(standardized_x)
    
    return X_train_standardized, X_test_standardized, means, stds

def standardize_target(y_train, y_test):
    """标准化目标值"""
    y_mean = mean(y_train)
    y_std = math.sqrt(mean([(y - y_mean) ** 2 for y in y_train]))
    y_std = y_std if y_std != 0 else 1
    
    y_train_std = [(y - y_mean) / y_std for y in y_train]
    y_test_std = [(y - y_mean) / y_std for y in y_test]
    
    return y_train_std, y_test_std, y_mean, y_std

def destandardize_predictions(y_pred_std, y_mean, y_std):
    """将标准化的预测值转换回原始尺度"""
    return [y * y_std + y_mean for y in y_pred_std]

def dot_product(vector1, vector2):
    """计算两个向量的点积"""
    if len(vector1) != len(vector2):
        raise ValueError("向量长度不一致，无法计算点积")
    return sum(x * y for x, y in zip(vector1, vector2))

def matrix_multiply(A, B):
    """矩阵乘法"""
    if len(A[0]) != len(B):
        raise ValueError("矩阵维度不匹配，无法相乘")
    
    result = []
    for i in range(len(A)):
        row = []
        for j in range(len(B[0])):
            val = sum(A[i][k] * B[k][j] for k in range(len(B)))
            row.append(val)
        result.append(row)
    
    return result

def matrix_vector_product(matrix, vector):
    """计算矩阵和向量的乘积"""
    return [dot_product(row, vector) for row in matrix]

def transform(matrix):
    """计算矩阵的转置"""
    return [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]

def add_bias_column(matrix):
    """为矩阵添加偏置列"""
    return [[1] + row for row in matrix]

def scale_multiply(vector, scalar):
    """向量数乘"""
    return [x * scalar for x in vector]

def vector_subtract(vector1, vector2):
    """向量相减"""
    if len(vector1) != len(vector2):
        raise ValueError("向量长度不一致，无法相减")
    return [x - y for x, y in zip(vector1, vector2)]

def inverse(matrix):
    """计算矩阵的逆"""
    n = len(matrix)
    
    # 创建增广矩阵 [matrix | I]
    augmented = []
    for i in range(n):
        row = matrix[i][:]  # 复制原矩阵的行
        # 添加单位矩阵部分
        row.extend([1.0 if i == j else 0.0 for j in range(n)])
        augmented.append(row)
    
   
    for i in range(n):
        # 寻找主元
        max_row = i
        for j in range(i+1, n):
            if abs(augmented[j][i]) > abs(augmented[max_row][i]):
                max_row = j
        
        # 交换行
        augmented[i], augmented[max_row] = augmented[max_row], augmented[i]
        
        # 检查主元是否为0
        if abs(augmented[i][i]) < 1e-10:
            raise ValueError("矩阵不可逆")
        
        # 将主元化为1
        pivot = augmented[i][i]
        for j in range(i, 2*n):
            augmented[i][j] /= pivot
        
        # 消元
        for j in range(n):
            if i != j:
                factor = augmented[j][i]
                for k in range(i, 2*n):
                    augmented[j][k] -= factor * augmented[i][k]
    
    # 提取逆矩阵部分
    inv_matrix = []
    for i in range(n):
        inv_matrix.append(augmented[i][n:])
    
    return inv_matrix

def compute_weights(X, y):
    """正规方程计算线性回归权重"""
    X_bias = add_bias_column(X)
    X_T = transform(X_bias)
    X_T_X = matrix_multiply(X_T, X_bias)  # 使用矩阵乘法，不是矩阵向量乘积
    X_T_y = matrix_vector_product(X_T, y)
    
    # 计算逆矩阵
    X_T_X_inv = inverse(X_T_X)
    
    # 计算权重
    weights = matrix_vector_product(X_T_X_inv, X_T_y)
    return weights

class LinearRegression:
    def __init__(self, nfeatures):
        self.nfeatures = nfeatures
        """初始化线性回归模型，随机生成权重"""
        self.weights = [random.uniform(-0.1, 0.1) for _ in range(nfeatures + 1)]
        """初始化损失历史记录"""
        self.loss_history = []
        self.gradient = None  # 初始化梯度变量

    def predict(self, x):
        """预测函数"""
        x_bias = [1] + x  # 添加偏置项
        return dot_product(self.weights, x_bias)
    
    def compute_loss(self, X, y):
        """计算损失函数"""
        return sum([(self.predict(x) - z) ** 2 for x, z in zip(X, y)]) / len(X)
    
    def compute_gradient(self, X, y):
        """计算梯度"""
        n_samples = len(X)
        # 计算预测值
        y_pred = [self.predict(x) for x in X]
        # 添加偏置项
        X_bias = add_bias_column(X)
        # 计算误差
        errors = vector_subtract(y_pred, y)
        # 计算梯度
        X_bias_T = transform(X_bias)
        gradient_unscaled = matrix_vector_product(X_bias_T, errors)
        gradient = scale_multiply(gradient_unscaled, 2 / n_samples)
        return gradient

    def train(self, X, y, learning_rate=0.01, epochs=400, verbose=True):
        """训练模型"""
        if verbose:
            print("开始训练模型...")
            
        for epoch in range(epochs):
            # 计算梯度
            self.gradient = self.compute_gradient(X, y)
            # 更新权重
            self.weights = vector_subtract(self.weights, scale_multiply(self.gradient, learning_rate))
            # 计算并记录损失
            if epoch % 50 == 0:
                loss = self.compute_loss(X, y)
                self.loss_history.append((epoch, loss))

                if verbose:
                    print(f"轮次 {epoch}: 损失 = {loss:.4f}")
                    
        final_loss = self.compute_loss(X, y)
        if verbose:
            print(f"训练完成! 最终损失: {final_loss:.4f}")
        
        return self.loss_history

    def get_weights(self):
        """获取模型权重"""
        return {
            'weights': self.weights,
            'bias': self.weights[0],
            'feature_weights': self.weights[1:]
        }

def load_data_from_file(filename):
    """从文件加载数据并分割为训练集和测试集"""
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    
    with open(filename, 'r') as file:
        lines = file.readlines()
        
        # 处理每一行数据
        for i, line in enumerate(lines):
            # 跳过空行
            if line.strip() == "":
                continue
            # 分割字符串并转换为浮点数
            values = [float(x) for x in line.split()]
            # 前13个值是特征，最后一个是目标值
            features = values[:-1]
            target = values[-1]
            
            # 前400条作为训练集，后106条作为测试集
            if i < 400:
                X_train.append(features)
                y_train.append(target)
            else:
                X_test.append(features)
                y_test.append(target)
    
    return X_train, y_train, X_test, y_test

# 主程序
if __name__ == "__main__":
    # 从文件加载数据
    filename = "data.txt"  
    try:
        X_train, y_train, X_test, y_test = load_data_from_file(filename)
        print(f"成功加载数据:")
        print(f"训练集: {len(X_train)} 个样本")
        print(f"测试集: {len(X_test)} 个样本")
        print(f"每个样本有 {len(X_train[0])} 个特征")
        
        # 标准化特征和目标值
        print("\n正在标准化数据...")
        X_train_std, X_test_std, x_means, x_stds = standardize_features(X_train, X_test)
        y_train_std, y_test_std, y_mean, y_std = standardize_target(y_train, y_test)
        
        print(f"目标值标准化参数: 均值 = {y_mean:.4f}, 标准差 = {y_std:.4f}")
        
        # 梯度下降法
      
        print("梯度下降法")
       
        
        # 创建并训练模型
        n_features = len(X_train_std[0])
        model_gd = LinearRegression(nfeatures=n_features)
        
        print("\n开始训练模型...")
        # 使用标准化后的数据训练模型
        loss_history = model_gd.train(
            X=X_train_std,
            y=y_train_std,
            learning_rate=0.005,
            epochs=3000,
            verbose=True
        )
        
        # 显示训练结果
        final_weights_gd = model_gd.get_weights()
        print(f"\n训练完成!")
        print(f"最优权重 θ*:")
        print(f"偏置项 (θ₀): {final_weights_gd['bias']:.6f}")
        for i, w in enumerate(final_weights_gd['feature_weights']):
            print(f"特征权重 θ{i+1}: {w:.6f}")
        
        # 计算训练集和测试集上的MSE（原始尺度）
        train_predictions_std_gd = [model_gd.predict(x) for x in X_train_std]
        test_predictions_std_gd = [model_gd.predict(x) for x in X_test_std]
        
        train_predictions_gd = destandardize_predictions(train_predictions_std_gd, y_mean, y_std)
        test_predictions_gd = destandardize_predictions(test_predictions_std_gd, y_mean, y_std)
        
        train_mse_gd = sum([(pred - actual) ** 2 for pred, actual in zip(train_predictions_gd, y_train)]) / len(y_train)
        test_mse_gd = sum([(pred - actual) ** 2 for pred, actual in zip(test_predictions_gd, y_test)]) / len(y_test)
        
        print(f"\n训练集MSE: {train_mse_gd:.4f}")
        print(f"测试集MSE: {test_mse_gd:.4f}")
        
        # 正规方程法
     
        print("正规方程法")
  
        
        print("\n计算正规方程权重...")
        weights_ne = compute_weights(X_train_std, y_train_std)
        
        print(f"\n最优权重 θ*:")
        print(f"偏置项 (θ₀): {weights_ne[0]:.6f}")
        for i, w in enumerate(weights_ne[1:]):
            print(f"特征权重 θ{i+1}: {w:.6f}")
        
        # 创建正规方程模型用于预测
        model_ne = LinearRegression(nfeatures=n_features)
        model_ne.weights = weights_ne
        
        # 计算训练集和测试集上的MSE（原始尺度）
        train_predictions_std_ne = [model_ne.predict(x) for x in X_train_std]
        test_predictions_std_ne = [model_ne.predict(x) for x in X_test_std]
        
        train_predictions_ne = destandardize_predictions(train_predictions_std_ne, y_mean, y_std)
        test_predictions_ne = destandardize_predictions(test_predictions_std_ne, y_mean, y_std)
        
        train_mse_ne = sum([(pred - actual) ** 2 for pred, actual in zip(train_predictions_ne, y_train)]) / len(y_train)
        test_mse_ne = sum([(pred - actual) ** 2 for pred, actual in zip(test_predictions_ne, y_test)]) / len(y_test)
        
        print(f"\n训练集MSE: {train_mse_ne:.4f}")
        print(f"测试集MSE: {test_mse_ne:.4f}")
        
        # 进行预测示例
        if len(X_test_std) > 0:
            print(f"\n测试集预测示例 (梯度下降法):")
            for i in range(min(5, len(X_test_std))):
                test_sample = X_test_std[i]
                prediction_std = model_gd.predict(test_sample)
                prediction = prediction_std * y_std + y_mean
                actual = y_test[i]
                error = abs(prediction - actual)
                print(f"样本 {i+1}: 实际值 = {actual:.4f}, 预测值 = {prediction:.4f}, 误差 = {error:.4f}")
            
    except FileNotFoundError:
        print(f"错误: 找不到文件 '{filename}'")
        print("请确保文件存在并且路径正确")
    except Exception as e:
        print(f"错误: {e}")