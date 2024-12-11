import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import time

# --- Загрузка и предобработка данных ---
def load_and_preprocess(dataset_name):
    if dataset_name == "iris":
        data = load_iris()
        X, y = data.data, data.target
        y = y.astype(np.int64)
        y = np.array(y, dtype=np.int64) #Дополнительная проверка для принудительного преобразования

    elif dataset_name == "housing":
        data = fetch_california_housing()
        X, y = data.data, data.target
    else:
        raise ValueError("Некорректное имя набора данных")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


# --- Определение модели MLP ---
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.float()
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


# --- Функция обучения для PyTorch ---
def train_pytorch(model, train_loader, criterion, optimizer, epochs=100):
    start_time = time.time()
    model.train()
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs.float())
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    end_time = time.time()
    training_time = end_time - start_time
    return training_time


# --- Функция оценки для PyTorch ---
def evaluate_pytorch(model, test_loader, dataset_name):
    model.eval()
    y_pred = []
    y_true = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs.float())
            if dataset_name == "iris":
                _, predicted = torch.max(outputs, 1)
                y_pred.extend(predicted.cpu().numpy())
                y_true.extend(labels.cpu().numpy())
            elif dataset_name == "housing":
                y_pred.extend(outputs.cpu().numpy().flatten())
                y_true.extend(labels.cpu().numpy().flatten())
            else:
                raise ValueError("Некорректное имя набора данных")

    if dataset_name == "iris":
        accuracy = accuracy_score(y_true, y_pred)
        return accuracy
    elif dataset_name == "housing":
        mse = mean_squared_error(y_true, y_pred)
        return mse
    else:
        raise ValueError("Некорректное имя набора данных")


# --- Запуск эксперимента PyTorch ---
def run_pytorch_experiment(dataset_name, epochs=100, hidden_size=10):
    X_train, X_test, y_train, y_test = load_and_preprocess(dataset_name)
    input_size = X_train.shape[1]

    if dataset_name == "iris":
        output_size = 3
        criterion = nn.CrossEntropyLoss()
        train_data = TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train).long())
        test_data = TensorDataset(torch.tensor(X_test).float(), torch.tensor(y_test).long())
    elif dataset_name == "housing":
        output_size = 1
        criterion = nn.MSELoss()
        train_data = TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train).float())
        test_data = TensorDataset(torch.tensor(X_test).float(), torch.tensor(y_test).float())
    else:
        raise ValueError("Некорректное имя набора данных")

    model = MLP(input_size, hidden_size, output_size)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32)

    training_time = train_pytorch(model, train_loader, criterion, optimizer, epochs=epochs)
    metric = evaluate_pytorch(model, test_loader, dataset_name)
    return metric, training_time


# --- Функция визуализации ---
def visualize_results(results):
    df = pd.DataFrame([results])
    print(df)

    melted_df = df.melt(var_name='Metric', value_name='Value')

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Metric', y='Value', data=melted_df)
    plt.title('Производительность модели')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Значение метрики')
    plt.tight_layout()
    plt.show()

# Запуск экспериментов и визуализация результатов
iris_metric, iris_time = run_pytorch_experiment("iris", epochs=100)
housing_metric, housing_time = run_pytorch_experiment("housing", epochs=50)

results = {
    "Точность Iris": iris_metric,
    "MSE Housing": housing_metric,
    "Время обучения Iris": iris_time,
    "Время обучения Housing": housing_time
}

visualize_results(results)