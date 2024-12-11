import pandas as pd
import numpy as np
from sklearn.datasets import load_iris, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Функция загрузки и предобработки данных
def load_and_preprocess_data(dataset_name):
    if dataset_name == "iris":
        data = load_iris()
        X, y = data.data, data.target
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


# Функция обучения и оценки моделей
def train_and_evaluate(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    if isinstance(model, (LogisticRegression, DecisionTreeClassifier, RandomForestClassifier)):
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        return accuracy, report
    elif isinstance(model, (LinearRegression, DecisionTreeRegressor, RandomForestRegressor)):
        mse = mean_squared_error(y_test, y_pred)
        return mse, None
    else:
        return None, None


# Функция выполнения эксперимента
def run_experiment(dataset_name):
    X_train, X_test, y_train, y_test = load_and_preprocess_data(dataset_name)
    results = {}
    if dataset_name == "iris":
      models = {
          "Логистическая регрессия": LogisticRegression(),
          "Дерево принятия решений": DecisionTreeClassifier(),
          "Случайный лес": RandomForestClassifier()
      }
      for name, model in models.items():
          accuracy, report = train_and_evaluate(model, X_train, y_train, X_test, y_test)
          results[name] = {"accuracy": accuracy, "report": report}
    elif dataset_name == "housing":
      models = {
          "Линейная регрессия": LinearRegression(),
          "Дерево принятия решений": DecisionTreeRegressor(),
          "Случайный лес": RandomForestRegressor()
      }
      for name, model in models.items():
          mse, _ = train_and_evaluate(model, X_train, y_train, X_test, y_test)
          results[name] = {"mse": mse}

    return results


# Функция визуализации результата
def visualize_results(results, dataset_name):
    if dataset_name == "iris":
        metric = "accuracy"
    else:
        metric = "mse"

    data = {
        "model": list(results.keys()),
        metric: [results[model][metric] for model in results]
    }
    df = pd.DataFrame(data)
    plt.figure(figsize=(10, 6))
    sns.barplot(x="model", y=metric, data=df)
    plt.title(f"{dataset_name.capitalize()} - Производительность модели")
    plt.show()
    for model, result in results.items():
        if "report" in result:
            print(f"\nClassification Report for {model} ({dataset_name}):\n{result['report']}")

# Запуск экспериментов и визуализация результатов
iris_results = run_experiment("iris")
housing_results = run_experiment("housing")
visualize_results(iris_results, "iris")
visualize_results(housing_results, "housing")