import tensorflow as tf
import numpy as np
from sklearn.datasets import load_iris, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import time
import pandas as pd

# Функция загрузки и предобработки данных
def load_and_preprocess(dataset_name):
    if dataset_name == "iris":
        data = load_iris()
        X, y = data.data, data.target
        y = tf.keras.utils.to_categorical(y, num_classes=3)  # One-hot encoding for multi-class
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


# Функция создания и обучения модели
def create_and_train_model(X_train, y_train, dataset_name, epochs=100, verbose=0):
    start_time = time.time()
    if dataset_name == "iris":
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(10, activation='relu', input_shape=(X_train.shape[1],)),
            tf.keras.layers.Dense(3, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=epochs, verbose=verbose,
                  validation_split=0.1,
                  callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)])

    elif dataset_name == "housing":
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(10, activation='relu', input_shape=(X_train.shape[1],)),
            tf.keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        model.fit(X_train, y_train, epochs=epochs, verbose=verbose,
                  validation_split=0.1,
                  callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)])
    else:
        raise ValueError("Некорректное имя набора данных")
    end_time = time.time()
    training_time = end_time - start_time
    return model, training_time


# Функция оценки модели
def evaluate_model(model, X_test, y_test, dataset_name):
    if dataset_name == "iris":
        _, accuracy = model.evaluate(X_test, y_test, verbose=0)
        return accuracy
    elif dataset_name == "housing":
        mse, mae = model.evaluate(X_test, y_test, verbose=0)
        return mse, mae
    else:
        raise ValueError("Некорректное имя набора данных")


# Функция выполнения эксперимента
def run_experiment(dataset_name, epochs=100, verbose=0):
    X_train, X_test, y_train, y_test = load_and_preprocess(dataset_name)
    model, training_time = create_and_train_model(X_train, y_train, dataset_name, epochs=epochs, verbose=verbose)
    if dataset_name == "housing":
        mse, mae = evaluate_model(model, X_test, y_test, dataset_name)
        return mse, mae, training_time
    else:
        accuracy = evaluate_model(model, X_test, y_test, dataset_name)
        return accuracy, training_time



# Функция визуализации результатов
def visualize_results(results):
    df = pd.DataFrame(results, index=[0])
    df = df.T.rename(columns={0: 'Значение'})
    plt.figure(figsize=(10, 6))
    sns.barplot(x=df.index, y='Значение', data=df)
    plt.xticks(rotation=45, ha='right')
    plt.title("Производительность модели")
    plt.tight_layout()
    plt.show()


# Запуск экспериментов и визуализация результатов
iris_accuracy, iris_training_time = run_experiment("iris", epochs=100, verbose=0)
housing_mse, housing_mae, housing_training_time = run_experiment("housing", epochs=50, verbose=0)

results = {
    "Точность Iris": iris_accuracy,
    "MSE Housing": housing_mse,
    "MAE Housing": housing_mae,
    "Время обучения Iris": iris_training_time,
    "Время обучения Housing": housing_training_time
}

visualize_results(results)