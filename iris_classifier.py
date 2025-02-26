"""
Простой классификатор цветов ириса с использованием логистической регрессии
"""

# Импорт необходимых библиотек
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score,
                             confusion_matrix,
                             ConfusionMatrixDisplay)
import joblib


# Загрузка данных
def load_data():
    """
    Загрузка набора данных Iris
    Возвращает признаки и целевые переменные
    """
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    return df, iris


# Предобработка данных
def preprocess_data(df, test_size=0.2, random_state=42):
    """
    Разделение данных на обучающую и тестовую выборки
    """
    X = df.drop('target', axis=1)
    y = df['target']
    return train_test_split(X, y,
                            test_size=test_size,
                            random_state=random_state)


# Обучение модели
def train_model(X_train, y_train):
    """
    Создание и обучение модели логистической регрессии
    """
    model = LogisticRegression(max_iter=200, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    return model


# Оценка модели
def evaluate_model(model, X_test, y_test):
    # Предсказания
    y_pred = model.predict(X_test)
    print("Предсказания:", y_pred)
    print("Реальные зна:", y_test.values)

    # Расчет точности
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    # Визуализация матрицы ошибок
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')  # Сохраняем график
    plt.show()  # Показываем график


# Основной блок выполнения
if __name__ == "__main__":
    # Загрузка данных
    df, iris = load_data()
    print("Данные успешно загружены:")
    print(f"- Количество образцов: {len(df)}")
    print(f"- Количество признаков: {len(df.columns) - 1}")
    print(f"- Названия классов: {iris.target_names.tolist()}")

    # Разделение данных
    X_train, X_test, y_train, y_test = preprocess_data(df)
    print("\nДанные разделены на:")
    print(f"- Обучающая выборка: {len(X_train)} образцов")
    print(f"- Тестовая выборка: {len(X_test)} образцов")

    # Обучение модели
    model = train_model(X_train, y_train)
    print("\nМодель успешно обучена")

    # Оценка модели
    print("\nОценка производительности модели:")
    evaluate_model(model, X_test, y_test)


    # Сохранение модели
    joblib.dump(model, 'iris_classifier.joblib')
    print("\nМодель сохранена как 'iris_classifier.joblib'")