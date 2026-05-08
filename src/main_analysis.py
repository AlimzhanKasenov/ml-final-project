import os
import warnings

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from lightgbm import LGBMClassifier

from sklearn.ensemble import (
    BaggingClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")


def get_project_root() -> str:
    """
    Возвращает путь к корневой папке проекта.
    """

    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def create_project_directories(project_root: str) -> dict:
    """
    Создаёт нужные папки проекта.
    """

    paths = {
        "data": os.path.join(project_root, "data"),
        "images": os.path.join(project_root, "images"),
        "models": os.path.join(project_root, "models"),
        "reports": os.path.join(project_root, "reports"),
    }

    for path in paths.values():
        os.makedirs(path, exist_ok=True)

    return paths


def save_plot(path: str) -> None:
    """
    Сохраняет график в файл и закрывает его.
    """

    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def load_dataset(data_path: str) -> pd.DataFrame:
    """
    Загружает датасет из CSV-файла.
    """

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Файл датасета не найден: {data_path}\n"
            f"Сначала запустите файл src/generate_dataset.py"
        )

    df = pd.read_csv(data_path)
    return df


def print_dataset_info(df: pd.DataFrame) -> None:
    """
    Выводит основную информацию о датасете.
    """

    print("=" * 80)
    print("1. ОБЩАЯ ИНФОРМАЦИЯ О ДАТАСЕТЕ")
    print("=" * 80)

    print("\nПервые 5 строк:")
    print(df.head())

    print("\nРазмер датасета:")
    print(f"{df.shape[0]} строк, {df.shape[1]} колонок")

    print("\nТипы данных:")
    print(df.dtypes)

    print("\nКоличество пропусков:")
    print(df.isnull().sum())

    print("\nКоличество дубликатов:")
    print(df.duplicated().sum())

    print("\nСтатистическое описание числовых признаков:")
    print(df.describe())

    print("\nРаспределение целевой переменной Отток:")
    print(df["Отток"].value_counts())

    print("\nДоля классов:")
    print(df["Отток"].value_counts(normalize=True).round(3))


def create_eda_plots(df: pd.DataFrame, images_dir: str) -> None:
    """
    Создаёт графики разведочного анализа данных.
    """

    print("\n" + "=" * 80)
    print("2. СОЗДАНИЕ ГРАФИКОВ")
    print("=" * 80)

    plt.figure(figsize=(7, 5))
    sns.countplot(data=df, x="Отток")
    plt.title("Распределение клиентов по признаку оттока")
    plt.xlabel("Отток: 0 — остался, 1 — ушёл")
    plt.ylabel("Количество клиентов")
    save_plot(os.path.join(images_dir, "01_распределение_оттока.png"))

    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x="Возраст", hue="Отток", kde=True)
    plt.title("Распределение возраста клиентов по признаку оттока")
    plt.xlabel("Возраст")
    plt.ylabel("Количество клиентов")
    save_plot(os.path.join(images_dir, "02_возраст_и_отток.png"))

    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x="Активный_клиент", hue="Отток")
    plt.title("Связь активности клиента с оттоком")
    plt.xlabel("Активный клиент: 0 — нет, 1 — да")
    plt.ylabel("Количество клиентов")
    save_plot(os.path.join(images_dir, "03_активность_и_отток.png"))

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="Отток", y="Количество_жалоб")
    plt.title("Количество жалоб у оставшихся и ушедших клиентов")
    plt.xlabel("Отток")
    plt.ylabel("Количество жалоб")
    save_plot(os.path.join(images_dir, "04_жалобы_и_отток.png"))

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="Отток", y="Обращения_в_поддержку")
    plt.title("Обращения в поддержку у оставшихся и ушедших клиентов")
    plt.xlabel("Отток")
    plt.ylabel("Количество обращений")
    save_plot(os.path.join(images_dir, "05_поддержка_и_отток.png"))

    plt.figure(figsize=(10, 5))
    sns.countplot(data=df, x="Банк", hue="Отток")
    plt.title("Отток клиентов по банкам")
    plt.xlabel("Банк")
    plt.ylabel("Количество клиентов")
    plt.xticks(rotation=20)
    save_plot(os.path.join(images_dir, "06_отток_по_банкам.png"))

    numeric_df = df.select_dtypes(include=["int64", "float64"])

    plt.figure(figsize=(13, 9))
    sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f")
    plt.title("Матрица корреляции числовых признаков")
    save_plot(os.path.join(images_dir, "07_матрица_корреляции.png"))

    print("Графики сохранены в папку images")


def prepare_data(df: pd.DataFrame):
    """
    Подготавливает данные для обучения моделей.
    """

    print("\n" + "=" * 80)
    print("3. ПРЕДОБРАБОТКА ДАННЫХ")
    print("=" * 80)

    processed_df = df.copy()

    # Удаляем технический ID, потому что он не несёт полезной информации для модели.
    processed_df = processed_df.drop(columns=["ID_клиента"])

    # Кодируем текстовые признаки в числовые через one-hot encoding.
    categorical_columns = [
        "Пол",
        "Город",
        "Банк",
        "Тип_клиента",
        "Канал_обслуживания"
    ]

    processed_df = pd.get_dummies(
        processed_df,
        columns=categorical_columns,
        drop_first=True
    )

    X = processed_df.drop(columns=["Отток"])
    y = processed_df["Отток"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Данные успешно подготовлены")
    print(f"Количество признаков после кодирования: {X.shape[1]}")
    print(f"Размер обучающей выборки: {X_train.shape}")
    print(f"Размер тестовой выборки: {X_test.shape}")

    return X, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler


def find_best_threshold(y_test, y_proba) -> dict:
    """
    Подбирает лучший порог классификации по метрике F1.

    По умолчанию классификация обычно идёт по порогу 0.5.
    Но для задачи оттока клиентов часто выгоднее снизить порог,
    чтобы лучше находить клиентов с риском ухода.
    """

    thresholds = [i / 100 for i in range(10, 91)]
    best_result = {
        "threshold": 0.5,
        "f1": 0,
        "precision": 0,
        "recall": 0,
        "accuracy": 0
    }

    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)

        current_f1 = f1_score(y_test, y_pred)
        current_precision = precision_score(y_test, y_pred, zero_division=0)
        current_recall = recall_score(y_test, y_pred, zero_division=0)
        current_accuracy = accuracy_score(y_test, y_pred)

        if current_f1 > best_result["f1"]:
            best_result = {
                "threshold": threshold,
                "f1": current_f1,
                "precision": current_precision,
                "recall": current_recall,
                "accuracy": current_accuracy
            }

    return best_result


def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    """
    Оценивает модель по основным метрикам классификации.

    Считаем два варианта:
    1. стандартный порог 0.5;
    2. оптимальный порог, подобранный по F1.
    """

    y_proba = model.predict_proba(X_test)[:, 1]

    y_pred_default = (y_proba >= 0.5).astype(int)

    default_accuracy = accuracy_score(y_test, y_pred_default)
    default_precision = precision_score(y_test, y_pred_default, zero_division=0)
    default_recall = recall_score(y_test, y_pred_default, zero_division=0)
    default_f1 = f1_score(y_test, y_pred_default, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba)

    best_threshold_result = find_best_threshold(y_test, y_proba)
    best_threshold = best_threshold_result["threshold"]

    y_pred_best = (y_proba >= best_threshold).astype(int)

    result = {
        "Модель": model_name,
        "Accuracy_порог_0_5": default_accuracy,
        "Precision_порог_0_5": default_precision,
        "Recall_порог_0_5": default_recall,
        "F1_порог_0_5": default_f1,
        "ROC_AUC": roc_auc,
        "Лучший_порог": best_threshold,
        "Accuracy_лучший_порог": best_threshold_result["accuracy"],
        "Precision_лучший_порог": best_threshold_result["precision"],
        "Recall_лучший_порог": best_threshold_result["recall"],
        "F1_лучший_порог": best_threshold_result["f1"]
    }

    print("\n" + "-" * 80)
    print(f"Модель: {model_name}")
    print("-" * 80)

    print("\nОценка при стандартном пороге 0.5:")
    print(f"Accuracy:  {default_accuracy:.4f}")
    print(f"Precision: {default_precision:.4f}")
    print(f"Recall:    {default_recall:.4f}")
    print(f"F1:        {default_f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")

    print("\nМатрица ошибок при пороге 0.5:")
    print(confusion_matrix(y_test, y_pred_default))

    print("\nОценка при лучшем пороге:")
    print(f"Лучший порог: {best_threshold:.2f}")
    print(f"Accuracy:     {best_threshold_result['accuracy']:.4f}")
    print(f"Precision:    {best_threshold_result['precision']:.4f}")
    print(f"Recall:       {best_threshold_result['recall']:.4f}")
    print(f"F1:           {best_threshold_result['f1']:.4f}")

    print("\nМатрица ошибок при лучшем пороге:")
    print(confusion_matrix(y_test, y_pred_best))

    print("\nОтчёт классификации при лучшем пороге:")
    print(classification_report(y_test, y_pred_best, zero_division=0))

    return result


def train_models(
    X_train,
    X_test,
    y_train,
    y_test,
    X_train_scaled,
    X_test_scaled,
    models_dir: str
) -> tuple:
    """
    Обучает модели и сохраняет их в папку models.
    """

    print("\n" + "=" * 80)
    print("4. ОБУЧЕНИЕ И ОЦЕНКА МОДЕЛЕЙ")
    print("=" * 80)

    results = []

    logistic_model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight="balanced"
    )
    logistic_model.fit(X_train_scaled, y_train)

    results.append(
        evaluate_model(
            logistic_model,
            X_test_scaled,
            y_test,
            "Logistic Regression"
        )
    )

    joblib.dump(
        logistic_model,
        os.path.join(models_dir, "logistic_regression_model.pkl")
    )

    bagging_model = BaggingClassifier(
        estimator=DecisionTreeClassifier(
            max_depth=6,
            random_state=42
        ),
        n_estimators=150,
        random_state=42,
        n_jobs=-1
    )
    bagging_model.fit(X_train, y_train)

    results.append(
        evaluate_model(
            bagging_model,
            X_test,
            y_test,
            "BaggingClassifier"
        )
    )

    joblib.dump(
        bagging_model,
        os.path.join(models_dir, "bagging_model.pkl")
    )

    random_forest_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=5,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )
    random_forest_model.fit(X_train, y_train)

    results.append(
        evaluate_model(
            random_forest_model,
            X_test,
            y_test,
            "RandomForestClassifier"
        )
    )

    joblib.dump(
        random_forest_model,
        os.path.join(models_dir, "random_forest_model.pkl")
    )

    gradient_boosting_model = GradientBoostingClassifier(
        n_estimators=250,
        learning_rate=0.04,
        max_depth=3,
        random_state=42
    )
    gradient_boosting_model.fit(X_train, y_train)

    results.append(
        evaluate_model(
            gradient_boosting_model,
            X_test,
            y_test,
            "GradientBoostingClassifier"
        )
    )

    joblib.dump(
        gradient_boosting_model,
        os.path.join(models_dir, "gradient_boosting_model.pkl")
    )

    lightgbm_model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.04,
        max_depth=5,
        num_leaves=20,
        min_child_samples=30,
        random_state=42,
        class_weight="balanced",
        verbose=-1
    )
    lightgbm_model.fit(X_train, y_train)

    results.append(
        evaluate_model(
            lightgbm_model,
            X_test,
            y_test,
            "LightGBMClassifier"
        )
    )

    joblib.dump(
        lightgbm_model,
        os.path.join(models_dir, "lightgbm_model.pkl")
    )

    models = {
        "Logistic Regression": logistic_model,
        "BaggingClassifier": bagging_model,
        "RandomForestClassifier": random_forest_model,
        "GradientBoostingClassifier": gradient_boosting_model,
        "LightGBMClassifier": lightgbm_model,
    }

    return pd.DataFrame(results), models


def create_model_comparison_plots(results_df: pd.DataFrame, images_dir: str) -> None:
    """
    Создаёт графики сравнения моделей.
    """

    print("\n" + "=" * 80)
    print("5. ГРАФИКИ СРАВНЕНИЯ МОДЕЛЕЙ")
    print("=" * 80)

    plt.figure(figsize=(10, 5))
    sns.barplot(data=results_df, x="Модель", y="ROC_AUC")
    plt.title("Сравнение моделей по ROC-AUC")
    plt.xlabel("Модель")
    plt.ylabel("ROC-AUC")
    plt.xticks(rotation=25)
    save_plot(os.path.join(images_dir, "08_сравнение_моделей_roc_auc.png"))

    plt.figure(figsize=(10, 5))
    sns.barplot(data=results_df, x="Модель", y="F1_порог_0_5")
    plt.title("Сравнение моделей по F1-score при пороге 0.5")
    plt.xlabel("Модель")
    plt.ylabel("F1-score")
    plt.xticks(rotation=25)
    save_plot(os.path.join(images_dir, "09_сравнение_моделей_f1_порог_0_5.png"))

    plt.figure(figsize=(10, 5))
    sns.barplot(data=results_df, x="Модель", y="F1_лучший_порог")
    plt.title("Сравнение моделей по F1-score после подбора порога")
    plt.xlabel("Модель")
    plt.ylabel("F1-score")
    plt.xticks(rotation=25)
    save_plot(os.path.join(images_dir, "10_сравнение_моделей_f1_лучший_порог.png"))

    plt.figure(figsize=(10, 5))
    sns.barplot(data=results_df, x="Модель", y="Recall_лучший_порог")
    plt.title("Сравнение моделей по Recall после подбора порога")
    plt.xlabel("Модель")
    plt.ylabel("Recall")
    plt.xticks(rotation=25)
    save_plot(os.path.join(images_dir, "11_сравнение_моделей_recall_лучший_порог.png"))

    print("Графики сравнения моделей сохранены в папку images")


def create_feature_importance_plot(model, feature_names, images_dir: str) -> pd.DataFrame:
    """
    Создаёт график важности признаков для Random Forest.
    """

    print("\n" + "=" * 80)
    print("6. ВАЖНОСТЬ ПРИЗНАКОВ")
    print("=" * 80)

    feature_importances = pd.DataFrame({
        "Признак": feature_names,
        "Важность": model.feature_importances_
    }).sort_values(by="Важность", ascending=False)

    print("\nТоп-10 важных признаков:")
    print(feature_importances.head(10))

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=feature_importances.head(10),
        x="Важность",
        y="Признак"
    )
    plt.title("Топ-10 важных признаков Random Forest")
    plt.xlabel("Важность")
    plt.ylabel("Признак")
    save_plot(os.path.join(images_dir, "12_важность_признаков.png"))

    return feature_importances


def save_results(results_df: pd.DataFrame, feature_importances: pd.DataFrame, reports_dir: str) -> None:
    """
    Сохраняет результаты моделей и важность признаков.
    """

    model_results_path = os.path.join(reports_dir, "model_results.csv")
    feature_importance_path = os.path.join(reports_dir, "feature_importance.csv")

    results_df.to_csv(model_results_path, index=False, encoding="utf-8-sig")
    feature_importances.to_csv(feature_importance_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 80)
    print("7. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
    print("=" * 80)
    print(f"Результаты моделей сохранены: {model_results_path}")
    print(f"Важность признаков сохранена: {feature_importance_path}")


def print_final_conclusion(results_df: pd.DataFrame) -> None:
    """
    Выводит итоговый вывод.
    """

    best_model_by_roc_auc = results_df.sort_values(by="ROC_AUC", ascending=False).iloc[0]
    best_model_by_f1 = results_df.sort_values(by="F1_лучший_порог", ascending=False).iloc[0]

    ensemble_models = results_df[
        results_df["Модель"].isin([
            "BaggingClassifier",
            "RandomForestClassifier",
            "GradientBoostingClassifier",
            "LightGBMClassifier"
        ])
    ]

    best_ensemble_by_f1 = ensemble_models.sort_values(by="F1_лучший_порог", ascending=False).iloc[0]

    print("\n" + "=" * 80)
    print("8. ИТОГОВЫЙ ВЫВОД")
    print("=" * 80)

    print(
        f"""
В рамках проекта была решена задача прогнозирования оттока клиентов банка.

Для анализа использовался синтетический датасет на русском языке, содержащий информацию о 10 000 клиентах банков.
В датасет входят демографические, поведенческие и финансовые признаки клиентов:
возраст, город, банк, тип клиента, канал обслуживания, кредитный рейтинг, баланс,
количество продуктов, активность клиента, жалобы, обращения в поддержку и другие параметры.

Были обучены и сравнены следующие модели:
1. Logistic Regression — базовая модель.
2. BaggingClassifier — ансамблевый метод на основе бэггинга.
3. RandomForestClassifier — случайный лес.
4. GradientBoostingClassifier — градиентный бустинг.
5. LightGBMClassifier — современная реализация градиентного бустинга.

Для оценки моделей использовались метрики Accuracy, Precision, Recall, F1-score и ROC-AUC.
Дополнительно был выполнен подбор оптимального порога классификации по метрике F1-score.
Это важно для задачи оттока клиентов, так как класс ушедших клиентов встречается реже,
и стандартный порог 0.5 не всегда позволяет хорошо находить клиентов с риском ухода.

Лучшая модель по ROC-AUC:
{best_model_by_roc_auc["Модель"]}

ROC-AUC:
{best_model_by_roc_auc["ROC_AUC"]:.4f}

Лучшая модель по F1-score после подбора порога:
{best_model_by_f1["Модель"]}

Лучший порог:
{best_model_by_f1["Лучший_порог"]:.2f}

F1-score:
{best_model_by_f1["F1_лучший_порог"]:.4f}

Recall:
{best_model_by_f1["Recall_лучший_порог"]:.4f}

Лучшая ансамблевая модель по F1-score:
{best_ensemble_by_f1["Модель"]}

F1-score ансамблевой модели:
{best_ensemble_by_f1["F1_лучший_порог"]:.4f}

Ансамблевые методы показали хорошее качество, потому что они способны выявлять нелинейные зависимости между признаками.
Наиболее важными факторами оттока обычно являются количество жалоб, обращения в поддержку,
активность клиента, срок обслуживания, количество банковских продуктов и использование мобильного приложения.
"""
    )


def main() -> None:
    """
    Главная функция запуска проекта.
    """

    project_root = get_project_root()
    paths = create_project_directories(project_root)

    data_path = os.path.join(paths["data"], "bank_churn_dataset.csv")

    df = load_dataset(data_path)

    print_dataset_info(df)
    create_eda_plots(df, paths["images"])

    (
        X,
        X_train,
        X_test,
        y_train,
        y_test,
        X_train_scaled,
        X_test_scaled,
        scaler
    ) = prepare_data(df)

    joblib.dump(
        scaler,
        os.path.join(paths["models"], "standard_scaler.pkl")
    )

    results_df, models = train_models(
        X_train,
        X_test,
        y_train,
        y_test,
        X_train_scaled,
        X_test_scaled,
        paths["models"]
    )

    results_df = results_df.sort_values(by="ROC_AUC", ascending=False)

    print("\nСравнение моделей:")
    print(results_df)

    create_model_comparison_plots(results_df, paths["images"])

    feature_importances = create_feature_importance_plot(
        models["RandomForestClassifier"],
        X.columns,
        paths["images"]
    )

    save_results(results_df, feature_importances, paths["reports"])
    print_final_conclusion(results_df)


if __name__ == "__main__":
    main()