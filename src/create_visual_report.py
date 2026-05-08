import base64
import os

import pandas as pd


def get_project_root() -> str:
    """
    Возвращает путь к корневой папке проекта.
    """

    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read_csv_table(file_path: str) -> pd.DataFrame:
    """
    Читает CSV-файл и возвращает DataFrame.
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Файл не найден: {file_path}")

    return pd.read_csv(file_path)


def image_to_base64(image_path: str) -> str:
    """
    Преобразует изображение в base64-строку.
    Это нужно, чтобы HTML-отчёт был самодостаточным и не зависел от папки images.
    """

    if not os.path.exists(image_path):
        print(f"Внимание: изображение не найдено: {image_path}")
        return ""

    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

    return f"data:image/png;base64,{encoded_image}"


def create_image_tag(images_dir: str, file_name: str, alt_text: str) -> str:
    """
    Создаёт HTML-тег img со встроенным изображением.
    """

    image_path = os.path.join(images_dir, file_name)
    image_src = image_to_base64(image_path)

    if not image_src:
        return f"<p><b>Изображение не найдено:</b> {file_name}</p>"

    return f'<img class="graph" src="{image_src}" alt="{alt_text}">'


def prepare_model_results_table(model_results: pd.DataFrame) -> str:
    """
    Готовит компактную таблицу результатов моделей для визуального отчёта.

    В исходном CSV много колонок, поэтому для HTML-отчёта оставляем только ключевые метрики,
    чтобы таблица красиво помещалась на экране.
    """

    required_columns = [
        "Модель",
        "ROC_AUC",
        "F1_порог_0_5",
        "Лучший_порог",
        "F1_лучший_порог",
        "Recall_лучший_порог"
    ]

    missing_columns = [
        column for column in required_columns
        if column not in model_results.columns
    ]

    if missing_columns:
        raise ValueError(
            "В файле model_results.csv не найдены нужные колонки: "
            + ", ".join(missing_columns)
        )

    model_results_view = model_results[required_columns].copy()

    model_results_view = model_results_view.rename(
        columns={
            "ROC_AUC": "ROC-AUC",
            "F1_порог_0_5": "F1 при пороге 0.5",
            "Лучший_порог": "Лучший порог",
            "F1_лучший_порог": "F1 после подбора",
            "Recall_лучший_порог": "Recall после подбора"
        }
    )

    return model_results_view.round(4).to_html(
        index=False,
        classes="table",
        border=0
    )


def prepare_feature_importance_table(feature_importance: pd.DataFrame) -> str:
    """
    Готовит таблицу топ-10 важных признаков.
    """

    return feature_importance.head(10).round(4).to_html(
        index=False,
        classes="table",
        border=0
    )


def create_html_report(project_root: str) -> None:
    """
    Создаёт визуальный HTML-отчёт по проекту.
    """

    reports_dir = os.path.join(project_root, "reports")
    images_dir = os.path.join(project_root, "images")

    model_results_path = os.path.join(reports_dir, "model_results.csv")
    feature_importance_path = os.path.join(reports_dir, "feature_importance.csv")

    model_results = read_csv_table(model_results_path)
    feature_importance = read_csv_table(feature_importance_path)

    model_results_html = prepare_model_results_table(model_results)
    feature_importance_html = prepare_feature_importance_table(feature_importance)

    graph_churn_distribution = create_image_tag(
        images_dir,
        "01_распределение_оттока.png",
        "Распределение оттока"
    )

    graph_age_churn = create_image_tag(
        images_dir,
        "02_возраст_и_отток.png",
        "Возраст и отток"
    )

    graph_activity_churn = create_image_tag(
        images_dir,
        "03_активность_и_отток.png",
        "Активность и отток"
    )

    graph_complaints_churn = create_image_tag(
        images_dir,
        "04_жалобы_и_отток.png",
        "Жалобы и отток"
    )

    graph_support_churn = create_image_tag(
        images_dir,
        "05_поддержка_и_отток.png",
        "Обращения в поддержку и отток"
    )

    graph_bank_churn = create_image_tag(
        images_dir,
        "06_отток_по_банкам.png",
        "Отток по банкам"
    )

    graph_correlation = create_image_tag(
        images_dir,
        "07_матрица_корреляции.png",
        "Матрица корреляции"
    )

    graph_roc_auc = create_image_tag(
        images_dir,
        "08_сравнение_моделей_roc_auc.png",
        "Сравнение моделей по ROC-AUC"
    )

    graph_f1_default = create_image_tag(
        images_dir,
        "09_сравнение_моделей_f1_порог_0_5.png",
        "Сравнение моделей по F1 при пороге 0.5"
    )

    graph_f1_best = create_image_tag(
        images_dir,
        "10_сравнение_моделей_f1_лучший_порог.png",
        "Сравнение моделей по F1 после подбора порога"
    )

    graph_recall_best = create_image_tag(
        images_dir,
        "11_сравнение_моделей_recall_лучший_порог.png",
        "Сравнение моделей по Recall после подбора порога"
    )

    graph_feature_importance = create_image_tag(
        images_dir,
        "12_важность_признаков.png",
        "Важность признаков"
    )

    html_content = f"""
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <title>Прогнозирование оттока клиентов банка</title>

    <style>
        body {{
            font-family: Arial, sans-serif;
            background: #f5f7fb;
            color: #222;
            margin: 0;
            padding: 0;
        }}

        .container {{
            max-width: 1150px;
            margin: 0 auto;
            padding: 30px;
        }}

        .header {{
            background: #1f2937;
            color: white;
            padding: 30px;
            border-radius: 16px;
            margin-bottom: 25px;
        }}

        .header h1 {{
            margin: 0 0 10px 0;
            font-size: 32px;
            line-height: 1.25;
        }}

        .header p {{
            margin: 0;
            font-size: 18px;
            line-height: 1.5;
        }}

        .card {{
            background: white;
            padding: 25px;
            border-radius: 16px;
            margin-bottom: 25px;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
            overflow: hidden;
        }}

        h2 {{
            color: #111827;
            margin-top: 0;
            border-bottom: 2px solid #e5e7eb;
            padding-bottom: 10px;
            font-size: 28px;
            line-height: 1.25;
        }}

        h3 {{
            color: #374151;
            margin-top: 25px;
            font-size: 21px;
        }}

        p, li {{
            font-size: 16px;
            line-height: 1.6;
        }}

        .table-wrapper {{
            width: 100%;
            overflow-x: auto;
            margin-top: 15px;
            border-radius: 10px;
        }}

        .table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 15px;
            table-layout: auto;
        }}

        .table th {{
            background: #111827;
            color: white;
            padding: 12px 10px;
            text-align: left;
            white-space: nowrap;
        }}

        .table td {{
            padding: 11px 10px;
            border-bottom: 1px solid #e5e7eb;
            white-space: nowrap;
        }}

        .table tr:nth-child(even) {{
            background: #f9fafb;
        }}

        .graph {{
            width: 100%;
            max-width: 1000px;
            display: block;
            border-radius: 12px;
            border: 1px solid #e5e7eb;
            margin-top: 15px;
        }}

        .note {{
            background: #eef2ff;
            border-left: 5px solid #4f46e5;
            padding: 15px 18px;
            border-radius: 8px;
            margin-top: 18px;
            font-size: 16px;
            line-height: 1.5;
        }}

        .footer {{
            text-align: center;
            color: #6b7280;
            padding: 20px;
            font-size: 14px;
        }}

        .metric-list {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
            gap: 12px;
            margin-top: 15px;
        }}

        .metric-item {{
            background: #f9fafb;
            padding: 14px;
            border-radius: 10px;
            border: 1px solid #e5e7eb;
        }}

        .metric-item b {{
            display: block;
            margin-bottom: 5px;
            color: #111827;
        }}
    </style>
</head>

<body>
<div class="container">

    <div class="header">
        <h1>Прогнозирование оттока клиентов банка</h1>
        <p>Проект на основе ансамблевых методов машинного обучения: Bagging, Random Forest, Gradient Boosting и LightGBM.</p>
    </div>

    <div class="card">
        <h2>1. Цель проекта</h2>
        <p>
            Цель проекта — построить модель машинного обучения, которая по характеристикам клиента банка
            прогнозирует вероятность его ухода.
        </p>
        <p>
            Такая задача полезна для банков, потому что позволяет заранее выявлять клиентов с высоким риском оттока
            и принимать меры для их удержания.
        </p>
    </div>

    <div class="card">
        <h2>2. Описание датасета</h2>
        <p>
            В проекте используется синтетический датасет на русском языке, содержащий информацию о 10 000 клиентах банков.
            Целевая переменная — <b>Отток</b>.
        </p>

        <ul>
            <li><b>0</b> — клиент остался;</li>
            <li><b>1</b> — клиент ушёл.</li>
        </ul>

        <p>
            В датасете используются демографические, финансовые и поведенческие признаки:
            возраст, город, банк, тип клиента, кредитный рейтинг, баланс, активность,
            количество жалоб, обращения в поддержку, использование мобильного приложения и другие параметры.
        </p>
    </div>

    <div class="card">
        <h2>3. Распределение целевой переменной</h2>
        <p>
            На графике показано количество клиентов, которые остались, и клиентов, которые ушли.
            Видно, что классы несбалансированы: клиентов без оттока значительно больше.
        </p>
        {graph_churn_distribution}
    </div>

    <div class="card">
        <h2>4. Разведочный анализ данных</h2>

        <h3>Возраст и отток</h3>
        <p>
            График показывает распределение возраста клиентов в зависимости от факта оттока.
        </p>
        {graph_age_churn}

        <h3>Активность клиента и отток</h3>
        <p>
            Активность клиента является важным фактором. Неактивные клиенты чаще попадают в группу риска.
        </p>
        {graph_activity_churn}

        <h3>Жалобы и отток</h3>
        <p>
            Количество жалоб может быть связано с вероятностью ухода клиента.
        </p>
        {graph_complaints_churn}

        <h3>Обращения в поддержку и отток</h3>
        <p>
            Частые обращения в поддержку могут указывать на проблемы клиента с сервисом.
        </p>
        {graph_support_churn}

        <h3>Отток по банкам</h3>
        <p>
            График показывает распределение оттока клиентов по разным банкам.
        </p>
        {graph_bank_churn}
    </div>

    <div class="card">
        <h2>5. Корреляционный анализ</h2>
        <p>
            Матрица корреляции показывает связи между числовыми признаками.
            Она помогает предварительно оценить, какие признаки могут быть связаны с целевой переменной.
        </p>
        {graph_correlation}
    </div>

    <div class="card">
        <h2>6. Использованные модели</h2>
        <p>В проекте были обучены и сравнены следующие модели:</p>

        <ul>
            <li><b>Logistic Regression</b> — базовая модель для сравнения;</li>
            <li><b>BaggingClassifier</b> — ансамблевый метод на основе бэггинга;</li>
            <li><b>RandomForestClassifier</b> — случайный лес;</li>
            <li><b>GradientBoostingClassifier</b> — градиентный бустинг;</li>
            <li><b>LightGBMClassifier</b> — современная реализация градиентного бустинга.</li>
        </ul>
    </div>

    <div class="card">
        <h2>7. Результаты моделей</h2>
        <p>
            Для оценки использовались Accuracy, Precision, Recall, F1-score и ROC-AUC.
            Также был выполнен подбор оптимального порога классификации по F1-score.
        </p>

        <div class="metric-list">
            <div class="metric-item">
                <b>ROC-AUC</b>
                Оценивает способность модели различать клиентов с оттоком и без оттока.
            </div>
            <div class="metric-item">
                <b>F1-score</b>
                Показывает баланс между Precision и Recall.
            </div>
            <div class="metric-item">
                <b>Recall</b>
                Показывает, какую долю клиентов с оттоком модель смогла найти.
            </div>
            <div class="metric-item">
                <b>Лучший порог</b>
                Порог вероятности, подобранный для улучшения F1-score.
            </div>
        </div>

        <div class="table-wrapper">
            {model_results_html}
        </div>

        <div class="note">
            По ROC-AUC лучший результат показала Logistic Regression.
            Среди ансамблевых моделей лучший F1-score после подбора порога показала LightGBMClassifier.
        </div>
    </div>

    <div class="card">
        <h2>8. Сравнение моделей на графиках</h2>

        <h3>ROC-AUC</h3>
        {graph_roc_auc}

        <h3>F1-score при стандартном пороге 0.5</h3>
        {graph_f1_default}

        <h3>F1-score после подбора порога</h3>
        {graph_f1_best}

        <h3>Recall после подбора порога</h3>
        {graph_recall_best}
    </div>

    <div class="card">
        <h2>9. Важность признаков</h2>
        <p>
            Важность признаков была рассчитана на основе модели RandomForestClassifier.
            Это позволяет определить, какие характеристики клиента сильнее всего влияли на прогноз.
        </p>

        <div class="table-wrapper">
            {feature_importance_html}
        </div>

        {graph_feature_importance}
    </div>

    <div class="card">
        <h2>10. Итоговый вывод</h2>
        <p>
            В рамках проекта была решена задача прогнозирования оттока клиентов банка.
            Были обучены базовая модель и несколько ансамблевых моделей.
        </p>
        <p>
            Результаты показали, что подбор порога классификации является важным этапом для задачи оттока,
            так как клиентов с оттоком меньше, чем клиентов без оттока.
        </p>
        <p>
            Ансамблевые методы позволяют выявлять нелинейные зависимости между признаками.
            Среди ансамблевых моделей лучший результат по F1-score после подбора порога показала LightGBMClassifier.
        </p>
    </div>

    <div class="footer">
        Проект: прогнозирование оттока клиентов банка на основе ансамблевых методов машинного обучения
    </div>

</div>
</body>
</html>
"""

    output_path = os.path.join(reports_dir, "visual_report.html")

    with open(output_path, "w", encoding="utf-8") as file:
        file.write(html_content)

    print("Визуальный HTML-отчёт успешно создан")
    print(f"Путь к файлу: {output_path}")
    print("Теперь этот HTML-файл можно копировать отдельно — графики уже встроены внутрь.")


def main() -> None:
    """
    Главная функция запуска генерации HTML-отчёта.
    """

    project_root = get_project_root()
    create_html_report(project_root)


if __name__ == "__main__":
    main()