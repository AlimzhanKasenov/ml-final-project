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
    Читает CSV-файл и возвращает таблицу DataFrame.
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Файл не найден: {file_path}")

    return pd.read_csv(file_path)


def image_to_base64(image_path: str) -> str:
    """
    Встраивает изображение внутрь HTML-файла через base64.
    """

    if not os.path.exists(image_path):
        print(f"Внимание: изображение не найдено: {image_path}")
        return ""

    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

    return f"data:image/png;base64,{encoded_image}"


def create_image_tag(images_dir: str, file_name: str, alt_text: str) -> str:
    """
    Создаёт HTML-тег изображения.
    """

    image_path = os.path.join(images_dir, file_name)
    image_src = image_to_base64(image_path)

    if not image_src:
        return f"<p><b>Изображение не найдено:</b> {file_name}</p>"

    return f'<img class="graph" src="{image_src}" alt="{alt_text}">'


def prepare_model_results_table(model_results: pd.DataFrame) -> str:
    """
    Готовит компактную таблицу результатов моделей для визуального отчёта.
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
            padding: 32px;
            border-radius: 16px;
            margin-bottom: 25px;
        }}

        .header h1 {{
            margin: 0 0 12px 0;
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
            padding: 26px;
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
            margin-top: 26px;
            font-size: 21px;
        }}

        p, li {{
            font-size: 16px;
            line-height: 1.65;
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

        .warning {{
            background: #fff7ed;
            border-left: 5px solid #f97316;
            padding: 15px 18px;
            border-radius: 8px;
            margin-top: 18px;
            font-size: 16px;
            line-height: 1.5;
        }}

        .success {{
            background: #ecfdf5;
            border-left: 5px solid #10b981;
            padding: 15px 18px;
            border-radius: 8px;
            margin-top: 18px;
            font-size: 16px;
            line-height: 1.5;
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

        .tool-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 14px;
            margin-top: 16px;
        }}

        .tool-item {{
            background: #f9fafb;
            border: 1px solid #e5e7eb;
            border-radius: 12px;
            padding: 15px;
        }}

        .tool-item b {{
            display: block;
            color: #111827;
            margin-bottom: 6px;
        }}

        .footer {{
            text-align: center;
            color: #6b7280;
            padding: 20px;
            font-size: 14px;
        }}
    </style>
</head>

<body>
<div class="container">

    <div class="header">
        <h1>Прогнозирование оттока клиентов банка</h1>
        <p>
            Учебный ML-проект по классификации клиентов банка с использованием предобработки данных,
            визуального анализа, логистической регрессии и ансамблевых методов машинного обучения.
        </p>
    </div>

    <div class="card">
        <h2>1. Цель проекта</h2>
        <p>
            Цель проекта — построить модель машинного обучения, которая по характеристикам клиента банка
            прогнозирует вероятность его ухода.
        </p>
        <p>
            Такая задача актуальна для банков, потому что удержание существующего клиента обычно дешевле,
            чем привлечение нового. Если заранее определить клиентов с высоким риском оттока, банк может
            предложить им индивидуальные условия, улучшить качество обслуживания или провести удерживающую кампанию.
        </p>

        <div class="success">
            В результате проекта был построен полный ML-процесс: создание датасета, анализ данных,
            предобработка, обучение моделей, сравнение метрик, подбор порога классификации и визуальный отчёт.
        </div>
    </div>

    <div class="card">
        <h2>2. Использованные инструменты</h2>
        <p>
            Для реализации проекта использовался язык Python и основные библиотеки для анализа данных,
            визуализации и машинного обучения.
        </p>

        <div class="tool-grid">
            <div class="tool-item">
                <b>Python</b>
                Основной язык программирования проекта.
            </div>

            <div class="tool-item">
                <b>NumPy</b>
                Использовался для генерации числовых данных и работы со случайными распределениями.
            </div>

            <div class="tool-item">
                <b>Pandas</b>
                Использовался для хранения, обработки и анализа табличных данных.
            </div>

            <div class="tool-item">
                <b>Matplotlib и Seaborn</b>
                Использовались для построения графиков и разведочного анализа данных.
            </div>

            <div class="tool-item">
                <b>Scikit-learn</b>
                Использовался для разделения выборки, масштабирования признаков, обучения моделей и расчёта метрик.
            </div>

            <div class="tool-item">
                <b>LightGBM</b>
                Использовался как модель градиентного бустинга для сравнения с другими алгоритмами.
            </div>

            <div class="tool-item">
                <b>Joblib</b>
                Использовался для сохранения обученных моделей в папку models.
            </div>

            <div class="tool-item">
                <b>HTML-отчёт</b>
                Используется для удобной демонстрации результатов проекта на защите.
            </div>
        </div>
    </div>

    <div class="card">
        <h2>3. Описание датасета</h2>
        <p>
            В проекте используется синтетический датасет на русском языке, содержащий информацию о 10 000 клиентах банков.
            Такой подход позволяет показать полный ML-пайплайн без использования закрытых банковских данных.
        </p>

        <p>
            Целевая переменная — <b>Отток</b>:
        </p>

        <ul>
            <li><b>0</b> — клиент остался;</li>
            <li><b>1</b> — клиент ушёл.</li>
        </ul>

        <p>
            В датасете используются демографические, финансовые и поведенческие признаки:
            возраст, город, банк, тип клиента, кредитный рейтинг, баланс, активность,
            количество жалоб, обращения в поддержку, использование мобильного приложения,
            сумма кредитов, просрочки и средняя оценка сервиса.
        </p>

        <div class="note">
            Важно: датасет является синтетическим. Он создан для учебной демонстрации методов машинного обучения
            и не содержит реальных персональных данных клиентов.
        </div>
    </div>

    <div class="card">
        <h2>4. Распределение целевой переменной</h2>
        <p>
            Этот график показывает, сколько клиентов осталось и сколько клиентов ушло.
            Видно, что клиентов без оттока значительно больше, чем клиентов с оттоком.
        </p>

        <p>
            Такая ситуация называется <b>дисбалансом классов</b>. Для задачи оттока это нормально:
            обычно большинство клиентов остаётся, а меньшая часть уходит. Из-за этого Accuracy не всегда
            является главным показателем качества модели.
        </p>

        {graph_churn_distribution}

        <div class="warning">
            Из-за дисбаланса классов в проекте дополнительно анализируются F1-score, Recall и ROC-AUC,
            а также выполняется подбор оптимального порога классификации.
        </div>
    </div>

    <div class="card">
        <h2>5. Разведочный анализ данных</h2>
        <p>
            Разведочный анализ помогает понять структуру данных и предварительно увидеть,
            какие признаки могут быть связаны с оттоком клиента.
        </p>

        <h3>Возраст и отток</h3>
        <p>
            График показывает распределение возраста клиентов для двух групп: оставшихся и ушедших.
            По нему можно оценить, есть ли заметные различия в возрасте между клиентами с оттоком и без оттока.
        </p>
        {graph_age_churn}

        <h3>Активность клиента и отток</h3>
        <p>
            График показывает связь между активностью клиента и фактом оттока.
            Если клиент неактивен, он потенциально может чаще попадать в группу риска.
            Для банка это важный сигнал: таких клиентов можно дополнительно вовлекать в продукты и сервисы.
        </p>
        {graph_activity_churn}

        <h3>Жалобы и отток</h3>
        <p>
            Этот график показывает, как количество жалоб связано с уходом клиента.
            Чем больше жалоб у клиента, тем выше вероятность, что он недоволен сервисом и может уйти.
        </p>
        {graph_complaints_churn}

        <h3>Обращения в поддержку и отток</h3>
        <p>
            Частые обращения в поддержку могут говорить о проблемах клиента с продуктом или сервисом.
            Если клиент часто обращается за помощью, это может быть признаком повышенного риска оттока.
        </p>
        {graph_support_churn}

        <h3>Отток по банкам</h3>
        <p>
            График показывает распределение клиентов с оттоком и без оттока по банкам.
            Такой анализ помогает сравнить группы клиентов по организациям и увидеть,
            есть ли различия в структуре клиентской базы.
        </p>
        {graph_bank_churn}
    </div>

    <div class="card">
        <h2>6. Корреляционный анализ</h2>
        <p>
            Матрица корреляции показывает взаимосвязи между числовыми признаками.
            Значения ближе к 1 означают положительную связь, значения ближе к -1 — отрицательную связь,
            а значения около 0 говорят о слабой линейной зависимости.
        </p>

        <p>
            Корреляционный анализ не доказывает причинно-следственную связь, но помогает предварительно понять,
            какие признаки могут быть полезны для модели.
        </p>

        {graph_correlation}
    </div>

    <div class="card">
        <h2>7. Подготовка данных</h2>
        <p>
            Перед обучением моделей данные были подготовлены:
        </p>

        <ul>
            <li>удалён технический признак <b>ID_клиента</b>, так как он не несёт полезной информации для модели;</li>
            <li>категориальные признаки преобразованы в числовой формат с помощью one-hot encoding;</li>
            <li>данные разделены на обучающую и тестовую выборки;</li>
            <li>для логистической регрессии выполнено масштабирование признаков;</li>
            <li>для оценки качества использовалась тестовая выборка, которая не участвовала в обучении.</li>
        </ul>
    </div>

    <div class="card">
        <h2>8. Использованные модели</h2>
        <p>
            В проекте были обучены и сравнены базовая модель и несколько ансамблевых моделей.
        </p>

        <ul>
            <li><b>Logistic Regression</b> — базовая модель для сравнения результатов.</li>
            <li><b>BaggingClassifier</b> — ансамблевый метод, который обучает несколько деревьев на разных подвыборках данных.</li>
            <li><b>RandomForestClassifier</b> — случайный лес, который строит много деревьев решений и объединяет их ответы.</li>
            <li><b>GradientBoostingClassifier</b> — градиентный бустинг, где новые модели последовательно исправляют ошибки предыдущих.</li>
            <li><b>LightGBMClassifier</b> — эффективная реализация градиентного бустинга, часто применяемая для табличных данных.</li>
        </ul>
    </div>

    <div class="card">
        <h2>9. Метрики оценки качества</h2>
        <p>
            Для оценки моделей использовались Accuracy, Precision, Recall, F1-score и ROC-AUC.
            Так как задача имеет дисбаланс классов, основной упор делался не только на Accuracy,
            но и на способность модели находить клиентов с оттоком.
        </p>

        <div class="metric-list">
            <div class="metric-item">
                <b>Accuracy</b>
                Общая доля правильных ответов модели. При дисбалансе классов может быть завышенной.
            </div>
            <div class="metric-item">
                <b>Precision</b>
                Показывает, насколько точны предсказания модели для класса оттока.
            </div>
            <div class="metric-item">
                <b>Recall</b>
                Показывает, какую долю реальных клиентов с оттоком модель смогла найти.
            </div>
            <div class="metric-item">
                <b>F1-score</b>
                Баланс между Precision и Recall. Полезен при несбалансированных классах.
            </div>
            <div class="metric-item">
                <b>ROC-AUC</b>
                Показывает способность модели разделять клиентов с оттоком и без оттока.
            </div>
            <div class="metric-item">
                <b>Лучший порог</b>
                Порог вероятности, подобранный для улучшения F1-score.
            </div>
        </div>
    </div>

    <div class="card">
        <h2>10. Результаты моделей</h2>
        <p>
            В таблице показаны ключевые результаты моделей. Для защиты оставлены основные показатели:
            ROC-AUC, F1-score при стандартном пороге 0.5, лучший найденный порог,
            F1-score после подбора порога и Recall после подбора порога.
        </p>

        <div class="table-wrapper">
            {model_results_html}
        </div>

        <div class="note">
            По ROC-AUC лучший результат показала Logistic Regression.
            Среди ансамблевых моделей лучший F1-score после подбора порога показала LightGBMClassifier.
        </div>
    </div>

    <div class="card">
        <h2>11. Сравнение моделей на графиках</h2>

        <h3>Сравнение по ROC-AUC</h3>
        <p>
            Этот график показывает, насколько хорошо модели разделяют клиентов с оттоком и без оттока.
            Чем выше ROC-AUC, тем лучше модель различает классы.
        </p>
        {graph_roc_auc}

        <h3>F1-score при стандартном пороге 0.5</h3>
        <p>
            Здесь показано качество моделей при стандартном пороге классификации 0.5.
            При таком пороге некоторые модели могут плохо находить редкий класс — клиентов с оттоком.
        </p>
        {graph_f1_default}

        <h3>F1-score после подбора порога</h3>
        <p>
            Этот график показывает F1-score после подбора оптимального порога.
            Подбор порога помогает лучше сбалансировать Precision и Recall.
        </p>
        {graph_f1_best}

        <h3>Recall после подбора порога</h3>
        <p>
            Recall показывает, какую долю клиентов с реальным оттоком модель смогла найти.
            Для банковской задачи это важная метрика, потому что пропуск клиента с высоким риском ухода
            может быть дороже, чем лишнее попадание клиента в группу риска.
        </p>
        {graph_recall_best}
    </div>

    <div class="card">
        <h2>12. Важность признаков</h2>
        <p>
            Важность признаков была рассчитана на основе модели RandomForestClassifier.
            Этот анализ показывает, какие признаки сильнее всего влияли на предсказания модели.
        </p>

        <div class="table-wrapper">
            {feature_importance_html}
        </div>

        {graph_feature_importance}

        <div class="note">
            Важность признаков помогает интерпретировать модель. Например, если высокую важность имеют жалобы,
            срок обслуживания, активность клиента или использование приложения, банк может использовать эти признаки
            для построения стратегии удержания клиентов.
        </div>
    </div>

    <div class="card">
        <h2>13. Итоговый вывод</h2>
        <p>
            В рамках проекта была решена задача прогнозирования оттока клиентов банка.
            Был создан синтетический датасет на 10 000 клиентов, проведён разведочный анализ,
            выполнена предобработка данных и обучены несколько моделей машинного обучения.
        </p>

        <p>
            В проекте были использованы как базовая модель Logistic Regression, так и ансамблевые методы:
            BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier и LightGBMClassifier.
        </p>

        <p>
            Так как в данных присутствует дисбаланс классов, основное внимание уделялось не только Accuracy,
            но и метрикам F1-score, Recall и ROC-AUC. Дополнительно был выполнен подбор порога классификации,
            что позволило улучшить качество поиска клиентов с риском оттока.
        </p>

        <div class="success">
            Главный итог: проект показывает полный цикл решения ML-задачи классификации —
            от подготовки данных до сравнения моделей, анализа метрик, интерпретации признаков
            и формирования визуального отчёта для защиты.
        </div>
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
    print("Графики встроены внутрь HTML-файла")


def main() -> None:
    """
    Запускает создание HTML-отчёта.
    """

    project_root = get_project_root()
    create_html_report(project_root)


if __name__ == "__main__":
    main()