import os

import numpy as np
import pandas as pd


def generate_bank_churn_dataset(rows_count: int = 10000, random_state: int = 42) -> pd.DataFrame:
    """
    Создаёт синтетический датасет клиентов банков для задачи прогнозирования оттока.
    """

    np.random.seed(random_state)

    customer_id = np.arange(1, rows_count + 1)

    age = np.random.normal(loc=42, scale=12, size=rows_count).astype(int)
    age = np.clip(age, 18, 75)

    gender = np.random.choice(
        ["Мужчина", "Женщина"],
        size=rows_count,
        p=[0.48, 0.52]
    )

    city = np.random.choice(
        ["Алматы", "Астана", "Шымкент", "Караганда", "Актобе", "Павлодар"],
        size=rows_count,
        p=[0.32, 0.25, 0.15, 0.10, 0.10, 0.08]
    )

    bank = np.random.choice(
        ["Kaspi Bank", "Halyk Bank", "ForteBank", "Jusan Bank", "Bereke Bank"],
        size=rows_count,
        p=[0.30, 0.28, 0.17, 0.15, 0.10]
    )

    client_type = np.random.choice(
        ["Новый", "Обычный", "Премиальный"],
        size=rows_count,
        p=[0.25, 0.60, 0.15]
    )

    service_channel = np.random.choice(
        ["Мобильное приложение", "Отделение", "Колл-центр", "Интернет-банк"],
        size=rows_count,
        p=[0.50, 0.20, 0.15, 0.15]
    )

    credit_score = np.random.normal(loc=650, scale=90, size=rows_count).astype(int)
    credit_score = np.clip(credit_score, 300, 850)

    tenure_years = np.random.randint(0, 11, size=rows_count)

    balance = np.random.normal(loc=120000, scale=65000, size=rows_count)
    balance = np.clip(balance, 0, 350000).round(2)

    products_count = np.random.choice(
        [1, 2, 3, 4],
        size=rows_count,
        p=[0.35, 0.40, 0.18, 0.07]
    )

    has_credit_card = np.random.choice(
        [0, 1],
        size=rows_count,
        p=[0.30, 0.70]
    )

    is_active_client = np.random.choice(
        [0, 1],
        size=rows_count,
        p=[0.42, 0.58]
    )

    estimated_salary = np.random.normal(loc=180000, scale=70000, size=rows_count)
    estimated_salary = np.clip(estimated_salary, 50000, 500000).round(2)

    complaints_count = np.random.poisson(lam=1.2, size=rows_count)
    complaints_count = np.clip(complaints_count, 0, 8)

    support_calls = np.random.poisson(lam=2.0, size=rows_count)
    support_calls = np.clip(support_calls, 0, 12)

    app_usage_per_month = np.random.normal(loc=12, scale=7, size=rows_count).astype(int)
    app_usage_per_month = np.clip(app_usage_per_month, 0, 30)

    loan_amount = np.random.normal(loc=900000, scale=450000, size=rows_count)
    loan_amount = np.clip(loan_amount, 0, 3000000).round(2)

    overdue_payments = np.random.poisson(lam=0.4, size=rows_count)
    overdue_payments = np.clip(overdue_payments, 0, 6)

    service_rating = np.random.normal(loc=4.0, scale=0.8, size=rows_count)
    service_rating = np.clip(service_rating, 1, 5).round(1)

    # Формируем вероятность оттока.
    # Жалобы, обращения и просрочки повышают риск ухода.
    # Активность, срок обслуживания, продукты и приложение снижают риск ухода.
    churn_score = (
        -1.1
        + 0.45 * complaints_count
        + 0.18 * support_calls
        + 0.28 * overdue_payments
        - 0.85 * is_active_client
        - 0.12 * tenure_years
        - 0.25 * products_count
        - 0.04 * app_usage_per_month
        - 0.002 * (credit_score - 650)
        + 0.015 * (age - 42)
        - 0.35 * (service_rating - 3)
    )

    churn_score += np.where(client_type == "Новый", 0.35, 0)
    churn_score += np.where(client_type == "Премиальный", -0.45, 0)
    churn_score += np.where(service_channel == "Колл-центр", 0.25, 0)
    churn_score += np.where(service_channel == "Мобильное приложение", -0.20, 0)

    churn_probability = 1 / (1 + np.exp(-churn_score))

    churn = np.random.binomial(
        n=1,
        p=churn_probability,
        size=rows_count
    )

    df = pd.DataFrame({
        "ID_клиента": customer_id,
        "Возраст": age,
        "Пол": gender,
        "Город": city,
        "Банк": bank,
        "Тип_клиента": client_type,
        "Канал_обслуживания": service_channel,
        "Кредитный_рейтинг": credit_score,
        "Срок_обслуживания_лет": tenure_years,
        "Баланс": balance,
        "Количество_продуктов": products_count,
        "Есть_кредитная_карта": has_credit_card,
        "Активный_клиент": is_active_client,
        "Примерная_зарплата": estimated_salary,
        "Количество_жалоб": complaints_count,
        "Обращения_в_поддержку": support_calls,
        "Использование_приложения_в_месяц": app_usage_per_month,
        "Сумма_кредитов": loan_amount,
        "Просрочки_по_кредитам": overdue_payments,
        "Средняя_оценка_сервиса": service_rating,
        "Отток": churn
    })

    return df


def main() -> None:
    """
    Создаёт CSV-файл с датасетом в папке data.
    """

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, "data")
    os.makedirs(data_dir, exist_ok=True)

    output_path = os.path.join(data_dir, "bank_churn_dataset.csv")

    df = generate_bank_churn_dataset(rows_count=10000, random_state=42)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("Датасет успешно создан")
    print(f"Путь к файлу: {output_path}")
    print(f"Размер датасета: {df.shape[0]} строк, {df.shape[1]} колонок")

    print("\nПервые 5 строк:")
    print(df.head())

    print("\nРаспределение целевой переменной Отток:")
    print(df["Отток"].value_counts())

    print("\nДоля классов:")
    print(df["Отток"].value_counts(normalize=True).round(3))


if __name__ == "__main__":
    main()