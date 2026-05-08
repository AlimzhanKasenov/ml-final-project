import pandas as pd
import numpy as np
import matplotlib
import seaborn as sns
import sklearn
import joblib
import lightgbm


def main() -> None:
    """
    Проверяет, что основные библиотеки проекта установлены и доступны.
    """

    print("Проверка библиотек проекта")
    print("-" * 50)

    print("pandas:", pd.__version__)
    print("numpy:", np.__version__)
    print("matplotlib:", matplotlib.__version__)
    print("seaborn:", sns.__version__)
    print("scikit-learn:", sklearn.__version__)
    print("joblib:", joblib.__version__)
    print("lightgbm:", lightgbm.__version__)

    print("-" * 50)
    print("Все библиотеки успешно подключены")


if __name__ == "__main__":
    main()