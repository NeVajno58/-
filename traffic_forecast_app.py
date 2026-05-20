import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")


# ============================================================
# Загрузка данных
# ============================================================

def load_data(file_path: str) -> pd.DataFrame:
    data = pd.read_csv(file_path)
    return data


def filter_road_segment(
    data: pd.DataFrame,
    segment: str | None = None
) -> pd.DataFrame:
    """
    Функция выбора участка дороги.
    В исходном Metro Interstate Traffic Volume Dataset отдельного столбца road_segment нет.
    Однако функция добавлена для поддержки таких данных в будущем.
    """
    if "road_segment" not in data.columns:
        print("Столбец road_segment не найден. Используется весь набор данных.")
        return data

    if segment is None or segment == "Все участки":
        return data

    return data[data["road_segment"] == segment]


# ============================================================
# Предобработка данных
# ============================================================

def remove_anomalies_iqr(series: pd.Series) -> pd.Series:
    """
    Простая обработка аномалий по правилу межквартильного размаха.
    Аномальные значения заменяются на NaN и затем восстанавливаются
    линейной интерполяцией.
    """

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    cleaned = series.copy()
    anomaly_mask = (cleaned < lower_bound) | (cleaned > upper_bound)

    print("Количество обнаруженных аномальных значений:", int(anomaly_mask.sum()))

    cleaned[anomaly_mask] = np.nan
    cleaned = cleaned.interpolate(method="linear")
    cleaned = cleaned.ffill().bfill()

    return cleaned


def prepare_series(
    data: pd.DataFrame,
    remove_anomalies: bool = False
) -> pd.Series:
    data = data.copy()

    if "date_time" not in data.columns:
        raise ValueError("В данных отсутствует столбец date_time.")

    if "traffic_volume" not in data.columns:
        raise ValueError("В данных отсутствует столбец traffic_volume.")

    data["date_time"] = pd.to_datetime(data["date_time"])
    data = data.sort_values("date_time")
    data = data.set_index("date_time")

    series = data["traffic_volume"]

    # Удаление дублирующихся временных меток.
    # Если для одного времени есть несколько записей, сохраняется первая.
    duplicates_count = int(series.index.duplicated().sum())
    print("Количество дублирующихся временных меток:", duplicates_count)

    series = series[~series.index.duplicated(keep="first")]

    # Приведение к регулярной почасовой сетке.
    series = series.resample("h").mean()

    # Линейная интерполяция пропусков.
    missing_count = int(series.isna().sum())
    print("Количество пропусков после ресемплирования:", missing_count)

    series = series.interpolate(method="linear")
    series = series.ffill().bfill()

    if remove_anomalies:
        series = remove_anomalies_iqr(series)

    return series


def _safe_feature_name(name: str) -> str:
    """
    Преобразует имя признака к безопасному виду для Prophet/SARIMAX.
    """
    safe = "".join(ch if ch.isalnum() else "_" for ch in str(name))
    safe = safe.strip("_")
    return safe or "feature"


def prepare_all_available_regressors(
    data: pd.DataFrame,
    target_column: str = "traffic_volume",
    date_column: str = "date_time",
    freq: str = "h",
    max_categories_per_column: int = 20,
    include_calendar_features: bool = True
) -> pd.DataFrame:
    """
    Подготовка внешних признаков из всех доступных столбцов датасета.

    В регрессоры включаются:
    1. календарные признаки: hour, day_of_week, is_weekend, month;
    2. все числовые признаки, кроме целевого столбца traffic_volume;
    3. категориальные признаки, преобразованные one-hot encoding;
    4. специальный бинарный признак is_holiday, если есть столбец holiday.

    Столбцы date_time и traffic_volume исключаются, так как date_time является
    временной осью, а traffic_volume — прогнозируемой величиной.
    """
    data = data.copy()

    if date_column not in data.columns:
        raise ValueError(f"В данных отсутствует столбец {date_column}.")

    data[date_column] = pd.to_datetime(data[date_column])
    data = data.sort_values(date_column).set_index(date_column)

    regular_index = pd.date_range(
        start=data.index.min().floor(freq),
        end=data.index.max().ceil(freq),
        freq=freq
    )

    regressors = pd.DataFrame(index=regular_index)

    if include_calendar_features:
        regressors["hour"] = regressors.index.hour
        regressors["day_of_week"] = regressors.index.dayofweek
        regressors["is_weekend"] = regressors.index.dayofweek.isin([5, 6]).astype(int)
        regressors["month"] = regressors.index.month

    excluded_columns = {target_column, date_column}
    feature_columns = [column for column in data.columns if column not in excluded_columns]

    for column in feature_columns:
        safe_column = _safe_feature_name(column)
        values = data[column]

        # Числовые и булевы признаки используются напрямую.
        if pd.api.types.is_numeric_dtype(values) or pd.api.types.is_bool_dtype(values):
            numeric_values = pd.to_numeric(values, errors="coerce")
            numeric_values = numeric_values.groupby(data.index).mean()
            numeric_values = numeric_values.resample(freq).mean()

            regressors[safe_column] = (
                numeric_values
                .reindex(regular_index)
                .interpolate(method="linear")
                .ffill()
                .bfill()
                .fillna(0)
            )

        # Текстовые признаки кодируются в набор бинарных признаков.
        else:
            categorical_values = (
                values
                .astype("string")
                .fillna("missing")
                .str.strip()
                .replace("", "missing")
            )

            if column == "holiday":
                holiday_binary = (
                    categorical_values
                    .str.lower()
                    .ne("none")
                    .astype(int)
                )
                holiday_binary = holiday_binary.groupby(data.index).max()
                holiday_binary = holiday_binary.resample(freq).max()
                regressors["is_holiday"] = (
                    holiday_binary
                    .reindex(regular_index)
                    .fillna(0)
                    .astype(int)
                )

            # Ограничение числа категорий защищает модель от слишком большого
            # количества разреженных признаков при загрузке пользовательских CSV.
            top_categories = categorical_values.value_counts().head(max_categories_per_column).index
            categorical_values = categorical_values.where(
                categorical_values.isin(top_categories),
                other="other"
            )

            encoded = pd.get_dummies(
                categorical_values,
                prefix=safe_column,
                dtype=float
            )
            encoded.index = data.index

            encoded = encoded.groupby(encoded.index).max()
            encoded = encoded.resample(freq).max()
            encoded = encoded.reindex(regular_index).fillna(0)

            for encoded_column in encoded.columns:
                regressors[_safe_feature_name(encoded_column)] = encoded[encoded_column]

    # Удаление полностью пустых и дублирующихся столбцов.
    regressors = regressors.replace([np.inf, -np.inf], np.nan)
    regressors = regressors.ffill().bfill().fillna(0)
    regressors = regressors.loc[:, ~regressors.columns.duplicated()]

    return regressors.astype(float)


def remove_constant_regressors(
    regressors: pd.DataFrame,
    reference_index: pd.Index | None = None
) -> pd.DataFrame:
    """
    Удаляет признаки, которые являются константными на обучающем окне.
    Такие признаки не несут информации для модели и могут ухудшать устойчивость
    оценки параметров, особенно в Prophet и SARIMAX.
    """
    if regressors is None or regressors.empty:
        return pd.DataFrame(index=reference_index)

    if reference_index is not None:
        frame = regressors.reindex(reference_index).ffill().bfill().fillna(0)
    else:
        frame = regressors.copy().ffill().bfill().fillna(0)

    variable_columns = [column for column in frame.columns if frame[column].nunique(dropna=False) > 1]
    return regressors[variable_columns].copy()


def prepare_prophet_regressors(data: pd.DataFrame) -> pd.DataFrame:
    """
    Подготовка регрессоров для Prophet из всех доступных признаков датасета.
    """
    return prepare_all_available_regressors(data)


def prepare_sarimax_regressors(data: pd.DataFrame) -> pd.DataFrame:
    """
    Подготовка внешних регрессоров для SARIMAX из всех доступных признаков датасета.
    """
    return prepare_all_available_regressors(data)


def split_train_test(series: pd.Series, train_part: float = 0.8):
    train_size = int(len(series) * train_part)
    train = series.iloc[:train_size]
    test = series.iloc[train_size:]
    return train, test


# ============================================================
# Метрики качества
# ============================================================

def calc_metrics(actual: pd.Series, predicted: pd.Series) -> tuple:
    predicted = pd.Series(predicted, index=actual.index)

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))

    # MAPE не определён при actual = 0, поэтому нулевые значения исключаются.
    mask = actual != 0

    if mask.sum() == 0:
        mape = np.nan
    else:
        mape = np.mean(
            np.abs((actual[mask] - predicted[mask]) / actual[mask])
        ) * 100

    return mae, rmse, mape




def select_best_model(metrics_df: pd.DataFrame, metric: str = "MAE") -> str:
    """
    Выбор лучшей модели по заданной метрике качества.
    Чем меньше MAE, RMSE или MAPE, тем лучше прогноз.
    """
    if metrics_df.empty:
        raise ValueError("Таблица метрик пуста.")

    if metric not in metrics_df.columns:
        raise ValueError(f"Метрика {metric} отсутствует в таблице результатов.")

    best_row = metrics_df.loc[metrics_df[metric].idxmin()]
    return str(best_row["model"])


def traffic_level(value: float) -> str:
    """
    Условная интерпретация уровня дорожной загруженности по traffic_volume.
    Пороговые значения являются демонстрационными и могут быть уточнены
    по квантилям конкретного датасета или пропускной способности дороги.
    """
    if value < 1500:
        return "Низкая загруженность"
    if value < 3500:
        return "Умеренная загруженность"
    if value < 5500:
        return "Высокая загруженность"
    return "Очень высокая загруженность"


def detect_congestion_periods(
    forecast: pd.Series,
    threshold: float = 5500
) -> pd.DataFrame:
    """
    Поиск временных интервалов с прогнозируемой высокой загруженностью.
    """
    congestion = forecast[forecast >= threshold]

    return pd.DataFrame({
        "date_time": congestion.index,
        "forecast_traffic_volume": congestion.values,
        "traffic_level": [traffic_level(value) for value in congestion.values]
    })


def error_by_hour(actual: pd.Series, predicted: pd.Series) -> pd.DataFrame:
    """
    Анализ средней абсолютной ошибки прогноза по часам суток.
    Позволяет определить, в какие часы модель ошибается сильнее.
    """
    predicted = pd.Series(predicted, index=actual.index)

    errors = pd.DataFrame({
        "actual": actual,
        "predicted": predicted
    })

    errors["absolute_error"] = (errors["actual"] - errors["predicted"]).abs()
    errors["hour"] = errors.index.hour

    return (
        errors
        .groupby("hour", as_index=False)["absolute_error"]
        .mean()
        .rename(columns={"absolute_error": "mean_absolute_error"})
    )


def summarize_validation_results(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Усреднение метрик по контрольным интервалам.
    Поддерживает как длинный формат: model, MAE, RMSE, MAPE,
    так и широкий формат из rolling_validation_models.
    """
    if results_df.empty:
        return pd.DataFrame(columns=["model", "MAE", "RMSE", "MAPE"])

    if {"model", "MAE", "RMSE", "MAPE"}.issubset(results_df.columns):
        return (
            results_df
            .groupby("model", as_index=False)[["MAE", "RMSE", "MAPE"]]
            .mean()
            .sort_values("MAE")
        )

    rows = []
    for model in ["ARIMA", "SARIMA", "Prophet"]:
        mae_col = f"{model}_MAE"
        rmse_col = f"{model}_RMSE"
        mape_col = f"{model}_MAPE"

        if {mae_col, rmse_col, mape_col}.issubset(results_df.columns):
            rows.append({
                "model": model,
                "MAE": results_df[mae_col].mean(),
                "RMSE": results_df[rmse_col].mean(),
                "MAPE": results_df[mape_col].mean(),
            })

    return pd.DataFrame(rows).sort_values("MAE") if rows else pd.DataFrame()


def build_text_report(
    best_model: str,
    metrics_df: pd.DataFrame,
    congestion_df: pd.DataFrame,
    error_hour_df: pd.DataFrame | None = None
) -> str:
    """
    Формирование текстового отчёта по результатам прогнозирования.
    """
    report = "ОТЧЁТ ПО ПРОГНОЗИРОВАНИЮ ТРАНСПОРТНОГО ПОТОКА\n"
    report += "=" * 58 + "\n\n"
    report += f"Лучшая модель по выбранной метрике: {best_model}\n\n"

    report += "Метрики качества моделей:\n"
    report += metrics_df.round(2).to_string(index=False)
    report += "\n\n"

    if congestion_df.empty:
        report += "Периоды прогнозируемой высокой загруженности не обнаружены.\n\n"
    else:
        report += "Периоды прогнозируемой высокой загруженности:\n"
        report += congestion_df.to_string(index=False)
        report += "\n\n"

    if error_hour_df is not None and not error_hour_df.empty:
        report += "Средняя абсолютная ошибка по часам суток:\n"
        report += error_hour_df.round(2).to_string(index=False)
        report += "\n"

    return report


def compare_prophet_with_and_without_regressors(
    train: pd.Series,
    test: pd.Series,
    horizon: int = 24,
    train_window_hours: int = 720,
    regressors: pd.DataFrame | None = None
) -> pd.DataFrame:
    """
    Сравнение Prophet без внешних признаков и Prophet с дополнительными регрессорами.
    Используется для оценки влияния погодных и календарных факторов на качество прогноза.
    """
    control_test = test.iloc[:horizon]
    rows = []

    forecast_basic, _ = forecast_prophet(
        train=train,
        test=test,
        horizon=horizon,
        train_window_hours=train_window_hours,
        regressors=None
    )
    rows.append(["Prophet без внешних регрессоров", *calc_metrics(control_test, forecast_basic)])

    if regressors is not None and not regressors.empty:
        forecast_extra, _ = forecast_prophet(
            train=train,
            test=test,
            horizon=horizon,
            train_window_hours=train_window_hours,
            regressors=regressors
        )
        rows.append(["Prophet с внешними регрессорами", *calc_metrics(control_test, forecast_extra)])

    return pd.DataFrame(rows, columns=["model", "MAE", "RMSE", "MAPE"])

def add_calendar_features(frame: pd.DataFrame, date_column: str = "ds") -> pd.DataFrame:
    result = frame.copy()
    dates = pd.to_datetime(result[date_column])

    result["hour"] = dates.dt.hour
    result["day_of_week"] = dates.dt.dayofweek
    result["is_weekend"] = dates.dt.dayofweek.isin([5, 6]).astype(int)
    result["month"] = dates.dt.month

    return result


# ============================================================
# Визуальный анализ временного ряда
# ============================================================

def plot_monthly_series(series: pd.Series) -> None:
    monthly_series = series.resample("ME").mean()
    monthly_rolling = monthly_series.rolling(window=3).mean()

    plt.figure(figsize=(14, 6))
    plt.plot(monthly_series, label="Среднемесячная интенсивность")
    plt.plot(
        monthly_rolling,
        label="Скользящее среднее за 3 месяца",
        linewidth=2
    )
    plt.title("Среднемесячная интенсивность дорожного движения")
    plt.xlabel("Дата")
    plt.ylabel("Среднее значение traffic_volume")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_week(series: pd.Series) -> None:
    week_series = series["2016-10-01":"2016-10-07"]

    plt.figure(figsize=(14, 6))
    plt.plot(week_series)
    plt.title("Интенсивность дорожного движения за одну неделю")
    plt.xlabel("Дата и время")
    plt.ylabel("Интенсивность транспортного потока")
    plt.grid(True)
    plt.show()


def plot_rolling_mean(series: pd.Series) -> None:
    rolling_mean = series.rolling(window=24).mean()

    plt.figure(figsize=(14, 6))
    plt.plot(series, label="Исходный ряд", alpha=0.5)
    plt.plot(
        rolling_mean,
        label="Скользящее среднее за 24 часа",
        linewidth=2
    )
    plt.title("Скользящее среднее интенсивности дорожного движения")
    plt.xlabel("Дата и время")
    plt.ylabel("Интенсивность транспортного потока")
    plt.legend()
    plt.grid(True)
    plt.show()


def check_stationarity(series: pd.Series) -> None:
    adf_result = adfuller(series.dropna())

    print("ADF statistic:", adf_result[0])
    print("p-value:", adf_result[1])
    print("Critical values:")

    for key, value in adf_result[4].items():
        print(key, value)

    diff_series = series.diff().dropna()
    adf_result_diff = adfuller(diff_series)

    print("\nADF statistic after differencing:", adf_result_diff[0])
    print("p-value after differencing:", adf_result_diff[1])


def plot_acf_pacf(series: pd.Series) -> None:
    diff_series = series.diff().dropna()

    plot_acf(diff_series, lags=40)
    plt.title("Автокорреляционная функция")
    plt.show()

    plot_pacf(diff_series, lags=40)
    plt.title("Частичная автокорреляционная функция")
    plt.show()


# ============================================================
# ARIMA
# ============================================================

def forecast_arima(
    train: pd.Series,
    test: pd.Series,
    horizon: int = 24,
    order=(2, 1, 2),
    train_window_hours: int = 720
) -> tuple[pd.Series, pd.DataFrame]:
    train_tail = train.iloc[-train_window_hours:]
    control_test = test.iloc[:horizon]

    model = ARIMA(train_tail, order=order)
    model_fit = model.fit()

    forecast_result = model_fit.get_forecast(steps=horizon)

    forecast_mean = forecast_result.predicted_mean
    forecast_interval = forecast_result.conf_int()

    forecast_mean = pd.Series(
        forecast_mean.values,
        index=control_test.index
    )

    forecast_interval.index = control_test.index

    return forecast_mean, forecast_interval


# ============================================================
# SARIMA
# ============================================================

def forecast_sarima(
    train: pd.Series,
    test: pd.Series,
    horizon: int = 24,
    order=(2, 1, 2),
    seasonal_order=(1, 0, 1, 24),
    train_window_hours: int = 720
) -> tuple[pd.Series, pd.DataFrame]:
    train_tail = train.iloc[-train_window_hours:]
    control_test = test.iloc[:horizon]

    model = SARIMAX(
        train_tail,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    model_fit = model.fit(disp=False)
    forecast_result = model_fit.get_forecast(steps=horizon)

    forecast_mean = forecast_result.predicted_mean
    forecast_interval = forecast_result.conf_int()

    forecast_mean = pd.Series(
        forecast_mean.values,
        index=control_test.index
    )

    forecast_interval.index = control_test.index

    return forecast_mean, forecast_interval




# ============================================================
# SARIMAX
# ============================================================

def forecast_sarimax(
    train: pd.Series,
    test: pd.Series,
    horizon: int = 24,
    order=(2, 1, 2),
    seasonal_order=(1, 0, 1, 24),
    train_window_hours: int = 720,
    regressors: pd.DataFrame | None = None
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Прогнозирование с помощью SARIMAX.

    SARIMAX отличается от SARIMA тем, что помимо прошлых значений ряда
    и сезонности использует внешние регрессоры: календарные, погодные
    и праздничные признаки.
    """
    if regressors is None or regressors.empty:
        raise ValueError(
            "Для SARIMAX нужны внешние регрессоры. "
            "Сформируйте их функцией prepare_sarimax_regressors()."
        )

    train_tail = train.iloc[-train_window_hours:]
    control_test = test.iloc[:horizon]

    # Используются все доступные информативные регрессоры.
    # Признаки, которые на обучающем окне являются константными, удаляются,
    # так как они не улучшают прогноз и могут ухудшать устойчивость модели.
    active_regressors = remove_constant_regressors(
        regressors,
        reference_index=train_tail.index
    )

    if active_regressors.empty:
        raise ValueError("Для SARIMAX не осталось информативных внешних регрессоров.")

    exog_train = active_regressors.reindex(train_tail.index).ffill().bfill().fillna(0)
    exog_test = active_regressors.reindex(control_test.index).ffill().bfill().fillna(0)

    model = SARIMAX(
        train_tail,
        exog=exog_train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    model_fit = model.fit(disp=False)
    forecast_result = model_fit.get_forecast(
        steps=horizon,
        exog=exog_test
    )

    forecast_mean = pd.Series(
        forecast_result.predicted_mean.values,
        index=control_test.index
    )

    forecast_interval = forecast_result.conf_int()
    forecast_interval.index = control_test.index

    return forecast_mean, forecast_interval


# ============================================================
# Prophet
# ============================================================

def forecast_prophet(
    train: pd.Series,
    test: pd.Series,
    horizon: int = 24,
    train_window_hours: int = 720,
    regressors: pd.DataFrame | None = None
) -> tuple[pd.Series, pd.DataFrame]:
    try:
        from prophet import Prophet
    except ImportError as exc:
        raise ImportError(
            "Библиотека prophet не установлена. Установите её командой: "
            "pip install prophet"
        ) from exc

    train_tail = train.iloc[-train_window_hours:]
    control_test = test.iloc[:horizon]

    prophet_train = train_tail.reset_index()
    prophet_train.columns = ["ds", "y"]
    prophet_train = add_calendar_features(prophet_train)

    future = pd.DataFrame({
        "ds": list(train_tail.index) + list(control_test.index)
    })
    future = add_calendar_features(future)

    regressor_columns = ["hour", "day_of_week", "is_weekend", "month"]

    if regressors is not None and not regressors.empty:
        # Prophet получает все доступные внешние признаки, подготовленные из CSV.
        # Константные признаки на обучающем окне исключаются автоматически.
        active_regressors = remove_constant_regressors(
            regressors,
            reference_index=train_tail.index
        )
        extra_columns = [
            column for column in active_regressors.columns
            if column not in regressor_columns
        ]

        if extra_columns:
            train_extra = (
                active_regressors
                .reindex(train_tail.index)[extra_columns]
                .ffill()
                .bfill()
                .fillna(0)
            )
            future_extra = (
                active_regressors
                .reindex(future["ds"])[extra_columns]
                .ffill()
                .bfill()
                .fillna(0)
            )

            for column in extra_columns:
                prophet_train[column] = train_extra[column].values
                future[column] = future_extra[column].values

            regressor_columns.extend(extra_columns)

    # Оставляем только признаки, которые меняются на обучающем окне.
    # Это важно для коротких окон и пользовательских датасетов.
    regressor_columns = [
        column for column in regressor_columns
        if column in prophet_train.columns and prophet_train[column].nunique(dropna=False) > 1
    ]

    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=False,
        interval_width=0.95
    )

    for regressor in regressor_columns:
        model.add_regressor(regressor)

    model.fit(prophet_train)

    forecast_df = model.predict(future)
    forecast_tail = forecast_df.tail(horizon)

    forecast_mean = pd.Series(
        forecast_tail["yhat"].values,
        index=control_test.index
    )

    forecast_interval = pd.DataFrame(
        {
            "lower": forecast_tail["yhat_lower"].values,
            "upper": forecast_tail["yhat_upper"].values
        },
        index=control_test.index
    )

    return forecast_mean, forecast_interval


# ============================================================
# Сравнение моделей
# ============================================================

def compare_models(
    series: pd.Series,
    horizon: int = 24,
    include_prophet: bool = True,
    sarimax_regressors: pd.DataFrame | None = None,
    prophet_regressors: pd.DataFrame | None = None
) -> pd.DataFrame:
    train, test = split_train_test(series)
    control_test = test.iloc[:horizon]

    results = []

    arima_forecast, _ = forecast_arima(
        train=train,
        test=test,
        horizon=horizon,
        order=(2, 1, 2)
    )

    results.append([
        "ARIMA(2,1,2)",
        *calc_metrics(control_test, arima_forecast)
    ])


    sarima_forecast, _ = forecast_sarima(
        train=train,
        test=test,
        horizon=horizon,
        order=(2, 1, 2),
        seasonal_order=(1, 0, 1, 24)
    )

    results.append([
        "SARIMA(2,1,2)(1,0,1,24)",
        *calc_metrics(control_test, sarima_forecast)
    ])

    sarimax_forecast = None

    if sarimax_regressors is not None and not sarimax_regressors.empty:
        try:
            sarimax_forecast, _ = forecast_sarimax(
                train=train,
                test=test,
                horizon=horizon,
                order=(2, 1, 2),
                seasonal_order=(1, 0, 1, 24),
                regressors=sarimax_regressors
            )

            results.append([
                "SARIMAX(2,1,2)(1,0,1,24)",
                *calc_metrics(control_test, sarimax_forecast)
            ])

        except Exception as error:
            print("SARIMAX не был рассчитан:", error)

    prophet_forecast = None

    if include_prophet:
        try:
            prophet_forecast, _ = forecast_prophet(
                train=train,
                test=test,
                horizon=horizon,
                regressors=prophet_regressors
            )

            results.append([
                "Prophet",
                *calc_metrics(control_test, prophet_forecast)
            ])

        except Exception as error:
            print("Prophet не был рассчитан:", error)

    results_df = pd.DataFrame(
        results,
        columns=["model", "MAE", "RMSE", "MAPE"]
    )

    print("\nСравнение моделей:")
    print(results_df)

    plt.figure(figsize=(14, 6))
    plt.plot(control_test, label="Фактические значения", marker="o")
    plt.plot(arima_forecast, label="ARIMA(2,1,2)", marker="o")
    plt.plot(sarima_forecast, label="SARIMA(2,1,2)(1,0,1,24)", marker="o")

    if sarimax_forecast is not None:
        plt.plot(sarimax_forecast, label="SARIMAX(2,1,2)(1,0,1,24)", marker="o")

    if prophet_forecast is not None:
        plt.plot(prophet_forecast, label="Prophet", marker="o")

    plt.title("Сравнение прогноза моделей на 24 часа")
    plt.xlabel("Дата и время")
    plt.ylabel("Интенсивность транспортного потока")
    plt.legend()
    plt.grid(True)
    plt.show()

    return results_df


def build_three_model_comparison(
    train: pd.Series,
    test: pd.Series,
    horizon: int,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
    train_window_hours: int,
    regressors: pd.DataFrame | None = None,
    sarimax_regressors: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    control_test = test.iloc[:horizon]

    arima_forecast, _ = forecast_arima(
        train=train,
        test=test,
        horizon=horizon,
        order=order,
        train_window_hours=train_window_hours
    )

    sarima_forecast, _ = forecast_sarima(
        train=train,
        test=test,
        horizon=horizon,
        order=order,
        seasonal_order=seasonal_order,
        train_window_hours=train_window_hours
    )

    prophet_forecast, _ = forecast_prophet(
        train=train,
        test=test,
        horizon=horizon,
        train_window_hours=train_window_hours,
        regressors=regressors
    )

    chart_data = {
        "Фактические значения": control_test,
        "ARIMA": arima_forecast,
        "SARIMA": sarima_forecast,
        "Prophet": prophet_forecast,
    }

    metric_rows = [
        ["ARIMA", *calc_metrics(control_test, arima_forecast)],
        ["SARIMA", *calc_metrics(control_test, sarima_forecast)],
        ["Prophet", *calc_metrics(control_test, prophet_forecast)],
    ]

    if sarimax_regressors is not None and not sarimax_regressors.empty:
        sarimax_forecast, _ = forecast_sarimax(
            train=train,
            test=test,
            horizon=horizon,
            order=order,
            seasonal_order=seasonal_order,
            train_window_hours=train_window_hours,
            regressors=sarimax_regressors
        )

        chart_data["SARIMAX"] = sarimax_forecast
        metric_rows.append([
            "SARIMAX",
            *calc_metrics(control_test, sarimax_forecast)
        ])

    chart_df = pd.DataFrame(chart_data)

    metrics_df = pd.DataFrame(
        metric_rows,
        columns=["model", "MAE", "RMSE", "MAPE"]
    )

    return chart_df, metrics_df


def build_interval_model_comparison(
    series: pd.Series,
    test: pd.Series,
    start_day,
    end_day,
    horizon: int,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
    train_window_hours: int,
    regressors: pd.DataFrame | None = None,
    sarimax_regressors: pd.DataFrame | None = None
) -> pd.DataFrame:
    rows = []
    start_day = pd.Timestamp(start_day)
    end_day = pd.Timestamp(end_day)

    for day in pd.date_range(start=start_day, end=end_day, freq="D"):
        day_test = test.loc[day:].iloc[:horizon]

        if len(day_test) < horizon:
            continue

        day_start = day_test.index[0]
        day_train = series.loc[series.index < day_start]

        if len(day_train) < 48:
            continue

        _, metrics_df = build_three_model_comparison(
            train=day_train,
            test=day_test,
            horizon=horizon,
            order=order,
            seasonal_order=seasonal_order,
            train_window_hours=train_window_hours,
            regressors=regressors
        )

        for _, row in metrics_df.iterrows():
            rows.append({
                "date": day_start.date(),
                "interval_start": day_test.index[0],
                "interval_end": day_test.index[-1],
                "model": row["model"],
                "MAE": row["MAE"],
                "RMSE": row["RMSE"],
                "MAPE": row["MAPE"],
            })

    return pd.DataFrame(
        rows,
        columns=["date", "interval_start", "interval_end", "model", "MAE", "RMSE", "MAPE"]
    )


# ============================================================
# Скользящая проверка
# ============================================================

def rolling_validation_models(
    series: pd.Series,
    intervals: int = 7,
    horizon: int = 24,
    history_days: int = 14,
    include_prophet: bool = False
) -> pd.DataFrame:
    train_size = int(len(series) * 0.8)
    rows = []

    for i in range(intervals):
        start = train_size + i * horizon
        end = start + horizon

        if end > len(series):
            break

        history_start = max(0, start - history_days * 24)

        history = series.iloc[history_start:start]
        actual = series.iloc[start:end]

        if len(history) < 48 or len(actual) < horizon:
            continue

        arima_forecast, _ = forecast_arima(
            train=history,
            test=actual,
            horizon=horizon,
            order=(2, 1, 2),
            train_window_hours=len(history)
        )

        sarima_forecast, _ = forecast_sarima(
            train=history,
            test=actual,
            horizon=horizon,
            order=(2, 1, 2),
            seasonal_order=(1, 0, 1, 24),
            train_window_hours=len(history)
        )

        arima_mae, arima_rmse, arima_mape = calc_metrics(actual, arima_forecast)
        sarima_mae, sarima_rmse, sarima_mape = calc_metrics(actual, sarima_forecast)

        row = {
            "interval": f"{actual.index[0]} - {actual.index[-1]}",
            "ARIMA_MAE": arima_mae,
            "ARIMA_RMSE": arima_rmse,
            "ARIMA_MAPE": arima_mape,
            "SARIMA_MAE": sarima_mae,
            "SARIMA_RMSE": sarima_rmse,
            "SARIMA_MAPE": sarima_mape,
        }

        if include_prophet:
            try:
                prophet_forecast, _ = forecast_prophet(
                    train=history,
                    test=actual,
                    horizon=horizon,
                    train_window_hours=len(history)
                )

                prophet_mae, prophet_rmse, prophet_mape = calc_metrics(
                    actual,
                    prophet_forecast
                )

                row["Prophet_MAE"] = prophet_mae
                row["Prophet_RMSE"] = prophet_rmse
                row["Prophet_MAPE"] = prophet_mape

            except Exception as error:
                print("Prophet не был рассчитан на интервале:", error)

        rows.append(row)

    results = pd.DataFrame(rows)

    if results.empty:
        print("Недостаточно данных для скользящей проверки.")
        return results

    print("\nСкользящая проверка:")
    print(results)

    if "SARIMA_MAE" in results.columns:
        print(
            "\nSARIMA MAE меньше ARIMA в",
            int((results["SARIMA_MAE"] < results["ARIMA_MAE"]).sum()),
            "из",
            len(results),
            "интервалов"
        )

        print(
            "SARIMA MAPE меньше ARIMA в",
            int((results["SARIMA_MAPE"] < results["ARIMA_MAPE"]).sum()),
            "из",
            len(results),
            "интервалов"
        )

    return results


# ============================================================
# Интервальный прогноз и визуализация
# ============================================================

def plot_forecast_with_interval(
    actual: pd.Series,
    forecast: pd.Series,
    interval: pd.DataFrame,
    title: str
) -> None:
    plt.figure(figsize=(14, 6))

    plt.plot(actual, label="Фактические значения", marker="o")
    plt.plot(forecast, label="Точечный прогноз", marker="o")

    plt.fill_between(
        interval.index,
        interval.iloc[:, 0],
        interval.iloc[:, 1],
        alpha=0.2,
        label="95% доверительный интервал"
    )

    plt.title(title)
    plt.xlabel("Дата и время")
    plt.ylabel("Интенсивность транспортного потока")
    plt.legend()
    plt.grid(True)
    plt.show()


# ============================================================
# Streamlit-интерфейс
# ============================================================

def run_streamlit_interface() -> None:
    import streamlit as st

    st.set_page_config(
        page_title="Прогнозирование интенсивности транспортного потока",
        layout="wide"
    )

    st.title("Прогнозирование интенсивности транспортного потока")

    st.write(
        "Прототип позволяет загрузить CSV-файл, выбрать модель прогнозирования, "
        "настроить параметры, построить точечный и интервальный прогноз, "
        "рассчитать метрики качества и выгрузить результаты."
    )

    uploaded_file = st.file_uploader(
        "Загрузите CSV-файл с транспортными данными",
        type=["csv"]
    )

    if uploaded_file is None:
        st.info("Загрузите CSV-файл для начала работы.")
        return

    data = pd.read_csv(uploaded_file)

    st.sidebar.header("Настройки данных")

    if "road_segment" in data.columns:
        segments = ["Все участки"] + sorted(
            data["road_segment"].dropna().unique().tolist()
        )

        selected_segment = st.sidebar.selectbox(
            "Выберите участок дороги",
            segments
        )

        data = filter_road_segment(data, selected_segment)
    else:
        st.sidebar.info(
            "В датасете нет столбца road_segment. "
            "Анализ выполняется для всего набора данных."
        )

    remove_anomalies = st.sidebar.checkbox(
        "Выполнить обработку аномалий по IQR",
        value=False
    )

    series = prepare_series(
        data,
        remove_anomalies=remove_anomalies
    )
    prophet_regressors = prepare_prophet_regressors(data)
    sarimax_regressors = prepare_sarimax_regressors(data)

    st.subheader("Фрагмент подготовленного временного ряда")
    st.line_chart(series.tail(24 * 7))

    train, test = split_train_test(series)

    st.sidebar.header("Настройки модели")

    model_type = st.sidebar.selectbox(
        "Выберите модель прогнозирования",
        [
            "ARIMA",
            "SARIMA",
            "SARIMAX",
            "Prophet"
        ]
    )

    horizon = st.sidebar.slider(
        "Горизонт прогноза, часов",
        min_value=1,
        max_value=72,
        value=24
    )

    train_window_days = st.sidebar.slider(
        "Размер обучающего окна, дней",
        min_value=7,
        max_value=90,
        value=30
    )

    train_window_hours = train_window_days * 24

    p = st.sidebar.number_input("p", min_value=0, max_value=5, value=2)
    d = st.sidebar.number_input("d", min_value=0, max_value=2, value=1)
    q = st.sidebar.number_input("q", min_value=0, max_value=5, value=2)

    P = st.sidebar.number_input("P", min_value=0, max_value=3, value=1)
    D = st.sidebar.number_input("D", min_value=0, max_value=2, value=0)
    Q = st.sidebar.number_input("Q", min_value=0, max_value=3, value=1)
    s = st.sidebar.number_input("s", min_value=1, max_value=168, value=24)

    order = (int(p), int(d), int(q))
    seasonal_order = (int(P), int(D), int(Q), int(s))

    control_test = test.iloc[:horizon]
    available_comparison_days = []

    for day in pd.Series(test.index.normalize()).drop_duplicates():
        day_start = pd.Timestamp(day)
        day_test = test.loc[day_start:].iloc[:horizon]

        if len(day_test) == horizon:
            available_comparison_days.append(day_start.date())

    if not available_comparison_days:
        st.error("Недостаточно данных в тестовой выборке для выбранного горизонта сравнения.")
        return

    selected_comparison_day = st.sidebar.selectbox(
        "День для сравнения моделей",
        available_comparison_days
    )

    st.sidebar.header("Интервальное сравнение")

    default_interval_end_index = min(6, len(available_comparison_days) - 1)
    interval_start_day = st.sidebar.date_input(
        "Начальная дата интервала",
        value=available_comparison_days[0],
        min_value=available_comparison_days[0],
        max_value=available_comparison_days[-1]
    )
    interval_end_day = st.sidebar.date_input(
        "Конечная дата интервала",
        value=available_comparison_days[default_interval_end_index],
        min_value=available_comparison_days[0],
        max_value=available_comparison_days[-1]
    )
    calculate_interval_comparison = st.sidebar.button(
        "Рассчитать таблицу сравнения"
    )

    comparison_day_start = pd.Timestamp(selected_comparison_day)
    comparison_test = test.loc[comparison_day_start:].iloc[:horizon]
    comparison_start = comparison_test.index[0]
    comparison_train = series.loc[series.index < comparison_start]

    st.subheader("Сравнение ARIMA, SARIMA, SARIMAX и Prophet")

    try:
        with st.spinner("Расчёт прогнозов моделей..."):
            comparison_chart_df, comparison_metrics_df = build_three_model_comparison(
                train=comparison_train,
                test=comparison_test,
                horizon=horizon,
                order=order,
                seasonal_order=seasonal_order,
                train_window_hours=train_window_hours,
                regressors=prophet_regressors,
                sarimax_regressors=sarimax_regressors
            )

        st.caption(
            f"Интервал сравнения: {comparison_test.index[0]} - {comparison_test.index[-1]}"
        )
        if not prophet_regressors.empty:
            st.caption(
                "Prophet использует дополнительные признаки: "
                + ", ".join(prophet_regressors.columns)
            )
        if not sarimax_regressors.empty:
            st.caption(
                "SARIMAX использует внешние регрессоры: "
                + ", ".join(sarimax_regressors.columns)
            )
        st.line_chart(comparison_chart_df)
        st.dataframe(comparison_metrics_df.round(2), hide_index=True)

        best_model_by_mae = select_best_model(comparison_metrics_df, metric="MAE")
        st.success(f"Лучшая модель на выбранном интервале по MAE: {best_model_by_mae}")

        if best_model_by_mae in comparison_chart_df.columns:
            congestion_df = detect_congestion_periods(
                comparison_chart_df[best_model_by_mae],
                threshold=5500
            )
            st.subheader("Периоды возможной высокой загруженности по лучшей модели")

            if congestion_df.empty:
                st.info("На выбранном интервале высокая загруженность по заданному порогу не обнаружена.")
            else:
                st.dataframe(congestion_df, hide_index=True)

    except Exception as error:
        st.error(f"Ошибка при сравнении моделей: {error}")

    if calculate_interval_comparison:
        if pd.Timestamp(interval_start_day) > pd.Timestamp(interval_end_day):
            st.error("Начальная дата интервала не может быть позже конечной.")
        else:
            st.subheader("Таблица сравнения моделей по интервалу дней")

            try:
                with st.spinner("Расчёт таблицы сравнения по выбранному интервалу..."):
                    interval_metrics_df = build_interval_model_comparison(
                        series=series,
                        test=test,
                        start_day=interval_start_day,
                        end_day=interval_end_day,
                        horizon=horizon,
                        order=order,
                        seasonal_order=seasonal_order,
                        train_window_hours=train_window_hours,
                        regressors=prophet_regressors,
                        sarimax_regressors=sarimax_regressors
                    )

                if interval_metrics_df.empty:
                    st.warning("Для выбранного интервала не удалось рассчитать метрики.")
                else:
                    st.dataframe(interval_metrics_df.round(2), hide_index=True)

                    interval_summary_df = summarize_validation_results(interval_metrics_df)
                    st.subheader("Средние метрики по выбранным интервалам")
                    st.dataframe(interval_summary_df.round(2), hide_index=True)

                    best_interval_model = select_best_model(interval_summary_df, metric="MAE")
                    st.success(f"Лучшая модель по средней MAE на выбранном интервале: {best_interval_model}")

                    csv_interval = interval_metrics_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="Скачать таблицу сравнения CSV",
                        data=csv_interval,
                        file_name="model_interval_comparison.csv",
                        mime="text/csv"
                    )

            except Exception as error:
                st.error(f"Ошибка при расчёте интервального сравнения: {error}")

    try:
        if model_type == "ARIMA":
            forecast, interval = forecast_arima(
                train=train,
                test=test,
                horizon=horizon,
                order=order,
                train_window_hours=train_window_hours
            )

        elif model_type == "SARIMA":
            forecast, interval = forecast_sarima(
                train=train,
                test=test,
                horizon=horizon,
                order=order,
                seasonal_order=seasonal_order,
                train_window_hours=train_window_hours
            )

        elif model_type == "SARIMAX":
            forecast, interval = forecast_sarimax(
                train=train,
                test=test,
                horizon=horizon,
                order=order,
                seasonal_order=seasonal_order,
                train_window_hours=train_window_hours,
                regressors=sarimax_regressors
            )

        else:
            forecast, interval = forecast_prophet(
                train=train,
                test=test,
                horizon=horizon,
                train_window_hours=train_window_hours,
                regressors=prophet_regressors
            )

        mae, rmse, mape = calc_metrics(control_test, forecast)

        st.subheader("Метрики качества")
        col1, col2, col3 = st.columns(3)

        col1.metric("MAE", f"{mae:.2f}")
        col2.metric("RMSE", f"{rmse:.2f}")
        col3.metric("MAPE", f"{mape:.2f} %")

        result_df = pd.DataFrame({
            "actual": control_test,
            "forecast": forecast,
            "lower_interval": interval.iloc[:, 0],
            "upper_interval": interval.iloc[:, 1],
            "absolute_error": (control_test - forecast).abs()
        })
        result_df["traffic_level"] = result_df["forecast"].apply(traffic_level)

        st.subheader("Точечный и интервальный прогноз")

        chart_df = pd.DataFrame({
            "Фактические значения": control_test,
            "Прогноз": forecast,
            "Нижняя граница интервала": interval.iloc[:, 0],
            "Верхняя граница интервала": interval.iloc[:, 1]
        })

        st.line_chart(chart_df)

        st.subheader("Таблица результатов")
        st.dataframe(result_df)

        st.subheader("Периоды возможной высокой загруженности")
        selected_congestion_threshold = st.slider(
            "Порог высокой загруженности traffic_volume",
            min_value=1000,
            max_value=8000,
            value=5500,
            step=100
        )
        congestion_df = detect_congestion_periods(
            forecast,
            threshold=selected_congestion_threshold
        )

        if congestion_df.empty:
            st.info("Периоды высокой загруженности по заданному порогу не обнаружены.")
        else:
            st.dataframe(congestion_df, hide_index=True)

        st.subheader("Анализ ошибок по часам суток")
        error_hour_df = error_by_hour(control_test, forecast)
        st.bar_chart(
            error_hour_df.set_index("hour")["mean_absolute_error"]
        )
        st.dataframe(error_hour_df.round(2), hide_index=True)

        single_model_metrics_df = pd.DataFrame(
            [[model_type, mae, rmse, mape]],
            columns=["model", "MAE", "RMSE", "MAPE"]
        )
        report_text = build_text_report(
            best_model=model_type,
            metrics_df=single_model_metrics_df,
            congestion_df=congestion_df,
            error_hour_df=error_hour_df
        )

        csv_result = result_df.to_csv(index=True).encode("utf-8")

        st.download_button(
            label="Скачать результаты прогноза CSV",
            data=csv_result,
            file_name="forecast_results.csv",
            mime="text/csv"
        )

        st.download_button(
            label="Скачать текстовый отчёт",
            data=report_text,
            file_name="traffic_forecast_report.txt",
            mime="text/plain"
        )

    except Exception as error:
        st.error(f"Ошибка при построении прогноза: {error}")


# ============================================================
# Консольная версия
# ============================================================

def run_console_version() -> None:
    data = load_data("Metro_Interstate_Traffic_Volume.csv")
    data = filter_road_segment(data)

    traffic_series = prepare_series(
        data,
        remove_anomalies=False
    )
    sarimax_regressors = prepare_sarimax_regressors(data)

    print("Количество наблюдений после подготовки:", len(traffic_series))
    print("Период:", traffic_series.index.min(), "-", traffic_series.index.max())

    plot_monthly_series(traffic_series)
    plot_week(traffic_series)
    plot_rolling_mean(traffic_series)

    check_stationarity(traffic_series)
    plot_acf_pacf(traffic_series)

    train, test = split_train_test(traffic_series)
    control_test = test.iloc[:24]

    sarima_forecast, sarima_interval = forecast_sarima(
        train=train,
        test=test,
        horizon=24,
        order=(2, 1, 2),
        seasonal_order=(1, 0, 1, 24)
    )

    print("\nМетрики SARIMA на контрольном интервале:")
    print(calc_metrics(control_test, sarima_forecast))

    plot_forecast_with_interval(
        actual=control_test,
        forecast=sarima_forecast,
        interval=sarima_interval,
        title="Точечный и интервальный прогноз SARIMA"
    )

    compare_models(
        traffic_series,
        horizon=24,
        include_prophet=True,
        sarimax_regressors=sarimax_regressors
    )

    rolling_validation_models(
        traffic_series,
        intervals=7,
        horizon=24,
        history_days=14,
        include_prophet=False
    )


if __name__ == "__main__":
    if "--web" in sys.argv:
        run_streamlit_interface()
    else:
        run_console_version()
