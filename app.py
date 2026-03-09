import io
import zipfile
from datetime import timedelta

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


def _try_read_csv(path_or_buffer):
    """Try reading a CSV file with a few common fallbacks.

    This handles:
    - Files saved with Windows encodings (cp1252 / latin1)
    - Files with inconsistent delimiters (auto-detect, tabs, semicolons)
    - Files with a few bad lines (skips malformed rows)
    - ZIP archives containing a CSV (common when downloading datasets)

    The goal is to give the user a useful error message instead of a hard crash.
    """

    # Detect ZIP archives immediately (useful when users upload a zipped dataset).
    if hasattr(path_or_buffer, "read"):
        try:
            pos = None
            if hasattr(path_or_buffer, "tell"):
                pos = path_or_buffer.tell()

            start = path_or_buffer.read(4)
            if hasattr(path_or_buffer, "seek"):
                path_or_buffer.seek(pos or 0)

            if start.startswith(b"PK") or zipfile.is_zipfile(path_or_buffer):
                with zipfile.ZipFile(path_or_buffer) as zf:
                    csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
                    if csv_names:
                        with zf.open(csv_names[0]) as f:
                            return _try_read_csv(io.BytesIO(f.read()))
        except Exception:
            if hasattr(path_or_buffer, "seek"):
                try:
                    path_or_buffer.seek(pos or 0)
                except Exception:
                    pass

    encodings = ["utf-8", "latin1", "cp1252"]
    read_attempts = [
        {},
        {"engine": "python", "on_bad_lines": "skip"},
        {"sep": None, "engine": "python"},
        {"sep": "\t", "engine": "python"},
        {"sep": ";", "engine": "python"},
    ]

    def _reset_buffer():
        if hasattr(path_or_buffer, "seek"):
            try:
                path_or_buffer.seek(0)
            except Exception:
                pass

    def _looks_like_bad_parse(df: pd.DataFrame) -> bool:
        # If pandas returns a single column with a delimiter in the header, it likely failed to parse.
        if df is None or df.shape[1] == 0:
            return True
        if df.shape[1] == 1:
            header = str(df.columns[0])
            if any(sep in header for sep in [",", ";", "\t"]):
                return True
            # If most rows still contain commas/semicolons, it's probably not parsed correctly.
            sample = df.iloc[:, 0].astype(str).head(5)
            if sample.str.contains(",|;|	").all():
                return True
        return False

    last_err = None
    for enc in encodings:
        for attempt_kwargs in read_attempts:
            _reset_buffer()
            try:
                df = pd.read_csv(path_or_buffer, encoding=enc, **attempt_kwargs)
                if _looks_like_bad_parse(df):
                    # Try a different strategy if parsing looks wrong.
                    last_err = pd.errors.ParserError("Parsed output looks incorrect (likely wrong delimiter)")
                    continue
                return df
            except UnicodeDecodeError as e:
                last_err = e
                break
            except pd.errors.ParserError as e:
                last_err = e
                continue
            except Exception as e:
                last_err = e
                continue

    if last_err is not None:
        raise last_err

    return pd.read_csv(path_or_buffer)


def _prepare_time_series(df: pd.DataFrame, date_col: str, value_col: str):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    df = df.sort_values(date_col)

    # Ensure the target column is numeric; coerce invalid values into NaN and drop them.
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df[[date_col, value_col]].dropna()

    df = df.set_index(date_col)
    return df


def forecast_naive(series: pd.Series, horizon: int):
    last_value = float(series.iloc[-1])
    future_index = pd.date_range(series.index[-1] + pd.Timedelta(days=1), periods=horizon)
    return pd.Series([last_value] * horizon, index=future_index, name=series.name)


def forecast_linear_lag(series: pd.Series, horizon: int, lags: int = 7):
    """Forecast using a simple linear model built on lag features.

    This is NOT a production forecasting algorithm, but it works well for
    quick demos and gives a reasonable-looking forecast.
    """

    y = series.astype(float).copy()
    df = pd.DataFrame({"y": y})

    for lag in range(1, lags + 1):
        df[f"lag_{lag}"] = df["y"].shift(lag)

    # Add simple seasonality features (weekday, month)
    df["weekday"] = df.index.weekday
    df["month"] = df.index.month

    df_train = df.dropna()
    if df_train.empty:
        return pd.Series(dtype="float64")

    X = df_train.drop(columns=["y"])
    y_train = df_train["y"]

    model = LinearRegression()
    model.fit(X, y_train)

    # Walk-forward forecast
    last_row = df.iloc[[-1]].copy()
    preds = []
    last_date = last_row.index[0]

    for step in range(horizon):
        next_date = last_date + timedelta(days=1)
        next_row = last_row.copy()
        next_row.index = pd.DatetimeIndex([next_date])
        next_row["weekday"] = next_date.weekday()
        next_row["month"] = next_date.month

        # shift lags
        for lag in range(lags, 1, -1):
            next_row[f"lag_{lag}"] = next_row[f"lag_{lag - 1}"]
        next_row["lag_1"] = next_row["y"]

        X_next = next_row.drop(columns=["y"])
        yhat = model.predict(X_next)[0]
        preds.append((next_date, float(yhat)))

        next_row["y"] = yhat
        last_row = next_row
        last_date = next_date

    return pd.Series(
        data=[v for _, v in preds],
        index=pd.DatetimeIndex([d for d, _ in preds]),
        name=series.name,
    )


def forecast_rf(series: pd.Series, horizon: int, lags: int = 7):
    """Forecast using a random forest on lag+time features."""

    y = series.astype(float).copy()
    df = pd.DataFrame({"y": y})

    for lag in range(1, lags + 1):
        df[f"lag_{lag}"] = df["y"].shift(lag)

    df["weekday"] = df.index.weekday
    df["month"] = df.index.month

    df_train = df.dropna()
    if df_train.empty:
        return pd.Series(dtype="float64")

    X = df_train.drop(columns=["y"])
    y_train = df_train["y"]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y_train)

    last_row = df.iloc[[-1]].copy()
    preds = []
    last_date = last_row.index[0]

    for step in range(horizon):
        next_date = last_date + timedelta(days=1)
        next_row = last_row.copy()
        next_row.index = pd.DatetimeIndex([next_date])
        next_row["weekday"] = next_date.weekday()
        next_row["month"] = next_date.month

        for lag in range(lags, 1, -1):
            next_row[f"lag_{lag}"] = next_row[f"lag_{lag - 1}"]
        next_row["lag_1"] = next_row["y"]

        X_next = next_row.drop(columns=["y"])
        yhat = model.predict(X_next)[0]
        preds.append((next_date, float(yhat)))

        next_row["y"] = yhat
        last_row = next_row
        last_date = next_date

    return pd.Series(
        data=[v for _, v in preds],
        index=pd.DatetimeIndex([d for d, _ in preds]),
        name=series.name,
    )


def forecast_exp_smoothing(series: pd.Series, horizon: int, alpha: float = 0.2):
    """Simple exponential smoothing (single): forecasts the last smoothed level."""

    level = float(series.iloc[0])
    for x in series:
        level = alpha * float(x) + (1 - alpha) * level

    future_index = pd.date_range(series.index[-1] + pd.Timedelta(days=1), periods=horizon)
    return pd.Series([level] * horizon, index=future_index, name=series.name)


def evaluate_forecast(actual: pd.Series, forecast: pd.Series):
    """Return MAE and MAPE comparing forecast against actual (same index)."""

    common_idx = actual.index.intersection(forecast.index)
    if len(common_idx) == 0:
        return None, None

    y_true = actual.loc[common_idx].astype(float)
    y_pred = forecast.loc[common_idx].astype(float)

    mae = mean_absolute_error(y_true, y_pred)
    mape = (abs((y_true - y_pred) / y_true.replace(0, np.nan)).mean()) * 100
    return mae, mape


def main():
    st.set_page_config(
        page_title="Sales Time Series Forecasting",
        page_icon="📈",
        layout="wide",
    )

    st.title("📊 Sales Time Series Forecasting")
    st.markdown(
        "Use this app to load your sales data, pick the date and value columns, and get a quick forecast."
    )

    st.sidebar.header("Data input")
    source = st.sidebar.radio("Data source", ["Use sample train.csv", "Upload CSV"])

    df = None
    if source == "Use sample train.csv":
        try:
            df = _try_read_csv("train.csv")
        except Exception as e:
            st.error(f"Unable to load train.csv: {e}")
    else:
        uploaded = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
        if uploaded is not None:
            upload_bytes = uploaded.read()
            df = _try_read_csv(io.BytesIO(upload_bytes))

    if df is None:
        st.info("Upload a CSV or place a `train.csv` file in this folder.")
        st.stop()

    st.sidebar.markdown("---")
    st.sidebar.write("**Preview**")
    st.sidebar.dataframe(df.head(5))

    st.subheader("Step 1: Choose columns")
    cols = df.columns.tolist()
    date_col = st.selectbox("Date column", cols, index=0)

    # Prefer numeric columns for forecasting
    numeric_cols = [
        c
        for c in cols
        if pd.to_numeric(df[c], errors="coerce").notna().any()
    ]
    if not numeric_cols:
        numeric_cols = cols

    if set(numeric_cols) != set(cols):
        st.info(
            "Non-numeric columns detected; only numeric columns are recommended for forecasting. "
            "If you need to use a non-numeric column, the app will attempt conversion but may result in no data."
        )

    value_col = st.selectbox("Value column", numeric_cols, index=min(1, len(numeric_cols) - 1))

    df_ts = _prepare_time_series(df, date_col, value_col)
    if df_ts.empty:
        st.error(
            "No usable time series data found after parsing the selected columns. "
            "Try selecting a different date column or a numeric value column."
        )
        st.stop()

    st.subheader("Step 2: Configure forecast")
    horizon = st.number_input("Forecast horizon (days)", value=30, min_value=1, max_value=365)

    model_options = [
        "Naive (last value)",
        "Linear (lags)",
        "Random Forest (lags)",
        "Exp smoothing",
    ]
    model_choices = st.multiselect(
        "Forecast methods (compare side-by-side)",
        model_options,
        default=["Naive (last value)", "Linear (lags)"],
    )

    max_eval = max(0, min(60, len(df_ts) // 2))
    eval_days = st.slider(
        "Evaluation window (days)",
        min_value=0,
        max_value=max_eval,
        value=min(30, max_eval),
        help="Use the last N days of history to compare model accuracy (MAE/MAPE).",
    )

    st.subheader("Time series preview")
    st.line_chart(df_ts)

    if st.button("Run forecast"):
        if not model_choices:
            st.warning("Select at least one forecast method.")
            st.stop()

        forecasts = {}
        for choice in model_choices:
            if choice == "Naive (last value)":
                forecasts[choice] = forecast_naive(df_ts[value_col], horizon=horizon)
            elif choice == "Linear (lags)":
                forecasts[choice] = forecast_linear_lag(df_ts[value_col], horizon=horizon)
            elif choice == "Random Forest (lags)":
                forecasts[choice] = forecast_rf(df_ts[value_col], horizon=horizon)
            elif choice == "Exp smoothing":
                forecasts[choice] = forecast_exp_smoothing(df_ts[value_col], horizon=horizon)

        st.subheader("Forecast")
        chart_df = df_ts.rename(columns={value_col: "history"}).copy()
        for name, fc in forecasts.items():
            chart_df[name] = fc
        st.line_chart(chart_df)

        st.markdown("### Forecast table")
        forecast_table = pd.concat(forecasts, axis=1)
        st.dataframe(forecast_table)

        if eval_days > 0 and len(df_ts) > eval_days:
            st.markdown("---")
            st.subheader("Model comparison")

            actual_eval = df_ts[value_col].iloc[-eval_days:]
            eval_rows = []

            train_series = df_ts[value_col].iloc[: -eval_days]
            for name, _ in forecasts.items():
                if name == "Naive (last value)":
                    eval_fc = forecast_naive(train_series, horizon=eval_days)
                elif name == "Linear (lags)":
                    eval_fc = forecast_linear_lag(train_series, horizon=eval_days)
                elif name == "Random Forest (lags)":
                    eval_fc = forecast_rf(train_series, horizon=eval_days)
                elif name == "Exp smoothing":
                    eval_fc = forecast_exp_smoothing(train_series, horizon=eval_days)
                else:
                    eval_fc = None

                mae, mape = (None, None)
                if eval_fc is not None:
                    mae, mape = evaluate_forecast(actual_eval, eval_fc)

                eval_rows.append({"model": name, "MAE": mae, "MAPE": mape})

            eval_df = pd.DataFrame(eval_rows).set_index("model")
            st.table(eval_df)

        st.markdown(
            "---\n" "Tip: Export the forecast data to CSV by copying the table or using the Streamlit sharing options."
        )


if __name__ == "__main__":
    main()
