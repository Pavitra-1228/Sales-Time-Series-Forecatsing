# Sales Time Series Forecasting

This project provides a *user-friendly* web app to explore and forecast sales time series data.

## ✅ Run the web app

From the project directory, run:

```bash
streamlit run app.py
```

Then open the address shown in your terminal (usually `http://localhost:8501`).

## 🔍 What it does

- Loads your **train.csv** (or any uploaded CSV file)
- Lets you pick the **date column** and **sales/value column**
- Builds a simple forecast and shows:
  - Plot of historical data
  - Forecast chart (next N days)
  - Forecast table

## 🛠 Troubleshooting

### `ModuleNotFoundError: No module named 'pandas'`

Install dependencies with:

```bash
python -m pip install pandas streamlit scikit-learn
```

### CSV encoding error

The app tries multiple encodings (UTF‑8, Latin‑1, CP1252) when reading your CSV.

If you still see errors, the file may be corrupt or use an unusual encoding.
