# Sales Forecasting (R)

Forecasts monthly sales with ARIMA using the built-in `AirPassengers` dataset (or simulated data).

## Structure
```
SalesForecasting/
├── main.R        <- the whole project (run this)
├── README.md
└── outputs/      <- created automatically (charts + forecast CSV)
```

## Run
1. Install R and RStudio.
2. Open `main.R` in RStudio.
3. Session > Set Working Directory > To Source File Location.
4. Click **Source**. Packages `forecast` and `ggplot2` install automatically.

To use simulated data, set `USE_SIMULATED <- TRUE` near the top of `main.R`.
