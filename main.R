# ============================================================
#  Sales Forecasting Project (R only)
#  - Data      : built-in AirPassengers (or simulated sales)
#  - Steps     : explore -> decompose -> train/test ARIMA -> forecast
#  - Packages  : forecast, ggplot2
#  Run in RStudio: open this file and click "Source"
#  (set working directory first: Session > Set Working Directory > To Source File Location)
# ============================================================

# ---------- 0. Setup ----------
required <- c("forecast", "ggplot2")
missing  <- required[!(required %in% rownames(installed.packages()))]
if (length(missing) > 0) install.packages(missing)

library(forecast)
library(ggplot2)

dir.create("outputs", showWarnings = FALSE)

# ---------- 1. Load data ----------
# Set TRUE to use simulated monthly sales instead of AirPassengers
USE_SIMULATED <- FALSE

if (USE_SIMULATED) {
  set.seed(42)
  n     <- 144                                   # 12 years of monthly data
  t     <- 1:n
  vals  <- 200 + 3 * t + 40 * sin(2 * pi * t / 12) + rnorm(n, 0, 12)
  sales <- ts(vals, start = c(2014, 1), frequency = 12)
  unit  <- "Sales"
} else {
  sales <- AirPassengers                          # monthly, 1949-1960
  unit  <- "Passengers (thousands)"
}

cat("\n===== DATA SUMMARY =====\n")
print(sales)
print(summary(sales))
cat("Start:", start(sales), " End:", end(sales), " Frequency:", frequency(sales), "\n")

# ---------- 2. Explore ----------
p1 <- autoplot(sales) +
  labs(title = "Monthly Series Over Time", x = "Year", y = unit) +
  theme_minimal()
ggsave("outputs/01_time_series.png", p1, width = 9, height = 5, dpi = 150)

p2 <- ggseasonplot(sales, year.labels = FALSE) +
  labs(title = "Seasonal Plot (each line = one year)", y = unit) +
  theme_minimal()
ggsave("outputs/02_seasonal_plot.png", p2, width = 9, height = 5, dpi = 150)

p3 <- ggsubseriesplot(sales) +
  labs(title = "Average by Month", y = unit) +
  theme_minimal()
ggsave("outputs/03_month_subseries.png", p3, width = 9, height = 5, dpi = 150)

# ---------- 3. Decompose: trend + seasonality + remainder ----------
# Multiplicative fits AirPassengers (seasonal swings grow with the level).
# For the simulated (additive) data it is switched automatically.
decomp_type <- if (USE_SIMULATED) "additive" else "multiplicative"
decomp <- decompose(sales, type = decomp_type)

p4 <- autoplot(decomp) +
  labs(title = paste("Classical Decomposition (", decomp_type, ")", sep = "")) +
  theme_minimal()
ggsave("outputs/04_decomposition.png", p4, width = 9, height = 7, dpi = 150)

# STL decomposition (more robust alternative)
stl_fit <- stl(if (USE_SIMULATED) sales else log(sales), s.window = "periodic")
p5 <- autoplot(stl_fit) +
  labs(title = "STL Decomposition") +
  theme_minimal()
ggsave("outputs/05_stl_decomposition.png", p5, width = 9, height = 7, dpi = 150)

cat("\nSeasonal index per month (", decomp_type, "):\n", sep = "")
print(round(decomp$figure, 3))

# ---------- 4. Train / test split (hold out last 12 months) ----------
h     <- 12
n_obs <- length(sales)
train <- window(sales, end   = time(sales)[n_obs - h])
test  <- window(sales, start = time(sales)[n_obs - h + 1])

# ---------- 5. ARIMA model ----------
# lambda = "auto" applies a Box-Cox transform to stabilise growing variance
fit_arima <- auto.arima(train, lambda = "auto", stepwise = FALSE, approximation = FALSE)

cat("\n===== ARIMA MODEL =====\n")
print(summary(fit_arima))

# Residual diagnostics (Ljung-Box p > 0.05 means residuals look like white noise)
png("outputs/06_residual_check.png", width = 1000, height = 700, res = 120)
checkresiduals(fit_arima)
dev.off()

# ---------- 6. Evaluate on the test set ----------
fc_test <- forecast(fit_arima, h = h)

cat("\n===== ARIMA ACCURACY (Train vs Test) =====\n")
acc_arima <- accuracy(fc_test, test)
print(round(acc_arima, 3))

# Benchmark: ETS (exponential smoothing)
fit_ets <- ets(train)
fc_ets  <- forecast(fit_ets, h = h)
cat("\n===== ETS BENCHMARK ACCURACY =====\n")
acc_ets <- accuracy(fc_ets, test)
print(round(acc_ets, 3))

cat("\nTest RMSE  ->  ARIMA:", round(acc_arima["Test set", "RMSE"], 2),
    " | ETS:", round(acc_ets["Test set", "RMSE"], 2), "\n")

p6 <- autoplot(fc_test) +
  autolayer(test, series = "Actual", color = "red") +
  labs(title = "ARIMA Forecast vs Actual (last 12 months held out)",
       x = "Year", y = unit) +
  theme_minimal()
ggsave("outputs/07_forecast_vs_actual.png", p6, width = 9, height = 5, dpi = 150)

# ---------- 7. Final model on ALL data -> forecast the future ----------
horizon   <- 24   # months ahead
fit_final <- auto.arima(sales, lambda = "auto", stepwise = FALSE, approximation = FALSE)
fc_future <- forecast(fit_final, h = horizon, level = c(80, 95))

cat("\n===== FINAL MODEL =====\n")
print(fit_final)

cat("\n===== NEXT", horizon, "MONTHS FORECAST =====\n")
print(round(as.data.frame(fc_future), 1))

p7 <- autoplot(fc_future) +
  labs(title = paste("ARIMA Forecast:", horizon, "Months Ahead"),
       x = "Year", y = unit) +
  theme_minimal()
ggsave("outputs/08_future_forecast.png", p7, width = 9, height = 5, dpi = 150)

# Save forecast table to CSV
out <- data.frame(
  Month    = rownames(as.data.frame(fc_future)),
  Forecast = round(as.data.frame(fc_future)[["Point Forecast"]], 1),
  Lo80     = round(as.data.frame(fc_future)[["Lo 80"]], 1),
  Hi80     = round(as.data.frame(fc_future)[["Hi 80"]], 1),
  Lo95     = round(as.data.frame(fc_future)[["Lo 95"]], 1),
  Hi95     = round(as.data.frame(fc_future)[["Hi 95"]], 1)
)
write.csv(out, "outputs/forecast_next_24_months.csv", row.names = FALSE)

cat("\nDone! Charts and forecast_next_24_months.csv are in the 'outputs' folder.\n")
