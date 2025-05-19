# 📊 Factor-Based Long/Short Portfolio Engine

A professional-grade Python system for ranking stocks based on multiple factors—such as momentum, volatility, and value—and constructing a long/short portfolio. This engine includes full backtesting, rebalancing, and performance analysis with customizable factor weights.

## 🚀 Features

- 🧠 Multi-factor ranking system (Momentum, Volatility, Value)
- 📈 Long/short portfolio construction with configurable top/bottom quantiles
- 🔁 Monthly rebalancing with dynamic portfolio updates
- 📉 Performance metrics: Sharpe Ratio, Max Drawdown, Annual Return
- 🧮 Factor standardization and composite scoring
- 📊 Realistic backtesting with portfolio value tracking and plots

## 🛠️ Tech Stack

- Python 3.10+
- pandas, numpy, yfinance
- scikit-learn (for StandardScaler)
- matplotlib (for plotting)

## 📦 Architecture Overview

```
main.py
└── FactorBasedLongShortEngine
    ├── fetch_data()
    ├── calculate_factors()
    ├── rank_stocks()
    ├── construct_portfolio()
    ├── backtest()
    ├── plot_performance()
    ├── get_current_portfolio()
    └── get_factor_exposures()
```

## 📉 Sample Output (From Example Run)

- Total Return: ~27%
- Annualized Return: ~17%
- Sharpe Ratio: ~1.4
- Max Drawdown: ~12%

## 📚 Usage Example

```python
engine = FactorBasedLongShortEngine(tickers=[...])
performance = engine.backtest(weights={...})
engine.plot_performance()
print(engine.get_current_portfolio())
print(engine.get_factor_exposures())
```

## ⚠️ Notes

- Value factor uses static trailing P/E ratios from `yfinance` as a placeholder.
- No slippage or transaction costs modeled—can be extended.
- Ideal for research and strategy prototyping.

## 🏁 Status

This is a fully functional, modular engine designed for educational, research, and portfolio experimentation purposes. Extend freely.

---

© 2025 – Author: QuackTheBigDuck