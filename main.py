import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


class FactorBasedLongShortEngine:

    def __init__(self, tickers, start_date=None, end_date=None, lookback_period=252):
        self.tickers = tickers

        if end_date is None:
            self.end_date = datetime.now().strftime('%Y-%m-%d')
        else:
            self.end_date = end_date

        if start_date is None:
            date_obj = datetime.strptime(self.end_date, '%Y-%m-%d') - timedelta(days=int(lookback_period * 1.5))
            self.start_date = date_obj.strftime('%Y-%m-%d')
        else:
            self.start_date = start_date

        self.lookback_period = lookback_period
        self.stock_data = None
        self.factor_data = None
        self.portfolio = None
        self.performance = None

    def fetch_data(self):
        print(f"Fetching data for {len(self.tickers)} stocks from {self.start_date} to {self.end_date}...")
        data = yf.download(self.tickers, start=self.start_date, end=self.end_date)
        print("Fetched data columns:", data.columns)
        if len(self.tickers) == 1:
            data = data.unstack().unstack().swaplevel(axis=1).sort_index(axis=1)
        self.stock_data = data
        print("Data fetching completed.")
        return data

    def calculate_factors(self):
        if self.stock_data is None:
            self.fetch_data()
        print("Calculating factors...")
        try:
            close_prices = self.stock_data['Close']
        except KeyError:
            raise KeyError("The 'Close' column is not available in the stock data.")
        factors = {}
        factors['momentum_1m'] = close_prices.pct_change(21)
        factors['momentum_3m'] = close_prices.pct_change(63)
        factors['momentum_6m'] = close_prices.pct_change(126)
        factors['momentum_12m'] = close_prices.pct_change(252)
        returns = close_prices.pct_change()
        factors['volatility_1m'] = returns.rolling(21).std()
        factors['volatility_3m'] = returns.rolling(63).std()
        factors['value'] = pd.DataFrame(index=close_prices.index, columns=self.tickers)

        for ticker in self.tickers:
            try:
                stock_info = yf.Ticker(ticker).info
                pe_ratio = stock_info.get('trailingPE', np.nan)
                if pe_ratio and not np.isnan(pe_ratio):
                    factors['value'][ticker] = pe_ratio
            except Exception as e:
                print(f"Error fetching fundamental data for {ticker}: {e}")

        self.factor_data = pd.concat(factors, axis=1)
        print("Factor calculation completed.")
        return self.factor_data

        for ticker in self.tickers:
            try:
                stock_info = yf.Ticker(ticker).info
                pe_ratio = stock_info.get('trailingPE', np.nan)
                if pe_ratio and not np.isnan(pe_ratio):
                    factors['value'][ticker] = pe_ratio
            except Exception as e:
                print(f"Error fetching fundamental data for {ticker}: {e}")

        self.factor_data = pd.concat(factors, axis=1)
        print("Factor calculation completed.")
        return self.factor_data

    def rank_stocks(self, date, factors_to_use=None, weights=None):
        if self.factor_data is None:
            self.calculate_factors()
        try:
            date_data = self.factor_data.loc[date].unstack(level=0)
        except KeyError:
            available_dates = self.factor_data.index.unique()
            available_dates = [d for d in available_dates if pd.notna(d)]
            closest_date = min(available_dates,
                               key=lambda x: abs((pd.to_datetime(x) - pd.to_datetime(date)).total_seconds()))
            print(f"Date {date} not found. Using closest date: {closest_date}")
            date_data = self.factor_data.loc[closest_date].unstack(level=0)
        if factors_to_use is None:
            factors_to_use = ['momentum_1m', 'momentum_3m', 'momentum_6m', 'momentum_12m',
                              'volatility_1m', 'volatility_3m', 'value']
        if weights is None:
            weights = {factor: 1 / len(factors_to_use) for factor in factors_to_use}
        rankings = pd.DataFrame(index=self.tickers)
        for factor in factors_to_use:
            if factor in date_data.index:
                factor_data = date_data.loc[factor]
                factor_data = factor_data.fillna(factor_data.mean())
                scaler = StandardScaler()
                factor_data_standardized = pd.Series(
                    scaler.fit_transform(factor_data.values.reshape(-1, 1)).flatten(),
                    index=factor_data.index
                )
                if factor.startswith('volatility') or factor == 'value':
                    factor_data_standardized = -factor_data_standardized
                rankings[factor] = factor_data_standardized * weights.get(factor, 1 / len(factors_to_use))
        rankings['composite_score'] = rankings.sum(axis=1)
        rankings['rank'] = rankings['composite_score'].rank(ascending=False)
        return rankings.sort_values('rank')

    def construct_portfolio(self, date, long_pct=0.2, short_pct=0.2, factors_to_use=None, weights=None):
        rankings = self.rank_stocks(date, factors_to_use, weights)
        n_stocks = len(self.tickers)
        n_long = int(n_stocks * long_pct)
        n_short = int(n_stocks * short_pct)
        portfolio = pd.DataFrame(0, index=self.tickers, columns=['weight'])
        long_stocks = rankings.head(n_long).index
        portfolio.loc[long_stocks, 'weight'] = float(1 / n_long)
        short_stocks = rankings.tail(n_short).index
        portfolio.loc[short_stocks, 'weight'] = float(-1 / n_short)
        self.portfolio = portfolio
        return portfolio

    def backtest(self, start_date=None, end_date=None, rebalance_freq='M',
                 long_pct=0.2, short_pct=0.2, factors_to_use=None, weights=None):
        if self.stock_data is None:
            self.fetch_data()
        if self.factor_data is None:
            self.calculate_factors()
        if start_date is None:
            start_date = (pd.to_datetime(self.start_date) + pd.Timedelta(days=self.lookback_period)).strftime(
                '%Y-%m-%d')
        if end_date is None:
            end_date = self.end_date
        try:
            price_data = self.stock_data['Close'].loc[start_date:end_date]
        except KeyError:
            raise KeyError("The 'Close' column is not available in the stock data.")
        rebalance_dates = pd.date_range(start=start_date, end=end_date, freq='ME')
        rebalance_dates = [date.strftime('%Y-%m-%d') for date in rebalance_dates]
        performance = pd.DataFrame(index=price_data.index, columns=['portfolio_value'])
        performance['portfolio_value'] = 100
        current_portfolio = None
        print(f"Running backtest from {start_date} to {end_date}...")

        for i, date in enumerate(price_data.index):
            date_str = date.strftime('%Y-%m-%d')
            if date_str in rebalance_dates or current_portfolio is None:
                current_portfolio = self.construct_portfolio(
                    date_str, long_pct, short_pct, factors_to_use, weights
                )
                print(f"Rebalanced portfolio on {date_str}")
            if i > 0:
                daily_returns = price_data.loc[date] / price_data.iloc[i - 1] - 1
                portfolio_return = (daily_returns * current_portfolio['weight']).sum()
                performance.loc[date, 'portfolio_value'] = float(performance.iloc[i - 1]['portfolio_value'] * (1 + portfolio_return))
        performance['daily_returns'] = performance['portfolio_value'].pct_change()
        performance['cumulative_returns'] = (1 + performance['daily_returns']).cumprod() - 1
        total_return = performance['portfolio_value'].iloc[-1] / performance['portfolio_value'].iloc[0] - 1
        annual_return = (1 + total_return) ** (252 / len(performance)) - 1
        annual_volatility = performance['daily_returns'].std() * np.sqrt(252)
        sharpe_ratio = annual_return / annual_volatility if annual_volatility != 0 else 0
        max_drawdown = (performance['portfolio_value'] / performance['portfolio_value'].cummax() - 1).min()

        print(f"Backtest completed with {len(rebalance_dates)} rebalances.")
        print(f"Total Return: {total_return:.2%}")
        print(f"Annual Return: {annual_return:.2%}")
        print(f"Annual Volatility: {annual_volatility:.2%}")
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"Max Drawdown: {max_drawdown:.2%}")
        self.performance = performance
        return performance

    def plot_performance(self):
        if self.performance is None:
            print("No backtest results available. Run backtest() first.")
            return
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plt.plot(self.performance['portfolio_value'])
        plt.title('Portfolio Value')
        plt.grid(True)
        plt.subplot(2, 1, 2)
        plt.plot(self.performance['cumulative_returns'])
        plt.title('Cumulative Returns')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def get_current_portfolio(self, long_pct=0.2, short_pct=0.2, factors_to_use=None, weights=None):
        if self.stock_data is None:
            self.fetch_data()
        latest_date = self.stock_data.index[-1].strftime('%Y-%m-%d')
        portfolio = self.construct_portfolio(latest_date, long_pct, short_pct, factors_to_use, weights)
        return portfolio

    def get_factor_exposures(self, portfolio=None):
        if self.factor_data is None:
            self.calculate_factors()
        if portfolio is None:
            if self.portfolio is None:
                portfolio = self.get_current_portfolio()
            else:
                portfolio = self.portfolio
        latest_date = self.factor_data.index[-1]
        latest_factors = self.factor_data.loc[latest_date].unstack(level=0)
        exposures = {}
        for factor in latest_factors.index:
            factor_values = latest_factors.loc[factor]
            factor_values = factor_values.fillna(factor_values.mean())
            exposures[factor] = (factor_values * portfolio['weight']).sum()
        return pd.Series(exposures)


def run_example():
    tickers = [
        'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'BRK-B', 'JNJ',
        'UNH', 'V', 'XOM', 'WMT', 'JPM', 'PG', 'MA', 'NVDA', 'HD', 'CVX',
        'LLY', 'MRK', 'PEP', 'KO', 'ABBV', 'BAC', 'PFE', 'COST', 'TMO',
        'AVGO', 'ABT', 'MCD', 'DHR', 'ACN', 'CSCO', 'DIS', 'VZ', 'NEE',
        'ADBE', 'TXN', 'CRM', 'PM', 'CMCSA', 'NKE', 'WFC', 'BMY', 'UPS'
    ]
    engine = FactorBasedLongShortEngine(tickers)
    weights = {
        'momentum_1m': 0.1,
        'momentum_3m': 0.2,
        'momentum_6m': 0.2,
        'momentum_12m': 0.1,
        'volatility_1m': 0.1,
        'volatility_3m': 0.1,
        'value': 0.2
    }
    performance = engine.backtest(
        rebalance_freq='M',
        long_pct=0.2,
        short_pct=0.2,
        weights=weights
    )
    engine.plot_performance()
    current_portfolio = engine.get_current_portfolio(
        long_pct=0.2,
        short_pct=0.2,
        weights=weights
    )
    print("\nCurrent Portfolio Allocation:")
    print(current_portfolio[current_portfolio['weight'] != 0])
    exposures = engine.get_factor_exposures(current_portfolio)
    print("\nFactor Exposures:")
    print(exposures)


if __name__ == "__main__":
    run_example()