# Efficient Frontier MPT

A simple toolkit for Modern Portfolio Theory (MPT) that demonstrates how to:

- Retrieve asset price data from Yahoo Finance.
- Calculate key metrics (returns, volatility, Sharpe ratio).
- Optimize portfolios for maximum Sharpe ratio or minimum variance.
- Visualize the Efficient Frontier in an interactive Plotly chart.
- Inspect correlations among assets.

This project uses Python, Streamlit, and Plotly to provide a user-friendly web application for portfolio optimization.

## Features

### Automatic Data Retrieval
- Fetches historical data for user-specified tickers using yfinance.

### Portfolio Optimization

- **Max Sharpe Ratio**: Find the portfolio weights that maximize risk-adjusted returns.
- **Min Variance**: Find the portfolio weights that minimize overall volatility.
- **Efficient Frontier**: Plot a range of portfolios with varying target returns and minimized variance.

### Correlation Matrix
- Visualize the correlation between assets in a heatmap.

### Asset Metrics
- View annualized return, volatility, and Sharpe ratio for each individual asset.

## Getting Started

### Prerequisites

Make sure you have the following installed:

- Python 3.7+
- pip (or another Python package manager)

### Installation

Clone or download this repository:

```bash
git clone https://github.com/<your-username>/efficient_frontier_mpt.git
```

Navigate to the project directory:

```bash
cd efficient_frontier_mpt
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

If you don't have a `requirements.txt`, you can manually install the main libraries:

```bash
pip install streamlit yfinance pandas numpy scipy plotly
```

## Usage

Run the Streamlit app:

```bash
streamlit run ef_streamlit.py
```

Open your web browser and go to the URL printed in the terminal (usually http://localhost:8501).

1. Enter your tickers (comma-separated) and date range in the sidebar.
2. Adjust the risk-free rate and weight constraints if needed.
3. Click **Run Portfolio Analysis** to see results:
   - Maximum Sharpe Ratio portfolio
   - Minimum Variance portfolio
   - Efficient Frontier chart
   - Correlation Matrix among the assets
   - Asset Metrics table

## Files

- **ef_streamlit.py**: Main Streamlit application. Handles user input, data retrieval, portfolio optimization, and chart plotting.
- **ef_test.ipynb**: Jupyter Notebook (if applicable) for testing functions and experimenting with the portfolio optimization code.
- **logowk.svg**: Sample logo image referenced in the Streamlit sidebar.

Feel free to adapt or remove these files to fit your project’s structure.

## How It Works

### Data Retrieval
- Uses `yfinance` to download historical price data for each ticker in the specified date range.

### Return & Covariance Calculation
- Daily returns are calculated from close prices.
- Mean returns and covariance matrix are derived from the daily returns.

### Optimization

#### Max Sharpe Ratio

The Sharpe ratio is calculated as:

$$
\text{Sharpe Ratio} = \frac{R_p - R_f}{\sigma_p}
$$

where:
- Rp is the portfolio return
- Rf is the risk-free rate
- σp is the portfolio volatility


We use `scipy.optimize.minimize` to maximize the Sharpe ratio (equivalently, minimize its negative).

#### Min Variance
- Minimizes the portfolio variance (or standard deviation).

#### Efficient Frontier
- For target returns from the min to max, finds the portfolio with the smallest variance.

### Visualization

Plotly is used to:
- Display the Efficient Frontier.
- Mark the Max Sharpe and Min Variance points.
- Create a correlation heatmap of the assets.

## Contributing

1. Fork this repository.
2. Create a new branch for your feature:
   ```bash
   git checkout -b feature/my-feature
   ```
3. Commit your changes:
   ```bash
   git commit -m 'Add some feature'
   ```
4. Push to your branch:
   ```bash
   git push origin feature/my-feature
   ```
5. Create a Pull Request.

All contributions, bug reports, and feature requests are welcome!

## Disclaimer

This application is for educational and informational purposes only. It does not constitute financial advice. Past performance of any asset is not indicative of future results. Always do your own research or consult a financial professional before making investment decisions.

## License

MIT License.

## Acknowledgments

- Streamlit for the easy-to-build data app framework.
- yfinance for quick stock/crypto data retrieval.
- Plotly for interactive data visualizations.
- Any other resources or references that inspired or helped with this project.
