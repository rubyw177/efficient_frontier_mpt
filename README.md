# Efficient Frontier MPT

A simple toolkit for Modern Portfolio Theory (MPT) that demonstrates how to:

- Retrieve asset price data from Yahoo Finance.
- Calculate key metrics (returns, volatility, Sharpe ratio).
- Optimize portfolios for maximum Sharpe ratio or minimum variance.
- Plot the Efficient Frontier interactively with Plotly.
- Dynamically select a portfolio along the efficient frontier using an interactive slider.
- Inspect correlations among assets.
- Manage expensive computations with caching and session state so that updates occur only when the user explicitly clicks **Run Portfolio Analysis**.

This project uses Python, Streamlit, and Plotly to provide a user-friendly web application for portfolio optimization.

## Features

### Automatic Data Retrieval
- **Yahoo Finance Integration:**  
  Automatically downloads historical data for user-specified tickers using `yfinance`.

### Portfolio Optimization
- **Max Sharpe Ratio:**  
  Find the portfolio weights that maximize risk-adjusted returns by minimizing the negative Sharpe ratio.
- **Min Variance:**  
  Determine the portfolio with the lowest volatility.
- **Efficient Frontier:**  
  Generate a range of portfolios with varying target returns, each optimized to minimize variance.

### Interactive Portfolio Selection
- **Efficient Frontier Slider:**  
  Use an interactive slider to select a target return along the efficient frontier. The app updates the portfolio’s weights, expected return, risk (volatility), and even highlights the selected portfolio on the efficient frontier plot.

### Visualization
- **Efficient Frontier Plot:**  
  The Plotly chart displays the efficient frontier curve along with markers for the Maximum Sharpe Ratio and Minimum Variance portfolios. A dynamically updated marker shows the user-selected portfolio.
- **Correlation Matrix:**  
  An interactive heatmap displays correlations among assets.
- **Asset Metrics:**  
  View annualized return, volatility, and Sharpe ratio for each asset in a table.
- **Portfolio Allocation:**  
  An interactive pie chart shows portfolio allocations.

### Smart Re-Calculation with Session State
- **User-Controlled Updates:**  
  Heavy computations (data retrieval, optimization) run only when the user clicks **Run Portfolio Analysis**.
- **Sidebar Change Detection:**  
  Although sidebar changes trigger a re-run of the script (Streamlit’s default behavior), caching and session state ensure that expensive operations are only re-computed when the user explicitly requests it.
- **Efficient Updates:**  
  Once the analysis is “locked in” by the run button, only the slider (and related lightweight updates) re-run the optimization for the selected portfolio.

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
   - Selected efficient frontier portfolio
   - Efficient Frontier chart
   - Correlation Matrix among the assets
   - Asset Metrics table
4. Use the interactive slider to choose a target return along the efficient frontier. The app updates the portfolio weights, expected return, volatility, and marks the selected point on the efficient frontier plot.

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

### Interactive Updates with Session State
- Lock-In Analysis:
When the user clicks Run Portfolio Analysis, heavy computations are executed once and stored in `st.session_state`.
- Sidebar Change Detection:
A helper function compares current sidebar inputs with stored values. If changes are detected, a message prompts the user to click the run button again to update the analysis.
- Slider-Driven Updates:
The efficient frontier slider triggers only a re-optimization for the selected target return without re-running the entire analysis.

### Visualization

Plotly is used to:
- Display the Efficient Frontier.
- Display portfolio allocation pie chart.
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
