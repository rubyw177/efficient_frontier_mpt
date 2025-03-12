import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# Cached data retrieval with error handling
@st.cache_data
def get_data(tickers, start=None, end=None):
    """
    Fetch historical close price data for given tickers using yfinance.

    Parameters:
        tickers (list or str): One or more ticker symbols.
        start (str, optional): Start date in 'YYYY-MM-DD' format.
        end (str, optional): End date in 'YYYY-MM-DD' format.

    Returns:
        DataFrame: Close prices for the specified tickers.
    """
    try:
        if start is None and end is None:
            data = yf.download(tickers=tickers)
        else:
            data = yf.download(tickers=tickers, start=start, end=end)
        return data["Close"] if "Close" in data.columns else None
    except Exception as e:
        st.error(f"Data retrieval failed: {str(e)}")
        return None

def get_return(price_data, mean_cov=True):
    """
    Calculate daily returns from price data and optionally return mean returns and covariance matrix.

    Parameters:
        price_data (DataFrame): Historical price data.
        mean_cov (bool): If True, return [mean_returns, covariance_matrix]; otherwise, return raw returns.

    Returns:
        list or DataFrame: [mean_returns, covariance_matrix] if mean_cov is True, else DataFrame of returns.
    """
    if price_data is None or price_data.empty:
        return None
    returns = price_data.pct_change().dropna()
    if mean_cov:
        return [returns.mean(), returns.cov()]
    return returns

def portfolio_performance(weights, mean_returns, cov_matrix, trading_days=365):
    """
    Calculates the annualized portfolio return and standard deviation.

    Parameters:
        weights (array-like): Portfolio weights.
        mean_returns (Series): Mean of daily returns.
        cov_matrix (DataFrame): Covariance matrix of daily returns.
        trading_days (int): Number of trading days per year.

    Returns:
        list: [annualized return, annualized standard deviation] in decimal form.
    """
    annual_return = np.sum(mean_returns * weights) * trading_days
    annual_std = np.sqrt(weights.T @ (cov_matrix @ weights)) * np.sqrt(trading_days)
    return [annual_return, annual_std]

def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate=0.0, trading_days=365):
    """
    Computes the negative Sharpe ratio for a given set of portfolio weights.

    Parameters:
        weights (array-like): Portfolio weights.
        mean_returns (Series): Mean of daily returns.
        cov_matrix (DataFrame): Covariance matrix of daily returns.
        risk_free_rate (float): Annualized risk-free rate.
        trading_days (int): Number of trading days per year.

    Returns:
        float: Negative Sharpe ratio.
    """
    annual_return, annual_std = portfolio_performance(weights, mean_returns, cov_matrix, trading_days)
    if np.isclose(annual_std, 0):
        return np.inf
    sharpe_ratio = (annual_return - risk_free_rate) / annual_std
    return -sharpe_ratio

def optimize_sharpe(mean_returns, cov_matrix, risk_free_rate=0.0, trading_days=365, constraint_set=(0, 1)):
    """
    Optimize the portfolio for maximum Sharpe ratio by minimizing the negative Sharpe ratio.

    Parameters:
        mean_returns (Series): Mean daily returns.
        cov_matrix (DataFrame): Covariance matrix of daily returns.
        risk_free_rate (float): Annualized risk-free rate.
        trading_days (int): Number of trading days per year.
        constraint_set (tuple): Bounds for the weights (default (0, 1)).

    Returns:
        OptimizeResult: The optimization result from scipy.optimize.minimize.
    """
    n_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free_rate, trading_days)
    constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1}]
    weights_bounds = tuple(constraint_set for _ in range(n_assets))
    init_guess = [1. / n_assets] * n_assets

    result = minimize(
        neg_sharpe_ratio,
        init_guess,
        args=args,
        method="SLSQP",
        bounds=weights_bounds,
        constraints=constraints
    )
    
    if not result.success:
        st.warning(f"Optimization for maximum Sharpe ratio failed: {result.message}")
    return result

def get_portfolio_variance(weights, mean_returns, cov_matrix, trading_days=365):
    """
    Calculate the annualized portfolio standard deviation (volatility).

    Parameters:
        weights (array-like): Portfolio weights.
        mean_returns (Series): Mean of daily returns.
        cov_matrix (DataFrame): Covariance matrix of daily returns.
        trading_days (int): Number of trading days per year.

    Returns:
        float: Annualized portfolio volatility.
    """
    _, annual_std = portfolio_performance(weights, mean_returns, cov_matrix, trading_days)
    return annual_std

def optimize_variance(mean_returns, cov_matrix, trading_days=365, constraint_set=(0, 1)):
    """
    Optimize the portfolio for minimum variance (risk).

    Parameters:
        mean_returns (Series): Mean of daily returns.
        cov_matrix (DataFrame): Covariance matrix of daily returns.
        trading_days (int): Number of trading days per year.
        constraint_set (tuple): Bounds for weights (default (0,1)).

    Returns:
        OptimizeResult: The optimization result from scipy.optimize.minimize.
    """
    n_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, trading_days)
    constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1}]
    weights_bounds = tuple(constraint_set for _ in range(n_assets))
    init_guess = [1. / n_assets] * n_assets

    result = minimize(
        get_portfolio_variance,
        init_guess,
        args=args,
        method="SLSQP",
        bounds=weights_bounds,
        constraints=constraints
    )
    
    if not result.success:
        st.warning(f"Optimization for minimum variance failed: {result.message}")
    return result

def get_portfolio_return(weights, mean_returns, cov_matrix, trading_days=365):
    """
    Calculate the annualized portfolio return.

    Parameters:
        weights (array-like): Portfolio weights.
        mean_returns (Series): Mean daily returns.
        cov_matrix (DataFrame): Covariance matrix of daily returns.
        trading_days (int): Number of trading days per year.

    Returns:
        float: Annualized portfolio return in decimal form.
    """
    annual_return, _ = portfolio_performance(weights, mean_returns, cov_matrix, trading_days)
    return annual_return

def efficient_optimization(mean_returns, cov_matrix, return_target, constraint_set=(0, 1), trading_days=365):
    """
    Optimize the portfolio to achieve at least a target annual return while minimizing variance.

    Parameters:
        mean_returns (Series): Mean daily returns.
        cov_matrix (DataFrame): Covariance matrix of daily returns.
        return_target (float): Desired annual return target in decimal form.
        constraint_set (tuple): Bounds for each asset weight (default (0,1)).
        trading_days (int): Number of trading days per year.

    Returns:
        OptimizeResult: The optimization result from scipy.optimize.minimize.
    """
    n_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, trading_days)
    constraints = [
        {"type": "ineq", "fun": lambda x: get_portfolio_return(x, mean_returns, cov_matrix, trading_days) - return_target},
        {"type": "eq", "fun": lambda x: np.sum(x) - 1}
    ]
    weights_bounds = tuple(constraint_set for _ in range(n_assets))
    init_guess = [1. / n_assets] * n_assets

    result = minimize(
        get_portfolio_variance,
        init_guess,
        args=args,
        method="SLSQP",
        bounds=weights_bounds,
        constraints=constraints
    )
    
    if not result.success:
        st.warning(f"Efficient optimization failed for target return {return_target}: {result.message}")
    return result

def calculated_result(mean_returns, cov_matrix, risk_free_rate=0.0, constraints_set=(0, 1), trading_days=365):
    """
    Compute portfolios optimized for:
      - Maximum Sharpe ratio.
      - Minimum variance.
      - Efficient frontier (minimizing variance for a series of target returns).

    Returns:
        dict: Contains performance metrics and allocations for the max Sharpe and min variance portfolios,
              as well as the efficient frontier data.
    """
    # Optimize maximum Sharpe ratio portfolio
    with st.spinner("Optimizing Sharpe Ratio..."):
        maxSR_portfolio = optimize_sharpe(mean_returns, cov_matrix, risk_free_rate, trading_days, constraints_set)
    maxSR_weights = maxSR_portfolio["x"]
    maxSR_return_dec, maxSR_std_dec = portfolio_performance(maxSR_weights, mean_returns, cov_matrix, trading_days)
    # Convert to percentages for display
    maxSR_return_disp, maxSR_std_disp = round(maxSR_return_dec * 100, 2), round(maxSR_std_dec * 100, 2)
    maxSR_allocation = pd.DataFrame(maxSR_weights, index=mean_returns.index, columns=["Weightings"])
    maxSR_allocation["Weightings"] = [round(x, 2) for x in maxSR_allocation["Weightings"]]

    # Optimize minimum variance portfolio
    with st.spinner("Optimizing Minimum Variance..."):
        minVar_portfolio = optimize_variance(mean_returns, cov_matrix, trading_days, constraints_set)
    minVar_weights = minVar_portfolio["x"]
    minVar_return_dec, minVar_std_dec = portfolio_performance(minVar_weights, mean_returns, cov_matrix, trading_days)
    minVar_return_disp, minVar_std_disp = round(minVar_return_dec * 100, 2), round(minVar_std_dec * 100, 2)
    minVar_allocation = pd.DataFrame(minVar_weights, index=mean_returns.index, columns=["Weightings"])
    minVar_allocation["Weightings"] = [round(x, 2) for x in minVar_allocation["Weightings"]]

    # Generate efficient frontier data using target returns in decimal form
    with st.spinner("Calculating Efficient Frontier..."):
        target_returns = np.linspace(minVar_return_dec, maxSR_return_dec, num=20)
        frontier_std = []
        progress_bar = st.progress(0)
        for i, target in enumerate(target_returns):
            cons = (
                {"type": "eq", "fun": lambda x: np.sum(x) - 1},
                {"type": "ineq", "fun": lambda x, t=target: portfolio_performance(x, mean_returns, cov_matrix, trading_days)[0] - t}
            )
            res = minimize(
                lambda w: portfolio_performance(w, mean_returns, cov_matrix, trading_days)[1],
                [1./len(mean_returns)] * len(mean_returns),
                method="SLSQP",
                bounds=[constraints_set] * len(mean_returns),
                constraints=cons
            )
            frontier_std.append(res.fun if res.success else np.nan)
            progress_bar.progress((i+1)/len(target_returns))
    
    return {
        "maxSR": {
            "return_disp": maxSR_return_disp,
            "return_ori": maxSR_return_dec,
            "std_disp": maxSR_std_disp,
            "std_ori": maxSR_std_dec,
            "allocation": maxSR_allocation
        },
        "minVar": {
            "return_disp": minVar_return_disp,
            "return_ori": minVar_return_dec,
            "std_disp": minVar_std_disp,
            "std_ori": minVar_std_dec,
            "allocation": minVar_allocation
        },
        "efficient_frontier": np.array(frontier_std),
        "target_returns": target_returns
    }

def plot_efficient_frontier(mean_returns, cov_matrix, risk_free_rate=0.0, constraints_set=(0, 1), trading_days=365, selected_point=None):
    """
    Create an interactive Plotly visualization of the efficient frontier,
    maximum Sharpe ratio portfolio, minimum variance portfolio, and selected point portfolio.
    """
    results = calculated_result(mean_returns, cov_matrix, risk_free_rate, constraints_set, trading_days)
    data = []

    # Maximum Sharpe Ratio point
    maxSR_plot = go.Scatter(
        name="Max Sharpe Ratio",
        mode="markers",
        x=[round(results["maxSR"]["std_ori"] * 100, 2)],
        y=[round(results["maxSR"]["return_ori"] * 100, 2)],
        marker=dict(color="#8883f0", size=18)
    )
    data.append(maxSR_plot)
    
    # Minimum Variance point
    minVar_plot = go.Scatter(
        name="Min Variance",
        mode="markers",
        x=[round(results["minVar"]["std_ori"] * 100, 2)],
        y=[round(results["minVar"]["return_ori"] * 100, 2)],
        marker=dict(color="#f39530", size=18)
    )
    data.append(minVar_plot)

    if selected_point is not None:
        selected_return = selected_point['return']
        selected_std = selected_point['risk']
        selected_plot = go.Scatter(
            name="Selected Point",
            mode="markers",
            x=[selected_std],
            y=[selected_return],
            marker=dict(color="#bd2d3c", size=18)
        )
        data.append(selected_plot)
    
    # Efficient frontier curve
    # Convert volatilities and target returns to percentages
    eff_vol = [round(v * 100, 2) if not np.isnan(v) else np.nan for v in results["efficient_frontier"]]
    eff_ret = [round(t * 100, 2) for t in results["target_returns"]]
    eff_curve = go.Scatter(
        name="Efficient Frontier",
        mode="lines",
        x=eff_vol,
        y=eff_ret,
        line=dict(color="#75badb", width=3, dash="dot")
    )
    data.append(eff_curve)

    layout = go.Layout(
        title="Portfolio Optimization with Efficient Frontier",
        xaxis=dict(title="Annualized Volatility (%)"),
        yaxis=dict(title="Annualized Return (%)"),
        showlegend=True,
        legend=dict(
            x=0.8,
            y=0.1,
            traceorder="normal",
            bgcolor="#e6e6e6",
            bordercolor="white",
            borderwidth=1,
            font=dict(color="#222222")
        ),
        width=750,
        height=600
    )

    fig = go.Figure(data=data, layout=layout)
    st.plotly_chart(fig)

def get_correlation(price_data, color_scale='BrBG_r', title="Correlation Matrix"):
    """
    Calculate and plot the correlation matrix as an interactive heatmap for a given price data DataFrame,
    with adaptive font and height size.

    Parameters:
        price_data (DataFrame): Historical price data.
        color_scale (str): The color scale to use for the heatmap (default 'BrBG_r').
        title (str): The title for the heatmap (default "Asset Correlation Matrix").

    Returns:
        Figure: The Plotly figure object.
    """
    with st.spinner("Getting Asset Correlation..."):
        if price_data is None or price_data.empty:
            st.error("No valid price data provided.")
            return None

        # Calculate daily percentage returns and drop missing values
        returns = price_data.pct_change().dropna()

        # Compute the correlation matrix
        corr_matrix = returns.corr()

        # Get the number of assets (tickers)
        num_assets = len(corr_matrix.columns)

        # Change font and height size based on the number of assets
        if num_assets >= 10:
            font_size = 7
            height_size = 700
        elif num_assets > 5 and num_assets < 10:
            font_size = 9
            height_size = 600
        else:
            font_size = 11
            height_size = 500

        # Create the heatmap using Plotly Express
        fig = px.imshow(
            corr_matrix,
            text_auto=".2f", 
            color_continuous_scale=color_scale,
            title=title
        )

        # Ensure Responsiveness
        fig.update_layout(
            xaxis=dict(
                tickangle=-45,  # Rotate x-axis labels for better readability
                tickfont=dict(size=font_size)
            ),
            yaxis=dict(
                tickfont=dict(size=font_size)
            ),
            autosize=True,
            height=height_size,
            margin=dict(l=10, r=10, t=50, b=10),
        )

        # Ensure text inside the matrix adjusts properly
        fig.update_traces(
            textfont_size=font_size,
            hoverinfo="text"  # Full value on hover
        )

        # Display the figure in Streamlit with full width
        st.plotly_chart(fig, use_container_width=True)
    return fig

def get_asset_metrics(price_data, risk_free_rate=0.0, trading_days=365):
    """
    Calculate annual return, annual volatility, and Sharpe ratio for each asset,
    and return a DataFrame with these metrics.

    Parameters:
        price_data (DataFrame): Historical price data with assets as columns.
        risk_free_rate (float): Annualized risk-free rate (in decimal, e.g., 0.02 for 2%).
        trading_days (int): Number of trading days per year (365 for default value).

    Returns:
        DataFrame: Contains each asset's annual return (%), annual volatility (%), and Sharpe ratio.
    """

    # Calculate daily returns, daily volatility and drop NaN values
    returns = price_data.pct_change().dropna()
    mean_returns = returns.mean()
    daily_std = returns.std()

    # Calculate annual return and annual volatility
    annual_return = (1 + mean_returns) ** trading_days - 1
    annual_std = daily_std * np.sqrt(trading_days)

    # Calculate sharpe ratio
    # (annual return - risk_free_rate) divided by annual volatility
    sharpe_ratio = (annual_return - risk_free_rate) / annual_std

    # Create a DataFrame with the metrics
    metrics_df = pd.DataFrame({
        "Annualized Return (%)": round(annual_return*100, 2),
        "Annualized Volatility (%)": round(annual_std*100, 2),
        "Sharpe Ratio": round(sharpe_ratio, 3)
    })
    
    return metrics_df

def plot_portfolio(tickers, weights):
    """
    Generate and display an interactive pie chart of portfolio allocations.
    
    Parameters:
        tickers (list or pd.Series): Asset symbols.
        weights (list or pd.Series of float): Allocation percentages in decimal form (e.g., 0.20 for 20%).
        
    Displays:
        A Plotly pie chart in Streamlit with proper formatting and error handling.
    """
    
    # Validate input
    if tickers is None or weights is None or len(tickers) == 0 or len(weights) == 0:
        st.error("Invalid allocation data - missing or empty values.")
        return
    
    # Create a DataFrame from the provided tickers and weights
    alloc_df = pd.DataFrame({
        'Ticker': tickers,
        'Weight': [round(w*100, 2) for w in weights]
    })
    
    # Convert decimal weights to a percentage display in hover info
    # (Plotly handles actual values in numeric form, so keep "Weight" numeric.)
    fig = px.pie(
        data_frame=alloc_df,
        values="Weight",
        names="Ticker",
        title=f"Portfolio Allocation ({len(alloc_df['Ticker'])} Assets)",
        hole=0.3,  # Donut chart style
        color_discrete_sequence=px.colors.qualitative.Pastel,
        hover_data={"Weight": ":.1f%"},
        labels={"Weight": "Allocation"}
    )
    
    # Format layout for clarity
    fig.update_layout(
        uniformtext_minsize=12,
        uniformtext_mode='hide',
        legend=dict(
            orientation="h",  # Horizontal legend
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        ),
        margin=dict(l=40, r=40, t=20, b=20)
    )
    
    # Add percentage labels inside slices
    fig.update_traces(
        textposition="inside",
        textinfo="percent+label",
        rotation=45,  # Rotate to help with label overlap
        sort=True     # Sort slices by size
    )
    
    # Display the chart in Streamlit
    st.write("")
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")

def sidebar_changed(tickers, start_date, end_date, risk_free_rate):
    # Create a dictionary with current sidebar values
    current_params = {
        "tickers": tickers,
        "start_date": str(start_date),
        "end_date": str(end_date),
        "risk_free_rate": risk_free_rate
    }
    # If these values haven’t been stored before, save them
    if "sidebar_params" not in st.session_state:
        st.session_state.sidebar_params = current_params
        return True  # first run, consider it a change
    # Compare the stored parameters with the current ones
    if st.session_state.sidebar_params != current_params:
        st.session_state.sidebar_params = current_params  # update the stored values
        return True
    return False

# -------------------------------
# Streamlit UI
# -------------------------------
# st.set_page_config(layout="wide")
st.title("Portfolio Optimization Toolkit")
st.markdown("Modern Portfolio Theory Implementation")

# Sidebar controls
with st.sidebar:
    st.logo(image="logowk.svg", size="large")
    st.header("Input Parameters")
    st.markdown("""
    **Ticker Lookup**  
    Need help finding tickers? Visit [Yahoo Finance](https://finance.yahoo.com/lookup) symbol lookup.
    """)
    st.write("")
    tickers = st.text_input(
        "Enter Stock/Crypto Tickers (comma-separated)",
        value="SPY, GOVT, BBCA.JK, NVDA, BTC-USD",
        help="Example: SPY, AAPL, BTC-USD. Check **Yahoo Finance** for ticker symbols."
    )
    ticker_list = [t.strip() for t in tickers.split(',')]

    start_date = st.date_input("Start Date", value=pd.to_datetime("2012-01-01"))
    end_date = st.date_input("End Date", value=pd.to_datetime(datetime.today().strftime('%Y-%m-%d')))
    risk_free_rate = st.number_input(
        "Risk-Free Rate (%)", 
        min_value=0.0,
        max_value=10.0,
        value=4.0,
        step=0.25
        ) / 100
    
    st.header("Portfolio Constraints")
    col1, col2 = st.columns(2)
    with col1:
        min_weight = st.number_input(
            "Minimum Weight (%)",
            min_value=0.0,
            max_value=10.0,
            value=2.0,
            step=1.0,
            format="%.1f"
        ) / 100
    with col2:
        max_weight = st.number_input(
            "Maximum Weight (%)", 
            min_value=100.0/len(ticker_list)+5.0,
            max_value=100.0,
            value=100.0,
            step=5.0,
            format="%.1f"
        ) / 100

    constraints_set = (min_weight, max_weight)

# Main execution
if st.button("Run Portfolio Analysis 🔍"):
    with st.spinner("Downloading market data..."):
        price_data = get_data(ticker_list, start=start_date, end=end_date)
    
    if price_data is not None and not price_data.empty:
        returns = get_return(price_data)
        if returns is not None:
            mean_returns, cov_matrix = returns
            results = calculated_result(mean_returns, cov_matrix, risk_free_rate, constraints_set)

            # Save computed data in session_state
            st.session_state['price_data'] = price_data
            st.session_state['mean_returns'] = mean_returns
            st.session_state['cov_matrix'] = cov_matrix
            st.session_state['results'] = results
            st.session_state['analysis_done'] = True
        
        else:
            st.error("Failed to calculate returns from price data.")
    else:
        st.error("No valid price data retrieved. Check ticker symbols and date range.")

# Check any changes in sidebar and update cache
if sidebar_changed(tickers, start_date, end_date, risk_free_rate) and st.session_state.get('analysis_done'):
    price_data = get_data(ticker_list, start=start_date, end=end_date)
    if price_data is not None and not price_data.empty:
        returns = get_return(price_data)
        if returns is not None:
            mean_returns, cov_matrix = returns
            results = calculated_result(mean_returns, cov_matrix, risk_free_rate, constraints_set)
            
            # Save computed results to session_state for later use.
            st.session_state['price_data'] = price_data
            st.session_state['mean_returns'] = mean_returns
            st.session_state['cov_matrix'] = cov_matrix
            st.session_state['results'] = results
            st.session_state['analysis_done'] = True
        
        else:
            st.error("Failed to calculate returns from price data.")
    else:
        st.error("No valid price data retrieved. Check ticker symbols and date range.")

# If analysis is done, display results and allow slider updates
if st.session_state.get('analysis_done'): 
    price_data = st.session_state['price_data']
    mean_returns = st.session_state['mean_returns']
    cov_matrix = st.session_state['cov_matrix']
    results = st.session_state['results']

    # Display optimization results
    st.write("")
    st.write("")
    st.subheader("Optimization Results")
    st.write("")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### Maximum Sharpe Ratio Portfolio (Highest Return/Risk Ratio)")
        st.metric("Expected Return", f"{results['maxSR']['return_disp']:.2f}%")
        st.metric("Volatility", f"{results['maxSR']['std_disp']:.2f}%")
        st.metric(
            "Expected Return Fluctuation Range", 
            f"{(results['maxSR']['return_disp'] - results['maxSR']['std_disp']):.2f}% ~ {(results['maxSR']['return_disp'] + results['maxSR']['std_disp']):.2f}%")
        st.write("")
        st.write("**Allocations:**")
        alloc_df = pd.DataFrame({
            'Ticker': mean_returns.index,
            'Weight': [f"{w:.1%}" for w in results['maxSR']['allocation']["Weightings"]]
        })
        st.dataframe(alloc_df, hide_index=True)
        plot_portfolio(alloc_df['Ticker'], results['maxSR']['allocation']["Weightings"])\
        
    with col2:
        st.markdown("##### Minimum Variance Portfolio (Lowest Volatility)")
        st.metric("Expected Return", f"{results['minVar']['return_disp']:.2f}%")
        st.metric("Volatility", f"{results['minVar']['std_disp']:.2f}%")
        st.metric(
            "Expected Return Fluctuation Range ", 
            f"{(results['minVar']['return_disp'] - results['minVar']['std_disp']):.2f}% ~ {(results['minVar']['return_disp'] + results['minVar']['std_disp']):.2f}%")
        st.write("")
        st.write("**Allocations:**")
        alloc_df = pd.DataFrame({
            'Ticker': mean_returns.index,
            'Weight': [f"{w:.1%}" for w in results['minVar']['allocation']["Weightings"]]
        })
        st.dataframe(alloc_df, hide_index=True)
        plot_portfolio(alloc_df['Ticker'], results['minVar']['allocation']["Weightings"])
    
    # # Plot efficient frontier
    # st.write("")
    # st.subheader("Efficient Frontier")
    # plot_efficient_frontier(mean_returns, cov_matrix, risk_free_rate, constraints_set)

    # Return slider
    st.write("")
    st.subheader("Select Efficient Frontier Portfolio")
    target_return_slider = st.slider(
        "Target Return (%)",
        min_value=results['minVar']['return_disp'],
        max_value=results['maxSR']['return_disp'],
        value=(results['minVar']['return_disp'] + results['maxSR']['return_disp']) / 2,
        step=0.1
    )
    selected_target = target_return_slider / 100

    # Display updated metrics
    with st.spinner("Optimizing portfolio for selected target return..."):
        new_portfolio = efficient_optimization(mean_returns, cov_matrix, selected_target, constraints_set)
    new_weights = new_portfolio['x']
    new_return = round(get_portfolio_return(new_weights, mean_returns, cov_matrix)*100, 2)
    new_std = round(get_portfolio_variance(new_weights, mean_returns, cov_matrix)*100, 2)
    
    col3, col4 = st.columns(2)
    with col3:
        st.metric("Expected Return", f"{new_return:.2f}%")
        st.metric("Volatility", f"{new_std:.2f}%")
        st.metric(
            "Expected Return Fluctuation Range ", 
            f"{(new_return - new_std):.2f}% ~ {(new_return + new_std):.2f}%") 

    with col4:
        st.write("**Allocations:**")
        alloc_df = pd.DataFrame({
            'Ticker': mean_returns.index,
            'Weight': [f"{w:.1%}" for w in new_weights]
        })
        st.dataframe(alloc_df, hide_index=True)
    
    st.write("")
    plot_portfolio(alloc_df['Ticker'], new_weights)

    # Update efficient frontier plot
    selected_point = {
        "return": new_return,
        "risk": new_std
    }
    st.write("")
    st.subheader("Efficient Frontier with Selected Portfolio")
    plot_efficient_frontier(mean_returns, cov_matrix, risk_free_rate, constraints_set, selected_point=selected_point)

    # Define custom color scale for correlation matrix
    custom_colorscale = [
        [0.0, "#f1b3a1"], 
        [0.5, "#e6e6e6"],  
        [1.0, "#a6c9e7"]   
    ]

    # Plot asset correlation matrix
    st.write("")
    st.subheader("Asset Correlation Matrix")
    get_correlation(price_data, color_scale=custom_colorscale)

    # Show asset metrics
    st.write("")
    st.subheader("Asset Metrics")
    metrics_df = get_asset_metrics(price_data, risk_free_rate=risk_free_rate, trading_days=365)
    st.dataframe(metrics_df, use_container_width=True)

    # Disclaimer warning
    st.write("")
    st.write("")
    st.warning(
        "**Disclaimer:** The results shown in this application are based on **historical data** and are intended for "
        "informational purposes only. The **expected returns** are calculated using past performance, but **future performance** "
        "may differ significantly. Market conditions, economic factors, and unforeseen events can impact actual returns. "
        "Please consider this before making any investment decisions."
    )
