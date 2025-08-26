import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import requests
import os
from datetime import datetime, timedelta
import json

# Page configuration
st.set_page_config(
    page_title="Real Estate Market Analyzer",
    page_icon="🏠",
    layout="wide"
)

# Title and description
st.title("🏠 Real Estate Market Analyzer - San Diego")
st.markdown("Analyze housing market trends and forecast future prices using advanced analytics")

# Sidebar for configuration
st.sidebar.header("Configuration")
st.sidebar.markdown("### Zillow API Settings")
api_key = st.sidebar.text_input("Zillow API Key", type="password", help="Enter your Zillow API key")
use_mock_data = st.sidebar.checkbox("Use Mock Data (for demo purposes)", value=True)

# Mock data generation function
def generate_mock_data():
    """Generate realistic mock data for demonstration purposes"""
    np.random.seed(42)
    
    # Generate dates for the last 2 years
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)
    dates = pd.date_range(start=start_date, end=end_date, freq='M')
    
    # Generate realistic price data with trends
    base_price = 750000  # Base price for San Diego
    trend = 0.02  # 2% monthly growth trend
    seasonal_factor = 0.05  # Seasonal variation
    
    prices = []
    for i, date in enumerate(dates):
        # Add trend
        trend_component = base_price * (1 + trend) ** i
        
        # Add seasonal variation
        seasonal_component = seasonal_factor * base_price * np.sin(2 * np.pi * i / 12)
        
        # Add random noise
        noise = np.random.normal(0, 0.02 * base_price)
        
        price = trend_component + seasonal_component + noise
        prices.append(max(price, 500000))  # Ensure minimum price
    
    # Generate additional metrics
    days_on_market = np.random.randint(15, 60, len(dates))
    inventory_levels = np.random.randint(800, 1200, len(dates))
    price_per_sqft = [price / np.random.uniform(1200, 2000) for price in prices]
    
    data = pd.DataFrame({
        'date': dates,
        'median_price': prices,
        'days_on_market': days_on_market,
        'inventory_levels': inventory_levels,
        'price_per_sqft': price_per_sqft,
        'month': dates.month,
        'year': dates.year
    })
    
    return data

# Data fetching function (placeholder for real API integration)
def fetch_zillow_data(api_key, location="San Diego, CA"):
    """Fetch data from Zillow API (placeholder implementation)"""
    if not api_key or api_key.strip() == "":
        return generate_mock_data()
    
    # This would be the actual API call to Zillow
    # For now, we'll use mock data
    st.info("Zillow API integration would be implemented here with real API key")
    return generate_mock_data()

# Load data
@st.cache_data
def load_data():
    if use_mock_data:
        return generate_mock_data()
    else:
        return fetch_zillow_data(api_key)

# Load the data
data = load_data()

# Main dashboard
if not data.empty:
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        current_price = data['median_price'].iloc[-1]
        price_change = ((current_price - data['median_price'].iloc[-2]) / data['median_price'].iloc[-2]) * 100
        st.metric(
            label="Current Median Price",
            value=f"${current_price:,.0f}",
            delta=f"{price_change:+.1f}%"
        )
    
    with col2:
        current_inventory = data['inventory_levels'].iloc[-1]
        inventory_change = ((current_inventory - data['inventory_levels'].iloc[-2]) / data['inventory_levels'].iloc[-2]) * 100
        st.metric(
            label="Active Listings",
            value=f"{current_inventory:,}",
            delta=f"{inventory_change:+.1f}%"
        )
    
    with col3:
        current_dom = data['days_on_market'].iloc[-1]
        dom_change = current_dom - data['days_on_market'].iloc[-2]
        st.metric(
            label="Days on Market",
            value=f"{current_dom}",
            delta=f"{dom_change:+d} days"
        )
    
    with col4:
        current_ppsf = data['price_per_sqft'].iloc[-1]
        ppsf_change = ((current_ppsf - data['price_per_sqft'].iloc[-2]) / data['price_per_sqft'].iloc[-2]) * 100
        st.metric(
            label="Price per Sq Ft",
            value=f"${current_ppsf:.0f}",
            delta=f"{ppsf_change:+.1f}%"
        )
    
    st.markdown("---")
    
    # Charts section
    st.header("📊 Market Trends Analysis")
    
    # Price trend chart
    fig_price = px.line(
        data, 
        x='date', 
        y='median_price',
        title='Median Home Prices Over Time',
        labels={'median_price': 'Median Price ($)', 'date': 'Date'}
    )
    fig_price.update_layout(height=400)
    st.plotly_chart(fig_price, use_container_width=True)
    
    # Multi-metric chart
    fig_multi = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Price Trends', 'Inventory Levels', 'Days on Market', 'Price per Sq Ft'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Price trends
    fig_multi.add_trace(
        go.Scatter(x=data['date'], y=data['median_price'], name='Median Price', line=dict(color='blue')),
        row=1, col=1
    )
    
    # Inventory levels
    fig_multi.add_trace(
        go.Scatter(x=data['date'], y=data['inventory_levels'], name='Inventory', line=dict(color='green')),
        row=1, col=2
    )
    
    # Days on market
    fig_multi.add_trace(
        go.Scatter(x=data['date'], y=data['days_on_market'], name='Days on Market', line=dict(color='orange')),
        row=2, col=1
    )
    
    # Price per sq ft
    fig_multi.add_trace(
        go.Scatter(x=data['date'], y=data['price_per_sqft'], name='Price per Sq Ft', line=dict(color='red')),
        row=2, col=2
    )
    
    fig_multi.update_layout(height=600, showlegend=False)
    st.plotly_chart(fig_multi, use_container_width=True)
    
    # Forecasting section
    st.header("🔮 Price Forecasting")
    
    # Prepare data for forecasting
    X = np.arange(len(data)).reshape(-1, 1)
    y = data['median_price'].values
    
    # Train linear regression model
    model = LinearRegression()
    model.fit(X, y)
    
    # Make predictions
    future_months = 12
    future_X = np.arange(len(data), len(data) + future_months).reshape(-1, 1)
    future_predictions = model.predict(future_X)
    
    # Calculate confidence interval (simplified)
    mse = mean_squared_error(y, model.predict(X))
    confidence_interval = 1.96 * np.sqrt(mse)  # 95% confidence interval
    
    # Create forecast chart
    fig_forecast = go.Figure()
    
    # Historical data
    fig_forecast.add_trace(go.Scatter(
        x=data['date'],
        y=data['median_price'],
        mode='lines+markers',
        name='Historical Prices',
        line=dict(color='blue', width=2)
    ))
    
    # Forecast
    future_dates = pd.date_range(
        start=data['date'].iloc[-1] + pd.DateOffset(months=1),
        periods=future_months,
        freq='M'
    )
    
    fig_forecast.add_trace(go.Scatter(
        x=future_dates,
        y=future_predictions,
        mode='lines+markers',
        name='Forecast',
        line=dict(color='red', width=2, dash='dash')
    ))
    
    # Confidence interval
    fig_forecast.add_trace(go.Scatter(
        x=future_dates,
        y=future_predictions + confidence_interval,
        mode='lines',
        name='Upper Bound (95% CI)',
        line=dict(color='red', width=1, dash='dot'),
        showlegend=False
    ))
    
    fig_forecast.add_trace(go.Scatter(
        x=future_dates,
        y=future_predictions - confidence_interval,
        mode='lines',
        fill='tonexty',
        fillcolor='rgba(255,0,0,0.1)',
        name='Lower Bound (95% CI)',
        line=dict(color='red', width=1, dash='dot'),
        showlegend=False
    ))
    
    fig_forecast.update_layout(
        title='12-Month Price Forecast with Confidence Intervals',
        xaxis_title='Date',
        yaxis_title='Median Price ($)',
        height=500
    )
    
    st.plotly_chart(fig_forecast, use_container_width=True)
    
    # Model performance metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        r2 = r2_score(y, model.predict(X))
        st.metric("Model R² Score", f"{r2:.3f}")
    
    with col2:
        rmse = np.sqrt(mse)
        st.metric("RMSE", f"${rmse:,.0f}")
    
    with col3:
        forecast_6m = future_predictions[5]  # 6 months out
        forecast_change = ((forecast_6m - current_price) / current_price) * 100
        st.metric("6-Month Forecast", f"${forecast_6m:,.0f}", f"{forecast_change:+.1f}%")
    
    # Market insights
    st.header("💡 Market Insights")
    
    # Calculate insights
    price_volatility = data['median_price'].pct_change().std() * 100
    avg_dom = data['days_on_market'].mean()
    price_trend = ((data['median_price'].iloc[-1] - data['median_price'].iloc[0]) / data['median_price'].iloc[0]) * 100
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Market Statistics")
        st.write(f"**Price Volatility:** {price_volatility:.1f}%")
        st.write(f"**Average Days on Market:** {avg_dom:.0f} days")
        st.write(f"**Overall Price Trend:** {price_trend:+.1f}%")
        
        if price_trend > 5:
            st.success("📈 Strong upward price trend detected")
        elif price_trend > 0:
            st.info("📊 Moderate price growth")
        else:
            st.warning("📉 Declining prices detected")
    
    with col2:
        st.subheader("Recommendations")
        if current_inventory < 1000:
            st.info("🏠 Low inventory suggests seller's market")
        else:
            st.info("🏠 Higher inventory suggests buyer's market")
        
        if current_dom < 30:
            st.success("⚡ Fast-moving market")
        else:
            st.warning("⏳ Slower market activity")
        
        if price_volatility > 3:
            st.warning("📊 High market volatility")
        else:
            st.success("📊 Stable market conditions")

else:
    st.error("Failed to load data. Please check your configuration.")

# Footer
st.markdown("---")
st.markdown(
    "**Note:** This is a demonstration application. For production use, integrate with the actual Zillow API and implement proper error handling and data validation."
)
import plotly.graph_objects as go
# Date range filtering functionality
import plotly.graph_objects as go
# Date range filtering functionality
# Export functionality for reports
import plotly.graph_objects as go
# Date range filtering functionality
# Export functionality for reports
# Performance optimizations
# Add caching for expensive operations
# Improve user interface
# Add dark mode support
# Fix edge cases in data validation
# Improve error handling
# Add bar chart for monthly comparisons
# Add heatmap for seasonal trends
# Add PDF export capability
# Add date range picker widget
# Add basic market metrics display
# Add simple line chart for prices
# Add multiple chart types
# Implement date filtering
