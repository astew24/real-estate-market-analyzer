# 🏠 Real Estate Market Analyzer

A comprehensive web application that analyzes and visualizes housing market trends in San Diego using advanced analytics and forecasting techniques. Built with Streamlit, this application provides real-time insights into real estate market dynamics.

## ✨ Features

- **Interactive Dashboard**: Single-page dashboard with real-time market metrics
- **Market Trend Analysis**: Visualize median home prices, inventory levels, and days on market
- **Price Forecasting**: Implement linear regression models to predict future market prices
- **Confidence Intervals**: Statistical analysis with 95% confidence intervals for forecasts
- **Market Insights**: Automated analysis and recommendations based on current market conditions
- **Zillow API Integration**: Ready for real-time data integration (currently uses mock data for demonstration)
- **Responsive Design**: Optimized for desktop and mobile viewing

## 🛠️ Technologies Used

- **Language**: Python 3.8+
- **Web Framework**: Streamlit
- **Data Analysis**: Pandas, NumPy
- **Machine Learning**: Scikit-learn
- **Data Visualization**: Plotly
- **API Integration**: Requests (for Zillow API)
- **Environment Management**: python-dotenv

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/real-estate-market-analyzer.git
   cd real-estate-market-analyzer
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables (optional)**
   ```bash
   cp .env.example .env
   # Edit .env file with your Zillow API key
   ```

### Running the Application

1. **Start the Streamlit app**
   ```bash
   streamlit run app.py
   ```

2. **Open your browser**
   Navigate to `http://localhost:8501`

3. **Configure the app**
   - Use the sidebar to toggle between mock data and real API data
   - Enter your Zillow API key if you want to use real data

## 📊 Features in Detail

### Market Metrics Dashboard
- Current median home price with month-over-month change
- Active listing inventory levels
- Average days on market
- Price per square foot trends

### Interactive Charts
- **Price Trends**: Line charts showing median price evolution over time
- **Multi-Metric View**: 2x2 grid showing all key metrics simultaneously
- **Forecasting**: 12-month price predictions with confidence intervals

### Machine Learning Models
- **Linear Regression**: Trained on historical price data
- **Model Performance**: R² score, RMSE, and forecast accuracy metrics
- **Statistical Analysis**: Confidence intervals and trend analysis

### Market Insights
- **Volatility Analysis**: Price fluctuation patterns
- **Market Conditions**: Buyer's vs. seller's market indicators
- **Recommendations**: Automated insights based on current metrics

## 🔧 Configuration

### Mock Data Mode
By default, the application runs in mock data mode, generating realistic sample data for demonstration purposes. This allows you to explore all features without an API key.

### Zillow API Integration
To use real data:

1. Obtain a Zillow API key from [Zillow Developer Portal](https://developer.zillow.com/)
2. Enter the API key in the sidebar
3. Uncheck "Use Mock Data"
4. The application will attempt to fetch real-time data

**Note**: Zillow API integration is implemented as a placeholder. You'll need to implement the actual API calls based on Zillow's current API documentation.

## 📈 Data Sources

- **Mock Data**: Realistically generated sample data for demonstration
- **Zillow API**: Real estate market data (when implemented)
- **Historical Analysis**: 2+ years of monthly data points

## 🚀 Deployment

### Local Development
```bash
streamlit run app.py
```

### Production Deployment
1. **Streamlit Cloud**: Deploy directly to Streamlit Cloud
2. **Heroku**: Use the provided Procfile and requirements.txt
3. **Docker**: Containerize the application for cloud deployment

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Streamlit team for the amazing web framework
- Plotly for interactive data visualization
- Scikit-learn for machine learning capabilities
- Zillow for real estate data APIs

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/yourusername/real-estate-market-analyzer/issues) page
2. Create a new issue with detailed information
3. Contact the maintainers

---

**Built with ❤️ for real estate professionals and market analysts**
