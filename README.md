# AlphaQuant

AlphaQuant is an AI-powered personalized investment advisor that provides data-driven insights, predictions, and portfolio optimization using cutting-edge techniques in machine learning, Bayesian forecasting, and reinforcement learning.

## 🌟 Features

- **Data Pipeline**: Collect, clean, and process market data from various sources
- **Sentiment Analysis**: Analyze news and social media for market sentiment
- **Bayesian Forecasting**: Generate probabilistic price forecasts with uncertainty quantification
- **Machine Learning Predictions**: Predict market movements using traditional ML and deep learning
- **Portfolio Optimization**: Optimize portfolios based on user preferences and risk tolerance
- **Reinforcement Learning**: Develop adaptive trading strategies using RL techniques
- **REST API**: Access all features through a well-documented FastAPI interface
- **Streamlit Dashboard**: Interactive web interface for visualizing insights and predictions

## 🏗️ Project Structure

```
AlphaQuant/
├── src/
│   ├── data_pipeline/       # Market data acquisition and processing
│   ├── models/              # Price forecasting models
│   ├── reinforcement_learning/  # RL agents for portfolio management
│   ├── api/                 # FastAPI application and endpoints
│   └── utils/               # Helper utilities
├── data/                    # Data storage (cached and processed)
├── notebooks/               # Jupyter notebooks for exploration
├── tests/                   # Unit and integration tests
├── output/                  # Generated plots and results
├── app.py                   # Streamlit dashboard application
├── demo.py                  # Demo script showcasing functionality
└── requirements.txt         # Project dependencies
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Pip package manager

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/alphaquant.git
   cd alphaquant
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   
   # On Windows:
   venv\Scripts\activate
   
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   - Copy the `.env.example` file to create a new `.env` file:
     ```bash
     cp .env.example .env
     ```
   - Edit the `.env` file with your API keys:
     ```
     NEWS_API_KEY=your_news_api_key
     ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
     POLYGON_API_KEY=your_polygon_key
     ```
   - You'll need at minimum a News API key for the demo to work properly. Sign up at [newsapi.org](https://newsapi.org/) to get a free API key.

## 💻 Running the Project

### 1. Generate Data with Demo Script

First, run the demo script to generate the necessary data files and visualizations:

```bash
python demo.py
```

The demo will:
1. Fetch and process data for selected stocks
2. Generate market overview and sentiment analysis
3. Create Bayesian forecasts with uncertainty estimates
4. Train ML models for price prediction
5. Generate stock recommendations
6. Demonstrate a simplified reinforcement learning trading environment

Check the `output/` directory for generated visualizations and CSV files.

### 2. Launch the Streamlit Dashboard

After running the demo script, launch the interactive Streamlit dashboard:

```bash
streamlit run app.py
```

The dashboard will be available at:
- Local URL: http://localhost:8501
- Network URL: http://your-ip-address:8501

The Streamlit dashboard provides:
- Market overview with sentiment analysis
- Interactive stock analysis tools
- Bayesian forecasting visualization
- Portfolio performance simulation
- Market map visualization

## 🌐 API Usage

Start the API server:

```bash
cd src/api
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000` with interactive documentation at `http://localhost:8000/docs`.

Example API endpoints:
- `GET /stock/{ticker}`: Get historical stock data with technical indicators
- `POST /predict`: Generate stock price predictions
- `POST /recommend`: Get investment recommendations for a stock
- `POST /portfolio/optimize`: Optimize portfolio allocation
- `GET /market/overview`: Get current market conditions overview

## 💻 Development

### Running Tests

```bash
pytest tests/
```

### Adding New Features

1. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Implement your changes and add tests
3. Run tests to ensure functionality
4. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- Open-source libraries that made this project possible
- Financial data providers for making market data accessible
- Academic research in finance, ML, and reinforcement learning that inspired this project 