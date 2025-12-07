# 🚀 Crypto Vision

**Cryptocurrency price prediction and technical analysis platform powered by machine learning.**

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)

## Features

- 🤖 **Machine Learning**: Random Forest & Gradient Boosting models with cross-validation
- 📊 **Technical Indicators**: SMA, RSI, MACD, Bollinger Bands, Stochastic
- ⚖️ **Weighted Trend Analysis**: Multi-indicator voting system with confidence scoring
- 📈 **Risk Management**: Sharpe ratio, drawdown, ATR-based stop-loss/take-profit
- 💾 **Export/Import**: Save and share configurations
- 🤝 **AI Optimization**: Generate prompts for parameter tuning

## Supported Assets

BTC, ETH, BNB, XRP, DOGE, ADA, SOL, MATIC, DOT, LTC, SHIB, AVAX, TRX, UNI, LINK, XLM

## Installation

```bash
# Clone repository
git clone https://github.com/rhetmenium/crypto-vision.git
cd crypto-vision

# Install dependencies
pip install PyQt5 pandas numpy ta ccxt scikit-learn

# Run application
python crypto_vision.py
```

## Quick Start

1. Select cryptocurrency and timeframe
2. Configure technical indicators
3. Choose ML model (Random Forest or Gradient Boosting)
4. Click "Run Prediction"
5. Review trend analysis, price projection, and risk metrics

## Technical Indicators

| Indicator | Purpose | Default Settings |
|-----------|---------|-----------------|
| **SMA** | Trend identification | Short: 20, Long: 50 |
| **RSI** | Momentum & reversals | Window: 14, Levels: 30/70 |
| **MACD** | Trend strength | Fast: 12, Slow: 26, Signal: 9 |
| **Bollinger Bands** | Volatility analysis | Window: 20, Std: 2 |
| **Stochastic** | Momentum extremes | Window: 14, Levels: 20/80 |

## Output

- **Trend Analysis**: Bullish/Bearish scores with confidence %
- **Price Projection**: Next-day prediction with MAE/RMSE metrics
- **Risk Metrics**: Volatility, Sharpe ratio, suggested stop-loss/take-profit levels

## ⚠️ Disclaimer

**EDUCATIONAL TOOL ONLY. NOT FINANCIAL ADVICE.**

- Past performance ≠ future results
- Crypto markets are highly volatile
- ML models have limitations
- Always do your own research (DYOR)
- Never invest more than you can afford to lose

## Contributing

Pull requests welcome! For major changes, please open an issue first.

Educational purposes only. Not for commercial use.
