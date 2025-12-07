import sys
import json
import os
from datetime import datetime
import numpy as np
import pandas as pd
import ta
import ccxt
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

class PredictionWorker(QThread):
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)
    
    def __init__(self, parent_window):
        super().__init__()
        self.parent = parent_window
    
    def run(self):
        try:
            sym = self.parent.asset_combo.currentText()
            tf = self.parent.timeframe_combo.currentText()
            model_type = self.parent.model_combo.currentText()
            
            self.progress.emit("Fetching data from exchange...")
            df = self.parent.fetch_data(sym, tf)
            if df.empty or len(df) < 50:
                self.error.emit(f"Insufficient data from exchange (got {len(df)} candles, need 50+). Try longer timeframe.")
                return
            
            self.progress.emit(f"Retrieved {len(df)} candles. Adding indicators...")
            df = self.parent.add_indicators(df)
            if df.empty or len(df) < 30:
                self.error.emit("Insufficient data after indicator calculation (need 30+ samples).")
                return
            
            self.progress.emit("Building feature matrix with engineering...")
            X_df = self.parent.build_feature_matrix(df)
            
            self.progress.emit("Running trend analysis...")
            trend_txt = self.parent._predict_trend(df, X_df, sym, tf)
            
            self.progress.emit(f"Training {model_type} model...")
            price_txt = self.parent._predict_price(df, X_df, sym, tf, model_type)
            
            self.progress.emit("Calculating risk metrics...")
            risk_txt = self.parent._calculate_risk_metrics(df, X_df)
            
            result = f"Asset: {sym}\nHistory: {tf}\nModel: {model_type}\nIndicators: {self.parent._selected_indicator_names()}\n\n{trend_txt}\n\n{price_txt}\n\n{risk_txt}"
            
            if self.parent.save_predictions_cb.isChecked():
                self.parent._save_prediction(sym, tf, result)
                result += f"\n\n✓ Saved to predictions/"
            
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(f"Prediction failed: {str(e)}")

class CryptoPredictionLab(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Crypto Vision")
        self.setGeometry(200, 200, 620, 580)
        self.setStyleSheet("""
            QWidget {background-color: #111; color: #f4f4f4; font-weight: bold;}
            QGroupBox {border: 1px solid #2a2a2a; margin-top: 12px; padding: 8px; font-weight: bold;}
            QLabel {font-weight: bold;}
            QComboBox, QTextEdit {background-color: #1b1b1b; border: 1px solid #333; padding: 4px; color: #f4f4f4; font-weight: bold;}
            QCheckBox {font-weight: bold;}
            QPushButton {background-color: #2563eb; border: none; border-radius: 6px; padding: 8px 12px; color: #fff; font-weight: bold;}
            QPushButton:hover {background-color: #1d4ed8;}
            QPushButton:disabled {background-color: #444; color: #888;}
            QTextEdit {min-height: 90px;}
            QSpinBox {background-color: #1b1b1b; border: 1px solid #333; padding: 2px; color: #f4f4f4; font-weight: bold;}
            QProgressBar {border: 1px solid #333; border-radius: 3px; background-color: #1b1b1b; text-align: center;}
            QProgressBar::chunk {background-color: #2563eb;}
        """)

        ml = QVBoxLayout()
        tr = QHBoxLayout()

        ig = QGroupBox("Technical Indicators")
        il = QVBoxLayout()

        def add_indicator(label, cb_attr, w_vals, w_range):
            row = QHBoxLayout()
            cb = QCheckBox(label)
            cb.setChecked(True)
            setattr(self, cb_attr, cb)
            row.addWidget(cb)
            for lbl, attr, val, rng in w_vals:
                row.addWidget(QLabel(lbl))
                spin = QSpinBox()
                spin.setRange(*rng)
                spin.setValue(val)
                spin.setMaximumWidth(w_range)
                setattr(self, attr, spin)
                row.addWidget(spin)
            row.addStretch()
            il.addLayout(row)

        add_indicator("SMA Short", "cb_sma_short", [("W:", "sma_short_window", 20, (3, 200))], 60)
        add_indicator("SMA Long", "cb_sma_long", [("W:", "sma_long_window", 50, (3, 200))], 60)
        add_indicator("RSI", "cb_rsi", [("W:", "rsi_window", 14, (3, 50)), ("Low:", "rsi_low", 30, (0, 50)), ("High:", "rsi_high", 70, (50, 100))], 50)
        add_indicator("MACD", "cb_macd", [("Fast:", "macd_fast", 12, (3, 50)), ("Slow:", "macd_slow", 26, (5, 100)), ("Sig:", "macd_signal", 9, (3, 20))], 50)
        add_indicator("Bollinger", "cb_bbands", [("W:", "bb_window", 20, (5, 100)), ("Std:", "bb_std", 2, (1, 5))], 60)
        add_indicator("Stochastic", "cb_stoch", [("W:", "stoch_window", 14, (5, 50)), ("Low:", "stoch_low", 20, (0, 50)), ("High:", "stoch_high", 80, (50, 100))], 50)

        ig.setLayout(il)
        tr.addWidget(ig, stretch=2)

        mg = QGroupBox("Market & Model")
        mgl = QVBoxLayout()
        mgl.addWidget(QLabel("Symbol"))
        self.asset_combo = QComboBox()
        self.asset_combo.addItems(["BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "DOGE-USD", "ADA-USD", "SOL-USD", "MATIC-USD", "DOT-USD", "LTC-USD", "SHIB-USD", "AVAX-USD", "TRX-USD", "UNI-USD", "LINK-USD", "XLM-USD"])
        mgl.addWidget(self.asset_combo)
        mgl.addWidget(QLabel("History"))
        self.timeframe_combo = QComboBox()
        self.timeframe_combo.addItems(["1 month", "3 months", "6 months", "1 year", "2 years", "5 years"])
        mgl.addWidget(self.timeframe_combo)
        mgl.addWidget(QLabel("ML Model"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["Random Forest", "Gradient Boosting"])
        mgl.addWidget(self.model_combo)
        mg.setLayout(mgl)
        tr.addWidget(mg, stretch=1)

        ml.addLayout(tr)

        pbr = QHBoxLayout()
        eb = QPushButton("Export")
        eb.clicked.connect(self.export_parameters)
        ib = QPushButton("Import")
        ib.clicked.connect(self.import_parameters)
        apb = QPushButton("AI Prompt")
        apb.clicked.connect(self.show_ai_prompt)
        pbr.addWidget(eb)
        pbr.addWidget(ib)
        pbr.addWidget(apb)
        ml.addLayout(pbr)

        self.save_predictions_cb = QCheckBox("Save predictions to file")
        ml.addWidget(self.save_predictions_cb)

        self.run_button = QPushButton("Run Prediction")
        self.run_button.clicked.connect(self.run_prediction)
        ml.addWidget(self.run_button)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(False)
        ml.addWidget(self.progress_bar)

        ml.addWidget(QLabel("Results"))
        self.result_box = QTextEdit()
        self.result_box.setReadOnly(True)
        ml.addWidget(self.result_box)

        disc = QLabel("⚠ Educational tool only. Not financial advice. User responsible for all decisions. Past performance ≠ future results.")
        disc.setWordWrap(True)
        disc.setStyleSheet("color: #888; font-size: 9px; padding: 4px;")
        disc.setAlignment(Qt.AlignCenter)
        ml.addWidget(disc)

        container = QWidget()
        container.setLayout(ml)
        self.setCentralWidget(container)
        
        self.worker = None

    def _get_history_days(self, tf):
        return {"1 month": 30, "3 months": 90, "6 months": 180, "1 year": 365, "2 years": 730, "5 years": 1825}.get(tf, 365)

    def fetch_data(self, symbol, tf):
        days = self._get_history_days(tf)
        ms = symbol.replace("-USD", "/USDT") if symbol.endswith("-USD") else symbol.replace("-", "/")
        ex = ccxt.binance({'enableRateLimit': True})
        
        try:
            all_ohlcv = []
            actual_days = max(days, 100)
            since = ex.milliseconds() - actual_days * 86400000
            limit = 1000
            
            while True:
                ohlcv = ex.fetch_ohlcv(ms, timeframe="1d", since=since, limit=limit)
                if not ohlcv:
                    break
                all_ohlcv.extend(ohlcv)
                if len(ohlcv) < limit:
                    break
                since = ohlcv[-1][0] + 86400000
                
            if not all_ohlcv or len(all_ohlcv) < 50:
                return pd.DataFrame()
            
            df = pd.DataFrame([{"Date": pd.to_datetime(t, unit="ms"), "Open": o, "High": h, "Low": l, "Close": c, "Volume": v} for t, o, h, l, c, v in all_ohlcv])
            df = df.drop_duplicates(subset=['Date']).set_index("Date").sort_index()
            
            if len(df) > days:
                df = df.tail(max(days, 100))
            
            return df.dropna()
        except Exception as e:
            print(f"Fetch error: {e}")
            return pd.DataFrame()

    def add_indicators(self, df):
        n = len(df)
        if n < 50:
            return pd.DataFrame()
        d = df.copy()
        
        try:
            max_window = n // 3
            
            if self.cb_sma_short.isChecked():
                w = min(self.sma_short_window.value(), max_window)
                if n > w:
                    d["SMA_short"] = ta.trend.sma_indicator(d["Close"], window=w)
            
            if self.cb_sma_long.isChecked():
                w = min(self.sma_long_window.value(), max_window)
                if n > w:
                    d["SMA_long"] = ta.trend.sma_indicator(d["Close"], window=w)
            
            if self.cb_rsi.isChecked():
                w = min(self.rsi_window.value(), max_window, 14)
                if n > w:
                    d["RSI"] = ta.momentum.RSIIndicator(d["Close"], window=w).rsi()
            
            if self.cb_macd.isChecked():
                f = min(self.macd_fast.value(), max_window, 12)
                sl = min(self.macd_slow.value(), max_window, 26)
                sig = min(self.macd_signal.value(), 9)
                if n > sl and f < sl:
                    macd = ta.trend.MACD(d["Close"], window_slow=sl, window_fast=f, window_sign=sig)
                    d["MACD"] = macd.macd()
                    d["MACD_Signal"] = macd.macd_signal()
                    d["MACD_Diff"] = macd.macd_diff()
            
            if self.cb_bbands.isChecked():
                w = min(self.bb_window.value(), max_window, 20)
                if n > w:
                    bb = ta.volatility.BollingerBands(d["Close"], window=w, window_dev=self.bb_std.value())
                    d["BB_High"] = bb.bollinger_hband()
                    d["BB_Low"] = bb.bollinger_lband()
                    d["BB_Mid"] = bb.bollinger_mavg()
            
            if self.cb_stoch.isChecked():
                w = min(self.stoch_window.value(), max_window, 14)
                if n > w:
                    d["Stoch"] = ta.momentum.StochasticOscillator(d["High"], d["Low"], d["Close"], window=w).stoch()
            
            atr_w = min(14, max_window)
            vol_w = min(20, max_window)
            
            if n > atr_w:
                d["ATR"] = ta.volatility.AverageTrueRange(d["High"], d["Low"], d["Close"], window=atr_w).average_true_range()
            if n > vol_w:
                d["Volume_SMA"] = ta.trend.sma_indicator(d["Volume"], window=vol_w)
            
            d = d.fillna(method='ffill').fillna(method='bfill')
            
            return d if len(d) >= 30 else pd.DataFrame()
        except Exception as e:
            print(f"Indicator error: {e}")
            return pd.DataFrame()

    def build_feature_matrix(self, df):
        fc = []
        for col in ["SMA_short", "SMA_long", "RSI", "MACD", "MACD_Signal", "MACD_Diff", "BB_High", "BB_Low", "BB_Mid", "Stoch", "ATR", "Volume_SMA"]:
            if col in df.columns:
                fc.append(col)
        
        feat_df = df[fc + ["Close", "Volume"]].copy()
        
        feat_df["Returns"] = feat_df["Close"].pct_change()
        feat_df["Log_Returns"] = np.log(feat_df["Close"] / feat_df["Close"].shift(1))
        feat_df["Volatility"] = feat_df["Returns"].rolling(20).std()
        feat_df["Momentum"] = feat_df["Close"] - feat_df["Close"].shift(10)
        feat_df["Volume_Change"] = feat_df["Volume"].pct_change()
        
        if "RSI" in feat_df.columns:
            feat_df["RSI_MA"] = feat_df["RSI"].rolling(5).mean()
        if "MACD" in feat_df.columns:
            feat_df["MACD_Histogram"] = feat_df["MACD"] - feat_df["MACD_Signal"]
        
        feat_df = feat_df.fillna(method='ffill').fillna(method='bfill')
        return feat_df

    def run_prediction(self):
        if self.worker and self.worker.isRunning():
            return
        
        self.run_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.result_box.clear()
        
        self.worker = PredictionWorker(self)
        self.worker.finished.connect(self._on_prediction_finished)
        self.worker.error.connect(self._on_prediction_error)
        self.worker.progress.connect(self._on_progress)
        self.worker.start()

    def _on_progress(self, msg):
        self.result_box.append(msg)

    def _on_prediction_finished(self, result):
        self.result_box.setPlainText(result)
        self.run_button.setEnabled(True)
        self.progress_bar.setVisible(False)

    def _on_prediction_error(self, error):
        self.result_box.setPlainText(f"❌ Error: {error}")
        self.run_button.setEnabled(True)
        self.progress_bar.setVisible(False)

    def _selected_indicator_names(self):
        names = []
        if self.cb_sma_short.isChecked(): names.append("SMA Short")
        if self.cb_sma_long.isChecked(): names.append("SMA Long")
        if self.cb_rsi.isChecked(): names.append("RSI")
        if self.cb_macd.isChecked(): names.append("MACD")
        if self.cb_bbands.isChecked(): names.append("BB")
        if self.cb_stoch.isChecked(): names.append("Stoch")
        return ", ".join(names) if names else "None"

    def _predict_price(self, df, X_df, sym, tf, model_type):
        dt = df.copy()
        dt["Target"] = dt["Close"].shift(-1)
        dt = dt.dropna()
        
        common_idx = X_df.index.intersection(dt.index)
        if len(common_idx) < 30:
            return "Insufficient aligned data for prediction."
        
        X = X_df.loc[common_idx].values
        y = dt.loc[common_idx, "Target"].values
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        tscv = TimeSeriesSplit(n_splits=3)
        mae_scores, rmse_scores = [], []
        
        for train_idx, test_idx in tscv.split(X_scaled):
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            if model_type == "Random Forest":
                model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
            else:
                model = GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            mae_scores.append(mean_absolute_error(y_test, y_pred))
            rmse_scores.append(np.sqrt(mean_squared_error(y_test, y_pred)))
        
        if model_type == "Random Forest":
            final_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        else:
            final_model = GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
        
        final_model.fit(X_scaled, y)
        
        last_features = X_df.iloc[-1:].values
        last_scaled = scaler.transform(last_features)
        pred = final_model.predict(last_scaled)[0]
        
        curr = df["Close"].iloc[-1]
        diff = pred - curr
        pct = (diff / curr) * 100
        
        return f"--- Price Projection ({model_type}) ---\nCurrent: ${curr:.4f}\nPredicted Next: ${pred:.4f}\nChange: ${diff:.4f} ({pct:+.2f}%)\n\nCross-Validation:\nMAE: ${np.mean(mae_scores):.4f} (±${np.std(mae_scores):.4f})\nRMSE: ${np.mean(rmse_scores):.4f}\n\nNote: ML models capture non-linear patterns but past ≠ future."

    def _predict_trend(self, df, X_df, sym, tf):
        last = df.iloc[-1]
        vu, vd, signals = 0, 0, []
        weights = {"SMA": 1.5, "RSI": 1.0, "MACD": 1.5, "BB": 1.0, "Stoch": 0.8}
        
        if "SMA_short" in df.columns and "SMA_long" in df.columns:
            w = weights["SMA"]
            if last["SMA_short"] > last["SMA_long"]: 
                vu += w
                signals.append(f"SMA: Bullish (↑{w})")
            else: 
                vd += w
                signals.append(f"SMA: Bearish (↓{w})")
        
        if "RSI" in df.columns:
            w = weights["RSI"]
            rsi = last["RSI"]
            if rsi < self.rsi_low.value(): 
                vu += w * 1.5
                signals.append(f"RSI: Oversold {rsi:.1f} (↑{w*1.5:.1f})")
            elif rsi > self.rsi_high.value(): 
                vd += w * 1.5
                signals.append(f"RSI: Overbought {rsi:.1f} (↓{w*1.5:.1f})")
            elif rsi < 50:
                vu += w * 0.5
                signals.append(f"RSI: Below neutral {rsi:.1f} (↑{w*0.5:.1f})")
            else:
                vd += w * 0.5
                signals.append(f"RSI: Above neutral {rsi:.1f} (↓{w*0.5:.1f})")
        
        if "MACD" in df.columns and "MACD_Signal" in df.columns:
            w = weights["MACD"]
            if last["MACD"] > last["MACD_Signal"]: 
                vu += w
                signals.append(f"MACD: Bullish (↑{w})")
            else: 
                vd += w
                signals.append(f"MACD: Bearish (↓{w})")
        
        if "BB_High" in df.columns and "BB_Low" in df.columns:
            w = weights["BB"]
            bb_range = last["BB_High"] - last["BB_Low"]
            if bb_range > 0:
                bb_pos = (last["Close"] - last["BB_Low"]) / bb_range
                if bb_pos < 0.2: 
                    vu += w * 1.3
                    signals.append(f"BB: Lower band (↑{w*1.3:.1f})")
                elif bb_pos > 0.8: 
                    vd += w * 1.3
                    signals.append(f"BB: Upper band (↓{w*1.3:.1f})")
                elif bb_pos < 0.5:
                    vu += w * 0.5
                    signals.append(f"BB: Below mid (↑{w*0.5:.1f})")
                else:
                    vd += w * 0.5
                    signals.append(f"BB: Above mid (↓{w*0.5:.1f})")
        
        if "Stoch" in df.columns:
            w = weights["Stoch"]
            st = last["Stoch"]
            if st < self.stoch_low.value(): 
                vu += w * 1.2
                signals.append(f"Stoch: Oversold {st:.1f} (↑{w*1.2:.1f})")
            elif st > self.stoch_high.value(): 
                vd += w * 1.2
                signals.append(f"Stoch: Overbought {st:.1f} (↓{w*1.2:.1f})")
            elif st < 50:
                vu += w * 0.4
                signals.append(f"Stoch: Low {st:.1f} (↑{w*0.4:.1f})")
            else:
                vd += w * 0.4
                signals.append(f"Stoch: High {st:.1f} (↓{w*0.4:.1f})")
        
        total = vu + vd
        conf = abs(vu - vd) / total * 100 if total > 0 else 0
        trend = "Bullish" if vu > vd else ("Bearish" if vd > vu else "Neutral")
        
        return f"--- Weighted Trend Analysis ---\nPrice: ${last['Close']:.4f}\nBullish Score: {vu:.1f}\nBearish Score: {vd:.1f}\nTrend: {trend} (Confidence: {conf:.1f}%)\n\nSignals:\n  • " + "\n  • ".join(signals)

    def _calculate_risk_metrics(self, df, X_df):
        returns = df["Close"].pct_change().dropna()
        
        volatility = returns.std() * np.sqrt(365)
        sharpe = (returns.mean() * 365) / (returns.std() * np.sqrt(365)) if returns.std() > 0 else 0
        
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.min() * 100
        
        curr_price = df["Close"].iloc[-1]
        atr = df["ATR"].iloc[-1] if "ATR" in df.columns else curr_price * 0.02
        
        stop_loss = curr_price - (2 * atr)
        take_profit = curr_price + (3 * atr)
        risk_reward = 3.0 / 2.0
        
        return f"--- Risk & Money Management ---\nAnnualized Volatility: {volatility*100:.2f}%\nSharpe Ratio: {sharpe:.2f}\nMax Drawdown: {max_dd:.2f}%\nATR (14): ${atr:.4f}\n\nSuggested Levels:\nStop Loss: ${stop_loss:.4f} (-{(curr_price-stop_loss)/curr_price*100:.2f}%)\nTake Profit: ${take_profit:.4f} (+{(take_profit-curr_price)/curr_price*100:.2f}%)\nRisk/Reward: 1:{risk_reward:.1f}\n\n⚠ Use 1-2% of capital per trade"

    def _save_prediction(self, sym, tf, result):
        try:
            folder = "predictions"
            os.makedirs(folder, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            fn = f"{folder}/pred_{sym.replace('-', '_')}_{ts}.txt"
            with open(fn, "w", encoding="utf-8") as f:
                f.write("="*60 + f"\nCRYPTO PREDICTION - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n" + "="*60 + "\n\n" + result + "\n" + "="*60)
        except Exception as e:
            print(f"Save error: {e}")

    def export_parameters(self):
        params = {
            "asset": self.asset_combo.currentText(),
            "timeframe": self.timeframe_combo.currentText(),
            "model": self.model_combo.currentText(),
            "indicators": {
                "sma_short": {"enabled": self.cb_sma_short.isChecked(), "window": self.sma_short_window.value()},
                "sma_long": {"enabled": self.cb_sma_long.isChecked(), "window": self.sma_long_window.value()},
                "rsi": {"enabled": self.cb_rsi.isChecked(), "window": self.rsi_window.value(), "oversold": self.rsi_low.value(), "overbought": self.rsi_high.value()},
                "macd": {"enabled": self.cb_macd.isChecked(), "fast": self.macd_fast.value(), "slow": self.macd_slow.value(), "signal": self.macd_signal.value()},
                "bollinger": {"enabled": self.cb_bbands.isChecked(), "window": self.bb_window.value(), "std": self.bb_std.value()},
                "stochastic": {"enabled": self.cb_stoch.isChecked(), "window": self.stoch_window.value(), "oversold": self.stoch_low.value(), "overbought": self.stoch_high.value()},
            }
        }
        QApplication.clipboard().setText(json.dumps(params, indent=2))
        QMessageBox.information(self, "Exported", "Parameters copied!")

    def import_parameters(self):
        d = QDialog(self)
        d.setWindowTitle("Import JSON")
        d.setGeometry(300, 300, 500, 400)
        d.setStyleSheet(self.styleSheet())
        lo = QVBoxLayout()
        te = QTextEdit()
        lo.addWidget(te)
        bl = QHBoxLayout()
        impb = QPushButton("Import")
        canb = QPushButton("Cancel")
        bl.addWidget(impb)
        bl.addWidget(canb)
        lo.addLayout(bl)
        
        def do_imp():
            try:
                p = json.loads(te.toPlainText().strip())
                if "asset" in p: self.asset_combo.setCurrentText(p["asset"])
                if "timeframe" in p: self.timeframe_combo.setCurrentText(p["timeframe"])
                if "model" in p: self.model_combo.setCurrentText(p["model"])
                if "indicators" in p:
                    i = p["indicators"]
                    if "sma_short" in i:
                        self.cb_sma_short.setChecked(i["sma_short"].get("enabled", True))
                        if "window" in i["sma_short"]: self.sma_short_window.setValue(i["sma_short"]["window"])
                    if "sma_long" in i:
                        self.cb_sma_long.setChecked(i["sma_long"].get("enabled", True))
                        if "window" in i["sma_long"]: self.sma_long_window.setValue(i["sma_long"]["window"])
                    if "rsi" in i:
                        self.cb_rsi.setChecked(i["rsi"].get("enabled", True))
                        if "window" in i["rsi"]: self.rsi_window.setValue(i["rsi"]["window"])
                        if "oversold" in i["rsi"]: self.rsi_low.setValue(i["rsi"]["oversold"])
                        if "overbought" in i["rsi"]: self.rsi_high.setValue(i["rsi"]["overbought"])
                    if "macd" in i:
                        self.cb_macd.setChecked(i["macd"].get("enabled", True))
                        if "fast" in i["macd"]: self.macd_fast.setValue(i["macd"]["fast"])
                        if "slow" in i["macd"]: self.macd_slow.setValue(i["macd"]["slow"])
                        if "signal" in i["macd"]: self.macd_signal.setValue(i["macd"]["signal"])
                    if "bollinger" in i:
                        self.cb_bbands.setChecked(i["bollinger"].get("enabled", True))
                        if "window" in i["bollinger"]: self.bb_window.setValue(i["bollinger"]["window"])
                        if "std" in i["bollinger"]: self.bb_std.setValue(i["bollinger"]["std"])
                    if "stochastic" in i:
                        self.cb_stoch.setChecked(i["stochastic"].get("enabled", True))
                        if "window" in i["stochastic"]: self.stoch_window.setValue(i["stochastic"]["window"])
                        if "oversold" in i["stochastic"]: self.stoch_low.setValue(i["stochastic"]["oversold"])
                        if "overbought" in i["stochastic"]: self.stoch_high.setValue(i["stochastic"]["overbought"])
                QMessageBox.information(d, "Success", "Imported!")
                d.accept()
            except Exception as e:
                QMessageBox.critical(d, "Error", str(e))
        
        impb.clicked.connect(do_imp)
        canb.clicked.connect(d.reject)
        d.setLayout(lo)
        d.exec_()

    def show_ai_prompt(self):
        d = QDialog(self)
        d.setWindowTitle("AI Optimization")
        d.setGeometry(300, 300, 650, 550)
        d.setStyleSheet(self.styleSheet())
        lo = QVBoxLayout()
        
        asset = self.asset_combo.currentText()
        tf = self.timeframe_combo.currentText()
        model = self.model_combo.currentText()
        
        cp = {
            "asset": asset,
            "timeframe": tf,
            "model": model,
            "indicators": {}
        }
        
        if self.cb_sma_short.isChecked(): cp["indicators"]["sma_short"] = {"enabled": True, "window": self.sma_short_window.value()}
        if self.cb_sma_long.isChecked(): cp["indicators"]["sma_long"] = {"enabled": True, "window": self.sma_long_window.value()}
        if self.cb_rsi.isChecked(): cp["indicators"]["rsi"] = {"enabled": True, "window": self.rsi_window.value(), "oversold": self.rsi_low.value(), "overbought": self.rsi_high.value()}
        if self.cb_macd.isChecked(): cp["indicators"]["macd"] = {"enabled": True, "fast": self.macd_fast.value(), "slow": self.macd_slow.value(), "signal": self.macd_signal.value()}
        if self.cb_bbands.isChecked(): cp["indicators"]["bollinger"] = {"enabled": True, "window": self.bb_window.value(), "std": self.bb_std.value()}
        if self.cb_stoch.isChecked(): cp["indicators"]["stochastic"] = {"enabled": True, "window": self.stoch_window.value(), "oversold": self.stoch_low.value(), "overbought": self.stoch_high.value()}
        
        prompt = f"""Professional crypto prediction system optimization request.

CURRENT SETUP:
```json
{json.dumps(cp, indent=2)}
```

SYSTEM DETAILS:
- Asset: {asset} over {tf}
- ML Model: {model} (ensemble method with cross-validation)
- Features: Technical indicators + engineered features (returns, volatility, momentum)
- Weighted voting system with confidence scoring
- Risk metrics: Sharpe, drawdown, ATR-based stops

OPTIMIZATION REQUEST:
1. Optimize parameters for {asset}'s volatility profile
2. Consider {tf} lookback period characteristics
3. Balance sensitivity vs noise reduction
4. Suggest indicator weights for voting

CONSTRAINTS:
- Integer values only (no decimals)
- SMA/BB windows: 3-200
- RSI window: 3-50, thresholds: 0-100
- MACD fast: 3-50, slow: 5-100, signal: 3-20
- BB std dev: 1-5
- Stochastic window: 5-50, thresholds: 0-100

Return optimized JSON in same format for direct import."""
        
        te = QTextEdit()
        te.setPlainText(prompt)
        te.setReadOnly(True)
        lo.addWidget(te)
        
        cb = QPushButton("Copy to Clipboard")
        cb.clicked.connect(lambda: (QApplication.clipboard().setText(prompt), QMessageBox.information(d, "Copied", "Prompt copied! If you want a more custom prompt, \nyou can make manual changes.")))
        lo.addWidget(cb)
        
        clb = QPushButton("Close")
        clb.clicked.connect(d.accept)
        lo.addWidget(clb)
        
        d.setLayout(lo)
        d.exec_()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CryptoPredictionLab()
    window.show()
    sys.exit(app.exec_())