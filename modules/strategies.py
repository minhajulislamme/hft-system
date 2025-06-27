from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
import logging
import warnings
import traceback
import math
from collections import deque

# Setup logging
logger = logging.getLogger(__name__)
warnings.simplefilter(action='ignore', category=FutureWarning)

# Import price action configuration values
try:
    from modules.config import (
        PRICE_ACTION_LOOKBACK,
        BREAKOUT_THRESHOLD,
        VOLATILITY_WINDOW,
        MOMENTUM_WINDOW,
        SUPPORT_RESISTANCE_STRENGTH
    )
except ImportError:
    # Fallback values for pure price action strategies
    PRICE_ACTION_LOOKBACK = 20
    BREAKOUT_THRESHOLD = 0.02  # 2% breakout threshold
    VOLATILITY_WINDOW = 14
    MOMENTUM_WINDOW = 10
    SUPPORT_RESISTANCE_STRENGTH = 3


class TradingStrategy:
    """Base trading strategy class for pure price action strategies"""
    
    def __init__(self, name="BaseStrategy"):
        self.name = name
        self.risk_manager = None
        self.last_signal = None
        self.signal_history = deque(maxlen=100)  # Keep last 100 signals for analysis
    
    @property
    def strategy_name(self):
        """Property to access strategy name (for compatibility)"""
        return self.name
        
    def set_risk_manager(self, risk_manager):
        """Set the risk manager for this strategy"""
        self.risk_manager = risk_manager
        
    def get_signal(self, klines):
        """Get trading signal from klines data. Override in subclasses."""
        return None
        
    def add_indicators(self, df):
        """Add mathematical price action calculations to dataframe. Override in subclasses."""
        return df
    
    def calculate_price_momentum(self, prices, window=10):
        """Calculate pure price momentum without indicators"""
        if len(prices) < window + 1:
            return 0
        
        current_price = prices[-1]
        past_price = prices[-window-1]
        
        # Prevent division by zero
        if past_price == 0 or past_price is None:
            return 0
        
        momentum = (current_price - past_price) / past_price
        return momentum
    
    def calculate_volatility(self, prices, window=14):
        """Calculate price volatility using standard deviation"""
        if len(prices) < window:
            return 0
        
        recent_prices = prices[-window:]
        returns = []
        
        for i in range(1, len(recent_prices)):
            prev_price = recent_prices[i-1]
            curr_price = recent_prices[i]
            
            # Prevent division by zero
            if prev_price == 0 or prev_price is None:
                continue
                
            return_val = (curr_price - prev_price) / prev_price
            returns.append(return_val)
        
        if not returns:
            return 0
        
        volatility = np.std(returns)
        return volatility
    
    def find_support_resistance(self, highs, lows, strength=3):
        """Find support and resistance levels using price action"""
        if len(highs) < strength * 2 + 1 or len(lows) < strength * 2 + 1:
            return [], []
        
        resistance_levels = []
        support_levels = []
        
        # Find resistance levels (local highs)
        for i in range(strength, len(highs) - strength):
            is_resistance = True
            for j in range(i - strength, i + strength + 1):
                if j != i and highs[j] >= highs[i]:
                    is_resistance = False
                    break
            if is_resistance:
                resistance_levels.append(highs[i])
        
        # Find support levels (local lows)
        for i in range(strength, len(lows) - strength):
            is_support = True
            for j in range(i - strength, i + strength + 1):
                if j != i and lows[j] <= lows[i]:
                    is_support = False
                    break
            if is_support:
                support_levels.append(lows[i])
        
        return resistance_levels, support_levels
    
    def detect_candlestick_patterns(self, ohlc_data):
        """Detect basic candlestick patterns using pure price action"""
        if len(ohlc_data) < 2:
            return None
        
        try:
            current = ohlc_data[-1]
            prev = ohlc_data[-2] if len(ohlc_data) >= 2 else current
            
            # Validate OHLC data
            required_keys = ['open', 'high', 'low', 'close']
            for key in required_keys:
                if key not in current or current[key] is None or current[key] <= 0:
                    return None
                if key not in prev or prev[key] is None or prev[key] <= 0:
                    return None
            
            o, h, l, c = current['open'], current['high'], current['low'], current['close']
            prev_o, prev_h, prev_l, prev_c = prev['open'], prev['high'], prev['low'], prev['close']
            
            # Validate OHLC relationships
            if not (l <= min(o, c) <= max(o, c) <= h):
                return None
            if not (prev_l <= min(prev_o, prev_c) <= max(prev_o, prev_c) <= prev_h):
                return None
            
            body_size = abs(c - o)
            prev_body_size = abs(prev_c - prev_o)
            total_range = h - l
            
            # Avoid patterns on very small ranges
            if total_range == 0:
                return None
            
            # Bullish patterns
            if c > o:  # Green candle
                # Hammer pattern (need meaningful lower shadow)
                lower_shadow = min(o, c) - l
                upper_shadow = h - max(o, c)
                
                if (body_size > 0 and lower_shadow > 2 * body_size and 
                    upper_shadow < body_size * 0.2):
                    return "BULLISH_HAMMER"
                
                # Bullish engulfing (need meaningful previous body)
                if (prev_body_size > 0 and prev_c < prev_o and 
                    c > prev_o and o < prev_c and body_size > prev_body_size):
                    return "BULLISH_ENGULFING"
            
            # Bearish patterns
            elif c < o:  # Red candle
                # Hanging man pattern
                lower_shadow = min(o, c) - l
                upper_shadow = h - max(o, c)
                
                if (body_size > 0 and lower_shadow > 2 * body_size and 
                    upper_shadow < body_size * 0.2):
                    return "BEARISH_HANGING_MAN"
                
                # Bearish engulfing
                if (prev_body_size > 0 and prev_c > prev_o and 
                    c < prev_o and o > prev_c and body_size > prev_body_size):
                    return "BEARISH_ENGULFING"
            
            # Doji pattern (very small body relative to range)
            if total_range > 0 and body_size < total_range * 0.1:
                return "DOJI"
            
            return None
            
        except (KeyError, TypeError, ValueError) as e:
            logger.warning(f"Error in candlestick pattern detection: {e}")
            return None


class PurePriceActionStrategy(TradingStrategy):
    """
    Pure Price Action Strategy:
    
    Core Strategy:
    - Uses only price movements, support/resistance, and mathematical analysis
    - No traditional indicators - pure price action and statistical analysis
    - Combines multiple price action signals for confirmation
    
    Mathematical Components:
    - Price momentum calculation using percentage change
    - Volatility measurement using standard deviation of returns
    - Support/resistance level detection using local highs/lows
    - Candlestick pattern recognition
    - Breakout detection using price thresholds
    
    Signal Generation Logic:
    - BUY: Bullish breakout above resistance + positive momentum + bullish patterns
    - SELL: Bearish breakdown below support + negative momentum + bearish patterns  
    - HOLD: Price within support/resistance range or conflicting signals

    Benefits of Pure Price Action Strategy:
    - No lagging indicators - responds immediately to price changes
    - Works in all market conditions (trending, ranging, volatile)
    - Mathematical approach reduces emotional bias
    - Focuses on what actually moves the market: price and volume
    """
    
    def __init__(self, 
                 lookback_period=20,        # Lookback period for analysis
                 breakout_threshold=0.02,   # 2% breakout threshold
                 volatility_window=14,      # Volatility calculation window
                 momentum_window=10,        # Momentum calculation window
                 sr_strength=3):            # Support/resistance strength
        
        super().__init__("PurePriceActionStrategy")
        
        # Parameter validation
        if lookback_period <= 0:
            raise ValueError("Lookback period must be positive")
        if breakout_threshold <= 0:
            raise ValueError("Breakout threshold must be positive") 
        if volatility_window <= 0:
            raise ValueError("Volatility window must be positive")
        if momentum_window <= 0:
            raise ValueError("Momentum window must be positive")
        if sr_strength <= 0:
            raise ValueError("Support/resistance strength must be positive")
        
        # Store parameters
        self.lookback_period = lookback_period
        self.breakout_threshold = breakout_threshold
        self.volatility_window = volatility_window
        self.momentum_window = momentum_window
        self.sr_strength = sr_strength
        self._warning_count = 0
        
        logger.info(f"{self.name} initialized with:")
        logger.info(f"  Lookback Period: {lookback_period} candles")
        logger.info(f"  Breakout Threshold: {breakout_threshold*100}%")
        logger.info(f"  Volatility Window: {volatility_window} periods")
        logger.info(f"  Momentum Window: {momentum_window} periods")
        logger.info(f"  Support/Resistance Strength: {sr_strength}")
    
    def add_indicators(self, df):
        """Add pure price action calculations"""
        try:
            # Ensure sufficient data
            min_required = max(self.lookback_period, self.volatility_window, self.momentum_window) + 5
            if len(df) < min_required:
                logger.warning(f"Insufficient data: need {min_required}, got {len(df)}")
                return df
            
            # Data cleaning
            if df['close'].isna().any():
                logger.warning("Found NaN values in close prices, cleaning data")
                df['close'] = df['close'].interpolate(method='linear').bfill().ffill()
                
            # Ensure positive prices
            price_cols = ['open', 'high', 'low', 'close']
            for col in price_cols:
                if (df[col] <= 0).any():
                    logger.warning(f"Found zero or negative values in {col}, using interpolation")
                    df[col] = df[col].replace(0, np.nan)
                    df[col] = df[col].interpolate(method='linear').bfill().ffill()
            
            # Calculate price momentum
            df['price_momentum'] = df['close'].pct_change(periods=self.momentum_window)
            
            # Calculate volatility using rolling standard deviation
            df['returns'] = df['close'].pct_change()
            df['volatility'] = df['returns'].rolling(window=self.volatility_window).std()
            
            # Calculate price range
            df['price_range'] = df['high'] - df['low']
            df['avg_range'] = df['price_range'].rolling(window=self.lookback_period).mean()
            
            # Calculate support and resistance levels
            df['resistance'] = df['high'].rolling(window=self.lookback_period).max()
            df['support'] = df['low'].rolling(window=self.lookback_period).min()
            
            # Distance from support/resistance (with division by zero protection)
            df['dist_from_resistance'] = np.where(
                df['close'] != 0, 
                (df['resistance'] - df['close']) / df['close'], 
                0
            )
            df['dist_from_support'] = np.where(
                df['close'] != 0,
                (df['close'] - df['support']) / df['close'],
                0
            )
            
            # Breakout detection
            df['near_resistance'] = df['dist_from_resistance'] < self.breakout_threshold
            df['near_support'] = df['dist_from_support'] < self.breakout_threshold
            
            # Price position within range (with division by zero protection)
            range_size = df['resistance'] - df['support']
            df['price_position'] = np.where(
                range_size != 0,
                (df['close'] - df['support']) / range_size,
                0.5  # Default to middle position if no range
            )
            
            # Volume-price analysis (if volume available)
            if 'volume' in df.columns:
                df['avg_volume'] = df['volume'].rolling(window=self.lookback_period).mean()
                # Protect against division by zero for volume ratio
                df['volume_ratio'] = np.where(
                    df['avg_volume'] != 0,
                    df['volume'] / df['avg_volume'],
                    1.0  # Default to 1.0 if no average volume
                )
                df['price_volume_momentum'] = df['price_momentum'] * df['volume_ratio']
            
            # Candlestick body analysis
            df['body_size'] = abs(df['close'] - df['open'])
            df['upper_shadow'] = df['high'] - np.maximum(df['open'], df['close'])
            df['lower_shadow'] = np.minimum(df['open'], df['close']) - df['low']
            df['total_range'] = df['high'] - df['low']
            
            # Relative body size (with division by zero protection)
            df['body_ratio'] = np.where(
                df['total_range'] != 0,
                df['body_size'] / df['total_range'],
                0.5  # Default to 0.5 if no range
            )
            
            # Generate signals based on pure price action
            self._generate_price_action_signals(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding price action calculations: {e}")
            return df
    
    def _generate_price_action_signals(self, df):
        """Generate signals based on pure price action analysis"""
        try:
            # Initialize signal columns
            df['buy_signal'] = False
            df['sell_signal'] = False
            df['hold_signal'] = True
            
            for i in range(max(self.lookback_period, self.momentum_window), len(df)):
                current_row = df.iloc[i]
                
                # Skip if any critical values are NaN or invalid
                critical_values = [
                    current_row.get('price_momentum', 0),
                    current_row.get('close', 0),
                    current_row.get('resistance', 0),
                    current_row.get('support', 0),
                    current_row.get('price_position', 0.5),
                    current_row.get('volatility', 0)
                ]
                
                if any(pd.isna(val) or val is None for val in critical_values):
                    continue
                
                buy_score = 0
                sell_score = 0
                
                # Momentum analysis (with NaN protection)
                momentum = current_row.get('price_momentum', 0)
                if not pd.isna(momentum):
                    if momentum > 0.005:  # Positive momentum > 0.5%
                        buy_score += 2
                    elif momentum < -0.005:  # Negative momentum < -0.5%
                        sell_score += 2
                
                # Breakout analysis (with safety checks)
                near_resistance = current_row.get('near_resistance', False)
                near_support = current_row.get('near_support', False)
                close_price = current_row.get('close', 0)
                resistance = current_row.get('resistance', 0)
                support = current_row.get('support', 0)
                
                if not pd.isna(close_price) and not pd.isna(resistance) and resistance > 0:
                    if near_resistance and close_price > resistance * 0.999:
                        buy_score += 3  # Strong bullish breakout
                        
                if not pd.isna(close_price) and not pd.isna(support) and support > 0:
                    if near_support and close_price < support * 1.001:
                        sell_score += 3  # Strong bearish breakdown
                
                # Price position analysis (with safety checks)
                price_position = current_row.get('price_position', 0.5)
                if not pd.isna(price_position) and not pd.isna(momentum):
                    if price_position > 0.8:  # Near top of range
                        if momentum > 0:
                            buy_score += 1  # Continued strength
                        else:
                            sell_score += 1  # Potential reversal
                    elif price_position < 0.2:  # Near bottom of range
                        if momentum < 0:
                            sell_score += 1  # Continued weakness
                        else:
                            buy_score += 1  # Potential reversal
                
                # Volatility analysis (with safety checks)
                volatility = current_row.get('volatility', 0)
                if not pd.isna(volatility) and volatility > 0:
                    # Calculate mean volatility safely
                    vol_slice = df['volatility'].iloc[max(0, i-20):i]
                    vol_mean = vol_slice.mean() if len(vol_slice) > 0 else volatility
                    
                    if not pd.isna(vol_mean) and vol_mean > 0:
                        if volatility > vol_mean * 1.5:
                            # High volatility - add to existing momentum
                            if not pd.isna(momentum):
                                if momentum > 0:
                                    buy_score += 1
                                elif momentum < 0:
                                    sell_score += 1
                
                # Volume confirmation (if available, with safety checks)
                if 'volume_ratio' in df.columns:
                    volume_ratio = current_row.get('volume_ratio', 1.0)
                    if not pd.isna(volume_ratio) and volume_ratio > 1.5:  # Above average volume
                        if not pd.isna(momentum):
                            if momentum > 0:
                                buy_score += 1
                            elif momentum < 0:
                                sell_score += 1
                
                # Candlestick analysis (with safety checks)
                body_ratio = current_row.get('body_ratio', 0.5)
                open_price = current_row.get('open', 0)
                
                if (not pd.isna(body_ratio) and not pd.isna(open_price) and 
                    not pd.isna(close_price) and body_ratio > 0.7):  # Strong body (decisive move)
                    if close_price > open_price:  # Bullish candle
                        buy_score += 1
                    else:  # Bearish candle
                        sell_score += 1
                
                # Generate final signal
                if buy_score >= 4 and buy_score > sell_score:
                    df.at[i, 'buy_signal'] = True
                    df.at[i, 'hold_signal'] = False
                elif sell_score >= 4 and sell_score > buy_score:
                    df.at[i, 'sell_signal'] = True
                    df.at[i, 'hold_signal'] = False
                # Otherwise keep hold_signal = True (default)
            
        except Exception as e:
            logger.error(f"Error generating price action signals: {e}")
    
    def get_signal(self, klines):
        """Generate pure price action signals"""
        try:
            min_required = max(self.lookback_period, self.volatility_window, self.momentum_window) + 5
            if not klines or len(klines) < min_required:
                if self._warning_count % 10 == 0:
                    logger.warning(f"Insufficient data for price action signal (need {min_required}, have {len(klines) if klines else 0})")
                self._warning_count += 1
                return None
            
            # Convert and validate data
            df = pd.DataFrame(klines)
            if len(df.columns) != 12:
                logger.error(f"Invalid klines format: expected 12 columns, got {len(df.columns)}")
                return None
                
            df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 
                         'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore']
            
            # Data cleaning
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                if df[col].isna().any():
                    logger.warning(f"Cleaning NaN values in {col}")
                    df[col] = df[col].interpolate(method='linear').bfill().ffill()
            
            # Final validation after cleaning
            if df[numeric_columns].isna().any().any():
                logger.error("Failed to clean price data after interpolation")
                return None
            
            # Add price action calculations
            df = self.add_indicators(df)
            
            if len(df) < 2:
                return None
            
            latest = df.iloc[-1]
            
            # Validate required columns with more lenient checks
            required_columns = ['buy_signal', 'sell_signal', 'hold_signal']
            
            for col in required_columns:
                if col not in df.columns:
                    logger.warning(f"Missing required column: {col}")
                    return None
            
            # Check if we have valid signal data
            if pd.isna(latest.get('buy_signal')) or pd.isna(latest.get('sell_signal')):
                logger.warning("Invalid signal data - NaN values found")
                return None
            
            # Generate signal based on pure price action
            signal = None
            
            # BUY Signal: Strong bullish price action
            if latest.get('buy_signal', False):
                signal = 'BUY'
                logger.info(f"🟢 BUY Signal - Strong Bullish Price Action")
                momentum = latest.get('price_momentum', 0)
                if not pd.isna(momentum):
                    logger.info(f"   Price Momentum: {momentum*100:.2f}%")
                
                close_price = latest.get('close', 0)
                resistance = latest.get('resistance', 0)
                price_position = latest.get('price_position', 0.5)
                
                if not pd.isna(close_price):
                    logger.info(f"   Current Price: {close_price:.6f}")
                if not pd.isna(resistance):
                    logger.info(f"   Resistance Level: {resistance:.6f}")
                if not pd.isna(price_position):
                    logger.info(f"   Price Position in Range: {price_position*100:.1f}%")
                
                if 'volume_ratio' in df.columns:
                    volume_ratio = latest.get('volume_ratio', 1.0)
                    if not pd.isna(volume_ratio):
                        logger.info(f"   Volume Ratio: {volume_ratio:.2f}x average")
            
            # SELL Signal: Strong bearish price action
            elif latest.get('sell_signal', False):
                signal = 'SELL'
                logger.info(f"🔴 SELL Signal - Strong Bearish Price Action")
                momentum = latest.get('price_momentum', 0)
                if not pd.isna(momentum):
                    logger.info(f"   Price Momentum: {momentum*100:.2f}%")
                
                close_price = latest.get('close', 0)
                support = latest.get('support', 0)
                price_position = latest.get('price_position', 0.5)
                
                if not pd.isna(close_price):
                    logger.info(f"   Current Price: {close_price:.6f}")
                if not pd.isna(support):
                    logger.info(f"   Support Level: {support:.6f}")
                if not pd.isna(price_position):
                    logger.info(f"   Price Position in Range: {price_position*100:.1f}%")
                
                if 'volume_ratio' in df.columns:
                    volume_ratio = latest.get('volume_ratio', 1.0)
                    if not pd.isna(volume_ratio):
                        logger.info(f"   Volume Ratio: {volume_ratio:.2f}x average")
            
            # HOLD Signal: No clear price action signal
            else:
                signal = 'HOLD'
                logger.info(f"⚪ HOLD Signal - No Clear Price Action")
                momentum = latest.get('price_momentum', 0)
                if not pd.isna(momentum):
                    logger.info(f"   Price Momentum: {momentum*100:.2f}%")
                
                close_price = latest.get('close', 0)
                support = latest.get('support', 0)
                resistance = latest.get('resistance', 0)
                price_position = latest.get('price_position', 0.5)
                
                if not pd.isna(close_price):
                    logger.info(f"   Current Price: {close_price:.6f}")
                if not pd.isna(support) and not pd.isna(resistance):
                    logger.info(f"   Support: {support:.6f} | Resistance: {resistance:.6f}")
                if not pd.isna(price_position):
                    logger.info(f"   Price Position in Range: {price_position*100:.1f}%")
            
            # Store signal for history (with safe data)
            timestamp = latest.get('timestamp')
            close_price = latest.get('close', 0)
            momentum = latest.get('price_momentum', 0)
            
            if not pd.isna(timestamp) and not pd.isna(close_price) and not pd.isna(momentum):
                self.signal_history.append({
                    'timestamp': timestamp,
                    'signal': signal,
                    'price': close_price,
                    'momentum': momentum
                })
            
            self.last_signal = signal
            return signal
            
        except Exception as e:
            logger.error(f"Error in price action signal generation: {e}")
            logger.error(traceback.format_exc())
            return None


class MathematicalMomentumStrategy(TradingStrategy):
    """
    Mathematical Momentum Strategy:
    
    Pure mathematical approach using:
    - Rate of change calculations
    - Statistical momentum analysis
    - Price acceleration/deceleration
    - Fibonacci ratios in price movements
    - Standard deviation breakouts
    """
    
    def __init__(self, momentum_window=14, acceleration_window=7, fibonacci_levels=True):
        super().__init__("MathematicalMomentumStrategy")
        self.momentum_window = momentum_window
        self.acceleration_window = acceleration_window
        self.fibonacci_levels = fibonacci_levels
        
        logger.info(f"{self.name} initialized with mathematical analysis")
    
    def add_indicators(self, df):
        """Add mathematical momentum calculations"""
        try:
            # Rate of change
            df['roc'] = df['close'].pct_change(periods=self.momentum_window)
            
            # Price acceleration (change in momentum)
            df['momentum'] = df['close'].pct_change()
            df['acceleration'] = df['momentum'].diff()
            
            # Statistical measures (with division by zero protection)
            rolling_mean = df['close'].rolling(20).mean()
            rolling_std = df['close'].rolling(20).std()
            df['z_score'] = np.where(
                rolling_std != 0,
                (df['close'] - rolling_mean) / rolling_std,
                0  # Default to 0 if no standard deviation
            )
            
            # Fibonacci-based analysis
            if self.fibonacci_levels:
                swing_high = df['high'].rolling(20).max()
                swing_low = df['low'].rolling(20).min()
                range_size = swing_high - swing_low
                
                df['fib_618'] = swing_low + 0.618 * range_size
                df['fib_382'] = swing_low + 0.382 * range_size
            
            # Generate mathematical signals
            df['buy_signal'] = (df['roc'] > 0.01) & (df['acceleration'] > 0) & (df['z_score'] > 1)
            df['sell_signal'] = (df['roc'] < -0.01) & (df['acceleration'] < 0) & (df['z_score'] < -1)
            df['hold_signal'] = ~(df['buy_signal'] | df['sell_signal'])
            
            return df
            
        except Exception as e:
            logger.error(f"Error in mathematical calculations: {e}")
            return df
    
    def get_signal(self, klines):
        """Generate mathematical momentum signals"""
        try:
            if not klines or len(klines) < 30:
                return None
            
            df = pd.DataFrame(klines)
            df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 
                         'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore']
            
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df = self.add_indicators(df)
            latest = df.iloc[-1]
            
            if latest['buy_signal']:
                logger.info(f"🟢 MATHEMATICAL BUY - ROC: {latest['roc']*100:.2f}%, Z-Score: {latest['z_score']:.2f}")
                return 'BUY'
            elif latest['sell_signal']:
                logger.info(f"🔴 MATHEMATICAL SELL - ROC: {latest['roc']*100:.2f}%, Z-Score: {latest['z_score']:.2f}")
                return 'SELL'
            else:
                return 'HOLD'
                
        except Exception as e:
            logger.error(f"Error in mathematical momentum signal: {e}")
            return None


# Factory function to get a strategy by name
def get_strategy(strategy_name):
    """Factory function to get a strategy by name"""
    strategies = {
        'PurePriceActionStrategy': PurePriceActionStrategy(
            lookback_period=PRICE_ACTION_LOOKBACK,
            breakout_threshold=BREAKOUT_THRESHOLD,
            volatility_window=VOLATILITY_WINDOW,
            momentum_window=MOMENTUM_WINDOW,
            sr_strength=SUPPORT_RESISTANCE_STRENGTH
        ),
        'MathematicalMomentumStrategy': MathematicalMomentumStrategy(
            momentum_window=MOMENTUM_WINDOW,
            acceleration_window=7,
            fibonacci_levels=True
        ),
        # Keep compatibility with old name
        'SmartTrendCatcher': PurePriceActionStrategy(
            lookback_period=PRICE_ACTION_LOOKBACK,
            breakout_threshold=BREAKOUT_THRESHOLD,
            volatility_window=VOLATILITY_WINDOW,
            momentum_window=MOMENTUM_WINDOW,
            sr_strength=SUPPORT_RESISTANCE_STRENGTH
        ),
    }
    
    if strategy_name in strategies:
        return strategies[strategy_name]
    
    logger.warning(f"Strategy {strategy_name} not found. Defaulting to PurePriceActionStrategy.")
    return strategies['PurePriceActionStrategy']


def get_strategy_for_symbol(symbol, strategy_name=None):
    """Get the appropriate strategy based on the trading symbol"""
    # If a specific strategy is requested, use it
    if strategy_name:
        return get_strategy(strategy_name)
    
    # Default to PurePriceActionStrategy for any symbol
    return get_strategy('PurePriceActionStrategy')