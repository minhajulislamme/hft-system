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
        SUPPORT_RESISTANCE_STRENGTH,
        # Enhanced configurable thresholds
        RESISTANCE_BREAKOUT_THRESHOLD,
        RESISTANCE_REJECTION_THRESHOLD,
        RESISTANCE_TOUCH_THRESHOLD,
        SUPPORT_BREAKDOWN_THRESHOLD,
        SUPPORT_REJECTION_THRESHOLD,
        SUPPORT_TOUCH_THRESHOLD,
        MOMENTUM_POSITIVE_THRESHOLD,
        MOMENTUM_NEGATIVE_THRESHOLD,
        MOMENTUM_CONFIRMATION_THRESHOLD,
        VOLATILITY_MULTIPLIER,
        VOLUME_RATIO_THRESHOLD,
        BODY_RATIO_THRESHOLD,
        PRICE_POSITION_HIGH,
        PRICE_POSITION_LOW,
        TOUCH_TOLERANCE
    )
except ImportError:
    # Fallback values for pure price action strategies
    PRICE_ACTION_LOOKBACK = 20
    BREAKOUT_THRESHOLD = 0.02  # 2% breakout threshold
    VOLATILITY_WINDOW = 14
    MOMENTUM_WINDOW = 10
    SUPPORT_RESISTANCE_STRENGTH = 3
    # Enhanced configurable thresholds fallbacks
    RESISTANCE_BREAKOUT_THRESHOLD = 1.002
    RESISTANCE_REJECTION_THRESHOLD = 0.995
    RESISTANCE_TOUCH_THRESHOLD = 0.998
    SUPPORT_BREAKDOWN_THRESHOLD = 0.998
    SUPPORT_REJECTION_THRESHOLD = 1.005
    SUPPORT_TOUCH_THRESHOLD = 1.002
    MOMENTUM_POSITIVE_THRESHOLD = 0.005
    MOMENTUM_NEGATIVE_THRESHOLD = -0.005
    MOMENTUM_CONFIRMATION_THRESHOLD = 0.003
    VOLATILITY_MULTIPLIER = 1.5
    VOLUME_RATIO_THRESHOLD = 1.5
    BODY_RATIO_THRESHOLD = 0.7
    PRICE_POSITION_HIGH = 0.8
    PRICE_POSITION_LOW = 0.2
    TOUCH_TOLERANCE = 0.005


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
    
    # Class constants for configurable thresholds (loaded from config)
    RESISTANCE_BREAKOUT_THRESHOLD = RESISTANCE_BREAKOUT_THRESHOLD  # From config
    RESISTANCE_REJECTION_THRESHOLD = RESISTANCE_REJECTION_THRESHOLD  # From config
    RESISTANCE_TOUCH_THRESHOLD = RESISTANCE_TOUCH_THRESHOLD  # From config
    
    SUPPORT_BREAKDOWN_THRESHOLD = SUPPORT_BREAKDOWN_THRESHOLD  # From config
    SUPPORT_REJECTION_THRESHOLD = SUPPORT_REJECTION_THRESHOLD  # From config
    SUPPORT_TOUCH_THRESHOLD = SUPPORT_TOUCH_THRESHOLD  # From config
    
    MOMENTUM_POSITIVE_THRESHOLD = MOMENTUM_POSITIVE_THRESHOLD  # From config
    MOMENTUM_NEGATIVE_THRESHOLD = MOMENTUM_NEGATIVE_THRESHOLD  # From config
    MOMENTUM_CONFIRMATION_THRESHOLD = MOMENTUM_CONFIRMATION_THRESHOLD  # From config
    
    VOLATILITY_MULTIPLIER = VOLATILITY_MULTIPLIER  # From config
    VOLUME_RATIO_THRESHOLD = VOLUME_RATIO_THRESHOLD  # From config
    BODY_RATIO_THRESHOLD = BODY_RATIO_THRESHOLD  # From config
    
    PRICE_POSITION_HIGH = PRICE_POSITION_HIGH  # From config
    PRICE_POSITION_LOW = PRICE_POSITION_LOW  # From config
    
    TOUCH_TOLERANCE = TOUCH_TOLERANCE  # From config
    
    def __init__(self, 
                 lookback_period=20,        # Lookback period for analysis
                 breakout_threshold=0.02,   # 2% breakout threshold
                 volatility_window=14,      # Volatility calculation window
                 momentum_window=10,        # Momentum calculation window
                 sr_strength=3):            # Support/resistance strength
        
        super().__init__("PurePriceActionStrategy")
        
        # Enhanced parameter validation with configuration checks
        self._validate_parameters(lookback_period, breakout_threshold, volatility_window, 
                                momentum_window, sr_strength)
        
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
    
    def _validate_parameters(self, lookback_period, breakout_threshold, volatility_window, 
                           momentum_window, sr_strength):
        """Enhanced parameter validation with detailed error messages"""
        if not isinstance(lookback_period, int) or lookback_period <= 0:
            raise ValueError(f"Lookback period must be a positive integer, got {lookback_period}")
        if not isinstance(breakout_threshold, (int, float)) or breakout_threshold <= 0:
            raise ValueError(f"Breakout threshold must be positive, got {breakout_threshold}")
        if not isinstance(volatility_window, int) or volatility_window <= 0:
            raise ValueError(f"Volatility window must be a positive integer, got {volatility_window}")
        if not isinstance(momentum_window, int) or momentum_window <= 0:
            raise ValueError(f"Momentum window must be a positive integer, got {momentum_window}")
        if not isinstance(sr_strength, int) or sr_strength <= 0:
            raise ValueError(f"Support/resistance strength must be a positive integer, got {sr_strength}")
        
        # Logical relationship checks
        if lookback_period < max(volatility_window, momentum_window):
            logger.warning(f"Lookback period ({lookback_period}) is less than max window size "
                          f"({max(volatility_window, momentum_window)}). This may cause issues.")
        
        if breakout_threshold > 0.1:  # 10%
            logger.warning(f"Breakout threshold ({breakout_threshold*100}%) seems very high.")
        
        if sr_strength > lookback_period // 4:
            logger.warning(f"Support/resistance strength ({sr_strength}) is high relative to "
                          f"lookback period ({lookback_period}).")
    
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
            
            # Enhanced Support/Resistance Analysis Indicators
            
            # Calculate more precise support/resistance levels using pivots
            df['pivot_high'] = df['high'].rolling(window=5, center=True).max() == df['high']
            df['pivot_low'] = df['low'].rolling(window=5, center=True).min() == df['low']
            
            # Dynamic support/resistance based on recent price action
            df['dynamic_resistance'] = df['high'].rolling(window=self.lookback_period//2).max()
            df['dynamic_support'] = df['low'].rolling(window=self.lookback_period//2).min()
            
            # Vectorized Support/Resistance strength calculation
            df['resistance_touches'] = self._calculate_resistance_touches_vectorized(df)
            df['support_touches'] = self._calculate_support_touches_vectorized(df)
            
            # Rejection detection indicators
            df['at_resistance'] = (df['high'] >= df['resistance'] * 0.998) & (df['high'] <= df['resistance'] * 1.002)
            df['at_support'] = (df['low'] >= df['support'] * 0.998) & (df['low'] <= df['support'] * 1.002)
            
            # Breakout confirmation indicators
            df['resistance_broken'] = df['close'] > df['resistance'] * 1.002
            df['support_broken'] = df['close'] < df['support'] * 0.998
            
            # Previous candle analysis for rejection patterns
            df['prev_close'] = df['close'].shift(1)
            df['prev_high'] = df['high'].shift(1)
            df['prev_low'] = df['low'].shift(1)
            
            # Generate signals based on pure price action
            self._generate_price_action_signals(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding price action calculations: {e}")
            return df
    
    def _calculate_resistance_touches_vectorized(self, df):
        """Vectorized calculation of resistance touches for better performance"""
        touches = pd.Series(0, index=df.index)
        
        for i in range(self.lookback_period, len(df)):
            resistance_level = df.iloc[i]['resistance']
            if pd.isna(resistance_level) or resistance_level <= 0:
                continue
            
            # Vectorized calculation of touches in lookback window
            start_idx = max(0, i - self.lookback_period)
            recent_highs = df['high'].iloc[start_idx:i]
            
            # Count touches within tolerance
            tolerance = resistance_level * self.TOUCH_TOLERANCE
            touches.iloc[i] = ((recent_highs >= resistance_level - tolerance) & 
                              (recent_highs <= resistance_level + tolerance)).sum()
        
        return touches
    
    def _calculate_support_touches_vectorized(self, df):
        """Vectorized calculation of support touches for better performance"""
        touches = pd.Series(0, index=df.index)
        
        for i in range(self.lookback_period, len(df)):
            support_level = df.iloc[i]['support']
            if pd.isna(support_level) or support_level <= 0:
                continue
            
            # Vectorized calculation of touches in lookback window
            start_idx = max(0, i - self.lookback_period)
            recent_lows = df['low'].iloc[start_idx:i]
            
            # Count touches within tolerance
            tolerance = support_level * self.TOUCH_TOLERANCE
            touches.iloc[i] = ((recent_lows >= support_level - tolerance) & 
                              (recent_lows <= support_level + tolerance)).sum()
        
        return touches
    
    def _generate_price_action_signals(self, df):
        """Generate signals based on pure price action analysis"""
        try:
            # Initialize signal columns
            df['buy_signal'] = False
            df['sell_signal'] = False
            df['hold_signal'] = True
            
            for i in range(max(self.lookback_period, self.momentum_window), len(df)):
                current_row = df.iloc[i]
                
                # Streamlined critical values check (removed redundant None checks)
                critical_values = [
                    current_row.get('price_momentum', 0),
                    current_row.get('close', 0),
                    current_row.get('resistance', 0),
                    current_row.get('support', 0),
                    current_row.get('price_position', 0.5),
                    current_row.get('volatility', 0)
                ]
                
                if any(pd.isna(val) for val in critical_values):
                    continue
                
                buy_score = 0
                sell_score = 0
                
                # Momentum analysis using class constants
                momentum = current_row.get('price_momentum', 0)
                if not pd.isna(momentum):
                    if momentum > self.MOMENTUM_POSITIVE_THRESHOLD:
                        buy_score += 2
                    elif momentum < self.MOMENTUM_NEGATIVE_THRESHOLD:
                        sell_score += 2
                
                # Enhanced Support/Resistance Analysis with flexible thresholds
                close_price = current_row.get('close', 0)
                resistance = current_row.get('resistance', 0)
                support = current_row.get('support', 0)
                
                # Get previous candle data for rejection/breakout analysis
                if i > 0:
                    prev_row = df.iloc[i-1]
                    prev_close = prev_row.get('close', 0)
                    current_low = current_row.get('low', 0)
                    current_high = current_row.get('high', 0)
                    
                    # RESISTANCE LEVEL ANALYSIS with configurable thresholds
                    if not pd.isna(close_price) and not pd.isna(resistance) and resistance > 0:
                        
                        # 1. RESISTANCE BREAKOUT → BUY Signal (Strong Bullish)
                        if close_price > resistance * self.RESISTANCE_BREAKOUT_THRESHOLD:
                            buy_score += 4  # Strong bullish breakout
                            logger.debug(f"Resistance breakout detected: {close_price:.6f} > {resistance:.6f}")
                        
                        # 2. RESISTANCE REJECTION → SELL Signal (More flexible logic)
                        elif (not pd.isna(current_high) and 
                              current_high >= resistance * self.RESISTANCE_TOUCH_THRESHOLD and
                              close_price < resistance * self.RESISTANCE_REJECTION_THRESHOLD):
                            sell_score += 3  # Bearish rejection at resistance
                            logger.debug(f"Resistance rejection detected: touched {resistance:.6f}, closed at {close_price:.6f}")
                    
                    # SUPPORT LEVEL ANALYSIS with configurable thresholds
                    if not pd.isna(close_price) and not pd.isna(support) and support > 0:
                        
                        # 3. SUPPORT BREAKDOWN → SELL Signal (Strong Bearish)
                        if close_price < support * self.SUPPORT_BREAKDOWN_THRESHOLD:
                            sell_score += 4  # Strong bearish breakdown
                            logger.debug(f"Support breakdown detected: {close_price:.6f} < {support:.6f}")
                        
                        # 4. SUPPORT REJECTION → BUY Signal (More flexible logic)
                        elif (not pd.isna(current_low) and 
                              current_low <= support * self.SUPPORT_TOUCH_THRESHOLD and
                              close_price > support * self.SUPPORT_REJECTION_THRESHOLD):
                            buy_score += 3  # Bullish bounce from support
                            logger.debug(f"Support rejection detected: touched {support:.6f}, closed at {close_price:.6f}")
                    
                    # Additional confirmation for breakouts with momentum
                    if not pd.isna(momentum):
                        # Confirm resistance breakout with positive momentum
                        if (close_price > resistance * (self.RESISTANCE_BREAKOUT_THRESHOLD - 0.001) and 
                            momentum > self.MOMENTUM_CONFIRMATION_THRESHOLD):
                            buy_score += 1  # Momentum confirmation
                        
                        # Confirm support breakdown with negative momentum  
                        if (close_price < support * (self.SUPPORT_BREAKDOWN_THRESHOLD + 0.001) and 
                            momentum < -self.MOMENTUM_CONFIRMATION_THRESHOLD):
                            sell_score += 1  # Momentum confirmation
                
                # Price position analysis using class constants
                price_position = current_row.get('price_position', 0.5)
                if not pd.isna(price_position) and not pd.isna(momentum):
                    if price_position > self.PRICE_POSITION_HIGH:  # Near top of range
                        if momentum > 0:
                            buy_score += 1  # Continued strength
                        else:
                            sell_score += 1  # Potential reversal
                    elif price_position < self.PRICE_POSITION_LOW:  # Near bottom of range
                        if momentum < 0:
                            sell_score += 1  # Continued weakness
                        else:
                            buy_score += 1  # Potential reversal
                
                # Volatility analysis with class constants
                volatility = current_row.get('volatility', 0)
                if not pd.isna(volatility) and volatility > 0:
                    # Calculate mean volatility safely
                    vol_slice = df['volatility'].iloc[max(0, i-20):i]
                    vol_mean = vol_slice.mean() if len(vol_slice) > 0 else volatility
                    
                    if not pd.isna(vol_mean) and vol_mean > 0:
                        if volatility > vol_mean * self.VOLATILITY_MULTIPLIER:
                            # High volatility - add to existing momentum
                            if not pd.isna(momentum):
                                if momentum > 0:
                                    buy_score += 1
                                elif momentum < 0:
                                    sell_score += 1
                
                # Volume confirmation using class constants
                if 'volume_ratio' in df.columns:
                    volume_ratio = current_row.get('volume_ratio', 1.0)
                    if not pd.isna(volume_ratio) and volume_ratio > self.VOLUME_RATIO_THRESHOLD:
                        if not pd.isna(momentum):
                            if momentum > 0:
                                buy_score += 1
                            elif momentum < 0:
                                sell_score += 1
                
                # Candlestick analysis using class constants
                body_ratio = current_row.get('body_ratio', 0.5)
                open_price = current_row.get('open', 0)
                
                if (not pd.isna(body_ratio) and not pd.isna(open_price) and 
                    not pd.isna(close_price) and body_ratio > self.BODY_RATIO_THRESHOLD):
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
                support = latest.get('support', 0)
                price_position = latest.get('price_position', 0.5)
                
                if not pd.isna(close_price):
                    logger.info(f"   Current Price: {close_price:.6f}")
                if not pd.isna(resistance):
                    logger.info(f"   Resistance Level: {resistance:.6f}")
                if not pd.isna(support):
                    logger.info(f"   Support Level: {support:.6f}")
                if not pd.isna(price_position):
                    logger.info(f"   Price Position in Range: {price_position*100:.1f}%")
                
                # Check for specific buy reasons
                if not pd.isna(close_price) and not pd.isna(resistance) and resistance > 0:
                    if close_price > resistance * self.RESISTANCE_BREAKOUT_THRESHOLD:
                        logger.info(f"   🚀 RESISTANCE BREAKOUT: Price broke above {resistance:.6f}")
                
                if not pd.isna(close_price) and not pd.isna(support) and support > 0:
                    current_low = latest.get('low', 0)
                    if (not pd.isna(current_low) and current_low <= support * self.SUPPORT_TOUCH_THRESHOLD and 
                        close_price > support * self.SUPPORT_REJECTION_THRESHOLD):
                        logger.info(f"   💪 SUPPORT REJECTION: Price bounced from {support:.6f}")
                
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
                resistance = latest.get('resistance', 0)
                price_position = latest.get('price_position', 0.5)
                
                if not pd.isna(close_price):
                    logger.info(f"   Current Price: {close_price:.6f}")
                if not pd.isna(support):
                    logger.info(f"   Support Level: {support:.6f}")
                if not pd.isna(resistance):
                    logger.info(f"   Resistance Level: {resistance:.6f}")
                if not pd.isna(price_position):
                    logger.info(f"   Price Position in Range: {price_position*100:.1f}%")
                
                # Check for specific sell reasons
                if not pd.isna(close_price) and not pd.isna(support) and support > 0:
                    if close_price < support * self.SUPPORT_BREAKDOWN_THRESHOLD:
                        logger.info(f"   📉 SUPPORT BREAKDOWN: Price broke below {support:.6f}")
                
                if not pd.isna(close_price) and not pd.isna(resistance) and resistance > 0:
                    current_high = latest.get('high', 0)
                    if (not pd.isna(current_high) and current_high >= resistance * self.RESISTANCE_TOUCH_THRESHOLD and 
                        close_price < resistance * self.RESISTANCE_REJECTION_THRESHOLD):
                        logger.info(f"   � RESISTANCE REJECTION: Price rejected at {resistance:.6f}")
                
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
    
    def analyze_support_resistance_action(self, df, index):
        """
        Analyze support/resistance price action for the current candle
        Returns a dictionary with analysis results
        """
        if index < 1 or index >= len(df):
            return {}
        
        current = df.iloc[index]
        previous = df.iloc[index-1]
        
        analysis = {
            'resistance_breakout': False,
            'resistance_rejection': False,
            'support_breakdown': False,
            'support_rejection': False,
            'action_type': 'none',
            'strength': 0
        }
        
        close_price = current.get('close', 0)
        high_price = current.get('high', 0)
        low_price = current.get('low', 0)
        resistance = current.get('resistance', 0)
        support = current.get('support', 0)
        prev_close = previous.get('close', 0)
        
        # Skip if any critical values are invalid
        if any(pd.isna(val) or val <= 0 for val in [close_price, resistance, support]):
            return analysis
        
        # Resistance Analysis using class constants
        if resistance > 0:
            # Resistance Breakout (Close above resistance)
            if close_price > resistance * self.RESISTANCE_BREAKOUT_THRESHOLD:
                analysis['resistance_breakout'] = True
                analysis['action_type'] = 'resistance_breakout'
                analysis['strength'] = min(5, int((close_price - resistance) / resistance * 500))
            
            # Resistance Rejection (Touched resistance but closed below) - More flexible
            elif (high_price >= resistance * self.RESISTANCE_TOUCH_THRESHOLD and
                  close_price < resistance * self.RESISTANCE_REJECTION_THRESHOLD):
                analysis['resistance_rejection'] = True
                analysis['action_type'] = 'resistance_rejection'
                analysis['strength'] = min(5, int((resistance - close_price) / resistance * 500))
        
        # Support Analysis using class constants
        if support > 0:
            # Support Breakdown (Close below support)
            if close_price < support * self.SUPPORT_BREAKDOWN_THRESHOLD:
                analysis['support_breakdown'] = True
                analysis['action_type'] = 'support_breakdown'
                analysis['strength'] = min(5, int((support - close_price) / support * 500))
            
            # Support Rejection (Touched support but closed above) - More flexible
            elif (low_price <= support * self.SUPPORT_TOUCH_THRESHOLD and
                  close_price > support * self.SUPPORT_REJECTION_THRESHOLD):
                analysis['support_rejection'] = True
                analysis['action_type'] = 'support_rejection'
                analysis['strength'] = min(5, int((close_price - support) / support * 500))
        
        return analysis


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