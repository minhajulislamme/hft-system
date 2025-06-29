import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API configuration
API_KEY = os.getenv('BINANCE_API_KEY', '')
API_SECRET = os.getenv('BINANCE_API_SECRET', '')

# Testnet configuration
API_TESTNET = os.getenv('BINANCE_API_TESTNET', 'False').lower() == 'true'

# API URLs - Automatically determined based on testnet setting
if API_TESTNET:
    # Testnet URLs
    API_URL = 'https://testnet.binancefuture.com'
    WS_BASE_URL = 'wss://stream.binancefuture.com'
else:
    # Production URLs
    API_URL = os.getenv('BINANCE_API_URL', 'https://fapi.binance.com')
    WS_BASE_URL = 'wss://fstream.binance.com'

# API request settings
RECV_WINDOW = int(os.getenv('BINANCE_RECV_WINDOW', '10000'))

# Trading parameters
TRADING_SYMBOL = os.getenv('TRADING_SYMBOL', 'SOLUSDT')
TRADING_TYPE = 'FUTURES'  # Use futures trading
LEVERAGE = int(os.getenv('LEVERAGE', '20'))
MARGIN_TYPE = os.getenv('MARGIN_TYPE', 'CROSSED')  # ISOLATED or CROSSED
STRATEGY = os.getenv('STRATEGY', 'PurePriceActionStrategy')

# Position sizing - Enhanced risk management (aligned with SmartTrendCatcher)
INITIAL_BALANCE = float(os.getenv('INITIAL_BALANCE', '50.0'))
FIXED_TRADE_PERCENTAGE = float(os.getenv('FIXED_TRADE_PERCENTAGE', '0.20'))  # 40% to match strategy base_position_pct
MAX_OPEN_POSITIONS = int(os.getenv('MAX_OPEN_POSITIONS', '3'))  # Conservative for better risk management

# Margin safety settings - More conservative
MARGIN_SAFETY_FACTOR = float(os.getenv('MARGIN_SAFETY_FACTOR', '0.90'))  # Use at most 90% of available margin
MAX_POSITION_SIZE_PCT = float(os.getenv('MAX_POSITION_SIZE_PCT', '0.50'))  # Max 50% position size (matches strategy)
MIN_FREE_BALANCE_PCT = float(os.getenv('MIN_FREE_BALANCE_PCT', '0.10'))  # Keep at least 10% free

# Multi-instance configuration for running separate bot instances per trading pair
MULTI_INSTANCE_MODE = os.getenv('MULTI_INSTANCE_MODE', 'True').lower() == 'true'
MAX_POSITIONS_PER_SYMBOL = int(os.getenv('MAX_POSITIONS_PER_SYMBOL', '3'))  # Updated to match .env

# Auto-compounding settings - Enhanced with performance-based adjustments
AUTO_COMPOUND = os.getenv('AUTO_COMPOUND', 'True').lower() == 'true'
COMPOUND_REINVEST_PERCENT = float(os.getenv('COMPOUND_REINVEST_PERCENT', '0.75'))
COMPOUND_INTERVAL = os.getenv('COMPOUND_INTERVAL', 'DAILY')

# Dynamic compounding adjustments
COMPOUND_PERFORMANCE_WINDOW = int(os.getenv('COMPOUND_PERFORMANCE_WINDOW', '7'))  # Look back 7 days
COMPOUND_MIN_WIN_RATE = float(os.getenv('COMPOUND_MIN_WIN_RATE', '0.6'))  # Require 60% win rate
COMPOUND_MAX_DRAWDOWN = float(os.getenv('COMPOUND_MAX_DRAWDOWN', '0.15'))  # Pause if >15% drawdown
COMPOUND_SCALING_FACTOR = float(os.getenv('COMPOUND_SCALING_FACTOR', '0.5'))  # Reduce compounding if performance poor

# Pure Price Action Strategy Parameters - No Traditional Indicators

# Price action analysis parameters
PRICE_ACTION_LOOKBACK = int(os.getenv('PRICE_ACTION_LOOKBACK', '20'))    # Lookback period for price analysis
BREAKOUT_THRESHOLD = float(os.getenv('BREAKOUT_THRESHOLD', '0.02'))      # 2% breakout threshold
VOLATILITY_WINDOW = int(os.getenv('VOLATILITY_WINDOW', '14'))            # Volatility calculation window
MOMENTUM_WINDOW = int(os.getenv('MOMENTUM_WINDOW', '10'))                # Momentum calculation window
SUPPORT_RESISTANCE_STRENGTH = int(os.getenv('SUPPORT_RESISTANCE_STRENGTH', '3'))  # S/R level strength

# Enhanced Configurable Thresholds for Pure Price Action Strategy
# Resistance Level Thresholds
RESISTANCE_BREAKOUT_THRESHOLD = float(os.getenv('RESISTANCE_BREAKOUT_THRESHOLD', '1.002'))  # 0.2% above resistance
RESISTANCE_REJECTION_THRESHOLD = float(os.getenv('RESISTANCE_REJECTION_THRESHOLD', '0.995'))  # 0.5% below resistance
RESISTANCE_TOUCH_THRESHOLD = float(os.getenv('RESISTANCE_TOUCH_THRESHOLD', '0.998'))  # Within 0.2% of resistance

# Support Level Thresholds
SUPPORT_BREAKDOWN_THRESHOLD = float(os.getenv('SUPPORT_BREAKDOWN_THRESHOLD', '0.998'))  # 0.2% below support
SUPPORT_REJECTION_THRESHOLD = float(os.getenv('SUPPORT_REJECTION_THRESHOLD', '1.005'))  # 0.5% above support
SUPPORT_TOUCH_THRESHOLD = float(os.getenv('SUPPORT_TOUCH_THRESHOLD', '1.002'))  # Within 0.2% of support

# Momentum Analysis Thresholds
MOMENTUM_POSITIVE_THRESHOLD = float(os.getenv('MOMENTUM_POSITIVE_THRESHOLD', '0.005'))  # 0.5% positive momentum
MOMENTUM_NEGATIVE_THRESHOLD = float(os.getenv('MOMENTUM_NEGATIVE_THRESHOLD', '-0.005'))  # 0.5% negative momentum
MOMENTUM_CONFIRMATION_THRESHOLD = float(os.getenv('MOMENTUM_CONFIRMATION_THRESHOLD', '0.003'))  # 0.3% for confirmation

# Market Condition Thresholds
VOLATILITY_MULTIPLIER = float(os.getenv('VOLATILITY_MULTIPLIER', '1.5'))  # High volatility threshold
VOLUME_RATIO_THRESHOLD = float(os.getenv('VOLUME_RATIO_THRESHOLD', '1.5'))  # Above average volume threshold
BODY_RATIO_THRESHOLD = float(os.getenv('BODY_RATIO_THRESHOLD', '0.7'))  # Strong candle body threshold

# Price Position Thresholds
PRICE_POSITION_HIGH = float(os.getenv('PRICE_POSITION_HIGH', '0.8'))  # Near top of range (80%)
PRICE_POSITION_LOW = float(os.getenv('PRICE_POSITION_LOW', '0.2'))  # Near bottom of range (20%)

# Support/Resistance Touch Detection
TOUCH_TOLERANCE = float(os.getenv('TOUCH_TOLERANCE', '0.005'))  # 0.5% tolerance for S/R touches

TIMEFRAME = os.getenv('TIMEFRAME', '5m')  # Default to 15 minutes, can be overridden

# Risk management - Enhanced stop loss and take profit settings
USE_STOP_LOSS = os.getenv('USE_STOP_LOSS', 'True').lower() == 'true'
STOP_LOSS_PCT = float(os.getenv('STOP_LOSS_PCT', '0.005'))  # 0.5% stop loss (more conservative)
TRAILING_STOP = os.getenv('TRAILING_STOP', 'True').lower() == 'true'
TRAILING_STOP_PCT = float(os.getenv('TRAILING_STOP_PCT', '0.01'))  # 1% trailing stop
UPDATE_TRAILING_ON_HOLD = os.getenv('UPDATE_TRAILING_ON_HOLD', 'True').lower() == 'true'  # Update trailing stop on HOLD signals

# Take profit settings - Fixed take profit (not trailing)
USE_TAKE_PROFIT = os.getenv('USE_TAKE_PROFIT', 'True').lower() == 'true'
TAKE_PROFIT_PCT = float(os.getenv('TAKE_PROFIT_PCT', '0.015'))  # 1.5% fixed take profit

# Enhanced backtesting parameters
BACKTEST_START_DATE = os.getenv('BACKTEST_START_DATE', '2023-01-01')
BACKTEST_END_DATE = os.getenv('BACKTEST_END_DATE', '')
BACKTEST_INITIAL_BALANCE = float(os.getenv('BACKTEST_INITIAL_BALANCE', '50.0'))
BACKTEST_COMMISSION = float(os.getenv('BACKTEST_COMMISSION', '0.0004'))
BACKTEST_USE_AUTO_COMPOUND = os.getenv('BACKTEST_USE_AUTO_COMPOUND', 'True').lower() == 'true'  # Enabled for enhanced auto-compounding test

# Enhanced validation requirements - Optimized for pure price action strategies
BACKTEST_BEFORE_LIVE = os.getenv('BACKTEST_BEFORE_LIVE', 'True').lower() == 'true'
BACKTEST_MIN_PROFIT_PCT = float(os.getenv('BACKTEST_MIN_PROFIT_PCT', '10.0'))  # Suitable for price action
BACKTEST_MIN_WIN_RATE = float(os.getenv('BACKTEST_MIN_WIN_RATE', '40.0'))  # Realistic for pure price action
BACKTEST_MAX_DRAWDOWN = float(os.getenv('BACKTEST_MAX_DRAWDOWN', '30.0'))  # Allow for volatility
BACKTEST_MIN_PROFIT_FACTOR = float(os.getenv('BACKTEST_MIN_PROFIT_FACTOR', '1.2'))  # Conservative
BACKTEST_PERIOD = os.getenv('BACKTEST_PERIOD', '90 days')  # Default to 90 days for comprehensive testing

# Logging and notifications
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
USE_TELEGRAM = os.getenv('USE_TELEGRAM', 'True').lower() == 'true'
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')
SEND_DAILY_REPORT = os.getenv('SEND_DAILY_REPORT', 'True').lower() == 'true'
DAILY_REPORT_TIME = os.getenv('DAILY_REPORT_TIME', '00:00')  # 24-hour format

# Other settings
RETRY_COUNT = int(os.getenv('RETRY_COUNT', '3'))
RETRY_DELAY = int(os.getenv('RETRY_DELAY', '5'))  # seconds