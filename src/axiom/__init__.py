"""axiom-backtest: an event-driven backtesting framework.

Public API is intentionally small. Import the pieces you need:

    from axiom import (
        HistoricCSVDataHandler, MovingAverageCrossover, Portfolio,
        SimulatedExecutionHandler, BacktestEngine, analyze, walk_forward,
    )
"""

from __future__ import annotations

from .data import DataHandler, HistoricCSVDataHandler
from .engine import BacktestEngine
from .event import (
    Direction,
    Event,
    EventType,
    FillEvent,
    MarketEvent,
    OrderEvent,
    SignalEvent,
)
from .execution import (
    CommissionModel,
    ExecutionHandler,
    NextBarExecutionHandler,
    PercentageSlippage,
    PerShareCommission,
    SimulatedExecutionHandler,
    SlippageModel,
    VolatilitySlippage,
    VolumeShareSlippage,
)
from .metrics import PerformanceReport, analyze, max_drawdown, sharpe_ratio, sortino_ratio
from .portfolio import Portfolio
from .strategy import MovingAverageCrossover, Strategy
from .validation import WalkForwardWindow, stitch_oos_equity, walk_forward

__version__ = "0.1.0"

__all__ = [
    "DataHandler",
    "HistoricCSVDataHandler",
    "BacktestEngine",
    "Direction",
    "Event",
    "EventType",
    "FillEvent",
    "MarketEvent",
    "OrderEvent",
    "SignalEvent",
    "CommissionModel",
    "ExecutionHandler",
    "NextBarExecutionHandler",
    "PercentageSlippage",
    "PerShareCommission",
    "SimulatedExecutionHandler",
    "SlippageModel",
    "VolatilitySlippage",
    "VolumeShareSlippage",
    "PerformanceReport",
    "analyze",
    "max_drawdown",
    "sharpe_ratio",
    "sortino_ratio",
    "Portfolio",
    "MovingAverageCrossover",
    "Strategy",
    "WalkForwardWindow",
    "stitch_oos_equity",
    "walk_forward",
    "__version__",
]
