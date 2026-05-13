from datetime import datetime
from unittest import TestCase


from vnpy.trader.constant import Direction, Exchange, Interval, Offset
from vnpy.trader.object import BarData
from vnpy_portfoliostrategy.backtesting import BacktestingEngine
from vnpy_portfoliostrategy.template import StrategyTemplate


VT_SYMBOL = "510300.SSE"


class CashTestStrategy(StrategyTemplate):
    """Minimal strategy for backtesting cash accounting tests."""

    def on_init(self) -> None:
        return

    def on_bars(self, bars: dict[str, BarData]) -> None:
        return


def create_engine() -> BacktestingEngine:
    engine = BacktestingEngine()
    engine.set_parameters(
        vt_symbols=[VT_SYMBOL],
        interval=Interval.DAILY,
        start=datetime(2024, 1, 1),
        rates={VT_SYMBOL: 0.001},
        slippages={VT_SYMBOL: 0.02},
        sizes={VT_SYMBOL: 10},
        priceticks={VT_SYMBOL: 0.01},
        capital=1_000,
        end=datetime(2024, 1, 2),
    )
    engine.add_strategy(CashTestStrategy, {})
    engine.strategy.inited = True
    engine.strategy.trading = True
    engine.datetime = datetime(2024, 1, 2)
    return engine


def create_bar(open_price: float, high_price: float, low_price: float, close_price: float) -> BarData:
    return BarData(
        symbol="510300",
        exchange=Exchange.SSE,
        datetime=datetime(2024, 1, 2),
        open_price=open_price,
        high_price=high_price,
        low_price=low_price,
        close_price=close_price,
        gateway_name=BacktestingEngine.gateway_name,
    )


class BacktestingCashTest(TestCase):
    def test_cross_limit_order_deducts_cash_for_long_trade(self) -> None:
        engine = create_engine()
        engine.bars[VT_SYMBOL] = create_bar(open_price=10.2, high_price=10.5, low_price=10.0, close_price=10.4)
        engine.send_order(engine.strategy, VT_SYMBOL, Direction.LONG, Offset.OPEN, 10.5, 2, False, False)

        engine.cross_limit_order()

        expected_turnover = 10.2 * 2 * 10
        expected_commission = expected_turnover * 0.001
        expected_slippage = 2 * 10 * 0.02
        self.assertAlmostEqual(
            engine.get_cash_available(),
            1_000 - expected_turnover - expected_commission - expected_slippage,
        )
        self.assertEqual(engine.strategy.pos_data[VT_SYMBOL], 2)

    def test_cross_limit_order_adds_cash_for_short_trade(self) -> None:
        engine = create_engine()
        engine.bars[VT_SYMBOL] = create_bar(open_price=10.3, high_price=10.5, low_price=10.0, close_price=10.1)
        engine.send_order(engine.strategy, VT_SYMBOL, Direction.SHORT, Offset.CLOSE, 10.0, 2, False, False)

        engine.cross_limit_order()

        expected_turnover = 10.3 * 2 * 10
        expected_commission = expected_turnover * 0.001
        expected_slippage = 2 * 10 * 0.02
        self.assertAlmostEqual(
            engine.get_cash_available(),
            1_000 + expected_turnover - expected_commission - expected_slippage,
        )
        self.assertEqual(engine.strategy.pos_data[VT_SYMBOL], -2)

    def test_portfolio_value_combines_cash_and_holding_value(self) -> None:
        engine = create_engine()
        engine.bars[VT_SYMBOL] = create_bar(open_price=10.0, high_price=11.5, low_price=9.8, close_price=11.0)
        engine.strategy.pos_data[VT_SYMBOL] = 3

        self.assertAlmostEqual(engine.get_holding_value(), 3 * 11.0 * 10)
        self.assertAlmostEqual(engine.get_portfolio_value(), engine.get_cash_available() + 3 * 11.0 * 10)
        self.assertAlmostEqual(engine.strategy.get_portfolio_value(), engine.get_portfolio_value())
