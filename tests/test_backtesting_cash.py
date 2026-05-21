from datetime import date, datetime
from math import log, sqrt
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from pandas import DataFrame, ExcelFile

import vnpy_portfoliostrategy.backtesting as backtesting_module
from vnpy.trader.constant import Direction, Exchange, Interval, Offset
from vnpy.trader.object import BarData, TradeData
from vnpy_portfoliostrategy.backtesting import BacktestingEngine, PortfolioDailyResult
from vnpy_portfoliostrategy.template import StrategyTemplate


VT_SYMBOL = "510300.SSE"
SECOND_VT_SYMBOL = "159915.SZSE"


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


def create_two_symbol_engine() -> BacktestingEngine:
    engine = BacktestingEngine()
    engine.set_parameters(
        vt_symbols=[VT_SYMBOL, SECOND_VT_SYMBOL],
        interval=Interval.DAILY,
        start=datetime(2024, 1, 1),
        rates={VT_SYMBOL: 0.001, SECOND_VT_SYMBOL: 0.002},
        slippages={VT_SYMBOL: 0.02, SECOND_VT_SYMBOL: 0.01},
        sizes={VT_SYMBOL: 10, SECOND_VT_SYMBOL: 100},
        priceticks={VT_SYMBOL: 0.01, SECOND_VT_SYMBOL: 0.001},
        capital=10_000,
        end=datetime(2024, 1, 3),
    )
    engine.add_strategy(CashTestStrategy, {})
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


def create_symbol_trade(
    tradeid: str,
    vt_symbol: str,
    direction: Direction,
    price: float,
    volume: float,
    trade_datetime: datetime,
) -> TradeData:
    symbol, exchange = vt_symbol.split(".")

    return TradeData(
        gateway_name=BacktestingEngine.gateway_name,
        symbol=symbol,
        exchange=Exchange(exchange),
        orderid=tradeid,
        tradeid=tradeid,
        direction=direction,
        price=price,
        volume=volume,
        datetime=trade_datetime,
    )


def add_two_symbol_daily_results(engine: BacktestingEngine) -> None:
    first_day = PortfolioDailyResult(
        date(2024, 1, 2),
        {
            VT_SYMBOL: 10,
            SECOND_VT_SYMBOL: 5,
        },
    )
    second_day = PortfolioDailyResult(
        date(2024, 1, 3),
        {
            VT_SYMBOL: 11,
            SECOND_VT_SYMBOL: 4,
        },
    )

    first_day.add_trade(
        create_symbol_trade("1", VT_SYMBOL, Direction.LONG, 9, 2, datetime(2024, 1, 2, 9, 30))
    )
    first_day.add_trade(
        create_symbol_trade("2", SECOND_VT_SYMBOL, Direction.LONG, 6, 1, datetime(2024, 1, 2, 9, 31))
    )

    first_day.calculate_pnl({}, {}, engine.sizes, engine.rates, engine.slippages)
    second_day.calculate_pnl(
        first_day.close_prices,
        first_day.end_poses,
        engine.sizes,
        engine.rates,
        engine.slippages,
    )

    engine.daily_results = {
        first_day.date: first_day,
        second_day.date: second_day,
    }
    engine.daily_df = DataFrame.from_dict(
        {
            "date": [first_day.date, second_day.date],
            "trade_count": [first_day.trade_count, second_day.trade_count],
            "turnover": [first_day.turnover, second_day.turnover],
            "commission": [first_day.commission, second_day.commission],
            "slippage": [first_day.slippage, second_day.slippage],
            "trading_pnl": [first_day.trading_pnl, second_day.trading_pnl],
            "holding_pnl": [first_day.holding_pnl, second_day.holding_pnl],
            "total_pnl": [first_day.total_pnl, second_day.total_pnl],
            "net_pnl": [first_day.net_pnl, second_day.net_pnl],
        }
    ).set_index("date")


def create_daily_df(net_pnl: float = 0) -> DataFrame:
    return DataFrame(
        {
            "net_pnl": [net_pnl],
            "commission": [0],
            "slippage": [0],
            "turnover": [0],
            "trade_count": [0],
        },
        index=[date(2024, 1, 2)],
    )


def create_trade(tradeid: str, direction: Direction, price: float, volume: float) -> TradeData:
    return TradeData(
        gateway_name=BacktestingEngine.gateway_name,
        symbol="510300",
        exchange=Exchange.SSE,
        orderid=tradeid,
        tradeid=tradeid,
        direction=direction,
        price=price,
        volume=volume,
        datetime=datetime(2024, 1, 2, 9, int(tradeid)),
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

    def test_benchmark_curve_normalizes_to_initial_capital(self) -> None:
        engine = create_engine()
        df = DataFrame(index=[date(2024, 1, 2), date(2024, 1, 3)])
        original_load_bar_data = backtesting_module.load_bar_data

        def fake_load_bar_data(
            vt_symbol: str,
            interval: Interval,
            start: datetime,
            end: datetime
        ) -> list[BarData]:
            return [
                create_bar(open_price=3000, high_price=3000, low_price=3000, close_price=3000),
                BarData(
                    symbol="510300",
                    exchange=Exchange.SSE,
                    datetime=datetime(2024, 1, 3),
                    open_price=3300,
                    high_price=3300,
                    low_price=3300,
                    close_price=3300,
                    gateway_name=BacktestingEngine.gateway_name,
                ),
            ]

        try:
            backtesting_module.load_bar_data = fake_load_bar_data
            benchmark_df = engine.calculate_benchmark_curve("000300.SSE", df)
        finally:
            backtesting_module.load_bar_data = original_load_bar_data

        self.assertIsNotNone(benchmark_df)
        self.assertAlmostEqual(benchmark_df["benchmark_balance"].iloc[0], 1_000)
        self.assertAlmostEqual(benchmark_df["benchmark_balance"].iloc[1], 1_100)

    def test_benchmark_curve_returns_none_when_dates_do_not_overlap(self) -> None:
        engine = create_engine()
        df = DataFrame(index=[date(2024, 1, 2)])
        original_load_bar_data = backtesting_module.load_bar_data

        def fake_load_bar_data(
            vt_symbol: str,
            interval: Interval,
            start: datetime,
            end: datetime
        ) -> list[BarData]:
            return [
                BarData(
                    symbol="510300",
                    exchange=Exchange.SSE,
                    datetime=datetime(2024, 1, 1),
                    open_price=3000,
                    high_price=3000,
                    low_price=3000,
                    close_price=3000,
                    gateway_name=BacktestingEngine.gateway_name,
                )
            ]

        try:
            backtesting_module.load_bar_data = fake_load_bar_data
            benchmark_df = engine.calculate_benchmark_curve("000300.SSE", df)
        finally:
            backtesting_module.load_bar_data = original_load_bar_data

        self.assertIsNone(benchmark_df)

    def test_calculate_statistics_pairs_partial_long_closes_by_closing_trade(self) -> None:
        engine = create_engine()
        engine.trades = {
            "1": create_trade("1", Direction.LONG, 10, 500),
            "2": create_trade("2", Direction.SHORT, 11, 300),
            "3": create_trade("3", Direction.LONG, 12, 200),
            "4": create_trade("4", Direction.SHORT, 13, 400),
        }

        statistics = engine.calculate_statistics(create_daily_df(), output=False)

        self.assertEqual(statistics["total_closed_trade_count"], 2)
        self.assertEqual(statistics["profit_trade_count"], 2)
        self.assertEqual(statistics["loss_trade_count"], 0)
        self.assertAlmostEqual(statistics["win_rate"], 1)
        self.assertAlmostEqual(statistics["average_profit"], 5280.5)
        self.assertAlmostEqual(statistics["average_loss"], 0)
        self.assertAlmostEqual(statistics["profit_loss_ratio"], 0)
        self.assertAlmostEqual(statistics["average_net_profit_per_volume"], 15.087142857142858)

    def test_calculate_statistics_returns_zero_for_no_closed_trades(self) -> None:
        engine = create_engine()

        statistics = engine.calculate_statistics(create_daily_df(), output=False)

        self.assertEqual(statistics["total_closed_trade_count"], 0)
        self.assertEqual(statistics["profit_trade_count"], 0)
        self.assertEqual(statistics["loss_trade_count"], 0)
        self.assertEqual(statistics["win_rate"], 0)
        self.assertEqual(statistics["average_profit"], 0)
        self.assertEqual(statistics["average_loss"], 0)
        self.assertEqual(statistics["profit_loss_ratio"], 0)
        self.assertEqual(statistics["average_net_profit_per_volume"], 0)

    def test_calculate_statistics_pairs_short_closes(self) -> None:
        engine = create_engine()
        engine.trades = {
            "1": create_trade("1", Direction.SHORT, 12, 5),
            "2": create_trade("2", Direction.LONG, 10, 5),
        }

        statistics = engine.calculate_statistics(create_daily_df(), output=False)

        self.assertEqual(statistics["total_closed_trade_count"], 1)
        self.assertEqual(statistics["profit_trade_count"], 1)
        self.assertEqual(statistics["loss_trade_count"], 0)
        self.assertAlmostEqual(statistics["win_rate"], 1)
        self.assertAlmostEqual(statistics["average_profit"], 96.9)
        self.assertAlmostEqual(statistics["average_net_profit_per_volume"], 19.38)

    def test_set_parameters_defaults_risk_free_to_three_percent(self) -> None:
        engine = create_engine()

        self.assertEqual(engine.risk_free, 3)

    def test_calculate_statistics_adds_calmar_and_annual_risk_free_sharpe(self) -> None:
        engine = create_engine()
        df = DataFrame(
            {
                "net_pnl": [100, -50],
                "commission": [0, 0],
                "slippage": [0, 0],
                "turnover": [0, 0],
                "trade_count": [0, 0],
            },
            index=[date(2024, 1, 2), date(2024, 1, 3)],
        )

        statistics = engine.calculate_statistics(df, output=False)

        expected_annual_return = 5 / 2 * 240
        expected_max_ddpercent = -50 / 1100 * 100
        expected_return_std = abs(log(1050 / 1100)) / sqrt(2) * 100
        expected_daily_return = log(1050 / 1100) / 2 * 100
        expected_sharpe_ratio = (
            (expected_daily_return - 3 / 240) / expected_return_std * sqrt(240)
        )

        self.assertAlmostEqual(statistics["calmar_ratio"], expected_annual_return / abs(expected_max_ddpercent))
        self.assertAlmostEqual(statistics["sharpe_ratio"], expected_sharpe_ratio)

    def test_calculate_symbol_daily_pnl_expands_contract_results(self) -> None:
        engine = create_two_symbol_engine()
        add_two_symbol_daily_results(engine)

        detail_df = engine.calculate_symbol_daily_pnl()

        self.assertEqual(
            list(detail_df.columns),
            [
                "date", "vt_symbol", "start_pos", "end_pos", "pre_close", "close_price",
                "trade_count", "turnover", "commission", "slippage", "trading_pnl",
                "holding_pnl", "total_pnl", "net_pnl", "contribution_pct",
            ],
        )
        self.assertEqual(len(detail_df), 4)

        grouped_net_pnl = detail_df.groupby("date")["net_pnl"].sum()
        for result_date, net_pnl in engine.daily_df["net_pnl"].items():
            self.assertAlmostEqual(grouped_net_pnl[result_date], net_pnl)

        first_symbol = detail_df[
            (detail_df["date"] == date(2024, 1, 2))
            & (detail_df["vt_symbol"] == VT_SYMBOL)
        ].iloc[0]
        self.assertEqual(first_symbol["start_pos"], 0)
        self.assertEqual(first_symbol["end_pos"], 2)
        self.assertAlmostEqual(first_symbol["trading_pnl"], 20)
        self.assertAlmostEqual(first_symbol["holding_pnl"], 0)
        self.assertAlmostEqual(first_symbol["commission"], 0.18)
        self.assertAlmostEqual(first_symbol["slippage"], 0.4)
        self.assertAlmostEqual(first_symbol["net_pnl"], 19.42)

    def test_calculate_daily_pnl_summary_identifies_winners_and_losers(self) -> None:
        engine = create_two_symbol_engine()
        add_two_symbol_daily_results(engine)

        summary_df = engine.calculate_daily_pnl_summary()

        first_day = summary_df[summary_df["date"] == date(2024, 1, 2)].iloc[0]
        self.assertEqual(first_day["profit_symbol_count"], 1)
        self.assertEqual(first_day["loss_symbol_count"], 1)
        self.assertEqual(first_day["top_profit_symbol"], VT_SYMBOL)
        self.assertAlmostEqual(first_day["top_profit_pnl"], 19.42)
        self.assertEqual(first_day["top_loss_symbol"], SECOND_VT_SYMBOL)
        self.assertAlmostEqual(first_day["top_loss_pnl"], -102.2)

    def test_calculate_daily_top_contributors_sorts_by_absolute_net_pnl(self) -> None:
        engine = create_two_symbol_engine()
        add_two_symbol_daily_results(engine)

        top_df = engine.calculate_daily_top_contributors(top_n=1)

        self.assertEqual(len(top_df), 2)
        first_day = top_df[top_df["date"] == date(2024, 1, 2)].iloc[0]
        self.assertEqual(first_day["vt_symbol"], SECOND_VT_SYMBOL)
        self.assertAlmostEqual(first_day["net_pnl"], -102.2)

    def test_export_pnl_report_uses_default_excel_filename_and_expected_sheets(self) -> None:
        engine = create_two_symbol_engine()
        add_two_symbol_daily_results(engine)

        with TemporaryDirectory() as directory:
            report_path = engine.export_pnl_report(path=Path(directory), top_n=1)

            self.assertEqual(report_path.parent, Path(directory))
            self.assertRegex(report_path.name, r"^detail_CashTestStrategy_\d{8}_\d{6}\.xlsx$")

            workbook = ExcelFile(report_path, engine="openpyxl")
            self.assertEqual(
                set(workbook.sheet_names),
                {"daily_summary", "symbol_daily_pnl", "daily_top_contributors"},
            )
