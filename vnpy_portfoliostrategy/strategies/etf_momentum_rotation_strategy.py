from math import floor

from vnpy.trader.constant import Direction, Interval
from vnpy.trader.object import BarData, TickData
from vnpy.trader.utility import ArrayManager

from vnpy_portfoliostrategy import StrategyEngine, StrategyTemplate


class EtfMomentumRotationStrategy(StrategyTemplate):
    """ETF动量轮动策略"""

    author = "Steve"

    EQUITY_SYMBOLS = [
        "510300.SSE",
        "510500.SSE",
        "512100.SSE",
        "159915.SZSE",
        "588000.SSE",
        "512880.SSE",
        "510150.SSE",
        "512010.SSE",
        "512480.SSE",
        "515700.SSE",
        "512660.SSE",
        "512800.SSE",
        "510880.SSE",
    ]
    BOND_SYMBOL = "511010.SSE"
    MONEY_SYMBOL = "511880.SSE"

    initial_capital = 1_000_000
    rebalance_days = 10
    short_window = 20
    middle_window = 60
    long_window = 120
    short_weight = 0.3
    middle_weight = 0.4
    long_weight = 0.3
    top_n = 2
    lot_size = 100
    price_add = 0.01

    rebalance_count = 0
    selected_symbols: list[str] = []
    defensive_symbol = ""
    momentum_scores: dict[str, float] = {}
    target_values: dict[str, float] = {}

    parameters = [
        "initial_capital",
        "rebalance_days",
        "short_window",
        "middle_window",
        "long_window",
        "short_weight",
        "middle_weight",
        "long_weight",
        "top_n",
        "lot_size",
        "price_add",
    ]
    variables = [
        "rebalance_count",
        "selected_symbols",
        "defensive_symbol",
        "momentum_scores",
        "target_values",
    ]

    def __init__(
        self,
        strategy_engine: StrategyEngine,
        strategy_name: str,
        vt_symbols: list[str],
        setting: dict
    ) -> None:
        """构造函数"""
        super().__init__(strategy_engine, strategy_name, vt_symbols, setting)

        size: int = max(self.short_window, self.middle_window, self.long_window) + 1
        self.ams: dict[str, ArrayManager] = {
            vt_symbol: ArrayManager(size) for vt_symbol in self.vt_symbols
        }

        self.equity_symbols: list[str] = [
            vt_symbol for vt_symbol in self.EQUITY_SYMBOLS if vt_symbol in self.vt_symbols
        ]

        self.selected_symbols = []
        self.defensive_symbol = ""
        self.momentum_scores = {}
        self.target_values = {}

    def on_init(self) -> None:
        """策略初始化回调"""
        self.write_log("策略初始化")

        init_days: int = max(self.long_window * 3, 250)
        self.load_bars(init_days, Interval.DAILY)

    def on_start(self) -> None:
        """策略启动回调"""
        self.write_log("策略启动")

    def on_stop(self) -> None:
        """策略停止回调"""
        self.write_log("策略停止")

    def on_tick(self, tick: TickData) -> None:
        """行情推送回调"""
        return

    def on_bars(self, bars: dict[str, BarData]) -> None:
        """K线切片回调"""
        for vt_symbol, bar in bars.items():
            am: ArrayManager | None = self.ams.get(vt_symbol)
            if am is None:
                continue

            am.update_bar(bar)

        self.rebalance_count += 1
        if self.rebalance_count >= max(int(self.rebalance_days), 1):
            self.rebalance_count = 0
            self.update_targets(bars)

        self.rebalance_portfolio(bars)
        self.put_event()

    def update_targets(self, bars: dict[str, BarData]) -> None:
        """计算动量排名并更新目标持仓"""
        for vt_symbol in self.vt_symbols:
            self.set_target(vt_symbol, 0)

        self.momentum_scores = self.calculate_equity_scores(bars)
        self.selected_symbols = [
            vt_symbol
            for vt_symbol, score in sorted(
                self.momentum_scores.items(),
                key=lambda item: item[1],
                reverse=True,
            )
            if score > 0
        ][:self.get_slot_count()]

        slot_cash: float = self.initial_capital / self.get_slot_count()
        self.target_values = {}

        for vt_symbol in self.selected_symbols:
            target_value: float = slot_cash
            target_volume: int = self.calculate_target_volume(vt_symbol, target_value, bars)
            self.set_target(vt_symbol, target_volume)
            self.target_values[vt_symbol] = target_value

        defensive_slots: int = self.get_slot_count() - len(self.selected_symbols)
        self.defensive_symbol = ""

        if defensive_slots <= 0:
            return

        self.defensive_symbol = self.select_defensive_symbol()
        target_value = slot_cash * defensive_slots

        if self.defensive_symbol not in bars:
            return

        target_volume = self.calculate_target_volume(self.defensive_symbol, target_value, bars)
        self.set_target(self.defensive_symbol, target_volume)
        self.target_values[self.defensive_symbol] = target_value

    def calculate_equity_scores(self, bars: dict[str, BarData]) -> dict[str, float]:
        """计算权益ETF综合动量分数"""
        scores: dict[str, float] = {}

        for vt_symbol in self.equity_symbols:
            if vt_symbol not in bars:
                continue

            score: float | None = self.calculate_score(vt_symbol)
            if score is None:
                continue

            scores[vt_symbol] = score

        return scores

    def calculate_score(self, vt_symbol: str) -> float | None:
        """计算单个ETF的综合动量分数"""
        short_momentum: float | None = self.calculate_momentum(vt_symbol, self.short_window)
        middle_momentum: float | None = self.calculate_momentum(vt_symbol, self.middle_window)
        long_momentum: float | None = self.calculate_momentum(vt_symbol, self.long_window)

        if short_momentum is None or middle_momentum is None or long_momentum is None:
            return None

        return (
            self.short_weight * short_momentum
            + self.middle_weight * middle_momentum
            + self.long_weight * long_momentum
        )

    def calculate_momentum(self, vt_symbol: str, window: int) -> float | None:
        """计算指定窗口动量"""
        am: ArrayManager | None = self.ams.get(vt_symbol)
        if not am or am.count < window + 1:
            return None

        close_array = am.close
        current_price: float = close_array[-1]
        previous_price: float = close_array[-window - 1]

        if current_price <= 0 or previous_price <= 0:
            return None

        return current_price / previous_price - 1

    def select_defensive_symbol(self) -> str:
        """选择防守ETF"""
        bond_momentum: float | None = self.calculate_momentum(self.BOND_SYMBOL, self.middle_window)
        if bond_momentum is not None and bond_momentum > 0:
            return self.BOND_SYMBOL

        return self.MONEY_SYMBOL

    def calculate_target_volume(
        self,
        vt_symbol: str,
        target_value: float,
        bars: dict[str, BarData],
    ) -> int:
        """根据目标市值和收盘价换算目标持仓"""
        bar: BarData | None = bars.get(vt_symbol)
        if not bar or bar.close_price <= 0:
            return 0

        lot_size: int = max(int(self.lot_size), 1)
        return floor(target_value / bar.close_price / lot_size) * lot_size

    def get_slot_count(self) -> int:
        """获取组合仓位槽数量"""
        return max(int(self.top_n), 1)

    def calculate_price(
        self,
        vt_symbol: str,
        direction: Direction,
        reference: float
    ) -> float:
        """计算调仓委托价格"""
        if direction == Direction.LONG:
            return reference + self.price_add

        return max(reference - self.price_add, 0)
