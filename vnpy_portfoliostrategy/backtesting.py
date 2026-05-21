from collections import defaultdict, deque
from datetime import date, datetime, timedelta
from functools import lru_cache, partial
from copy import copy
from pathlib import Path
import traceback

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pandas import DataFrame, ExcelWriter
from collections.abc import Callable

from vnpy.trader.constant import Direction, Offset, Interval, Status
from vnpy.trader.database import get_database, BaseDatabase
from vnpy.trader.object import OrderData, TradeData, BarData
from vnpy.trader.utility import round_to, extract_vt_symbol
from vnpy.trader.optimize import (
    OptimizationSetting,
    check_optimization_setting,
    run_bf_optimization,
    run_ga_optimization
)

from .base import EngineType
from .locale import _
from .template import StrategyTemplate


INTERVAL_DELTA_MAP: dict[Interval, timedelta] = {
    Interval.MINUTE: timedelta(minutes=1),
    Interval.HOUR: timedelta(hours=1),
    Interval.DAILY: timedelta(days=1),
}


class BacktestingEngine:
    """组合策略回测引擎"""

    engine_type: EngineType = EngineType.BACKTESTING
    gateway_name: str = "BACKTESTING"

    def __init__(self) -> None:
        """构造函数"""
        self.vt_symbols: list[str] = []
        self.start: datetime
        self.end: datetime

        self.rates: dict[str, float]
        self.slippages: dict[str, float]
        self.sizes: dict[str, float]
        self.priceticks: dict[str, float]

        self.capital: float = 1_000_000
        self.cash: float = self.capital
        self.risk_free: float = 3
        self.annual_days: int = 240

        self.strategy_class: type[StrategyTemplate]
        self.strategy: StrategyTemplate
        self.bars: dict[str, BarData] = {}
        self.datetime: datetime = datetime(1970, 1, 1)

        self.interval: Interval
        self.days: int = 0
        self.history_data: dict[tuple, BarData] = {}
        self.dts: set[datetime] = set()

        self.limit_order_count: int = 0
        self.limit_orders: dict[str, OrderData] = {}
        self.active_limit_orders: dict[str, OrderData] = {}

        self.trade_count: int = 0
        self.trades: dict[str, TradeData] = {}

        self.logs: list = []

        self.daily_results: dict[date, PortfolioDailyResult] = {}
        self.daily_df: DataFrame = None

    def clear_data(self) -> None:
        """清理上次回测缓存数据"""
        self.limit_order_count = 0
        self.limit_orders.clear()
        self.active_limit_orders.clear()

        self.trade_count = 0
        self.trades.clear()
        self.cash = self.capital

        self.logs.clear()
        self.daily_results.clear()
        self.daily_df = None

    def set_parameters(
        self,
        vt_symbols: list[str],
        interval: Interval,
        start: datetime,
        rates: dict[str, float],
        slippages: dict[str, float],
        sizes: dict[str, float],
        priceticks: dict[str, float],
        capital: float = 0,
        end: datetime | None = None,
        risk_free: float = 3,
        annual_days: int = 240
    ) -> None:
        """设置参数"""
        self.vt_symbols = vt_symbols
        self.interval = interval

        self.rates = rates
        self.slippages = slippages
        self.sizes = sizes
        self.priceticks = priceticks

        self.start = start
        if not end:
            self.end = datetime.now()
        else:
            self.end = end.replace(hour=23, minute=59, second=59)

        self.capital = capital
        self.cash = capital
        self.risk_free = risk_free
        self.annual_days = annual_days

    def add_strategy(self, strategy_class: type[StrategyTemplate], setting: dict) -> None:
        """增加策略"""
        self.strategy_class = strategy_class
        self.strategy = strategy_class(
            self, strategy_class.__name__, copy(self.vt_symbols), setting
        )

    def load_data(self) -> None:
        """加载历史数据"""
        self.output(_("开始加载历史数据"))

        if not self.end:
            self.end = datetime.now()

        if self.start >= self.end:
            self.output(_("起始日期必须小于结束日期"))
            return

        # 清理上次加载的历史数据
        self.history_data.clear()
        self.dts.clear()

        # 每次加载30天历史数据
        progress_delta: timedelta = timedelta(days=30)
        total_delta: timedelta = self.end - self.start
        interval_delta: timedelta = INTERVAL_DELTA_MAP[self.interval]

        for vt_symbol in self.vt_symbols:
            if self.interval == Interval.MINUTE:
                start: datetime = self.start
                end: datetime = self.start + progress_delta
                progress: float = 0

                data_count = 0
                while start < self.end:
                    end = min(end, self.end)

                    data: list[BarData] = load_bar_data(
                        vt_symbol,
                        self.interval,
                        start,
                        end
                    )

                    for bar in data:
                        self.dts.add(bar.datetime)
                        self.history_data[(bar.datetime, vt_symbol)] = bar
                        data_count += 1

                    progress += progress_delta / total_delta
                    progress = min(progress, 1)
                    progress_bar = "#" * int(progress * 10)
                    self.output(_("{}加载进度：{} [{:.0%}]").format(
                        vt_symbol, progress_bar, progress
                    ))

                    start = end + interval_delta
                    end += (progress_delta + interval_delta)
            else:
                data = load_bar_data(
                    vt_symbol,
                    self.interval,
                    self.start,
                    self.end
                )

                for bar in data:
                    self.dts.add(bar.datetime)
                    self.history_data[(bar.datetime, vt_symbol)] = bar

                data_count = len(data)

            self.output(_("{}历史数据加载完成，数据量：{}").format(vt_symbol, data_count))

        self.output(_("所有历史数据加载完成"))

    def run_backtesting(self) -> None:
        """开始回测"""
        self.strategy.on_init()

        dts: list = list(self.dts)
        dts.sort()

        # 使用指定时间的历史数据初始化策略
        day_count: int = 0
        _ix: int = 0

        for _ix, dt in enumerate(dts):
            if self.datetime and dt.day != self.datetime.day:
                day_count += 1
                if day_count >= self.days:
                    break

            try:
                self.new_bars(dt)
            except Exception:
                self.output(_("触发异常，回测终止"))
                self.output(traceback.format_exc())
                return

        self.strategy.inited = True
        self.output(_("策略初始化完成"))

        self.strategy.on_start()
        self.strategy.trading = True
        self.output(_("开始回放历史数据"))

        # 使用剩余历史数据进行策略回测
        for dt in dts[_ix:]:
            try:
                self.new_bars(dt)
            except Exception:
                self.output(_("触发异常，回测终止"))
                self.output(traceback.format_exc())
                return

        self.output(_("历史数据回放结束"))

    def calculate_result(self) -> DataFrame:
        """计算逐日盯市盈亏"""
        self.output(_("开始计算逐日盯市盈亏"))

        if not self.trades:
            self.output(_("成交记录为空，无法计算"))
            return

        for trade in self.trades.values():
            d: date = trade.datetime.date()
            daily_result: PortfolioDailyResult = self.daily_results[d]
            daily_result.add_trade(trade)

        pre_closes: dict = {}
        start_poses: dict = {}

        for daily_result in self.daily_results.values():
            daily_result.calculate_pnl(
                pre_closes,
                start_poses,
                self.sizes,
                self.rates,
                self.slippages,
            )

            pre_closes = daily_result.close_prices
            start_poses = daily_result.end_poses

        results: dict = defaultdict(list)

        for daily_result in self.daily_results.values():
            fields: list = [
                "date", "trade_count", "turnover",
                "commission", "slippage", "trading_pnl",
                "holding_pnl", "total_pnl", "net_pnl"
            ]
            for key in fields:
                value = getattr(daily_result, key)
                results[key].append(value)

        if results:
            self.daily_df = DataFrame.from_dict(results).set_index("date")

        self.output(_("逐日盯市盈亏计算完成"))
        return self.daily_df

    def calculate_symbol_daily_pnl(self) -> DataFrame:
        """计算按合约展开的逐日盯市盈亏"""
        fields: list[str] = [
            "date", "vt_symbol", "start_pos", "end_pos", "pre_close", "close_price",
            "trade_count", "turnover", "commission", "slippage", "trading_pnl",
            "holding_pnl", "total_pnl", "net_pnl", "contribution_pct",
        ]
        results: list[dict] = []

        for result_date in sorted(self.daily_results):
            daily_result: PortfolioDailyResult = self.daily_results[result_date]
            daily_net_pnl: float = daily_result.net_pnl

            for vt_symbol, contract_result in sorted(daily_result.contract_results.items()):
                if daily_net_pnl:
                    contribution_pct: float = contract_result.net_pnl / daily_net_pnl
                else:
                    contribution_pct = 0

                results.append({
                    "date": daily_result.date,
                    "vt_symbol": vt_symbol,
                    "start_pos": contract_result.start_pos,
                    "end_pos": contract_result.end_pos,
                    "pre_close": contract_result.pre_close,
                    "close_price": contract_result.close_price,
                    "trade_count": contract_result.trade_count,
                    "turnover": contract_result.turnover,
                    "commission": contract_result.commission,
                    "slippage": contract_result.slippage,
                    "trading_pnl": contract_result.trading_pnl,
                    "holding_pnl": contract_result.holding_pnl,
                    "total_pnl": contract_result.total_pnl,
                    "net_pnl": contract_result.net_pnl,
                    "contribution_pct": contribution_pct,
                })

        return DataFrame(results, columns=fields)

    def calculate_daily_pnl_summary(self) -> DataFrame:
        """计算每日盈亏归因摘要"""
        detail_df: DataFrame = self.calculate_symbol_daily_pnl()
        fields: list[str] = [
            "date", "net_pnl", "holding_pnl", "trading_pnl", "commission", "slippage",
            "profit_symbol_count", "loss_symbol_count", "top_profit_symbol",
            "top_profit_pnl", "top_loss_symbol", "top_loss_pnl",
        ]
        results: list[dict] = []

        if detail_df.empty:
            return DataFrame(results, columns=fields)

        for result_date, group_df in detail_df.groupby("date", sort=True):
            profit_df: DataFrame = group_df[group_df["net_pnl"] > 0]
            loss_df: DataFrame = group_df[group_df["net_pnl"] < 0]

            top_profit_symbol: str = ""
            top_profit_pnl: float = 0
            if not profit_df.empty:
                top_profit = profit_df.loc[profit_df["net_pnl"].idxmax()]
                top_profit_symbol = top_profit["vt_symbol"]
                top_profit_pnl = top_profit["net_pnl"]

            top_loss_symbol: str = ""
            top_loss_pnl: float = 0
            if not loss_df.empty:
                top_loss = loss_df.loc[loss_df["net_pnl"].idxmin()]
                top_loss_symbol = top_loss["vt_symbol"]
                top_loss_pnl = top_loss["net_pnl"]

            results.append({
                "date": result_date,
                "net_pnl": group_df["net_pnl"].sum(),
                "holding_pnl": group_df["holding_pnl"].sum(),
                "trading_pnl": group_df["trading_pnl"].sum(),
                "commission": group_df["commission"].sum(),
                "slippage": group_df["slippage"].sum(),
                "profit_symbol_count": len(profit_df),
                "loss_symbol_count": len(loss_df),
                "top_profit_symbol": top_profit_symbol,
                "top_profit_pnl": top_profit_pnl,
                "top_loss_symbol": top_loss_symbol,
                "top_loss_pnl": top_loss_pnl,
            })

        return DataFrame(results, columns=fields)

    def calculate_daily_top_contributors(self, top_n: int = 10) -> DataFrame:
        """计算每日盈亏贡献最大的合约"""
        detail_df: DataFrame = self.calculate_symbol_daily_pnl()

        if detail_df.empty:
            return detail_df

        sorted_df: DataFrame = (
            detail_df.assign(abs_net_pnl=detail_df["net_pnl"].abs())
            .sort_values(["date", "abs_net_pnl"], ascending=[True, False])
        )
        top_df: DataFrame = sorted_df.groupby("date", sort=True).head(top_n)

        return top_df.drop(columns=["abs_net_pnl"]).reset_index(drop=True)

    def export_pnl_report(self, path: str | Path | None = None, top_n: int = 10) -> Path:
        """导出逐日盈亏归因Excel报表"""
        if path is None:
            report_path: Path = Path.cwd() / self.generate_pnl_report_filename()
        else:
            report_path = Path(path)
            if report_path.is_dir():
                report_path = report_path / self.generate_pnl_report_filename()

        report_path.parent.mkdir(parents=True, exist_ok=True)

        with ExcelWriter(report_path, engine="openpyxl") as writer:
            self.calculate_daily_pnl_summary().to_excel(
                writer,
                sheet_name="daily_summary",
                index=False,
            )
            self.calculate_symbol_daily_pnl().to_excel(
                writer,
                sheet_name="symbol_daily_pnl",
                index=False,
            )
            self.calculate_daily_top_contributors(top_n).to_excel(
                writer,
                sheet_name="daily_top_contributors",
                index=False,
            )

        return report_path

    def generate_pnl_report_filename(self) -> str:
        """生成逐日盈亏归因报表文件名"""
        strategy_name: str = self.strategy.strategy_name
        timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")

        return f"detail_{strategy_name}_{timestamp}.xlsx"

    def calculate_statistics(self, df: DataFrame = None, output: bool = True) -> dict:
        """计算策略统计指标"""
        self.output(_("开始计算策略统计指标"))

        if df is None:
            df = self.daily_df

        # 初始化统计指标
        start_date: str = ""
        end_date: str = ""
        total_days: int = 0
        profit_days: int = 0
        loss_days: int = 0
        end_balance: float = 0
        max_drawdown: float = 0
        max_ddpercent: float = 0
        max_drawdown_duration: int = 0
        total_net_pnl: float = 0
        daily_net_pnl: float = 0
        total_commission: float = 0
        daily_commission: float = 0
        total_slippage: float = 0
        daily_slippage: float = 0
        total_turnover: float = 0
        daily_turnover: float = 0
        total_trade_count: int = 0
        daily_trade_count: float = 0
        total_return: float = 0
        annual_return: float = 0
        daily_return: float = 0
        return_std: float = 0
        sharpe_ratio: float = 0
        return_drawdown_ratio: float = 0
        calmar_ratio: float = 0
        trade_statistics: dict = self.calculate_trade_statistics()

        # 检查是否发生过爆仓
        positive_balance: bool = False

        # 计算资金相关指标
        if df is not None:
            df["balance"] = df["net_pnl"].cumsum() + self.capital
            df["return"] = np.log(df["balance"] / df["balance"].shift(1)).fillna(0)
            df["highlevel"] = df["balance"].rolling(min_periods=1, window=len(df), center=False).max()
            df["drawdown"] = df["balance"] - df["highlevel"]
            df["ddpercent"] = df["drawdown"] / df["highlevel"] * 100

            # 检查是否发生过爆仓
            positive_balance = (df["balance"] > 0).all()
            if not positive_balance:
                self.output(_("回测中出现爆仓（资金小于等于0），无法计算策略统计指标"))

        # 计算统计指标
        if positive_balance:
            start_date = df.index[0]
            end_date = df.index[-1]

            total_days = len(df)
            profit_days = len(df[df["net_pnl"] > 0])
            loss_days= len(df[df["net_pnl"] < 0])

            end_balance = df["balance"].iloc[-1]
            max_drawdown = df["drawdown"].min()
            max_ddpercent = df["ddpercent"].min()
            max_drawdown_end = df["drawdown"].idxmin()

            if isinstance(max_drawdown_end, date):
                max_drawdown_start = df["balance"][:max_drawdown_end].idxmax()          # type: ignore
                max_drawdown_duration = (max_drawdown_end - max_drawdown_start).days
            else:
                max_drawdown_duration = 0

            total_net_pnl = df["net_pnl"].sum()
            daily_net_pnl = total_net_pnl / total_days

            total_commission = df["commission"].sum()
            daily_commission = total_commission / total_days

            total_slippage = df["slippage"].sum()
            daily_slippage = total_slippage / total_days

            total_turnover = df["turnover"].sum()
            daily_turnover = total_turnover / total_days

            total_trade_count = df["trade_count"].sum()
            daily_trade_count = total_trade_count / total_days

            total_return = (end_balance / self.capital - 1) * 100
            annual_return = total_return / total_days * self.annual_days
            daily_return = df["return"].mean() * 100
            return_std = df["return"].std() * 100

            if return_std:
                daily_risk_free: float = self.risk_free / self.annual_days
                sharpe_ratio = (daily_return - daily_risk_free) / return_std * np.sqrt(self.annual_days)
            else:
                sharpe_ratio = 0

            if max_drawdown:
                return_drawdown_ratio = -total_net_pnl / max_drawdown
            else:
                return_drawdown_ratio = 0

            if max_ddpercent:
                calmar_ratio = annual_return / abs(max_ddpercent)
            else:
                calmar_ratio = 0

        # 输出结果
        if output:
            self.output("-" * 30)
            self.output(_("首个交易日：\t{}").format(start_date))
            self.output(_("最后交易日：\t{}").format(end_date))

            self.output(_("总交易日：\t{}").format(total_days))
            # self.output(_("盈利交易日：\t{}").format(profit_days))
            # self.output(_("亏损交易日：\t{}").format(loss_days))

            self.output(_("起始资金：\t{:,.2f}").format(self.capital))
            self.output(_("结束资金：\t{:,.2f}").format(end_balance))

            self.output(_("总收益率：\t{:,.2f}%").format(total_return))
            self.output(_("年化收益：\t{:,.2f}%").format(annual_return))
            self.output(_("最大回撤: \t{:,.2f}").format(max_drawdown))
            self.output(_("百分比最大回撤: {:,.2f}%").format(max_ddpercent))
            self.output(_("最长回撤天数: \t{}").format(max_drawdown_duration))

            self.output(_("总盈亏：\t{:,.2f}").format(total_net_pnl))
            self.output(_("总手续费：\t{:,.2f}").format(total_commission))
            self.output(_("总滑点：\t{:,.2f}").format(total_slippage))
            self.output(_("总成交金额：\t{:,.2f}").format(total_turnover))
            self.output(_("总成交笔数：\t{}").format(total_trade_count))

            # self.output(_("日均盈亏：\t{:,.2f}").format(daily_net_pnl))
            # self.output(_("日均手续费：\t{:,.2f}").format(daily_commission))
            # self.output(_("日均滑点：\t{:,.2f}").format(daily_slippage))
            # self.output(_("日均成交金额：\t{:,.2f}").format(daily_turnover))
            # self.output(_("日均成交笔数：\t{}").format(daily_trade_count))

            self.output(_("已平仓次数：\t{}").format(trade_statistics["total_closed_trade_count"]))
            self.output(_("盈利次数：\t{}").format(trade_statistics["profit_trade_count"]))
            self.output(_("亏损次数：\t{}").format(trade_statistics["loss_trade_count"]))
            self.output(_("胜率：\t{:.2%}").format(trade_statistics["win_rate"]))
            self.output(_("平均盈利：\t{:,.2f}").format(trade_statistics["average_profit"]))
            self.output(_("平均亏损：\t{:,.2f}").format(trade_statistics["average_loss"]))
            self.output(_("盈亏比：\t{:,.2f}").format(trade_statistics["profit_loss_ratio"]))
            self.output(_("平均每手净利润：\t{:,.2f}").format(
                trade_statistics["average_net_profit_per_volume"]
            ))

            # self.output(_("日均收益率：\t{:,.2f}%").format(daily_return))
            self.output(_("收益标准差：\t{:,.2f}%").format(return_std))
            self.output(f"Sharpe Ratio：\t{sharpe_ratio:,.2f}")
            self.output(_("收益回撤比：\t{:,.2f}").format(return_drawdown_ratio))
            self.output(f"Calmar Ratio：\t{calmar_ratio:,.2f}")

        statistics: dict = {
            "start_date": start_date,
            "end_date": end_date,
            "total_days": total_days,
            "profit_days": profit_days,
            "loss_days": loss_days,
            "capital": self.capital,
            "end_balance": end_balance,
            "max_drawdown": max_drawdown,
            "max_ddpercent": max_ddpercent,
            "max_drawdown_duration": max_drawdown_duration,
            "total_net_pnl": total_net_pnl,
            "daily_net_pnl": daily_net_pnl,
            "total_commission": total_commission,
            "daily_commission": daily_commission,
            "total_slippage": total_slippage,
            "daily_slippage": daily_slippage,
            "total_turnover": total_turnover,
            "daily_turnover": daily_turnover,
            "total_trade_count": total_trade_count,
            "daily_trade_count": daily_trade_count,
            "total_return": total_return,
            "annual_return": annual_return,
            "daily_return": daily_return,
            "return_std": return_std,
            "sharpe_ratio": sharpe_ratio,
            "return_drawdown_ratio": return_drawdown_ratio,
            "calmar_ratio": calmar_ratio,
            **trade_statistics,
        }

        # 过滤极值
        for key, value in statistics.items():
            if value in (np.inf, -np.inf):
                value = 0
            statistics[key] = np.nan_to_num(value)

        self.output(_("策略统计指标计算完成"))
        return statistics

    def calculate_trade_statistics(self) -> dict:
        """计算逐笔已平仓交易统计指标"""
        position_queues: defaultdict[str, deque] = defaultdict(deque)
        closed_pnls: list[float] = []
        total_closed_volume: float = 0

        trades: list[TradeData] = sorted(
            self.trades.values(),
            key=lambda trade: (trade.datetime or datetime.min, trade.tradeid)
        )

        for trade in trades:
            if trade.direction not in {Direction.LONG, Direction.SHORT}:
                continue

            vt_symbol: str = trade.vt_symbol
            size: float = self.sizes.get(vt_symbol, 1)
            rate: float = self.rates.get(vt_symbol, 0)
            slippage: float = self.slippages.get(vt_symbol, 0)

            remaining_volume: float = trade.volume
            queue: deque = position_queues[vt_symbol]
            trade_cost_per_volume: float = trade.price * size * rate + size * slippage

            closed_pnl: float = 0
            closed_volume: float = 0

            while remaining_volume and queue and queue[0]["direction"] != trade.direction:
                open_trade: dict = queue[0]
                volume: float = min(remaining_volume, open_trade["volume"])

                if open_trade["direction"] == Direction.LONG:
                    gross_pnl: float = (trade.price - open_trade["price"]) * volume * size
                else:
                    gross_pnl = (open_trade["price"] - trade.price) * volume * size

                cost: float = (open_trade["cost_per_volume"] + trade_cost_per_volume) * volume
                closed_pnl += gross_pnl - cost
                closed_volume += volume

                remaining_volume -= volume
                open_trade["volume"] -= volume

                if not open_trade["volume"]:
                    queue.popleft()

            if closed_volume:
                closed_pnls.append(closed_pnl)
                total_closed_volume += closed_volume

            if remaining_volume:
                queue.append({
                    "direction": trade.direction,
                    "price": trade.price,
                    "volume": remaining_volume,
                    "cost_per_volume": trade_cost_per_volume,
                })

        total_closed_trade_count: int = len(closed_pnls)
        profit_pnls: list[float] = [pnl for pnl in closed_pnls if pnl > 0]
        loss_pnls: list[float] = [pnl for pnl in closed_pnls if pnl < 0]

        profit_trade_count: int = len(profit_pnls)
        loss_trade_count: int = len(loss_pnls)
        average_profit: float = sum(profit_pnls) / profit_trade_count if profit_trade_count else 0
        average_loss: float = abs(sum(loss_pnls) / loss_trade_count) if loss_trade_count else 0

        if total_closed_trade_count:
            win_rate: float = profit_trade_count / total_closed_trade_count
        else:
            win_rate = 0

        if average_loss:
            profit_loss_ratio: float = average_profit / average_loss
        else:
            profit_loss_ratio = 0

        total_net_profit: float = sum(closed_pnls)
        if total_closed_volume:
            average_net_profit_per_volume: float = total_net_profit / total_closed_volume
        else:
            average_net_profit_per_volume = 0

        return {
            "total_closed_trade_count": total_closed_trade_count,
            "profit_trade_count": profit_trade_count,
            "loss_trade_count": loss_trade_count,
            "win_rate": win_rate,
            "average_profit": average_profit,
            "average_loss": average_loss,
            "profit_loss_ratio": profit_loss_ratio,
            "average_net_profit_per_volume": average_net_profit_per_volume,
        }

    def calculate_benchmark_curve(self, benchmark_symbol: str, df: DataFrame = None) -> DataFrame | None:
        """计算基准归一化资金曲线"""
        if df is None:
            df = self.daily_df

        if df is None:
            return None

        benchmark_bars: list[BarData] = load_bar_data(
            benchmark_symbol,
            self.interval,
            self.start,
            self.end
        )
        benchmark_closes: dict[date, float] = {
            bar.datetime.date(): bar.close_price
            for bar in benchmark_bars
            if bar.close_price > 0
        }

        if not benchmark_closes:
            self.output(_("基准{}历史数据为空，无法绘制基准曲线").format(benchmark_symbol))
            return None

        closes: list[float] = []
        last_close: float | None = None
        for ix in df.index:
            d: date = ix.date() if isinstance(ix, datetime) else ix
            close: float | None = benchmark_closes.get(d, None)
            if close:
                last_close = close
            closes.append(last_close if last_close else np.nan)

        benchmark_df: DataFrame = DataFrame(index=df.index)
        benchmark_df["benchmark_close"] = closes

        valid_closes = benchmark_df["benchmark_close"].dropna()
        if valid_closes.empty:
            self.output(_("基准{}历史数据无法和回测日期对齐，无法绘制基准曲线").format(benchmark_symbol))
            return None

        first_valid_close: float = valid_closes.iloc[0]
        benchmark_df["benchmark_balance"] = benchmark_df["benchmark_close"] / first_valid_close * self.capital

        return benchmark_df

    def show_chart(self, df: DataFrame = None, benchmark_symbol: str | None = None) -> None:
        """显示图表"""
        if df is None:
            df = self.daily_df

        if df is None:
            return

        benchmark_df: DataFrame | None = None
        if benchmark_symbol:
            benchmark_df = self.calculate_benchmark_curve(benchmark_symbol, df)

        fig = make_subplots(
            rows=4,
            cols=1,
            subplot_titles=["Balance", "Drawdown", "Daily Pnl", "Pnl Distribution"],
            vertical_spacing=0.06
        )

        balance_line = go.Scatter(
            x=df.index,
            y=df["balance"],
            mode="lines",
            name="Balance"
        )
        if benchmark_df is not None:
            benchmark_line = go.Scatter(
                x=benchmark_df.index,
                y=benchmark_df["benchmark_balance"],
                mode="lines",
                name=benchmark_symbol
            )
        drawdown_scatter = go.Scatter(
            x=df.index,
            y=df["drawdown"],
            fillcolor="red",
            fill='tozeroy',
            mode="lines",
            name="Drawdown"
        )
        pnl_bar = go.Bar(y=df["net_pnl"], name="Daily Pnl")
        pnl_histogram = go.Histogram(x=df["net_pnl"], nbinsx=100, name="Days")

        fig.add_trace(balance_line, row=1, col=1)
        if benchmark_df is not None:
            fig.add_trace(benchmark_line, row=1, col=1)
        fig.add_trace(drawdown_scatter, row=2, col=1)
        fig.add_trace(pnl_bar, row=3, col=1)
        fig.add_trace(pnl_histogram, row=4, col=1)

        fig.update_layout(height=1000, width=1000)
        fig.show()

    def run_bf_optimization(
        self,
        optimization_setting: OptimizationSetting,
        output: bool = True,
        max_workers: int | None = None
    ) -> list:
        """暴力穷举优化"""
        if not check_optimization_setting(optimization_setting):
            return []

        evaluate_func: Callable = wrap_evaluate(self, optimization_setting.target_name)
        results: list = run_bf_optimization(
            evaluate_func,
            optimization_setting,
            get_target_value,
            max_workers=max_workers,
            output=self.output,
        )

        if output:
            for result in results:
                msg: str = _("参数：{}, 目标：{}").format(result[0], result[1])
                self.output(msg)

        return results

    run_optimization = run_bf_optimization

    def run_ga_optimization(
        self,
        optimization_setting: OptimizationSetting,
        max_workers: int | None = None,
        ngen: int = 30,
        output: bool = True
    ) -> list:
        """遗传算法优化"""
        if not check_optimization_setting(optimization_setting):
            return []

        evaluate_func: Callable = wrap_evaluate(self, optimization_setting.target_name)
        results: list = run_ga_optimization(
            evaluate_func,
            optimization_setting,
            get_target_value,
            max_workers=max_workers,
            ngen=ngen,
            output=self.output
        )

        if output:
            for result in results:
                msg: str = _("参数：{}, 目标：{}").format(result[0], result[1])
                self.output(msg)

        return results

    def update_daily_close(self, bars: dict[str, BarData], dt: datetime) -> None:
        """更新每日收盘价"""
        d: date = dt.date()

        close_prices: dict = {}
        for bar in bars.values():
            close_prices[bar.vt_symbol] = bar.close_price

        daily_result: PortfolioDailyResult | None = self.daily_results.get(d, None)

        if daily_result:
            daily_result.update_close_prices(close_prices)
        else:
            self.daily_results[d] = PortfolioDailyResult(d, close_prices)

    def new_bars(self, dt: datetime) -> None:
        """历史数据推送"""
        self.datetime = dt

        bars: dict[str, BarData] = {}
        for vt_symbol in self.vt_symbols:
            bar: BarData | None = self.history_data.get((dt, vt_symbol), None)

            # 判断是否获取到该合约指定时间的历史数据
            if bar:
                # 更新K线以供委托撮合
                self.bars[vt_symbol] = bar
                # 缓存K线数据以供strategy.on_bars更新
                bars[vt_symbol] = bar
            # 如果获取不到，但self.bars字典中已有合约数据缓存, 使用之前的数据填充
            elif vt_symbol in self.bars:
                old_bar: BarData = self.bars[vt_symbol]

                bar = BarData(
                    symbol=old_bar.symbol,
                    exchange=old_bar.exchange,
                    datetime=dt,
                    open_price=old_bar.close_price,
                    high_price=old_bar.close_price,
                    low_price=old_bar.close_price,
                    close_price=old_bar.close_price,
                    gateway_name=old_bar.gateway_name
                )
                self.bars[vt_symbol] = bar

        self.cross_limit_order()
        self.strategy.on_bars(bars)

        if self.strategy.inited:
            self.update_daily_close(self.bars, dt)

    def cross_limit_order(self) -> None:
        """撮合限价委托"""
        for order in list(self.active_limit_orders.values()):
            bar: BarData = self.bars[order.vt_symbol]

            long_cross_price: float = bar.low_price
            short_cross_price: float = bar.high_price
            long_best_price: float = bar.open_price
            short_best_price: float = bar.open_price

            # 推送委托未成交状态更新
            if order.status == Status.SUBMITTING:
                order.status = Status.NOTTRADED
                self.strategy.update_order(order)

            # 检查可以被撮合的限价委托
            long_cross: bool = (
                order.direction == Direction.LONG
                and order.price >= long_cross_price
                and long_cross_price > 0
            )

            short_cross: bool = (
                order.direction == Direction.SHORT
                and order.price <= short_cross_price
                and short_cross_price > 0
            )

            if not long_cross and not short_cross:
                continue

            # 推送委托成交状态更新
            order.traded = order.volume
            order.status = Status.ALLTRADED
            self.strategy.update_order(order)

            if order.vt_orderid in self.active_limit_orders:
                self.active_limit_orders.pop(order.vt_orderid)

            # 推送成交信息
            self.trade_count += 1

            if long_cross:
                trade_price = min(order.price, long_best_price)
            else:
                trade_price = max(order.price, short_best_price)

            trade: TradeData = TradeData(
                symbol=order.symbol,
                exchange=order.exchange,
                orderid=order.orderid,
                tradeid=str(self.trade_count),
                direction=order.direction,
                offset=order.offset,
                price=trade_price,
                volume=order.volume,
                datetime=self.datetime,
                gateway_name=self.gateway_name,
            )

            size: float = self.sizes[trade.vt_symbol]
            turnover: float = trade.price * trade.volume * size
            commission: float = turnover * self.rates[trade.vt_symbol]
            slippage: float = trade.volume * size * self.slippages[trade.vt_symbol]

            if trade.direction == Direction.LONG:
                self.cash -= turnover
            else:
                self.cash += turnover

            self.cash -= commission + slippage

            self.strategy.update_trade(trade)
            self.trades[trade.vt_tradeid] = trade

    def load_bars(
        self,
        strategy: StrategyTemplate,
        days: int,
        interval: Interval
    ) -> None:
        """加载历史数据"""
        self.days = days

    def send_order(
        self,
        strategy: StrategyTemplate,
        vt_symbol: str,
        direction: Direction,
        offset: Offset,
        price: float,
        volume: float,
        lock: bool,
        net: bool
    ) -> list[str]:
        """发送委托"""
        price = round_to(price, self.priceticks[vt_symbol])
        symbol, exchange = extract_vt_symbol(vt_symbol)

        self.limit_order_count += 1

        order: OrderData = OrderData(
            symbol=symbol,
            exchange=exchange,
            orderid=str(self.limit_order_count),
            direction=direction,
            offset=offset,
            price=price,
            volume=volume,
            status=Status.SUBMITTING,
            datetime=self.datetime,
            gateway_name=self.gateway_name,
        )

        self.active_limit_orders[order.vt_orderid] = order
        self.limit_orders[order.vt_orderid] = order

        return [order.vt_orderid]

    def cancel_order(self, strategy: StrategyTemplate, vt_orderid: str) -> None:
        """委托撤单"""
        if vt_orderid not in self.active_limit_orders:
            return
        order: OrderData = self.active_limit_orders.pop(vt_orderid)

        order.status = Status.CANCELLED
        self.strategy.update_order(order)

    def write_log(self, msg: str, strategy: StrategyTemplate | None = None) -> None:
        """输出日志"""
        msg = f"{self.datetime}\t{msg}"
        self.logs.append(msg)

    def send_email(self, msg: str, strategy: StrategyTemplate | None = None) -> None:
        """发送邮件"""
        pass

    def sync_strategy_data(self, strategy: StrategyTemplate) -> None:
        """保存策略数据到文件"""
        pass

    def get_engine_type(self) -> EngineType:
        """获取引擎类型"""
        return self.engine_type

    def get_pricetick(self, strategy: StrategyTemplate, vt_symbol: str) -> float:
        """获取合约价格跳动"""
        return self.priceticks[vt_symbol]

    def get_size(self, strategy: StrategyTemplate, vt_symbol: str) -> float:
        """获取合约乘数"""
        return self.sizes[vt_symbol]

    def get_cash_available(self, strategy: StrategyTemplate | None = None) -> float:
        """获取当前可用现金"""
        return self.cash

    def get_cash(self, strategy: StrategyTemplate | None = None) -> float:
        """获取当前可用现金"""
        return self.get_cash_available()

    def get_holding_value(self, strategy: StrategyTemplate | None = None) -> float:
        """获取当前持仓市值"""
        holding_value: float = 0

        strategy = strategy or self.strategy

        for vt_symbol, pos in strategy.pos_data.items():
            if not pos:
                continue

            bar: BarData | None = self.bars.get(vt_symbol)
            if not bar:
                continue

            size: float = self.sizes[vt_symbol]
            holding_value += bar.close_price * pos * size

        return holding_value

    def get_portfolio_value(self, strategy: StrategyTemplate | None = None) -> float:
        """获取当前组合权益"""
        return self.cash + self.get_holding_value(strategy)

    def put_strategy_event(self, strategy: StrategyTemplate) -> None:
        """推送事件更新策略界面"""
        pass

    def output(self, msg: str) -> None:
        """输出回测引擎信息"""
        print(f"{datetime.now()}\t{msg}")

    def get_all_trades(self) -> list[TradeData]:
        """获取所有成交信息"""
        return list(self.trades.values())

    def get_all_orders(self) -> list[OrderData]:
        """获取所有委托信息"""
        return list(self.limit_orders.values())

    def get_all_daily_results(self) -> list["PortfolioDailyResult"]:
        """获取所有每日盈亏信息"""
        return list(self.daily_results.values())


class ContractDailyResult:
    """合约每日盈亏结果"""

    def __init__(self, result_date: date, close_price: float) -> None:
        """构造函数"""
        self.date: date = result_date
        self.close_price: float = close_price
        self.pre_close: float = 0

        self.trades: list[TradeData] = []
        self.trade_count: int = 0

        self.start_pos: float = 0
        self.end_pos: float = 0

        self.turnover: float = 0
        self.commission: float = 0
        self.slippage: float = 0

        self.trading_pnl: float = 0
        self.holding_pnl: float = 0
        self.total_pnl: float = 0
        self.net_pnl: float = 0

    def add_trade(self, trade: TradeData) -> None:
        """添加成交信息"""
        self.trades.append(trade)

    def calculate_pnl(
        self,
        pre_close: float,
        start_pos: float,
        size: float,
        rate: float,
        slippage: float
    ) -> None:
        """计算盈亏"""
        # 记录昨收盘价
        self.pre_close = pre_close

        # 计算持仓盈亏
        self.start_pos = start_pos
        self.end_pos = start_pos

        self.holding_pnl = self.start_pos * (self.close_price - self.pre_close) * size

        # 计算交易盈亏
        self.trade_count = len(self.trades)

        for trade in self.trades:
            if trade.direction == Direction.LONG:
                pos_change = trade.volume
            else:
                pos_change = -trade.volume

            self.end_pos += pos_change

            turnover: float = trade.volume * size * trade.price

            self.trading_pnl += pos_change * (self.close_price - trade.price) * size
            self.slippage += trade.volume * size * slippage
            self.turnover += turnover
            self.commission += turnover * rate

        # 计算每日盈亏
        self.total_pnl = self.trading_pnl + self.holding_pnl
        self.net_pnl = self.total_pnl - self.commission - self.slippage

    def update_close_price(self, close_price: float) -> None:
        """更新每日收盘价"""
        self.close_price = close_price


class PortfolioDailyResult:
    """组合每日盈亏结果"""

    def __init__(self, result_date: date, close_prices: dict[str, float]) -> None:
        """"""
        self.date: date = result_date
        self.close_prices: dict[str, float] = close_prices
        self.pre_closes: dict[str, float] = {}
        self.start_poses: dict[str, float] = {}
        self.end_poses: dict[str, float] = {}

        self.contract_results: dict[str, ContractDailyResult] = {}

        for vt_symbol, close_price in close_prices.items():
            self.contract_results[vt_symbol] = ContractDailyResult(result_date, close_price)

        self.trade_count: int = 0
        self.turnover: float = 0
        self.commission: float = 0
        self.slippage: float = 0
        self.trading_pnl: float = 0
        self.holding_pnl: float = 0
        self.total_pnl: float = 0
        self.net_pnl: float = 0

    def add_trade(self, trade: TradeData) -> None:
        """添加成交信息"""
        contract_result: ContractDailyResult = self.contract_results[trade.vt_symbol]
        contract_result.add_trade(trade)

    def calculate_pnl(
        self,
        pre_closes: dict[str, float],
        start_poses: dict[str, float],
        sizes: dict[str, float],
        rates: dict[str, float],
        slippages: dict[str, float],
    ) -> None:
        """计算盈亏"""
        self.pre_closes = pre_closes
        self.start_poses = start_poses

        for vt_symbol, contract_result in self.contract_results.items():
            contract_result.calculate_pnl(
                pre_closes.get(vt_symbol, 0),
                start_poses.get(vt_symbol, 0),
                sizes[vt_symbol],
                rates[vt_symbol],
                slippages[vt_symbol]
            )

            self.trade_count += contract_result.trade_count
            self.turnover += contract_result.turnover
            self.commission += contract_result.commission
            self.slippage += contract_result.slippage
            self.trading_pnl += contract_result.trading_pnl
            self.holding_pnl += contract_result.holding_pnl
            self.total_pnl += contract_result.total_pnl
            self.net_pnl += contract_result.net_pnl

            self.end_poses[vt_symbol] = contract_result.end_pos

    def update_close_prices(self, close_prices: dict[str, float]) -> None:
        """更新每日收盘价"""
        self.close_prices.update(close_prices)

        for vt_symbol, close_price in close_prices.items():
            contract_result: ContractDailyResult | None = self.contract_results.get(vt_symbol, None)
            if contract_result:
                contract_result.update_close_price(close_price)
            else:
                self.contract_results[vt_symbol] = ContractDailyResult(self.date, close_price)


@lru_cache(maxsize=999)
def load_bar_data(
    vt_symbol: str,
    interval: Interval,
    start: datetime,
    end: datetime
) -> list[BarData]:
    """通过数据库获取历史数据"""
    symbol, exchange = extract_vt_symbol(vt_symbol)

    database: BaseDatabase = get_database()

    bars: list[BarData] = database.load_bar_data(
        symbol, exchange, interval, start, end
    )

    return bars


def evaluate(
    target_name: str,
    strategy_class: type[StrategyTemplate],
    vt_symbols: list[str],
    interval: Interval,
    start: datetime,
    rates: dict[str, float],
    slippages: dict[str, float],
    sizes: dict[str, float],
    priceticks: dict[str, float],
    capital: float,
    end: datetime,
    setting: dict
) -> tuple:
    """包装回测相关函数以供进程池内运行"""
    engine: BacktestingEngine = BacktestingEngine()

    engine.set_parameters(
        vt_symbols=vt_symbols,
        interval=interval,
        start=start,
        rates=rates,
        slippages=slippages,
        sizes=sizes,
        priceticks=priceticks,
        capital=capital,
        end=end,
    )

    engine.add_strategy(strategy_class, setting)
    engine.load_data()
    engine.run_backtesting()
    engine.calculate_result()
    statistics: dict = engine.calculate_statistics(output=False)

    target_value: float = statistics[target_name]
    return (str(setting), target_value, statistics)


def wrap_evaluate(engine: BacktestingEngine, target_name: str) -> Callable:
    """包装回测配置函数以供进程池内运行"""
    func: Callable = partial(
        evaluate,
        target_name,
        engine.strategy_class,
        engine.vt_symbols,
        engine.interval,
        engine.start,
        engine.rates,
        engine.slippages,
        engine.sizes,
        engine.priceticks,
        engine.capital,
        engine.end
    )
    return func


def get_target_value(result: list) -> float:
    """获取优化目标"""
    target_value: float = result[1]
    return target_value
