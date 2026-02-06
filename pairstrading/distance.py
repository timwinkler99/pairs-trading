from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Iterable

import duckdb
import numpy as np
import polars as pl

from .calendar import Period


@dataclass(frozen=True)
class DistanceTrade:
    pair_id: int
    period_id: int
    trade: int
    stock1: str
    stock2: str
    s1_instruction: int
    s2_instruction: int
    trade_start: date
    trade_end: date
    trading_days: int
    trade_close_type: str
    return_s1: float
    return_s2: float
    return_total: float


@dataclass(frozen=True)
class DistanceMonthly:
    pair_id: int
    period_id: int
    month_end: date
    excess_return: float
    trade_indicator: int


def _quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _trading_month_windows(start: date, months: int = 6) -> list[tuple[date, date]]:
    windows: list[tuple[date, date]] = []
    cur_start = start
    for _ in range(months):
        # month end is one day before next month start
        cur_end = (cur_start.replace(day=1) + timedelta(days=32)).replace(day=1) - timedelta(days=1)
        windows.append((cur_start, cur_end))
        cur_start = cur_end + timedelta(days=1)
    return windows


def _periods_from_table(con: duckdb.DuckDBPyConnection, table: str) -> dict[int, Period]:
    df = con.execute(f"SELECT id, formation_start FROM {table}").pl()
    return {row["id"]: Period(row["id"], row["formation_start"]) for row in df.iter_rows(named=True)}


def _compute_distance_for_pair(
    dates: list[date],
    prices: np.ndarray,
    std: float,
    trading_months: Iterable[tuple[date, date]],
):
    n = len(dates)
    if n < 2:
        return [], []

    daily_returns = (prices[1:] - prices[:-1]) / prices[:-1]
    daily_returns = np.vstack([np.zeros((1, 2)), daily_returns])
    cumret = np.cumsum(daily_returns, axis=0)
    spread = cumret[:, 0] - cumret[:, 1]

    convergence_threshold = 0.2 * std
    divergence_threshold = 2.0 * std

    trading = False
    s1_instruction = 0
    s2_instruction = 0
    trade_indicator = 0
    trade_count = 0

    trade_instructions: list[str] = []
    trade_indicators: list[int] = []
    s1_instructions: list[int] = []
    s2_instructions: list[int] = []
    trade_counts: list[int] = []

    for idx in range(n):
        s1_overvalued = spread[idx] >= divergence_threshold
        s2_overvalued = spread[idx] <= -divergence_threshold
        convergence = False
        if trading:
            convergence = -convergence_threshold <= spread[idx] <= convergence_threshold

        if idx == 0:
            s1_instruction = 0
            s2_instruction = 0
            trade_indicator = 0
            instruction = "nothing"
            trade_counts.append(0)
        elif idx == n - 1 and trading:
            instruction = "force"
            trade_counts.append(trade_count)
        elif not trading and s1_overvalued:
            trading = True
            s1_instruction = -1
            s2_instruction = 1
            trade_indicator = 1
            instruction = "open"
            trade_count += 1
            trade_counts.append(trade_count)
        elif not trading and s2_overvalued:
            trading = True
            s1_instruction = 1
            s2_instruction = -1
            trade_indicator = 1
            instruction = "open"
            trade_count += 1
            trade_counts.append(trade_count)
        elif trading and convergence:
            trading = False
            instruction = "close"
            trade_counts.append(trade_count)
        elif trading and not convergence:
            instruction = "keep"
            trade_counts.append(trade_count)
        else:
            s1_instruction = 0
            s2_instruction = 0
            trade_indicator = 0
            instruction = "nothing"
            trade_counts.append(0)

        if instruction in {"open", "keep", "close", "force"}:
            trade_indicator = 1
        elif instruction == "nothing":
            trade_indicator = 0

        trade_instructions.append(instruction)
        trade_indicators.append(trade_indicator)
        s1_instructions.append(s1_instruction)
        s2_instructions.append(s2_instruction)

    daily_plus1 = daily_returns + 1.0
    weights = np.ones_like(daily_plus1)
    weights[1:, 0] = np.cumprod(daily_plus1[:-1, 0])
    weights[1:, 1] = np.cumprod(daily_plus1[:-1, 1])

    trade_indicators_arr = np.asarray(trade_indicators, dtype=float)
    excess_daily = (
        daily_returns[:, 0] * weights[:, 0] * trade_indicators_arr
        + daily_returns[:, 1] * weights[:, 1] * trade_indicators_arr
    ) / (weights[:, 0] + weights[:, 1])
    excess_daily_plus1 = excess_daily + 1.0

    monthly_results: list[tuple[date, float, int]] = []
    date_arr = np.array(dates, dtype="datetime64[D]")
    for start, end in trading_months:
        mask = (date_arr >= np.datetime64(start)) & (date_arr <= np.datetime64(end))
        if not mask.any():
            monthly_excess = 0.0
            month_trade_indicator = 0
        else:
            monthly_excess = float(np.prod(excess_daily_plus1[mask]) - 1.0)
            month_trade_indicator = int(trade_indicators_arr[mask].sum() != 0)
        monthly_results.append((end, monthly_excess, month_trade_indicator))

    trade_events = [
        (idx, instr)
        for idx, instr in enumerate(trade_instructions)
        if instr in {"open", "close", "force"}
    ]

    trades: list[tuple[int, int, str, str, date, date, int, str, float, float, float]] = []
    for i in range(0, len(trade_events), 2):
        if i + 1 >= len(trade_events):
            break
        open_idx = trade_events[i][0]
        close_idx = trade_events[i + 1][0]
        trade_close_type = trade_events[i + 1][1]
        trading_days = close_idx - open_idx + 1

        s1_inst = s1_instructions[open_idx]
        s2_inst = s2_instructions[open_idx]

        p1_start = prices[open_idx, 0]
        p1_end = prices[close_idx, 0]
        p2_start = prices[open_idx, 1]
        p2_end = prices[close_idx, 1]

        if s1_inst == 1:
            return_s1 = (p1_end - p1_start) / p1_start
            return_s2 = (p2_start - p2_end) / p2_start
        else:
            return_s1 = (p1_start - p1_end) / p1_start
            return_s2 = (p2_end - p2_start) / p2_start
        return_total = return_s1 + return_s2

        trades.append(
            (
                trade_counts[open_idx],
                s1_inst,
                s2_inst,
                dates[open_idx],
                dates[close_idx],
                trading_days,
                trade_close_type,
                float(return_s1),
                float(return_s2),
                float(return_total),
            )
        )

    return trades, monthly_results


def run_distance_trading(
    duckdb_path: str,
    pairs_table: str = "pair_selection",
    prices_table: str = "prices_wide",
    periods_table: str = "periods",
    trades_table: str = "distance_trades",
    monthly_table: str = "distance_monthly_excess",
):
    con = duckdb.connect(duckdb_path)

    periods = _periods_from_table(con, periods_table)
    pairs = con.execute(
        f"""
        SELECT pair_id, period_id, stock1, stock2, std
        FROM {pairs_table}
        ORDER BY pair_id
        """
    ).fetchall()

    con.execute(f"DROP TABLE IF EXISTS {trades_table}")
    con.execute(
        f"""
        CREATE TABLE {trades_table} (
            pair_id INTEGER,
            period_id INTEGER,
            trade INTEGER,
            stock1 VARCHAR,
            stock2 VARCHAR,
            s1_instruction INTEGER,
            s2_instruction INTEGER,
            trade_start DATE,
            trade_end DATE,
            trading_days INTEGER,
            trade_close_type VARCHAR,
            return_s1 DOUBLE,
            return_s2 DOUBLE,
            return_total DOUBLE
        )
        """
    )

    con.execute(f"DROP TABLE IF EXISTS {monthly_table}")
    con.execute(
        f"""
        CREATE TABLE {monthly_table} (
            pair_id INTEGER,
            period_id INTEGER,
            month_end DATE,
            excess_return DOUBLE,
            trade_indicator INTEGER
        )
        """
    )

    trades_batch: list[DistanceTrade] = []
    monthly_batch: list[DistanceMonthly] = []

    for pair_id, period_id, stock1, stock2, std in pairs:
        period = periods[period_id]
        trading_months = _trading_month_windows(period.trading_start, months=6)

        s1 = _quote_ident(stock1)
        s2 = _quote_ident(stock2)
        df = con.execute(
            f"""
            SELECT date, {s1} AS s1, {s2} AS s2
            FROM {prices_table}
            WHERE date BETWEEN '{period.trading_start}' AND '{period.trading_end}'
            ORDER BY date
            """
        ).pl()

        dates = df["date"].to_list()
        prices = df[["s1", "s2"]].to_numpy()

        trades, monthly = _compute_distance_for_pair(
            dates,
            prices,
            float(std),
            trading_months,
        )

        for trade in trades:
            trade_id, s1_inst, s2_inst, t_start, t_end, t_days, t_close, r1, r2, r_total = trade
            trades_batch.append(
                DistanceTrade(
                    pair_id=pair_id,
                    period_id=period_id,
                    trade=trade_id,
                    stock1=stock1,
                    stock2=stock2,
                    s1_instruction=s1_inst,
                    s2_instruction=s2_inst,
                    trade_start=t_start,
                    trade_end=t_end,
                    trading_days=t_days,
                    trade_close_type=t_close,
                    return_s1=r1,
                    return_s2=r2,
                    return_total=r_total,
                )
            )

        for month_end, excess_return, trade_indicator in monthly:
            monthly_batch.append(
                DistanceMonthly(
                    pair_id=pair_id,
                    period_id=period_id,
                    month_end=month_end,
                    excess_return=excess_return,
                    trade_indicator=trade_indicator,
                )
            )

    if trades_batch:
        trades_df = pl.DataFrame([t.__dict__ for t in trades_batch])
        con.register("trades_pl", trades_df)
        con.execute(f"INSERT INTO {trades_table} SELECT * FROM trades_pl")

    if monthly_batch:
        monthly_df = pl.DataFrame([m.__dict__ for m in monthly_batch])
        con.register("monthly_pl", monthly_df)
        con.execute(f"INSERT INTO {monthly_table} SELECT * FROM monthly_pl")

    con.close()
