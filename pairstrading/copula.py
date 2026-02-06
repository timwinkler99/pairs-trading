from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Iterable

import duckdb
import numpy as np
import polars as pl
import pyvinecopulib as pv
from scipy import stats

from .calendar import Period


@dataclass(frozen=True)
class MarginalFit:
    period_id: int
    stock: str
    distribution: str
    aic: float
    params: tuple[float, ...]


@dataclass(frozen=True)
class CopulaFit:
    pair_id: int
    period_id: int
    stock1: str
    stock2: str
    name: str
    rotation: int
    par1: float
    par2: float
    loglik: float
    aic: float
    bic: float


@dataclass(frozen=True)
class CopulaTrade:
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
class CopulaMonthly:
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
        cur_end = (cur_start.replace(day=1) + timedelta(days=32)).replace(day=1) - timedelta(days=1)
        windows.append((cur_start, cur_end))
        cur_start = cur_end + timedelta(days=1)
    return windows


def _periods_from_table(con: duckdb.DuckDBPyConnection, table: str) -> dict[int, Period]:
    df = con.execute(f"SELECT id, formation_start FROM {table}").pl()
    return {row["id"]: Period(row["id"], row["formation_start"]) for row in df.iter_rows(named=True)}


def _fit_best_marginal(data: np.ndarray) -> tuple[str, tuple[float, ...], float, stats.rv_continuous]:
    candidates = [stats.t, stats.norm, stats.genlogistic, stats.genextreme]
    best_aic = np.inf
    best_name = ""
    best_params: tuple[float, ...] = ()
    best_dist = None

    for dist in candidates:
        try:
            params = dist.fit(data)
            dist_fit = dist(*params)
            log_pdf = np.log(dist_fit.pdf(data) + 1e-12)
            log_like = float(np.sum(log_pdf))
            aic = 2 * len(params) - 2 * log_like
        except Exception:
            continue

        if aic < best_aic:
            best_aic = aic
            best_name = dist.name
            best_params = tuple(float(p) for p in params)
            best_dist = dist

    if best_dist is None:
        raise RuntimeError("No marginal distribution could be fit")

    return best_name, best_params, best_aic, best_dist


def _compute_copula_trades(
    dates: list[date],
    prices: np.ndarray,
    hfunc1: np.ndarray,
    hfunc2: np.ndarray,
    trading_months: Iterable[tuple[date, date]],
):
    n = len(dates)
    if n < 2:
        return [], []

    daily_returns = (prices[1:] - prices[:-1]) / prices[:-1]
    daily_returns = np.vstack([np.zeros((1, 2)), daily_returns])

    m1 = hfunc1 - 0.5
    m2 = hfunc2 - 0.5
    M1 = np.cumsum(m1)
    M2 = np.cumsum(m2)

    divergence_lower = -0.5
    divergence_upper = 0.5
    convergence_lower = -0.1
    convergence_upper = 0.1

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
        s1_overvalued = (M1[idx] > divergence_upper) and (M2[idx] < divergence_lower)
        s2_overvalued = (M1[idx] < divergence_lower) and (M2[idx] > divergence_upper)

        convergence = False
        if trading:
            convergence = (
                convergence_lower < M1[idx] < convergence_upper
                and convergence_lower < M2[idx] < convergence_upper
            )

        if idx == n - 1 and trading:
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
        elif trading and not convergence:
            instruction = "keep"
            trade_counts.append(trade_count)
        elif trading and convergence:
            trading = False
            instruction = "close"
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


def run_copula_trading(
    duckdb_path: str,
    pairs_table: str = "pair_selection",
    prices_table: str = "prices_wide",
    periods_table: str = "periods",
    marginals_table: str = "copula_marginals",
    copula_table: str = "copula_fits",
    trades_table: str = "copula_trades",
    monthly_table: str = "copula_monthly_excess",
):
    con = duckdb.connect(duckdb_path)

    periods = _periods_from_table(con, periods_table)
    pairs = con.execute(
        f"""
        SELECT pair_id, period_id, stock1, stock2
        FROM {pairs_table}
        ORDER BY period_id, pair_id
        """
    ).fetchall()

    con.execute(f"DROP TABLE IF EXISTS {marginals_table}")
    con.execute(
        f"""
        CREATE TABLE {marginals_table} (
            period_id INTEGER,
            stock VARCHAR,
            distribution VARCHAR,
            aic DOUBLE,
            params VARCHAR
        )
        """
    )

    con.execute(f"DROP TABLE IF EXISTS {copula_table}")
    con.execute(
        f"""
        CREATE TABLE {copula_table} (
            pair_id INTEGER,
            period_id INTEGER,
            stock1 VARCHAR,
            stock2 VARCHAR,
            name VARCHAR,
            rotation INTEGER,
            par1 DOUBLE,
            par2 DOUBLE,
            loglik DOUBLE,
            aic DOUBLE,
            bic DOUBLE
        )
        """
    )

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

    marginals_batch: list[MarginalFit] = []
    copula_batch: list[CopulaFit] = []
    trades_batch: list[CopulaTrade] = []
    monthly_batch: list[CopulaMonthly] = []

    period_groups: dict[int, list[tuple[int, int, str, str]]] = {}
    for pair in pairs:
        pair_id, period_id, stock1, stock2 = pair
        period_groups.setdefault(period_id, []).append(pair)

    for period_id, period_pairs in period_groups.items():
        period = periods[period_id]

        stocks = sorted({p[2] for p in period_pairs} | {p[3] for p in period_pairs})
        if not stocks:
            continue

        cols = ", ".join([_quote_ident(s) for s in stocks])
        df = con.execute(
            f"""
            SELECT date, {cols}
            FROM {prices_table}
            WHERE date BETWEEN '{period.formation_start}' AND '{period.trading_end}'
            ORDER BY date
            """
        ).pl()

        dates = df["date"].to_list()
        price_mat = df.drop("date").to_numpy()
        daily_returns = (price_mat[1:] - price_mat[:-1]) / price_mat[:-1]
        daily_returns = np.vstack([np.zeros((1, len(stocks))), daily_returns])

        date_arr = np.array(dates, dtype="datetime64[D]")
        formation_mask = (date_arr >= np.datetime64(period.formation_start)) & (
            date_arr <= np.datetime64(period.formation_end)
        )
        trading_mask = (date_arr >= np.datetime64(period.trading_start)) & (
            date_arr <= np.datetime64(period.trading_end)
        )

        formation_returns = daily_returns[formation_mask]
        trading_returns = daily_returns[trading_mask]
        trading_dates = [d for d, m in zip(dates, trading_mask) if m]
        trading_prices = price_mat[trading_mask]

        cdf_formation: dict[str, np.ndarray] = {}
        cdf_trading: dict[str, np.ndarray] = {}

        for idx, stock in enumerate(stocks):
            data_f = formation_returns[:, idx]
            name, params, aic, dist = _fit_best_marginal(data_f)
            dist_fit = dist(*params)
            cdf_f = dist_fit.cdf(data_f)
            cdf_t = dist_fit.cdf(trading_returns[:, idx])
            cdf_f = np.clip(cdf_f, 1e-6, 1 - 1e-6)
            cdf_t = np.clip(cdf_t, 1e-6, 1 - 1e-6)

            cdf_formation[stock] = cdf_f
            cdf_trading[stock] = cdf_t

            marginals_batch.append(
                MarginalFit(
                    period_id=period_id,
                    stock=stock,
                    distribution=name,
                    aic=float(aic),
                    params=params,
                )
            )

        trading_months = _trading_month_windows(period.trading_start, months=6)

        for pair_id, _, stock1, stock2 in period_pairs:
            s1_f = cdf_formation[stock1]
            s2_f = cdf_formation[stock2]
            s1_t = cdf_trading[stock1]
            s2_t = cdf_trading[stock2]

            data_f = np.column_stack([s1_f, s2_f])
            data_t = np.column_stack([s1_t, s2_t])

            controls = pv.FitControlsBicop(
                family_set=[
                    pv.BicopFamily.student,
                    pv.BicopFamily.clayton,
                    pv.BicopFamily.gumbel,
                ],
                parametric_method="mle",
                selection_criterion="aic",
            )
            cop = pv.Bicop()
            cop.select(data=data_f, controls=controls)

            if cop.family == pv.BicopFamily.student:
                par1 = float(cop.parameters[0][0])
                par2 = float(cop.parameters[1][0])
            else:
                par1 = float(cop.parameters[0][0])
                par2 = 0.0

            # pyvinecopulib exposes these as methods (not properties)
            copula_batch.append(
                CopulaFit(
                    pair_id=pair_id,
                    period_id=period_id,
                    stock1=stock1,
                    stock2=stock2,
                    name=str(cop.family),
                    rotation=int(cop.rotation),
                    par1=par1,
                    par2=par2,
                    loglik=float(cop.loglik()),
                    aic=float(cop.aic()),
                    bic=float(cop.bic()),
                )
            )

            hfunc1 = cop.hfunc1(data_t)
            hfunc2 = cop.hfunc2(data_t)

            trades, monthly = _compute_copula_trades(
                trading_dates,
                trading_prices[:, [stocks.index(stock1), stocks.index(stock2)]],
                hfunc1,
                hfunc2,
                trading_months,
            )

            for trade in trades:
                trade_id, s1_inst, s2_inst, t_start, t_end, t_days, t_close, r1, r2, r_total = trade
                trades_batch.append(
                    CopulaTrade(
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
                    CopulaMonthly(
                        pair_id=pair_id,
                        period_id=period_id,
                        month_end=month_end,
                        excess_return=excess_return,
                        trade_indicator=trade_indicator,
                    )
                )

    if marginals_batch:
        marginals_df = pl.DataFrame(
            [
                {
                    "period_id": m.period_id,
                    "stock": m.stock,
                    "distribution": m.distribution,
                    "aic": m.aic,
                    "params": ",".join(str(p) for p in m.params),
                }
                for m in marginals_batch
            ]
        )
        con.register("marginals_pl", marginals_df)
        con.execute(f"INSERT INTO {marginals_table} SELECT * FROM marginals_pl")

    if copula_batch:
        copula_df = pl.DataFrame([c.__dict__ for c in copula_batch])
        con.register("copula_pl", copula_df)
        con.execute(f"INSERT INTO {copula_table} SELECT * FROM copula_pl")

    if trades_batch:
        trades_df = pl.DataFrame([t.__dict__ for t in trades_batch])
        con.register("trades_pl", trades_df)
        con.execute(f"INSERT INTO {trades_table} SELECT * FROM trades_pl")

    if monthly_batch:
        monthly_df = pl.DataFrame([m.__dict__ for m in monthly_batch])
        con.register("monthly_pl", monthly_df)
        con.execute(f"INSERT INTO {monthly_table} SELECT * FROM monthly_pl")

    con.close()
