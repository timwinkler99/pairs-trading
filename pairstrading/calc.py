from itertools import combinations
import duckdb
import polars as pl
import numpy as np


def process_period(period_data):
    """Process a single period and return pair statistics."""
    period, duckdb_path = period_data

    # Each process needs its own connection
    con = duckdb.connect(duckdb_path, read_only=True)

    # Get prices for entire period (formation + trading)
    df_period_prices = con.execute(f"""
        SELECT * FROM prices_wide
        WHERE date BETWEEN '{period.formation_start}' AND '{period.trading_end}'
    """).pl()

    con.close()

    # Drop stocks with any missing values during this period
    price_cols = df_period_prices.drop("date")

    # Keep only columns without nulls - vectorized operation
    valid_stocks = [
        col for col in price_cols.columns if price_cols[col].null_count() == 0
    ]

    if len(valid_stocks) < 2:
        return [], 0, 0

    # Filter to formation period only
    df_formation = df_period_prices.filter(
        pl.col("date") <= period.formation_end
    ).select(["date"] + valid_stocks)

    # Calculate cumulative returns more efficiently using numpy
    # Convert to numpy for faster computation
    prices = df_formation.drop("date").to_numpy()

    # Calculate returns: (price[t] - price[t-1]) / price[t-1]
    returns = np.diff(prices, axis=0) / prices[:-1]

    # Cumulative sum of returns
    cumret = np.cumsum(returns, axis=0)

    # Generate all pair combinations
    pair_combinations = list(combinations(range(len(valid_stocks)), 2))

    # Calculate statistics for all pairs vectorized
    batch_data = []
    for i, j in pair_combinations:
        spread = cumret[:, i] - cumret[:, j]
        ssd = float(np.sum(spread**2))
        std = float(np.std(spread, ddof=1))

        batch_data.append(
            {
                "period_id": period.id,
                "stock1": valid_stocks[i],
                "stock2": valid_stocks[j],
                "ssd": ssd,
                "std": std,
            }
        )

    print(
        f"Completed period {period.id}: {len(valid_stocks)} stocks, {len(pair_combinations)} pairs"
    )
    return batch_data, len(valid_stocks), len(pair_combinations)
