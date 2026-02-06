import duckdb


def select_top_pairs(
    duckdb_path: str,
    lower: int = 0,
    upper: int = 20,
    input_table: str = "pair_stats",
    output_table: str = "pair_selection",
):
    if upper <= lower:
        raise ValueError("upper must be greater than lower")
    if lower < 0:
        raise ValueError("lower must be >= 0")

    con = duckdb.connect(duckdb_path)

    con.execute(f"DROP TABLE IF EXISTS {output_table}")
    con.execute(
        f"""
        CREATE TABLE {output_table} AS
        WITH ranked AS (
            SELECT
                *,
                row_number() OVER (PARTITION BY period_id ORDER BY ssd) AS rn
            FROM {input_table}
        ),
        filtered AS (
            SELECT *
            FROM ranked
            WHERE rn > {lower} AND rn <= {upper}
        )
        SELECT
            row_number() OVER (ORDER BY period_id, rn, stock1, stock2) - 1 AS pair_id,
            period_id,
            stock1,
            stock2,
            ssd,
            std,
            rn
        FROM filtered
        """
    )

    df = con.execute(f"SELECT * FROM {output_table}").pl()
    con.close()
    return df
