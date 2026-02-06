from datetime import date, timedelta
from dateutil.relativedelta import relativedelta


class Period:
    def __init__(self, id: int, start_date: date) -> None:
        self.id = id
        self.formation_start: date = start_date
        self.formation_end: date = self.formation_start + relativedelta(
            months=12, days=-1
        )
        self.trading_start: date = self.formation_end + timedelta(days=1)
        self.trading_end: date = self.trading_start + relativedelta(months=6, days=-1)

    def __str__(self) -> str:
        return (
            f"Period {self.id}: "
            f"Formation: {self.formation_start} → {self.formation_end} | "
            f"Trading: {self.trading_start} → {self.trading_end}"
        )

    @staticmethod
    def seed_periods(num_periods: int, start_date: date = None):
        if start_date is None:
            start_date = date(2025, 1, 1)
        periods = [
            Period(i + 1, start_date + relativedelta(months=i))
            for i in range(num_periods)
        ]
        return periods
