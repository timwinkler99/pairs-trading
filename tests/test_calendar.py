from datetime import date
from pairstrading.calendar import Period


class TestPeriod:
    """Test suite for the Period class."""

    def test_period_creation(self):
        """Test basic period creation."""
        p = Period(1, date(2005, 1, 1))

        assert p.id == 1
        assert p.formation_start == date(2005, 1, 1)
        assert p.formation_end == date(2006, 1, 1)
        assert p.trading_start == date(2006, 1, 2)
        assert p.trading_end == date(2006, 7, 1)

    def test_formation_period_duration(self):
        """Test that formation period is exactly 12 months."""
        p = Period(1, date(2005, 1, 1))

        # Formation period should be 12 months
        assert p.formation_end == date(2006, 1, 1)

    def test_trading_period_duration(self):
        """Test that trading period is 6 months after formation."""
        p = Period(1, date(2005, 1, 1))

        # Trading should start 1 day after formation ends
        assert p.trading_start == date(2006, 1, 2)
        # Trading should end 6 months after formation ends
        assert p.trading_end == date(2006, 7, 1)

    def test_multiple_periods(self):
        """Test creating multiple consecutive periods."""
        p1 = Period(1, date(2005, 1, 1))
        p2 = Period(2, date(2005, 2, 1))

        assert p1.id == 1
        assert p2.id == 2
        assert p2.formation_start > p1.formation_start

    def test_period_str_representation(self):
        """Test string representation of Period."""
        p = Period(1, date(2005, 1, 1))
        result = str(p)

        assert "Period 1" in result
        assert "Formation:" in result
        assert "Trading:" in result
        assert "2005-01-01" in result
        assert "2006-01-01" in result

    def test_leap_year_handling(self):
        """Test period calculation across leap year."""
        p = Period(1, date(2004, 2, 29))  # Leap year

        # Should handle leap year correctly
        assert p.formation_start == date(2004, 2, 29)
        assert p.formation_end == date(2005, 2, 28)  # No Feb 29 in 2005
