import unittest
from command_line import cmd_handler
from datetime import datetime as dt


class TestCMD(unittest.TestCase):
    def setUp(self):
        self.ob = cmd_handler()

    def test_validate_ticker(self):
        self.assertEqual(self.ob.validate_ticker('AaPl'), "AAPL")
        self.assertIsNone(self.ob.validate_ticker('Google'))
        self.assertIsNone(self.ob.validate_ticker(''))

    def test_validate_date(self):
        # Correct entry-1
        dt_inp = "2020-01-01"
        dt_exp = dt.strptime(dt_inp, "%Y-%m-%d")
        self.assertEqual(self.ob.validate_date(dt_inp), dt_exp)

        # Correct entry-2 : extra spaces are allowed
        dt_inp = " 2020-01-01   "
        dt_exp = dt.strptime("2020-01-01", "%Y-%m-%d")
        self.assertEqual(self.ob.validate_date(dt_inp), dt_exp)

        # Invalid entry - out of range entry for year, month or day
        dt_inp = "2020-13-01"
        self.assertIsNone(self.ob.validate_date(dt_inp))

        # Invalid entry - User mistakes is the length of year, month or day
        dt_inp = "2020-01-011"
        self.assertIsNone(self.ob.validate_date(dt_inp))

        # Invalid entry - Input contains non-numeric characters
        dt_inp = "2020-Jan-01"
        self.assertIsNone(self.ob.validate_date(dt_inp))

        # Invalid entry - Space separator instead of hyphen
        dt_inp = "2020 13 01"
        self.assertIsNone(self.ob.validate_date(dt_inp))

    def test_check_hist_start(self):
        # Valid entry
        inp = "2020-01-01"
        dt_inp = dt.strptime(inp, "%Y-%m-%d")
        self.assertEqual(self.ob.check_hist_start(dt_inp), True)

        # Invalid entry - User entered a date before 2005-JAN-01 : Only later dates supported
        inp = "2004-01-01"
        dt_inp = dt.strptime(inp, "%Y-%m-%d")
        self.assertEqual(self.ob.check_hist_start(dt_inp), False)

        # Invalid entry - User entered a future date
        inp = "2021-01-01"
        dt_inp = dt.strptime(inp, "%Y-%m-%d")
        self.assertEqual(self.ob.check_hist_start(dt_inp), False)

    def test_check_hist_end(self):
        # Valid entry
        inp = "2020-01-01"
        dt_inp = dt.strptime(inp, "%Y-%m-%d")
        self.assertEqual(self.ob.check_hist_start(dt_inp), True)

    def test_check_indicators(self):
        # valid entry
        inp = "linear"
        self.assertEqual(self.ob.check_indicators(inp), ['linear'])

        # invalid entry - not a single valid indicator typed
        inp = 'trendline moving average'
        self.assertIsNone(self.ob.check_indicators(inp))


if __name__ == "__main__":
    unittest.main()
