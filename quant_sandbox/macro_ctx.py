import polars as pl
from datetime import date
from dateutil.relativedelta import relativedelta

from quant_sandbox import macro_df
from quant_sandbox import utils

_today = date.today()


def get_macro_context(target_date: date = _today):
    """Returns approximate macroeconomic context, given a target date."""

    # Data about any period is released in the subsequent period. Given a date, if a data point has not been
    # released yet, we would only know the data point from before the start of the previous period.
    start_of_month = (target_date - relativedelta(day=1))
    start_of_previous_month = (target_date - relativedelta(months=1, day=1))

    month_ctx = utils.CalendarMonthContext.from_date(target_date)

    # Consumer sentiment
    print(macro_df.UMCSENT.filter(pl.col("observation_date") < (
        # Typically released on 4th Friday of month
        start_of_previous_month if target_date < month_ctx.fourth_friday else start_of_month
    )).tail(1))

    # GDP
    print(macro_df.GDPC1.filter(pl.col("observation_date") < target_date).tail(1))
    print(macro_df.PCE.filter(pl.col("observation_date") < target_date).tail(1))

    # Inflation
    print(macro_df.CORESTICKM159SFRBATL.filter(pl.col("observation_date") < (
        # Typically released on 2nd Tuesday or Wednesday of month
        start_of_previous_month if target_date < month_ctx.second_wednesday else start_of_month
    )).tail(1))
    print(macro_df.MICH.filter(pl.col("observation_date") < (
        # Typically released on 4th Friday of month
        start_of_previous_month if target_date < month_ctx.fourth_friday else start_of_month
    )).tail(1))
    print(macro_df.DPCCRV1Q225SBEA.filter(pl.col("observation_date") < target_date).tail(1))

    # Interest rates
    print(macro_df.FEDFUNDS.filter(pl.col("observation_date") < target_date).tail(1))

    # Money supply
    print(macro_df.M2REAL.filter(pl.col("observation_date") < target_date).tail(1))

    # Unemployment
    print(macro_df.UNRATE.filter(pl.col("observation_date") < (
        # Typically released on 1st Friday of month
        start_of_previous_month if target_date < month_ctx.first_friday else start_of_month
    )).tail(1))


if __name__ == '__main__':
   #get_macro_context(date(2020, 3, 2))
   #get_macro_context(date(2020, 3, 29))
   get_macro_context()
