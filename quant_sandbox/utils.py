import calendar
from datetime import date
from functools import cache
from pydantic import BaseModel


class CalendarContext(BaseModel):

    month_context: 'CalendarMonthContext'

    @classmethod
    def from_date(cls, target_date: date) -> 'CalendarContext':
        return cls(
            month_context=CalendarMonthContext.from_date(target_date),
        )


class CalendarMonthContext(BaseModel):

    second_wednesday: date

    first_friday: date
    fourth_friday: date

    @classmethod
    def from_date(cls, target_date: date) -> 'CalendarMonthContext':
        return cls(
            second_wednesday=date(target_date.year, target_date.month, get_second_wednesday_of_month(
                target_date.year, target_date.month
            )),
            first_friday=date(target_date.year, target_date.month, get_first_friday_of_month(
                target_date.year, target_date.month
            )),
            fourth_friday=date(target_date.year, target_date.month, get_fourth_friday_of_month(
                target_date.year, target_date.month
            )),
        )


@cache
def get_first_friday_of_month(year: int, month: int):
    month_matrix = calendar.monthcalendar(year, month)
    # Return date in the Friday column of the first or second week
    if month_matrix[0][calendar.FRIDAY] != 0:
        return month_matrix[0][calendar.FRIDAY]
    else:
        return month_matrix[1][calendar.FRIDAY]


@cache
def get_fourth_friday_of_month(year: int, month: int):
    month_matrix = calendar.monthcalendar(year, month)
    # Extract only the dates in the Friday column that are not 0
    fridays = [week[calendar.FRIDAY] for week in month_matrix if week[calendar.FRIDAY] != 0]
    return fridays[3]


@cache
def get_second_wednesday_of_month(year: int, month: int):
    month_matrix = calendar.monthcalendar(year, month)
    # Extract only the dates in the Wednesday column that are not 0
    wednesdays = [week[calendar.WEDNESDAY] for week in month_matrix if week[calendar.WEDNESDAY] != 0]
    return wednesdays[1]
