import os
import polars as pl

from quant_sandbox.common import get_base_dir

base_dir = get_base_dir()
data_dir = os.path.join(base_dir, 'data')
fed_data_dir = os.path.join(data_dir, 'fred.stlouisfed.org')

# Consumer sentiment

UMCSENT = pl.read_csv(
    # Monthly
    os.path.join(fed_data_dir, 'UMCSENT.csv'), try_parse_dates=True
)

# Economic activity

BBKMCOIX = pl.read_csv(
    # Monthly
    os.path.join(fed_data_dir, 'BBKMCOIX.csv'), try_parse_dates=True
)

BBKMGDP = pl.read_csv(
    # Monthly
    os.path.join(fed_data_dir, 'BBKMGDP.csv'), try_parse_dates=True
)

BBKMLEIX = pl.read_csv(
    # Monthly
    os.path.join(fed_data_dir, 'BBKMLEIX.csv'), try_parse_dates=True
)

GDPC1 = pl.read_csv(
    # Quarterly
    os.path.join(fed_data_dir, 'GDPC1.csv'), try_parse_dates=True
)

# Inflation

CORESTICKM159SFRBATL = pl.read_csv(
    # Monthly
    os.path.join(fed_data_dir, 'CORESTICKM159SFRBATL.csv'), try_parse_dates=True
)

MICH = pl.read_csv(
    # Monthly
    os.path.join(fed_data_dir, 'MICH.csv'), try_parse_dates=True
)

PCEPILFE = pl.read_csv(
    # Monthly
    os.path.join(fed_data_dir, 'PCEPILFE.csv'), try_parse_dates=True
)

# Interest rates

FEDFUNDS = pl.read_csv(
    # Monthly
    os.path.join(fed_data_dir, 'FEDFUNDS.csv'), try_parse_dates=True
)

# Money supply

M2REAL = pl.read_csv(
    # Monthly
    os.path.join(fed_data_dir, 'M2REAL.csv'), try_parse_dates=True
)

# Unemployment

PAYEMS = pl.read_csv(
    # Monthly
    os.path.join(fed_data_dir, 'PAYEMS.csv'), try_parse_dates=True
)

U2RATE = pl.read_csv(
    # Monthly
    os.path.join(fed_data_dir, 'U2RATE.csv'), try_parse_dates=True
)

U6RATE = pl.read_csv(
    # Monthly
    os.path.join(fed_data_dir, 'U6RATE.csv'), try_parse_dates=True
)

UEMPMEAN = pl.read_csv(
    # Monthly
    os.path.join(fed_data_dir, 'UEMPMEAN.csv'), try_parse_dates=True
)

UNRATE = pl.read_csv(
    # Monthly
    os.path.join(fed_data_dir, 'UNRATE.csv'), try_parse_dates=True
)


if __name__ == '__main__':
    print(CORESTICKM159SFRBATL)
    print(FEDFUNDS)
    print(UNRATE)
