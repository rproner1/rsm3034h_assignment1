import sys
from pathlib import Path
from dotenv import load_dotenv
import os
import wrds
import numpy as np
import pandas as pd

from main.data.download import get_ff5_factors_monthly

load_dotenv()
sys.path.append(str(Path().resolve().parent))  # Adds the parent directory of 'main' to sys.path

def get_credentials():
    fred_api_key = os.getenv("FRED_API_KEY")
    wrds_username = os.getenv("WRDS_USERNAME")
    wrds_password = os.getenv("WRDS_PASSWORD")
    return fred_api_key, wrds_username, wrds_password

fred_api_key, wrds_username, wrds_password = get_credentials()

def get_compustat(
    wrds_username: str,
    wrds_password: str,
    START_DATE: str = "01/01/2000",
    END_DATE: str = "12/31/2024",
):
    query = f"""
        select gvkey, datadate, seq, ceq, at, lt, txditc, txdb, itcb, pstkrv,
        pstkl, pstk, capx, oancf, sale, cogs, xint, xsga
        from comp.funda
        where consol='C' and popsrc='D' and indfmt='INDL' and datafmt='STD'
        and datadate between '{START_DATE}' and '{END_DATE}'
    """
    conn = wrds.Connection(wrds_username=wrds_username, wrds_password=wrds_password)
    funda = conn.raw_sql(query, date_cols=["datadate", "fdatem", "rdq"])
    conn.close()
    return funda

def get_compustat_gic_codes(wrds_username: str, wrds_password: str):
    query = """
        select gvkey, gsector, ggroup, gind, gsubind, indfrom, indthru
        from comp.co_hgic
    """
    conn = wrds.Connection(wrds_username=wrds_username, wrds_password=wrds_password)
    gic = conn.raw_sql(query, date_cols=["indfrom", "indthru"])
    conn.close()
    return gic

def get_crsp_monthly(
    wrds_username: str,
    wrds_password: str,
    CRSP_START_DATE="20000101",
    CRSP_END_DATE="20241231",
) -> pd.DataFrame:
    query = f"""
    select a.date, a.permno, a.shrout, a.cfacpr, a.cfacshr, a.altprc, a.prc, a.vol, a.ret,
    b.ticker, b.comnam, b.exchcd, b.shrcd, b.ncusip
    from crsp.msf as a
    left join crsp.dsenames as b
    on a.PERMNO=b.PERMNO
    and b.namedt<=a.date
    and a.date<=b.nameendt
    where b.SHRCD between 10 and 11
    and b.EXCHCD between 1 and 3
    and a.date >= '{CRSP_START_DATE}' and a.date <= '{CRSP_END_DATE}'
    """
    conn = wrds.Connection(wrds_username=wrds_username, wrds_password=wrds_password)
    df = conn.raw_sql(query)
    df["date"] = pd.to_datetime(df["date"])
    df["date"] = df["date"] + pd.offsets.MonthEnd(0)
    df["prc"] = np.abs(df["prc"])
    df["permno"] = df["permno"].astype(int)
    conn.close()
    return df

def get_crsp_compu_link_table(wrds_username: str, wrds_password: str) -> pd.DataFrame:
    query = """
    SELECT lpermno AS permno, gvkey, linkdt, COALESCE(linkenddt, CURRENT_DATE) AS linkenddt
    FROM crsp.ccmxpf_linktable
    WHERE usedflag = 1
    AND linkprim IN ('P', 'C')
    """
    conn = wrds.Connection(wrds_username=wrds_username, wrds_password=wrds_password)
    data = conn.raw_sql(query)
    data["linkdt"] = pd.to_datetime(data["linkdt"])
    data["linkenddt"] = pd.to_datetime(data["linkenddt"])
    conn.close()
    return data

# --- CRSP Data ---
crsp = get_crsp_monthly(
    wrds_username=wrds_username,
    wrds_password=wrds_password
)

# Compute market cap (ME) and scale by millions
crsp = (
    crsp
    .assign(mktcap=lambda x: x["shrout"] * x["altprc"] / 1_000_000)
    .assign(mktcap=lambda x: x["mktcap"].replace(0, np.nan))
)

# Create lagged market cap variable for value-weighted portfolios
mktcap_lag = (
    crsp
    .assign(
        date=lambda x: x["date"] + pd.DateOffset(months=1),
        mktcap_lag=lambda x: x["mktcap"]
    )
    .loc[:, ["permno", "date", "mktcap_lag"]]
)

crsp = crsp.merge(mktcap_lag, how="left", on=["permno", "date"])

# Download risk-free rate
ff_factors_monthly = get_ff5_factors_monthly()
rf = ff_factors_monthly.loc[:, ["date", "rf"]]

# Calculate excess return
crsp = crsp.merge(rf, how="left", on="date")
crsp['ret_excess'] = crsp['ret'] - crsp['rf']
crsp['ret_excess'] = crsp['ret_excess'].clip(lower=-1)
crsp = crsp.drop(columns=['rf'])

print(crsp[crsp.date.dt.month == 12].head())

# Drop rows with missing excess return and market cap
crsp = crsp.dropna(subset=['ret_excess', 'mktcap', 'mktcap_lag'])

print(crsp[crsp.date.dt.month == 12].head())

# --- Preparing Compustat ---
comp = get_compustat(wrds_username=wrds_username, wrds_password=wrds_password)

comp['be'] = (
    comp['seq'].combine_first(comp['ceq'] + comp['pstk']).combine_first(comp['at'] - comp['lt'])
    + comp['txditc'].combine_first(comp['txdb'] + comp['itcb']).fillna(0)
    - comp['pstk'].combine_first(comp['pstkrv']).combine_first(comp['pstkl']).fillna(0)
)
comp['be'] = comp['be'].apply(lambda x: np.nan if x <= 0 else x)
comp['op'] = ((comp["sale"] - comp["cogs"].fillna(0) - comp["xsga"].fillna(0) - comp["xint"].fillna(0)) / comp["be"])

comp['year'] = pd.DatetimeIndex(comp['datadate']).year
comp = comp.sort_values('datadate').groupby(['gvkey', 'year']).tail(1).reset_index()

# Calculate inv
compustat_lag = comp[['gvkey', 'year', 'at']].copy()
compustat_lag['year'] += 1
compustat_lag = compustat_lag.rename(columns={'at': 'at_lag'})
comp = comp.merge(compustat_lag, on=['gvkey', 'year'], how='left')
comp = (
    comp
    .assign(inv=lambda x: (x['at'] - x['at_lag']) / x['at_lag'])
    .assign(inv=lambda x: x['inv'].where(x['at_lag'] > 0))
)

# --- Linking Compustat and CRSP ---
link_table = get_crsp_compu_link_table(
    wrds_username=wrds_username, wrds_password=wrds_password
)

ccm_links = crsp.merge(link_table, how="inner", on="permno")
ccm_links = ccm_links.query("~gvkey.isnull() & (date >= linkdt) & (date <= linkenddt)")
ccm_links = ccm_links.loc[:, ['permno', 'gvkey', 'date']]

crsp = crsp.merge(ccm_links, how='left', on=['permno', 'date'])

# --- Replication ---
size = crsp.query("date.dt.month == 6").copy()
size['sorting_date'] = size['date'] + pd.DateOffset(months=1)
size = size[['permno', 'exchcd', 'sorting_date', 'mktcap']].rename(columns={'mktcap': 'size'})

me = crsp.query("date.dt.month == 12").copy()
# Uncomment and adjust if you want to use sorting_date for me
# me['sorting_date'] = me['date'] + pd.DateOffset(months=7)
# me = me[['permno', 'gvkey', 'sorting_date', 'mktcap']].rename(columns={'mktcap': 'me'})

# Example output
print("CRSP December snapshot:")
print(crsp[crsp.date.dt.month == 12].head())

print("Compustat columns:")
print(comp.columns)

print("Size DataFrame:")
print(size.head())

print("ME DataFrame:")
print(me.head())