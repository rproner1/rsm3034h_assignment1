from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
import logging

from ..utils import get_latest_file

load_dotenv()

def assign_exchange(primaryexch):
    if primaryexch == "N":
        return "NYSE"
    elif primaryexch == "A":
        return "AMEX"
    elif primaryexch == "Q":
        return "NASDAQ"
    else:
        return "Other"
    
def assign_industry(siccd):
    if 1 <= siccd <= 999:
        return "Agriculture"
    elif 1000 <= siccd <= 1499:
        return "Mining"
    elif 1500 <= siccd <= 1799:
        return "Construction"
    elif 2000 <= siccd <= 3999:
        return "Manufacturing"
    elif 4000 <= siccd <= 4899:
        return "Transportation"
    elif 4900 <= siccd <= 4999:
        return "Utilities"
    elif 5000 <= siccd <= 5199:
        return "Wholesale"
    elif 5200 <= siccd <= 5999:
        return "Retail"
    elif 6000 <= siccd <= 6799:
        return "Finance"
    elif 7000 <= siccd <= 8999:
        return "Services"
    elif 9000 <= siccd <= 9999:
        return "Public"
    else:
        return "Missing"

def calculate_intangibles(df, firm_col='gvkey', date_col='datadate', sga_col='xsga', g=0.1, delta=0.2):
    """
    Compute intangible assets (INT) for each firm-year and return a DataFrame with keys for merging.
    Args:
        df: DataFrame with at least [firm_col, date_col, sga_col]
        firm_col: column name for firm identifier
        date_col: column name for date (must be sortable)
        sga_col: column name for SG&A expense
        g: average growth rate for SG&A
        delta: depreciation rate
    Returns:
        DataFrame with [firm_col, date_col, 'int']
    """
    df = df.sort_values([firm_col, date_col]).copy()
    results = []
    for _, group in df.groupby(firm_col):
        xsga = group[sga_col].values
        int_vals = []
        for i, sga in enumerate(xsga):
            if i == 0:
                int_val = sga / (g + delta) if not pd.isna(sga) else np.nan
            else:
                prev = int_vals[-1]
                int_val = (1 - delta) * prev + sga if not pd.isna(prev) and not pd.isna(sga) else np.nan
            int_vals.append(int_val)
        results.append(pd.DataFrame({
            firm_col: group[firm_col].values,
            date_col: group[date_col].values,
            'int': int_vals
        }))
    return pd.concat(results, ignore_index=True)

def assign_delisting_reason(dlstcd):
    """
    Assign delisting reason based on dlstcd code.
    Args:
        dlstcd: delisting code
    Returns:
        delisting reason as string
    """
    if pd.isna(dlstcd):
        return "Not Delisted"
    elif 200 <= dlstcd <= 240:
        return "Merger"
    elif 300 <= dlstcd <= 390:
        return "Exchange"
    elif dlstcd == 400:
        return "Liquidation"
    elif 501 <= dlstcd <= 519:
        return "Change Exchange"
    elif (dlstcd == 500) or (520 <= dlstcd <= 584):
        return "Performance"
        

# load crsp file
def load_crsp_file(path: Path, start_date: str, end_date: str) -> pd.DataFrame:
    logging.info("Loading CRSP data...")
    crsp = pd.read_parquet(get_latest_file(path / "crsp_monthly.parquet"))
    crsp = crsp[(crsp["date"] >= start_date) & (crsp["date"] <= end_date)] 
    crsp["year_month"] = crsp["date"].dt.to_period("M")
    crsp = crsp.rename(columns={"ncusip": "cusip"})

    # Rename exchange code and industry code.
    crsp['exchange'] = crsp['primaryexch'].apply(assign_exchange)
    crsp['industry'] = crsp['siccd'].apply(assign_industry)

    # Adust for delisting return
    crsp['dlrsn'] = crsp['dlstcd'].apply(assign_delisting_reason)
    crsp['dlret'][(crsp['dlrsn'] == 'Performance') & (crsp['exchange'].isin(['NYSE', 'AMEX']))] = -0.3
    crsp['dlret'][(crsp['dlrsn'] == 'Performance') & (crsp['exchange'] == 'NASDAQ')] = -0.55
    crsp['ret'] = (1+crsp['ret']) * (1+crsp['dlret'].fillna(0)) - 1

    # Compute market cap (ME) and scale by millions
    crsp['mktcap'] = crsp['shrout'] * crsp['prc'] / 1000000
    crsp['mktcap'] = crsp['mktcap'].replace(0, np.nan)
    
    # print(crsp[crsp.date.dt.month == 12])
    # create lagged market cap
    mktcap_lag = crsp.copy()
    mktcap_lag['date'] = mktcap_lag['date'] + pd.offsets.MonthEnd(1)
    mktcap_lag = mktcap_lag.rename(columns={"mktcap": "mktcap_lag"})
    mktcap_lag = mktcap_lag[["permno", "date", "mktcap_lag"]]
    crsp = crsp.merge(mktcap_lag, on=["permno", "date"], how="left")
    # print(crsp[crsp.date.dt.month == 12])

    # Compute excess returns
    ff = pd.read_parquet(get_latest_file(path / "ff5_monthly.parquet"))
    rf = ff[['date','rf']]

    crsp = crsp.merge(rf, how="left", on="date")
    crsp['ret_excess'] = crsp['ret'] - crsp['rf']
    crsp['ret_excess'] = crsp['ret_excess'].clip(lower=-1)
    crsp = crsp.drop(columns=['rf'])

    
    crsp = crsp.dropna(subset=['ret_excess', 'mktcap', 'mktcap_lag'])

    # merge on permno and select only the dates where the link is valid
    link_tab = pd.read_parquet(get_latest_file(path / "crsp_compu_link_table.parquet")).rename(columns={"lpermno": "permno"})
    link_tab["linkenddt"] = link_tab["linkenddt"].fillna(pd.to_datetime(end_date))

    ccm_links = crsp.merge(link_tab, on="permno", how="left")
    ccm_links = ccm_links.query("~gvkey.isnull() & (date >= linkdt) & (date <= linkenddt)")
    ccm_links = ccm_links[["gvkey", "permno", "date"]]

    crsp = crsp.merge(ccm_links, on=['permno', 'date'], how='left')

    # drop_duplicates
    # crsp = crsp.drop_duplicates(subset=["permno", "date"])

    return crsp

def load_compustat_data(path: Path) -> pd.DataFrame:

    logging.info("Loading Compustat data...")
    comp = pd.read_parquet(get_latest_file(path / "compustat.parquet"))

    # Calculate intangibles
    int = calculate_intangibles(comp)
    # print(int)
    comp = comp.merge(int, on=['gvkey', 'datadate'], how='left')

    # Calculate book equity (BE)
    comp['be'] = (
        comp['seq'].combine_first(comp['ceq'] + comp['pstk']).combine_first(comp['at'] - comp['lt'])
        + comp['txditc'].combine_first(comp['txdb'] + comp['itcb']).fillna(0)
        - comp['pstk'].combine_first(comp['pstkrv']).combine_first(comp['pstkl']).fillna(0)
    )
    # Note to self. Try replacing sale with reserves
    comp['be'] = comp['be'].apply(lambda x: np.nan if x <= 0 else x)

    comp['be_int'] = comp['be'] - comp['gdwl'].fillna(0) + comp['int'].fillna(0)

    # Calculate operating performance (OP)
    comp['op'] =  ((comp["sale"]-comp["cogs"].fillna(0)-
            comp["xsga"].fillna(0)-comp["xint"].fillna(0))/comp["be"])
    
    comp['year'] = pd.DatetimeIndex(comp['datadate']).year

    # Keep only the last observation for each firm-year combination
    comp = comp.sort_values('datadate').groupby(['gvkey', 'year']).tail(1).reset_index()

    # Calculate inv
    compustat_lag = comp[['gvkey', 'year', 'at']].copy()
    compustat_lag['year'] += 1
    compustat_lag = compustat_lag.rename(columns={'at': 'at_lag'})
    comp = comp.merge(compustat_lag, on=['gvkey', 'year'], how='left')

    comp['inv'] = (comp['at'] / comp['at_lag']) - 1
    comp['inv'] = np.where(comp['at_lag'].fillna(-1) <= 0, np.nan, comp['inv'])

    return comp

def load_fama_french_returns_data(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    ff = pd.read_parquet(get_latest_file(path / "ff5_monthly.parquet"))
    ff["date"] = pd.to_datetime(ff["date"])
    ff["mkt"] = ff["mkt_rf"] + ff["rf"]

    return df.merge(ff, on="date", how="left")


def load_fama_french_me_breakpoints(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    ff_me = pd.read_parquet(get_latest_file(path / "ff_size_breakpoints.parquet"))
    # keep only the 20th percentile cuts
    ff_me = ff_me[["date", "size_bp4", "size_bp8", "size_bp12", "size_bp16"]].rename(
        columns={
            "size_bp4": "ff_me_20",
            "size_bp8": "ff_me_40",
            "size_bp12": "ff_me_60",
            "size_bp16": "ff_me_80",
        }
    )

    ff_me["year_month"] = ff_me["date"].dt.to_period("M")

    return df.merge(ff_me.drop(columns="date"), on=["year_month"], how="left")


def assign_portfolio(df: pd.DataFrame, sorting_variable: str, percentiles: list) -> pd.Series:
    """Assign portfolios to a bin according to a sorting variable."""
    logging.info(f"Assigning portfolios based on {sorting_variable}...")
    breakpoints = df[df['exchange'] == 'NYSE']
    breakpoints = breakpoints[sorting_variable].quantile([0] + percentiles + [1], interpolation="linear").drop_duplicates()
    breakpoints.iloc[0] = -np.inf
    breakpoints.iloc[breakpoints.size-1] = np.inf
    
    assigned_portfolios = pd.cut(
      df[sorting_variable],
      bins=breakpoints,
      labels=pd.Series(range(1, breakpoints.size)),
      include_lowest=True,
      right=False
    )
    
    return assigned_portfolios


def build_panel(
    download_dir: Path,
    open_dir: Path,
    restricted_dir: Path,
    clean_dir: Path,
    preprocess_dir: Path,
    start_date: str,
    end_date: str,
) -> None:
    """
    Main function to process the panel data.
    """

    crsp = load_crsp_file(download_dir, start_date, end_date)
    comp = load_compustat_data(download_dir)

    # crsp = crsp.dropna(subset=['mktcap', 'mktcap_lag', 'ret_excess']) # Causes all me to be dropped

    logging.info("Creating size variable")
    size = crsp.query("date.dt.month == 6").copy()
    size['sorting_date'] = size['date'] + pd.offsets.MonthEnd(1)
    size = size[['permno', 'exchange', 'sorting_date', 'mktcap']].rename(columns={'mktcap': 'size'})
    # print(size.head())

    logging.info("Creating book-to-market variables")
    me = crsp.query("date.dt.month == 12").copy()
    me['sorting_date'] = me['date'] + pd.offsets.MonthEnd(7)
    me = me[['permno', 'gvkey', 'sorting_date', 'mktcap']].rename(columns={'mktcap': 'me'})
    # print(me.head())

    bm = comp.copy()
    bm['sorting_date'] = pd.to_datetime((bm['datadate'].dt.year + 1).astype(str) + '-07-31', format='%Y-%m-%d')
    bm = bm.merge(me, on=['gvkey','sorting_date'], how='inner')
    bm['bm'] = bm['be']/bm['me']
    bm['bm_int'] = bm['be_int']/bm['me']


    sorting_vars = size.merge(bm, on=['sorting_date', 'permno'], how='inner').dropna().drop_duplicates(['permno', 'sorting_date'])
    # print(sorting_vars.head())

    logging.info("Assigning portfolios based on size, book-to-market, and intangible book-to-market")
    sorting_vars['size_portfolio'] = assign_portfolio(sorting_vars, 'size', [0.0, 0.5, 1.0])
    sorting_vars['bm_portfolio'] = assign_portfolio(sorting_vars, 'bm', [0.0, 0.3, 0.7, 1.0])
    sorting_vars['bm_int_portfolio'] = assign_portfolio(sorting_vars, 'bm_int', [0.0, 0.3, 0.7, 1.0])
    # print(sorting_vars.head())

    portfolios = sorting_vars[['sorting_date', 'permno', 'size_portfolio', 'bm_portfolio', 'bm_int_portfolio']]
    
    # If current month is June or early, sort on previous fiscal year. Otherwise, sort on current year
    crsp['sorting_date'] = crsp['date'].apply(lambda x: pd.to_datetime(str(x.year - 1) + '-07-31') if x.month <= 6 else pd.to_datetime(str(x.year) + '-07-31'))

    portfolios = crsp.merge(portfolios, on=['sorting_date', 'permno'], how='inner')
    print(portfolios.head())

    return portfolios
