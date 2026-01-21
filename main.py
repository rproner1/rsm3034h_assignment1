    
import pandas as pd
import numpy as np
import logging
import os
import hydra
from typing import Union, Tuple
from pathlib import Path
from omegaconf import DictConfig
# from main.data.download import get_crsp_monthly
from main.data import download_files, build_panel
from main.utils import configure_pyplot, get_latest_file, timestamp_file
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

# Load environment variables from .env file
load_dotenv()


def get_directories() -> Tuple[Path, Path, Path, Path, Path, Path, Path]:
    datadir_path = os.getenv("DATADIR")
    if not datadir_path:
        raise ValueError("DATADIR environment variable not set")

    fig_dir = Path(os.getenv("FIGDIR"))
    tab_dir = Path(os.getenv("TBLDIR"))

    data_dir = Path(datadir_path)
    download_dir = data_dir / "download_cache/"
    open_dir = data_dir / "open/"
    clean_dir = data_dir / "clean/"
    restricted_dir = data_dir / "restricted/"
    preprocess_dir = data_dir / "preprocess_cache/"

    tmp_dir = Path(os.getenv("TMP_DIR", "./tmp/"))

    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)
    download_dir.mkdir(parents=True, exist_ok=True)
    open_dir.mkdir(parents=True, exist_ok=True)
    clean_dir.mkdir(parents=True, exist_ok=True)
    restricted_dir.mkdir(parents=True, exist_ok=True)
    preprocess_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    return (
        fig_dir,
        tab_dir,
        data_dir,
        download_dir,
        open_dir,
        clean_dir,
        restricted_dir,
        preprocess_dir,
        tmp_dir,
    )


def get_credentials() -> Tuple[Union[str, None], Union[str, None], Union[str, None]]:
    fred_api_key = os.getenv("FRED_API_KEY")
    wrds_username = os.getenv("WRDS_USERNAME")
    wrds_password = os.getenv("WRDS_PASSWORD")

    return fred_api_key, wrds_username, wrds_password

def replication_score(replicated_factor: np.ndarray, benchmark_factor: np.ndarray) -> float:

    score = (
        50 * np.corrcoef(replicated_factor, benchmark_factor) + 
        25 * (1 - np.abs(replicated_factor.mean() - benchmark_factor.mean())) +
        25 * (1 - np.abs(replicated_factor.std() - benchmark_factor.std()))    
    )

    return score[0,1]

@hydra.main(version_base=None, config_path="./conf", config_name="config")
def main(cfg: DictConfig):
    fig_dir, tab_dir, data_dir, download_dir, open_dir, clean_dir, restricted_dir, preprocess_dir, tmp_dir = get_directories()
    fred_api_key, wrds_username, wrds_password = get_credentials()

    configure_pyplot(
        font_family=cfg.matplotlib.font.family,
        font_serif=cfg.matplotlib.font.serif,
        font_sans_serif=cfg.matplotlib.font.sans_serif,
    )

    (
        fig_dir,
        tab_dir,
        data_dir,
        download_dir,
        open_dir,
        clean_dir,
        restricted_dir,
        preprocess_dir,
        tmp_dir,
    ) = get_directories()

    fred_api_key, wrds_username, wrds_password = get_credentials()

    panel_path = clean_dir / "panel_data.parquet"

    # download files
    if cfg.data.download:
        download_files(
            cache_dir=download_dir,
            tmp_dir=tmp_dir,
            ignore_cache=cfg.data.ignore_download_cache,
            fred_api_key=fred_api_key,
            wrds_username=wrds_username,
            wrds_password=wrds_password,
            start_date=cfg.data.start_date,
            end_date=cfg.data.end_date,
        )

    if cfg.tasks.build_panel:
        # build panel data
        logging.info("Building panel data...")
        panel = build_panel(
            download_dir, open_dir, restricted_dir, clean_dir, preprocess_dir,
            start_date=cfg.data.start_date,
            end_date=cfg.data.end_date
        )
        logging.info(f"Panel built. Shape: {panel.shape}")

        if cfg.tasks.save_panel:
            # save panel data

            panel.to_parquet(timestamp_file(panel_path), index=False, engine="pyarrow")
            logging.info(f"Panel data saved to {panel_path}")

    elif cfg.tasks.load_panel:
        # load existing panel data
        panel = pd.read_parquet(get_latest_file(panel_path))
        logging.info(f"Loaded existing panel data from {panel_path}")
    else:
        raise ValueError("No panel data task specified in the configuration.")


    # Form hml factor using function from factor module
    hml = panel.copy() 
    hml = (panel
        .groupby(['size_portfolio', 'bm_portfolio', 'date'])
        .apply(lambda x: pd.Series({
        "ret": np.average(x["ret_excess"], weights=x["mktcap_lag"])
        })
        )
    )
    # print(hml.head())
    hml_int = (panel
        .groupby(['size_portfolio', 'bm_int_portfolio', 'date'])
        .apply(lambda x: pd.Series({
        "ret": np.average(x["ret_excess"], weights=x["mktcap_lag"])
        })
        )
    )

    hml = (hml
        .reset_index()
        .groupby('date')
        .apply(lambda x: pd.Series({
            'hml': x['ret'][x['bm_portfolio'] == 3].mean() - x['ret'][x['bm_portfolio'] == 1].mean(),
        })
        )
        .reset_index()
    )

    hml_int = (hml_int
        .reset_index()
        .groupby('date')
        .apply(lambda x: pd.Series({
            'hml_int': x['ret'][x['bm_int_portfolio'] == 3].mean() - x['ret'][x['bm_int_portfolio'] == 1].mean(),
        })
        )
        .reset_index()
    )

    # Evaluate replication using evaluation function from evaluation module
    # load latest Fama-French CSV with names like "ff5_monthly_YYYYMMDD_HHMMSS.csv"
    files = list(download_dir.glob("ff5_monthly_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No Fama-French Parquet files found in {download_dir} matching ff5_monthly_*.parquet")

    # filenames include a sortable timestamp (YYYYMMDD_HHMMSS) so the max name is the latest
    latest_parquet = max(files, key=lambda p: p.name)
    ff = pd.read_parquet(latest_parquet)
    logging.info(f"Loaded Fama-French data from {latest_parquet}")

    # Load latest EKP HML 
    files = list(download_dir.glob("ekp_hml_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No EKP Parquet files found in {download_dir} matching ekp_hml_*.parquet")

    # filenames include a sortable timestamp (YYYYMMDD_HHMMSS) so the max name is the latest
    latest_parquet = max(files, key=lambda p: p.name)
    ekp_hml = pd.read_parquet(latest_parquet)
    logging.info(f"Loaded EKP HML data from {latest_parquet}")


    ff = ff.rename(columns={'hml': 'hml_ff'})
    hml = hml.merge(ff, on='date', how='left').dropna()
    print(hml.head())

    # Perform evaluation
    ff_score = replication_score(hml['hml'].values, hml['hml_ff'].values)
    print(f"HML replication grade: {ff_score}")

    fig, ax = plt.subplots()

    ax.plot(hml['date'], hml['hml'], label='HML')
    ax.plot(hml['date'], hml['hml_ff'], label='HML FF')
    ax.legend()
    ax.set_xlabel('Date')
    ax.set_ylabel('HML')
    ax.set_title('HML Factor Comparison')
    plt.show()

    # Regress hml_ff on hml
    lr = smf.ols('hml_ff ~ hml', data=hml).fit()
    print(lr.summary())

    hml_int = hml_int.merge(ekp_hml, on='date', how='inner')
    ekp_score = replication_score(hml_int['hml_int'].values, hml_int['hml_int_t100'].values)
    print(f"HML_INT replication grade: {ekp_score}")

    fig, ax = plt.subplots()

    ax.plot(hml_int['date'], hml_int['hml_int'], label='HML_INT')
    ax.plot(hml_int['date'], hml_int['hml_int_t100'], label='HML_INT EKP')
    ax.legend()
    ax.set_xlabel('Date')
    ax.set_ylabel('HML_INT')
    ax.set_title('HML_INT Factor Comparison')
    plt.show()

    print(f"Average replication score: {(ff_score + ekp_score) / 2}")

if __name__ == "__main__":
    main()