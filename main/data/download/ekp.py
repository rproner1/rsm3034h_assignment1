import pandas as pd
from pathlib import Path
import requests

def get_ekp_hml(cache_dir: Path, overwrite: bool = False) -> Path:
    """
    Download the EKP HML factor CSV from GitHub and save it to the cache_dir.
    Returns the path to the downloaded file.
    """
    url = "https://raw.githubusercontent.com/edwardtkim/intangiblevalue/main/output/int_factors.csv"
    out_path = cache_dir / "ekp_hml.csv"
    if out_path.exists() and not overwrite:
        return out_path
    # Download if not present or overwrite requested
    response = requests.get(url)
    response.raise_for_status()
    with open(out_path, "wb") as f:
        f.write(response.content)
    # Read CSV
    df = pd.read_csv(out_path)
    # Robustly rename date column
    for col in df.columns:
        if col.lower() in ['yearmonth', 'date', 'Date']:
            df = df.rename(columns={col: 'date'})
            break
    # Convert 'date' to datetime
    df['date'] = pd.to_datetime(df['date'])
    # Convert all other columns to decimal format
    for col in df.columns:
        if col != 'date':
            df[col] = pd.to_numeric(df[col], errors='coerce') / 100
    return df