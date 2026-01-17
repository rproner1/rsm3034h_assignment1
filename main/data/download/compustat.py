import os
import wrds

def get_compustat(
    wrds_username: str,
    wrds_password: str,
    start_date: str = "07/01/1926",
    end_date: str = "12/31/2025",
):
    """
    Retrieve data from Compustat.
    """
    query = f"""
        select gvkey, datadate, seq, ceq, at, lt, txditc, txdb, itcb,  pstkrv,
        pstkl, pstk, capx, oancf, sale, cogs, xint, xsga 
        from comp.funda
        where consol='C' and popsrc='D' and indfmt='INDL' and datafmt='STD'
        and datadate between '{start_date}' and '{end_date}'
    """

    conn = wrds.Connection(wrds_username=wrds_username, wrds_password=wrds_password)
    funda = conn.raw_sql(query, date_cols=["datadate"])
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