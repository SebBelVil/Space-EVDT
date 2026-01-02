
from scipy.stats import poisson
import pandas as pd
import numpy as np
from sgp4.api import Satrec
from sgp4.api import jday

def tle_to_orbit(TLE1, TLE2):
    """Return approximate altitude (km) and inclination (deg) from TLE lines."""
    if pd.isna(TLE1) or pd.isna(TLE2):
        return np.nan, np.nan
    try:
        sat = Satrec.twoline2rv(TLE1, TLE2)
        # Get position at epoch to calculate altitude
        jd, fr = sat.jdsatepoch, sat.jdsatepochF
        e, r, v = sat.sgp4(jd, fr)
        # Altitude = ||r|| - Earth radius (~6371 km)
        if e != 0:
            return np.nan, np.nan
        r_mag = np.linalg.norm(r)
        altitude = r_mag - 6371  # km
        inclination = np.degrees(sat.inclo)
        return altitude, inclination
    except:
        return np.nan, np.nan

def analyze_assets_risk_with_orbit(df_merged, asset_list):
    results = []

    for asset_name in asset_list:
        # Filter CDMs for this asset
        df_asset = df_merged[
            df_merged['OBJECT_NAME_1'].str.contains(asset_name, case=False, na=False, regex=False) |
            df_merged['OBJECT_NAME_2'].str.contains(asset_name, case=False, na=False, regex=False)
        ].copy()

        if df_asset.empty:
            print(f"No CDMs found for asset: {asset_name}")
            continue

        # Determine which side is the asset
        df_asset['application_used'] = df_asset.apply(
            lambda row: row['application_1'] if asset_name.lower() in str(row['OBJECT_NAME_1']).lower()
            else row['application_2'],
            axis=1
        )

        df_asset['gva_used'] = df_asset.apply(
            lambda row: row['gva_1'] if asset_name.lower() in str(row['OBJECT_NAME_1']).lower()
            else row['gva_2'],
            axis=1
        )

        # Extract TLE for the asset
        df_asset['TLE1_used'] = df_asset.apply(
            lambda row: row['TLE1_1'] if asset_name.lower() in str(row['OBJECT_NAME_1']).lower()
            else row['TLE1_2'],
            axis=1
        )
        df_asset['TLE2_used'] = df_asset.apply(
            lambda row: row['TLE2_1'] if asset_name.lower() in str(row['OBJECT_NAME_1']).lower()
            else row['TLE2_2'],
            axis=1
        )

        # Compute orbital info
        df_asset['altitude_km'], df_asset['inclination_deg'] = zip(*df_asset.apply(
            lambda row: tle_to_orbit(row['TLE1_used'], row['TLE2_used']), axis=1
        ))

        # Poisson risk
        mu_7d = df_asset['MAX_PROB'].sum()
        Prob_7days = 1 - np.prod(1 - df_asset['MAX_PROB'].values)
        mu_yr = mu_7d * (365/7)
        Prob_yr = poisson.sf(k=0, mu=mu_yr)
        Prob_yr_perc = Prob_yr * 100

        # EVaR
        application_mode = df_asset['application_used'].mode().iloc[0]
        gva_mode = df_asset[df_asset['application_used'] == application_mode]['gva_used'].mean()
        EVAR_asset = gva_mode * Prob_yr

        # Orbital averages
        avg_altitude = df_asset['altitude_km'].mean()
        avg_inclination = df_asset['inclination_deg'].mean()

        num_cdms = len(df_asset)
        results.append({
            "asset": asset_name,
            "application": application_mode,
            "num_cdms": num_cdms,
            "mu_7d": mu_7d,
            "Prob_7days": Prob_7days,
            "mu_yr": mu_yr,
            "Prob_yr": Prob_yr,
            "Prob_yr_perc": Prob_yr_perc,
            "GVA_millions": gva_mode,
            "EVAR_millions": EVAR_asset,
            "avg_altitude_km": avg_altitude,
            "avg_inclination_deg": avg_inclination
        })

    return pd.DataFrame(results)

def get_objects_with_gva(df):
    """
    Returns a set of all OBJECT_NAME_1 and OBJECT_NAME_2 where gva_1 or gva_2 is not NaN.
    """
    # Filter rows where gva_1 or gva_2 is not NaN
    mask_1 = df['gva_1'].notna()
    mask_2 = df['gva_2'].notna()
    
    names_1 = df.loc[mask_1, 'OBJECT_NAME_1']
    names_2 = df.loc[mask_2, 'OBJECT_NAME_2']
    
    # Combine into a set to avoid duplicates and ignore NaNs
    objects_with_gva = set(names_1.dropna()).union(set(names_2.dropna()))
    
    return objects_with_gva