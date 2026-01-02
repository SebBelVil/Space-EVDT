from sgp4.api import Satrec
from datetime import datetime
import numpy as np
import pandas as pd

def compute_conjunction_altitudes(df_cdm,
                                  tle1_1='TLE1_1', tle2_1='TLE2_1',
                                  tle1_2='TLE1_2', tle2_2='TLE2_2',
                                  tca_col='TCA'):
    """
    Compute satellite positions at TCA and approximate conjunction altitude.
    """

    df = df_cdm.copy()

    # Convert TCA to datetime
    df['TCA_dt'] = pd.to_datetime(df[tca_col])

    r1_list, r2_list, alt_list = [], [], []

    for idx, row in df.iterrows():
        try:
            # Satellite 1
            sat1 = Satrec.twoline2rv(row[tle1_1].strip(), row[tle2_1].strip())
            tca = row['TCA_dt']
            epoch = datetime(sat1.jdsatepoch_year, 1, 1) + pd.to_timedelta(sat1.jdsatepoch_day - 1, unit='D')
            tca_seconds = (tca - epoch).total_seconds()
            jd_tca = sat1.jdsatepoch + sat1.jdsatepochF + tca_seconds / 86400
            jd_int = int(jd_tca)
            jd_frac = jd_tca - jd_int
            e1, r1, v1 = sat1.sgp4(jd_int, jd_frac)
            if e1 != 0: r1 = None

            # Satellite 2
            sat2 = Satrec.twoline2rv(row[tle1_2].strip(), row[tle2_2].strip())
            epoch2 = datetime(sat2.jdsatepoch_year, 1, 1) + pd.to_timedelta(sat2.jdsatepoch_day - 1, unit='D')
            tca_seconds2 = (tca - epoch2).total_seconds()
            jd_tca2 = sat2.jdsatepoch + sat2.jdsatepochF + tca_seconds2 / 86400
            jd_int2 = int(jd_tca2)
            jd_frac2 = jd_tca2 - jd_int2
            e2, r2, v2 = sat2.sgp4(jd_int2, jd_frac2)
            if e2 != 0: r2 = None

            r1_list.append(r1)
            r2_list.append(r2)

            if r1 is not None and r2 is not None:
                alt = (np.linalg.norm(r1) + np.linalg.norm(r2))/2 - 6371
            else:
                alt = None
            alt_list.append(alt)

        except Exception as ex:
            r1_list.append(None)
            r2_list.append(None)
            alt_list.append(None)
            print(f"Row {idx} propagation error: {ex}")

    df['r1'] = r1_list
    df['r2'] = r2_list
    df['Conjunction_Altitude_km'] = alt_list

    print(f" Conjunction altitudes attempted for {len(df)} CDMs.")
    failed = sum(1 for a in alt_list if a is None)
    print(f"{failed} CDMs could not be propagated (altitude=None).")

    return df

import pandas as pd
from sgp4.api import Satrec, jday
import matplotlib.pyplot as plt

def tle_to_altitude(tle1, tle2, year=2025, month=12, day=11, hour=0, minute=0, second=0):
    """
    Compute Altitude based on TLE information
    """
    if pd.isna(tle1) or pd.isna(tle2):
        return None
    sat = Satrec.twoline2rv(tle1, tle2)
    jd, fr = jday(year, month, day, hour, minute, second)
    e, r, v = sat.sgp4(jd, fr)
    if e == 0:
        # Magnitude of position vector minus Earth radius (km)
        return (r[0]**2 + r[1]**2 + r[2]**2)**0.5 - 6371
    else:
        return None

