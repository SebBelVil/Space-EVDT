def merge_cdm_tle(df_CDM_unique, df_gp_filtered):
    """
    Merge TLE information onto the unique CDM dataframe.
    """
    # Merge TLEs for satellite 1
    df_merged = df_CDM_unique.merge(
        df_gp_filtered[['NORAD_CAT_ID','TLE_LINE1','TLE_LINE2']],
        left_on='NORAD_CAT_ID_1',
        right_on='NORAD_CAT_ID',
        how='left'
    )
    df_merged.rename(columns={'TLE_LINE1':'TLE1_1','TLE_LINE2':'TLE2_1'}, inplace=True)
    df_merged.drop(columns='NORAD_CAT_ID', inplace=True)

    # Merge TLEs for satellite 2
    df_merged = df_merged.merge(
        df_gp_filtered[['NORAD_CAT_ID','TLE_LINE1','TLE_LINE2']],
        left_on='NORAD_CAT_ID_2',
        right_on='NORAD_CAT_ID',
        how='left'
    )
    df_merged.rename(columns={'TLE_LINE1':'TLE1_2','TLE_LINE2':'TLE2_2'}, inplace=True)
    df_merged.drop(columns='NORAD_CAT_ID', inplace=True)

    print(f" TLE information was added to {len(df_merged)} unique CDMs")
    return df_merged


def map_applications_to_cdm(df_merged, df_payloads):
    """
    Adds SatCat numbers, mission categories, and satellite GVA to a merged CDM dataframe.
    """
    import pandas as pd

    # Helper to extract SatCat number from TLE line
    def extract_satno(tle_line):
        try:
            return int(tle_line[2:7].strip())
        except:
            return None

    # Lookup dictionaries
    mission_map = df_payloads.set_index("attributes.satno")["mission_category"].to_dict()
    gva_map = df_payloads.set_index("attributes.satno")["satellite_gva"].to_dict()

    # Extract SatCat numbers
    df_merged["satno_1"] = df_merged["TLE2_1"].apply(extract_satno)
    df_merged["satno_2"] = df_merged["TLE2_2"].apply(extract_satno)

    # Map mission categories
    df_merged["application_1"] = df_merged["satno_1"].map(mission_map)
    df_merged["application_2"] = df_merged["satno_2"].map(mission_map)

    # Map GVA
    df_merged["gva_1"] = df_merged["satno_1"].map(gva_map)
    df_merged["gva_2"] = df_merged["satno_2"].map(gva_map)

    print(" Snapshot of CDM with applications and GVA:\n")
    display(df_merged[[
        "OBJECT_NAME_1","application_1","gva_1",
        "OBJECT_NAME_2","application_2","gva_2",
        "MAX_PROB"
    ]].head())

    return df_merged   
    
import pandas as pd

def merge_payloads_with_tle(df_payloads, df_gp_filtered):
    """
    Merge payload dataframe with TLE information from GP catalog.
    """
    
    df_payloads = df_payloads.copy()
    df_gp_filtered = df_gp_filtered.copy()
    
    # Create NORAD_ID columns
    df_payloads['NORAD_ID'] = df_payloads['attributes.satno'].astype('Int64')
    df_gp_filtered['NORAD_ID'] = df_gp_filtered['NORAD_CAT_ID'].astype('Int64')
    
    # Merge TLE data
    df_merged = df_payloads.merge(
        df_gp_filtered[['NORAD_ID', 'TLE_LINE1', 'TLE_LINE2']],
        on='NORAD_ID',
        how='left'
    )
    
    return df_merged