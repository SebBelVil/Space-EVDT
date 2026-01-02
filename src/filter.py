

def filter_active_payloads(df):
    """
    Returns only the active payload objects from the DISCoS dataframe.
    """
    print("Filtering only active Payloads...")
    df_payloads = df[
        (df['attributes.active'] == True) &
        (df['attributes.objectClass'] == 'Payload')
    ].copy()

    print(f"Total active Payloads: {len(df_payloads)}")
    return df_payloads


def categorize_mission(mission):
    """
    Categorize satellites as Other, Navgigation, Communications, or Earth Observation, depending on their ESA DISCOSweb categorization.
    """
    if mission is None:
        return "Others"
    mission_lower = mission.lower()

    # Navigation
    if any(keyword.lower() in mission_lower for keyword in ['Navigation', 'Nav']):
        return "Navigation"

    # Communications
    elif any(keyword.lower() in mission_lower for keyword in ['Communications', 'Sigint', 'Com', 'SIGINT']):
        return "Communications"
    
    # Earth Observation
    elif any(keyword.lower() in mission_lower for keyword in ['Imaging', 'Weather', 'Meteo', 'Earth', 'EO']):
        return "Earth Observation"
    
    else:
        return "Others"


def categorize_relevant(relevant_categories, df):
    """
    Extracts only active satellite population.
    """
    # Count all categories (including Others)
    counts_all = df['mission_category'].value_counts()

    # Extract only relevant for weighting
    counts_relevant = counts_all[relevant_categories]
    
    # Compute weights **only using relevant categories**
    weights_relevant = counts_relevant / counts_relevant.sum()
    
    print("Counts (all categories):\n", counts_all)
    print("\nWeights (only relevant categories):\n", weights_relevant)

    return weights_relevant


import pandas as pd

def remove_duplicate_cdms(df_CDM):
    """
    Removes duplicate CDM entries by creating a normalized, order-independent
    identifier for each pair of objects and keeping the entry with the highest MAX_PROB.
    """
    # Ensure MAX_PROB is numeric
    df_CDM['MAX_PROB'] = pd.to_numeric(df_CDM['MAX_PROB'], errors='coerce')
    
    # Sort by MAX_PROB descending
    df_CDM = df_CDM.sort_values('MAX_PROB', ascending=False)
    
    # Create normalized pair identifier (order-independent)
    df_CDM['pair_id'] = df_CDM.apply(
        lambda x: tuple(sorted([x['NORAD_CAT_ID_1'], x['NORAD_CAT_ID_2']])), axis=1
    )
    
    # Drop duplicates, keep first (highest MAX_PROB)
    df_CDM_unique = df_CDM.drop_duplicates(subset='pair_id', keep='first').drop(columns='pair_id')
    
    # Reset index
    df_CDM_unique = df_CDM_unique.reset_index(drop=True)
    
    print(f"Extracted {len(df_CDM_unique)} unique CDMs")
    return df_CDM_unique


def filter_gp(df_gp):
    """
    Keep only active TLEs and relevant columns.
    """
    # Keep relevant columns
    cols = ['OBJECT_NAME','NORAD_CAT_ID','OBJECT_TYPE', 'TLE_LINE1', 'TLE_LINE2', 'DECAY_DATE']
    df_gp_filtered = df_gp[cols].copy()

    # Keep only rows where DECAY_DATE is empty / NaN
    df_gp_filtered = df_gp_filtered[df_gp_filtered['DECAY_DATE'].isna()]

    print(f" Extracted {len(df_gp_filtered)} TLEs from active objects")
    return df_gp_filtered
