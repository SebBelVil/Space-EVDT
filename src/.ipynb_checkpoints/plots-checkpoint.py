import matplotlib.pyplot as plt

def plot_satellite_weights(df):
    """
    Plots satellite-level GVA assignment for EO, Comms, and Nav.
    Each satellite = one point.
    Different color per category.
    """

    # Filter only major 3 categories
    df_plot = df[df["mission_category"].isin(
        ["Earth Observation", "Communications", "Navigation"]
    )].copy()

    if df_plot.empty:
        print("No EO/Comms/Nav satellites found in dataframe.")
        return

    # Assign colors
    colors = {
        "Earth Observation": "blue",
        "Communications": "red",
        "Navigation": "green"
    }

    # Create a categorical index for plotting
    df_plot["x"] = range(len(df_plot))

    plt.figure(figsize=(14, 6))

    for category, color in colors.items():
        df_cat = df_plot[df_plot["mission_category"] == category]

        plt.scatter(
            df_cat["x"],
            df_cat["satellite_gva"],
            c=color,
            label=category,
            alpha=0.7,
            s=40
        )

    plt.title("Satellite-Level GVA Allocation by Category", fontsize=14)
    plt.xlabel("Satellite ID")
    plt.ylabel("GVA Assigned (Millions USD)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.show()

# Plot 3d bar chart of the space population

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_total_satellite_launches(discos_df):
    """
    Plot total launches by 5-year bins and simplified object class.
    Bars are adjacent, colors are distinct, y-axis ordered as Other, Rocket, Unknown, Payload.
    Front bars hide the back bars.
    """

    # --- Prepare 5-year bins ---
    discos_df['launch_year'] = pd.to_datetime(discos_df['attributes.firstEpoch'], errors='coerce').dt.year
    min_year = int(discos_df['launch_year'].min())
    max_year = int(discos_df['launch_year'].max())
    bins = list(range(min_year, max_year + 5, 5))
    labels = [f"{b}-{b+4}" for b in bins[:-1]]

    # --- Group object classes ---
    df_total = discos_df.copy()
    df_total['class_group'] = df_total['attributes.objectClass'].fillna('Unknown')
    df_total['class_group'] = df_total['class_group'].apply(lambda x: 
        'Payload' if x=='Payload' else
        'Rocket' if 'Rocket' in x else
        'Other' if 'Other' in x or x=='Debris' else
        'Unknown'
    )
    df_total['year_bin'] = pd.cut(df_total['launch_year'], bins=bins, labels=labels, right=False)

    # --- Reorder columns ---
    ordered_classes = ['Other', 'Rocket', 'Unknown', 'Payload']
    counts_total = df_total.groupby(['year_bin','class_group']).size().unstack(fill_value=0)
    counts_total = counts_total.reindex(columns=ordered_classes, fill_value=0)

    class_colors = {'Payload':'green', 'Rocket':'red', 'Other':'gray', 'Unknown':'blue'}

    # --- Plot 3D bars ---
    fig = plt.figure(figsize=(14,7))
    ax = fig.add_subplot(111, projection='3d')

    xpos = np.arange(len(counts_total.index))
    dx = 0.8
    dy = 0.5

    # Draw bars in the order of ordered_classes, but front bars (lower y) last
    for i, cls in enumerate(ordered_classes):
        ypos = np.ones(len(xpos)) * i
        dz = counts_total[cls].values
        color = class_colors.get(cls, 'gray')
        ax.bar3d(xpos, ypos, np.zeros(len(xpos)), dx, dy, dz, color=color, alpha=0.8)

    # --- Axes formatting ---
    ax.set_xticks(xpos + dx/2)
    ax.set_xticklabels(counts_total.index, rotation=45, ha='right')
    ax.set_yticks(np.arange(len(ordered_classes)) + dy/2)
    ax.set_yticklabels(ordered_classes)
    ax.set_zlabel("Number of objects")
    ax.set_title("Total Space Population by 5-Year Bin and Object Class")
    plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3d backend)
import matplotlib.patches as mpatches

def plot_launches_3d_stacked(discos_df):
    """
    3D stacked bars grouped by 5-year bins and main class.
    Subtypes are aggregated into super-categories with consistent colors:
      - Debris (contains 'Debris' or 'Fragmentation')
      - Mission related (contains 'Mission Related')
      - Object (base objects: Payload, Rocket, Other, Unknown, and everything else)
    """
    df = discos_df.copy()

    # --- extract launch year and create 5-year bins ---
    df['launch_year'] = pd.to_datetime(df['attributes.firstEpoch'], errors='coerce').dt.year
    min_year = int(df['launch_year'].min(skipna=True))
    max_year = int(df['launch_year'].max(skipna=True))
    bins = list(range(min_year, max_year + 5, 5))
    labels = [f"{b}-{b+4}" for b in bins[:-1]]
    df['year_bin'] = pd.cut(df['launch_year'], bins=bins, labels=labels, right=False)

    # --- define main group mapping and keep original subtype ---
    def main_group(x):
        if pd.isna(x):
            return 'Unknown'
        x = str(x)
        if x == 'Payload' or x.startswith('Payload'):
            return 'Payload'
        if 'Rocket' in x:
            return 'Rocket'
        if 'Other' in x:
            return 'Other'
        if 'Debris' in x and 'Payload' not in x and 'Rocket' not in x:
            return 'Other'
        return 'Unknown'

    df['main_group'] = df['attributes.objectClass'].apply(main_group)
    # subtype: use original exact label (fillna -> 'Unknown')
    df['subtype'] = df['attributes.objectClass'].fillna('Unknown').astype(str)

    # desired main-group order on y-axis
    ordered_main = ['Other', 'Rocket', 'Unknown', 'Payload']

    # --- pivot to counts: index = year_bin, columns = (main_group, subtype) ---
    grouped = df.groupby(['year_bin', 'main_group', 'subtype']).size().reset_index(name='count')

    # ensure all year bins appear (even if empty)
    year_index = pd.Index(labels, name='year_bin')
    # build nested dict counts[year_bin][main_group][subtype] = count
    counts = {y: {m: {} for m in ordered_main} for y in labels}
    for _, row in grouped.iterrows():
        y = row['year_bin']
        m = row['main_group']
        s = row['subtype']
        if pd.isna(y):
            continue
        if m not in ordered_main:
            m = 'Other'
        counts[y].setdefault(m, {})
        counts[y][m][s] = counts[y][m].get(s, 0) + int(row['count'])

    # --- determine super-category for subtypes and color mapping ---
    def subtype_super(s):
        s_low = s.lower()
        if 'debris' in s_low or 'fragment' in s_low:
            return 'Debris'
        if 'mission related' in s_low or 'mission' in s_low and 'related' in s_low:
            return 'Mission related'
        # everything that looks like an object (Payload, Rocket, Other, Unknown) -> Object
        if any(tok in s_low for tok in ['payload', 'rocket', 'other', 'unknown']):
            return 'Object'
        # fallback
        return 'Object'

    # collect all subtypes present and map to super categories
    subtypes_per_main = {m: set() for m in ordered_main}
    for y in labels:
        for m in ordered_main:
            for s in counts[y][m]:
                subtypes_per_main[m].add(s)

    # build subtype -> super mapping
    subtype_to_super = {}
    for m in ordered_main:
        for s in sorted(subtypes_per_main[m]):
            subtype_to_super[s] = subtype_super(s)

    # Define colors for super-categories (consistent)
    color_map = {
        'Debris': '#f17925',         # orange
        'Mission related': '#57b956',# green
        'Object': '#438fff',         # blue
    }
    # ensure unknown fallback
    color_map.setdefault('Unknown', (0.6,0.6,0.6,1.0))

    # --- prepare arrays for plotting ---
    x_labels = labels
    n_x = len(x_labels)
    n_y = len(ordered_main)
    x_pos = np.arange(n_x) * 1.2
    dx = 0.9
    dy = 0.6  # more y spacing for readability

    fig = plt.figure(figsize=(18,9))
    ax = fig.add_subplot(111, projection='3d')

    # For each main group (y index), and each x, draw stacked bars by super-category
    for y_idx, m in enumerate(ordered_main):
        for xi, xlab in enumerate(x_labels):
            subtype_counts = counts[xlab][m]
            if not subtype_counts:
                continue
            # aggregate counts by super-category, but keep deterministic order
            super_order = ['Object','Mission related object','Debris']  # choose stacking order
            bottom = 0.0
            for sup in super_order:
                # sum all subtypes that map to this super category
                h = sum(v for s,v in subtype_counts.items() if subtype_to_super.get(s,'Object') == sup)
                if h == 0:
                    continue
                ax.bar3d(x_pos[xi], y_idx, bottom, dx, dy, h,
                         color=color_map.get(sup, 'lightgray'), shade=True)
                bottom += h

    # --- axes formatting ---
    ax.set_xticks(x_pos + dx/2)
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    #ax.xaxis.set_tick_params(pad=20)  # adjust this number to bring ticks closer/further
    ax.set_yticks(np.arange(n_y) + dy/2)
    ax.set_yticklabels(ordered_main)
    ax.set_zlabel("Number of objects")
    ax.set_xlabel("First Epoch", labelpad=-13)
    ax.set_ylabel("")  # no y label
    ax.set_title("Space Objects per 5-year Bin — stacked by category (Debris / Mission related Object / Object)")

    # legend for super-categories
    patches = [
        mpatches.Patch(color=color_map['Object'], label='Object'),
        mpatches.Patch(color=color_map['Mission related'], label='Mission related'),
        mpatches.Patch(color=color_map['Debris'], label='Debris'),
    ]
    ax.legend(handles=patches, bbox_to_anchor=(1.02, 0.5), loc='center left')

    # reduce grid clutter
    ax.grid(False)
    plt.tight_layout()
    plt.show()


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

def plot_payload_orbits(df_payload_with_tle, bin_size=500, max_alt=40000):
    """
    Plots payloads in orbit using a stacked polar plot.
    """

    # Copy dataframe to avoid modifying original
    df = df_payload_with_tle.copy()

    # --- Step 1: Define orbit bins ---
    bins = np.arange(0, max_alt + bin_size, bin_size)
    df['orbit_bin'] = pd.cut(df['ALTITUDE_KM'], bins=bins, labels=bins[:-1])

    # --- Step 2: Simplify applications ---
    def map_app(app):
        if app in ['Earth Observation', 'EO']:
            return 'EO'
        elif app in ['Communications', 'Comms']:
            return 'Comms'
        elif app in ['Navigation', 'Nav']:
            return 'Nav'
        else:
            return 'Other'

    df['app_simple'] = df['mission_category'].apply(map_app)

    # --- Step 3: Count satellites per orbit bin and application ---
    stack_counts = df.groupby(['orbit_bin', 'app_simple']).size().unstack(fill_value=0)
    orbit_bins = stack_counts.index.astype(float)

    # --- Step 4: Prepare plot ---
    fig, ax = plt.subplots(figsize=(8,8), subplot_kw={'projection':'polar'})
    colors = {'EO':'green', 'Comms':'orange', 'Nav':'purple', 'Other':'grey'}
    earth_radius = 6371

    # --- Step 5: Plot each orbit bin as a stacked donut ---
    for i, alt in enumerate(orbit_bins):
        counts = stack_counts.loc[alt]
        total = counts.sum()
        if total == 0:
            continue
        start_angle = 0
        for app in counts.index:
            frac = counts[app] / total  # fraction of the orbit
            if frac == 0:
                continue
            theta = np.linspace(start_angle, start_angle + 2*np.pi*frac, 100)
            r_inner = earth_radius + alt
            r_outer = r_inner + bin_size * 0.8  # thickness of orbit ring
            ax.fill_between(theta, r_inner, r_outer, color=colors[app], alpha=0.8)
            start_angle += 2*np.pi * frac

    # --- Step 6: Draw Earth ---
    theta = np.linspace(0, 2*np.pi, 100)
    ax.fill_between(theta, 0, earth_radius, color='lightblue', alpha=0.5)

    # --- Step 7: Customize plot ---
    ax.set_theta_offset(np.pi/2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, earth_radius + max_alt + bin_size)
    ax.set_xticks([])
    ax.set_yticks([])
    total_objects = len(df)  # or df.shape[0]
    ax.set_title(
        f'Active Payloads by Orbit and Application\n'
        f'(Bin size: {bin_size} km, Max altitude: {max_alt} km, Total objects: {total_objects})',
        fontsize=14
    ) 
    # ax.set_title(
    #     f'Active Payloads by Orbit and Application - Low Earth Orbit\n'
    #     f'(Bin size: {bin_size} km, Max altitude: {max_alt} km)',
    #     fontsize=14
    # )
    # --- Step 8: Legend ---
    legend_elements = [Patch(facecolor=c, label=l) for l,c in colors.items()]
    #ax.legend(handles=legend_elements, loc='center right', bbox_to_anchor=(1.20, 0.5))
    ax.legend(
        handles=legend_elements,
        loc='upper center',       # align the legend box's top center
        bbox_to_anchor=(0.5, -0.0),  # center horizontally, below the axes
        ncol=len(legend_elements),   # put all items in a single row
        frameon=False,               # optional: remove the box
        fontsize=10
    )
    plt.show()

import matplotlib.pyplot as plt
import numpy as np

def plot_eo_satellite_bins(raw_counts):
    """
    Takes raw_counts {satellite: paper_count} and plots how many satellites
    """
    # Define bins
    bins = list(range(0, 1100, 100))  # 0,100,...,1000
    labels = [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins)-1)]

    # Get counts per bin
    values = list(raw_counts.values())
    hist, _ = np.histogram(values, bins=bins)

    # Plot
    plt.figure(figsize=(10,5))
    plt.bar(labels, hist, width=0.8, color="skyblue", edgecolor="black")
    plt.xticks(rotation=45)
    plt.xlabel("Paper count range")
    plt.ylabel("Number of satellites")
    plt.title("Distribution of EO satellites by paper count range")
    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

def plot_evar_vs_gva(df_risk):
    """
    Plot EVAR vs GVA for assets with annual collision probability as color scale.
    """
    df_plot = df_risk.copy()
    x = np.arange(len(df_plot))

    fig, ax1 = plt.subplots(figsize=(20, 8))

    # --- EVAR bars (left axis) with color = annual probability ---
    colors = plt.cm.viridis(df_plot["Prob_yr_perc"] / df_plot["Prob_yr_perc"].max())
    bars = ax1.bar(x, df_plot["EVAR_millions"], color=colors, label="EVAR (M USD)")

    ax1.set_xlabel("Assets")
    ax1.set_ylabel("EVAR (M USD)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(df_plot["asset"], rotation=90, fontsize=8)
    ax1.set_title("Assets: EVAR vs GVA with Annual Collision Probability")

    # --- GVA markers (right axis) as large dots ---
    ax2 = ax1.twinx()
    gva_markers = ax2.scatter(
        x, df_plot["GVA_millions"], color="red", marker="D", s=100, label="GVA (M USD)"
    )
    ax2.set_ylabel("GVA (M USD)")

    # --- Combine legends on ax1 ---
    handles1, labels1 = ax1.get_legend_handles_labels()
    ax1.legend(handles1 + [gva_markers], labels1 + ["GVA (M USD)"], loc="upper left")

    # --- Colorbar below the figure ---
    sm = plt.cm.ScalarMappable(
        cmap="viridis", norm=Normalize(vmin=0, vmax=df_plot["Prob_yr_perc"].max())
    )
    sm.set_array([])

    fig.subplots_adjust(bottom=0.2)  # make space for colorbar
    cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.03])  # [left, bottom, width, height]
    cbar = plt.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Annual collision probability (%)")

    plt.show()


import numpy as np
import matplotlib.pyplot as plt

def plot_risk_density_capped(df_risk, alt_bin_size=25, inc_bin_size=2, min_alt=0, max_alt=2000, cap_prob=10):
    """
    Plots a 2D density map of Prob_yr_perc as a function of altitude and inclination.
    Probabilities above cap_prob (%) are capped for better visualization.
    """

    # Filter altitude
    df_plot = df_risk[(df_risk['avg_altitude_km'] >= min_alt) &
                      (df_risk['avg_altitude_km'] <= max_alt)].copy()
    
    # Cap probabilities
    df_plot['Prob_yr_perc'] = df_plot['Prob_yr_perc'].clip(upper=cap_prob)

    # Define bins
    alt_bins = np.arange(min_alt, max_alt + alt_bin_size, alt_bin_size)
    inc_bins = np.arange(0, 181, inc_bin_size)  # 0 to 180 deg

    # Digitize
    alt_idx = np.digitize(df_plot['avg_altitude_km'], alt_bins) - 1
    inc_idx = np.digitize(df_plot['avg_inclination_deg'], inc_bins) - 1

    # Initialize 2D grid for averaging
    density_grid = np.full((len(inc_bins)-1, len(alt_bins)-1), np.nan)
    count_grid = np.zeros((len(inc_bins)-1, len(alt_bins)-1))

    # Sum Prob_yr_perc per bin
    for a, i, p in zip(alt_idx, inc_idx, df_plot['Prob_yr_perc']):
        if 0 <= i < density_grid.shape[0] and 0 <= a < density_grid.shape[1]:
            if np.isnan(density_grid[i, a]):
                density_grid[i, a] = p
            else:
                density_grid[i, a] += p
            count_grid[i, a] += 1

    # Average per bin
    with np.errstate(divide='ignore', invalid='ignore'):
        density_grid = np.divide(density_grid, count_grid)
    
    # Fill empty bins with 0
    density_grid = np.nan_to_num(density_grid, nan=0)

    # Plot
    plt.figure(figsize=(12,6))
    extent = [alt_bins[0], alt_bins[-1], inc_bins[0], inc_bins[-1]]
    plt.imshow(density_grid, origin='lower', aspect='auto', extent=extent,
               cmap='viridis', vmin=0, vmax=cap_prob)
    plt.colorbar(label=f'Annual Collision Probability (%) (capped at {cap_prob}%)')
    plt.xlabel('Altitude (km)')
    plt.ylabel('Inclination (deg)')
    plt.title(f"Collision Risk Density by Orbit (assuming no CAM)\n"
              f"Altitude bin size: {alt_bin_size} km | Inclination bin size: {inc_bin_size}°")   
    plt.show()