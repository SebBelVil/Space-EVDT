#------- Scenario 1 - all equal weights ---------
def gva_equal(df_payloads, total_gva, weights):
    """
    Scenario 1:
    Distribute category GVA equally among active satellites in that category.
    Satellites labeled 'Others' automatically receive 0.
    """

    import numpy as np

    # Verify weight sum = 1
    if not np.isclose(sum(weights.values()), 1.0):
        raise ValueError("Weights must sum to 1 exactly.")

    df = df_payloads.copy()
    df["satellite_gva"] = 0.0  # initialize

    for category, w in weights.items():
        cat_gva = total_gva * w

        df_cat = df[df["mission_category"] == category]
        n = len(df_cat)

        if n == 0:
            print(f"No satellites found in category '{category}'. Skipping.")
            continue

        per_sat_gva = cat_gva / n

        df.loc[df["mission_category"] == category, "satellite_gva"] = per_sat_gva

        print(f"[{category}]")
        print(f"  Satellites: {n}")
        print(f"  Category GVA: ${cat_gva:,.2f} M")
        print(f"  Per-satellite GVA: ${per_sat_gva:,.2f} M\n")

    # Ensure 'Others' = 0
    df.loc[df["mission_category"] == "Others", "satellite_gva"] = 0.0

    print(" Scenario 1 complete: Equal weighting applied.\n")
    return df

#------- Scenario 2 - Comms, Nav equal weight and EO based on citations ---------

import pandas as pd

def gva_eo_cite(df_payloads, total_gva, weights_gva):
    """
    Scenario 2:
    Mixed GVA distribution model:
    - EO: weighted by actual 'citations' column in df_payloads
    - Navigation + Communications: equal weight (uniform within category)
    - Others: 0
    """

    df = df_payloads.copy()
    df["satellite_gva"] = 0.0

    # 1. EO CITATION-WEIGHT SCENARIO
    if "Earth Observation" in weights_gva:
        w = weights_gva["Earth Observation"]
        eo_total = total_gva * w

        df_eo = df[df["mission_category"] == "Earth Observation"]

        if len(df_eo) > 0:
            total_cites = df_eo["citations"].sum()

            if total_cites > 0:
                df.loc[df_eo.index, "satellite_gva"] = (
                    eo_total * (df_eo["citations"] / total_cites)
                )

            print("[EO Citations]")
            print(f" Satellites: {len(df_eo)}")
            print(f" Total EO GVA: ${eo_total:,.2f} M")
            print(f" Total citations: {total_cites}\n")

    # 2. COMMUNICATIONS + NAVIGATION EQUAL-WEIGHT SCENARIO
    for category in ["Communications", "Navigation"]:
        if category in weights_gva:
            w = weights_gva[category]
            cat_total = total_gva * w

            df_cat = df[df["mission_category"] == category]
            n = len(df_cat)

            if n > 0:
                per_sat = cat_total / n
                df.loc[df_cat.index, "satellite_gva"] = per_sat

                print(f"[{category} Equal Weight]")
                print(f" Satellites: {n}")
                print(f" Category GVA: ${cat_total:,.2f} M")
                print(f" Per-satellite GVA: ${per_sat:,.2f} M\n")

    # 3. OTHERS = 0
    df.loc[df["mission_category"] == "Others", "satellite_gva"] = 0.0

    print("Scenario 2 complete: EO citation-weight + COM/NAV equal-weight.\n")
    return df

#------- Widget ---------

import ipywidgets as widgets
from IPython.display import display, clear_output
import os
import pandas as pd

from src.plots import plot_eo_satellite_bins
def scenario_selector(df_payloads, total_gva, weights_gva, gva_equal, gva_eo_cite, get_eo_paper_counts):
    """
    Interactive widget to select a GVA scenario.
    Scenario 1: Equal weighting
    Scenario 2: EO citations (CSV or API)
    """

    # --- Scenario dropdown ---
    scenario_dropdown = widgets.Dropdown(
        options=[
            "Equal weighting (all categories)",
            "EO: citations (CSV or API)"
        ],
        description="Scenario:",
        style={'description_width': '180px'},
        layout=widgets.Layout(width='450px')
    )

    # --- EO source dropdown ---
    source_dropdown = widgets.Dropdown(
        options=["CSV file", "Fetch latest (API)"],
        description="EO data source:",
        layout=widgets.Layout(width='400px')
    )

    # --- CSV file dropdown ---
    def list_csv_files():
        if os.path.exists("data"):
            files = [f for f in os.listdir("data") if f.startswith("EO_paper_count") and f.endswith(".csv")]
            return files if files else ["<no files found>"]
        return ["<no files found>"]

    csv_dropdown = widgets.Dropdown(
        options=list_csv_files(),
        description="CSV file:",
        layout=widgets.Layout(width='400px')
    )

    # --- API key input ---
    api_text = widgets.Text(
        description="API key:",
        layout=widgets.Layout(width='400px')
    )

    # --- Run button + output ---
    run_button = widgets.Button(
        description="Run Scenario",
        button_style="success",
        layout=widgets.Layout(width='200px')
    )
    output = widgets.Output()

    # --- UI layout ---
    ui = widgets.VBox([scenario_dropdown, run_button, output])
    display(ui)

    # --- Dynamic visibility ---
    def update_ui(change):
        with output:
            clear_output()
        if scenario_dropdown.value == "Equal weighting (all categories)":
            ui.children = [scenario_dropdown, run_button, output]
        else:
            # Show source dropdown first
            if source_dropdown.value == "CSV file":
                ui.children = [scenario_dropdown, source_dropdown, csv_dropdown, run_button, output]
            else:
                ui.children = [scenario_dropdown, source_dropdown, api_text, run_button, output]

    scenario_dropdown.observe(update_ui, names="value")
    source_dropdown.observe(update_ui, names="value")

    # --- Callback ---
    def on_run_clicked(b):
        global df_result
        with output:
            clear_output()
            scenario = scenario_dropdown.value
            print(f"Running scenario: {scenario}")

            if scenario == "Equal weighting (all categories)":
                df_result = gva_equal(df_payloads, total_gva, weights_gva)

            elif scenario == "EO: citations (CSV or API)":
                if source_dropdown.value == "CSV file":
                    selected_file = csv_dropdown.value
                    if selected_file == "<no files found>":
                        print(" No CSV files available in data folder.")
                        return
                    filepath = os.path.join("data", selected_file)
                    cite_df = pd.read_csv(filepath)
                    df_payloads["citations"] = df_payloads["attributes.name"].map(
                        dict(zip(cite_df["Satellite"], cite_df["PaperCount"]))
                    ).fillna(0)
                    df_result = gva_eo_cite(df_payloads, total_gva, weights_gva)
                    raw_counts = dict(zip(cite_df["Satellite"], cite_df["PaperCount"]))
                    plot_eo_satellite_bins(raw_counts)

                elif source_dropdown.value == "Fetch latest (API)":
                    if not api_text.value.strip():
                        print("Please provide a valid API key.")
                        return
                    raw_counts, weights = get_eo_paper_counts(df_payloads, api_text.value)
                    df_payloads["citations"] = df_payloads["attributes.name"].map(raw_counts).fillna(0)
                    df_result = gva_eo_cite(df_payloads, total_gva, weights_gva)
                    plot_eo_satellite_bins(raw_counts)

            else:
                print("Unknown scenario.")
                return

            print("Scenario completed. Satellite GVA available in df_result.")
            #display(df_result.head())

    run_button.on_click(on_run_clicked)
