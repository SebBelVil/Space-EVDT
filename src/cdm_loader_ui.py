import ipywidgets as widgets
from IPython.display import display, clear_output
import pandas as pd
import requests
from datetime import datetime
import os

def show_cdm_ui():
    """
    Interactive UI for loading CDM (SOCRATES) data:
    - Choose a local CSV
    - Or fetch latest data from CelesTrak
    """
    
    # --- Widgets ---
    choice_widget = widgets.ToggleButtons(
        options=[("Local CSV", 0), ("Fetch Latest", 1)],
        description="CDM Source:"
    )
    
    fetch_button = widgets.Button(description="Load CDM")
    output = widgets.Output()
    
    # File selector for local CSV
    file_widget = widgets.Dropdown(
        options=[f for f in os.listdir("data") if f.startswith("SOCRATES_sort-maxProb")],
        description="Select file:"
    )
    
    # Layout
    ui = widgets.VBox([choice_widget, file_widget, fetch_button, output])
    
    file_widget.layout.display = 'flex'  # default visible for local CSV
    
    # --- Functions ---
    def on_choice_change(change):
        file_widget.layout.display = 'flex' if change['new'] == 0 else 'none'
    
    def on_fetch_clicked(b):
        global cdm_df
        with output:
            clear_output()
            if choice_widget.value == 0:
                # Local CSV
                file_path = os.path.join("data", file_widget.value)
                cdm_df = pd.read_csv(file_path)
                print(f"Loaded local CDM file: {file_widget.value}, {len(cdm_df)} objects.")
            else:
                # Fetch latest from CelesTrak
                url = "https://celestrak.org/SOCRATES/sort-maxProb.csv"
                response = requests.get(url)
                if response.status_code != 200:
                    print(f"Error fetching data: {response.status_code}")
                    return
                
                # Convert to DataFrame
                from io import StringIO
                cdm_df = pd.read_csv(StringIO(response.text))
                
                # Save locally with today's date
                today = datetime.today().strftime("%Y%m%d")
                file_name = f"SOCRATES_sort-maxProb_{today}.csv"
                file_path = os.path.join("data", file_name)
                cdm_df.to_csv(file_path, index=False)
                
                print(f"Fetched latest CDM from CelesTrak ({len(cdm_df)} objects)")
                print(f"Saved as: {file_name}")
            
            print("CDM loaded successfully!")
            return cdm_df
    
    # --- Connect widgets ---
    choice_widget.observe(on_choice_change, names='value')
    fetch_button.on_click(on_fetch_clicked)
    
    # --- Display ---
    display(ui)