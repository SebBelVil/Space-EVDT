import ipywidgets as widgets
from IPython.display import display, clear_output
import pandas as pd
from datetime import datetime
import os
import io

def show_gp_ui():
    """
    Interactive UI for loading GP (General Perturbations) data.
    User can select local CSV (from multiple files in data folder)
    or fetch latest via Space-Track session.
    """
    # --- Widgets ---
    choice_widget = widgets.ToggleButtons(
        options=[("Local CSV", 0), ("Fetch Latest (requires Space-Track login)", 1)],
        description="Data Source:"
    )

    # Local CSV dropdown (populated later)
    local_files = [f for f in os.listdir("data") if f.startswith("GP_SPACETRACK") and f.endswith(".csv")]
    local_dropdown = widgets.Dropdown(
        options=local_files,
        description="Select File:"
    )

    username_widget = widgets.Text(description="Username:")
    password_widget = widgets.Password(description="Password:")

    fetch_button = widgets.Button(description="Load GP Data")
    output = widgets.Output()

    # --- Layout ---
    local_box = widgets.VBox([local_dropdown])
    cred_box = widgets.VBox([username_widget, password_widget])
    ui = widgets.VBox([choice_widget, local_box, cred_box, fetch_button, output])

    local_box.layout.display = 'flex' if local_files else 'none'
    cred_box.layout.display = 'none'  # hide initially

    # --- Functions ---
    def on_choice_change(change):
        if change['new'] == 0:  # Local CSV
            local_box.layout.display = 'flex'
            cred_box.layout.display = 'none'
        else:  # Fetch latest
            local_box.layout.display = 'none'
            cred_box.layout.display = 'flex'

    def on_fetch_clicked(b):
        global df_gp  # store dataframe globally
        with output:
            clear_output()
            
            if choice_widget.value == 0:
                # Load selected local file
                file_path = os.path.join("data", local_dropdown.value)
                df_gp = pd.read_csv(file_path)
                print(f"Loaded {len(df_gp)} GP entries from {file_path}")
            
            else:
                # Session-based fetch via Space-Track
                import requests
                session = requests.Session()
                login_url = "https://www.space-track.org/ajaxauth/login"
                login_data = {"identity": username_widget.value, "password": password_widget.value}
                r = session.post(login_url, data=login_data)
                if r.status_code != 200:
                    print(f"Login failed: {r.status_code}")
                    return
                
                # Fetch GP data
                gp_url = "https://www.space-track.org/basicspacedata/query/class/gp/orderby/CCSDS_OMM_VERS%20asc/format/csv/emptyresult/show"
                response = session.get(gp_url)
                if response.status_code != 200:
                    print(f"Failed to fetch GP data: {response.status_code}")
                    return
                
                df_gp = pd.read_csv(io.StringIO(response.text))
                
                # Save locally with today’s date
                today_str = datetime.today().strftime("%Y%m%d")
                file_path = f"data/GP_SPACETRACK_{today_str}.csv"
                df_gp.to_csv(file_path, index=False)
                print(f"Fetched {len(df_gp)} GP entries and saved to {file_path}")

    # --- Connect widgets ---
    choice_widget.observe(on_choice_change, names='value')
    fetch_button.on_click(on_fetch_clicked)

    # --- Display UI ---
    display(ui)