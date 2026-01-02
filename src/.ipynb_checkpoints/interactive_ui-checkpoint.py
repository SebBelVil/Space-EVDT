import ipywidgets as widgets
from IPython.display import display, clear_output
from .discos_loader import load_discos_data

def show_discos_ui():
    """
    UI with widget to load DISCOS data 
    """
    # --- Widgets ---
    choice_widget = widgets.ToggleButtons(
        options=[("Local CSV", 0), ("DISCOS Web API", 1)],
        description="Data Source:"
    )

    username_widget = widgets.Text(description="Username:")
    password_widget = widgets.Password(description="Password:")
    token_widget = widgets.Password(description="Token:")

    fetch_button = widgets.Button(description="Load Data")
    output = widgets.Output()

    # --- Layout ---
    cred_box = widgets.VBox([username_widget, password_widget, token_widget])
    ui = widgets.VBox([choice_widget, cred_box, fetch_button, output])
    cred_box.layout.display = 'none'  # hide initially

    # --- Functions ---
    def on_choice_change(change):
        cred_box.layout.display = 'flex' if change['new'] == 1 else 'none'

    def on_fetch_clicked(b):
        global discos_df  # declare discos_df as global here
        with output:
            clear_output()
            if choice_widget.value == 0:
                discos_df = load_discos_data(0)
            else:
                discos_df = load_discos_data(
                    choice=1,
                    username_widget=username_widget,
                    password_widget=password_widget,
                    token_widget=token_widget
                )
            print("Data loaded successfully!")
            print(f"Total objects: {len(discos_df)}")

    # --- Connect widgets ---
    choice_widget.observe(on_choice_change, names='value')
    fetch_button.on_click(on_fetch_clicked)

    # --- Display everything ---
    display(ui)