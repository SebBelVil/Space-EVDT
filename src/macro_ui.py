import ipywidgets as widgets
from IPython.display import display, clear_output

def gva_input_ui(default_gva=6266432, default_nav=0.641, default_comms=0.185, default_eo=0.174):
    """
    Interactive UI to set global GVA and sector weights with validation (weights sum = 1).
    """
    # --- Widgets ---
    gva_widget = widgets.FloatText(value=default_gva, description="Space GVA (M USD):")
    nav_widget = widgets.FloatText(value=default_nav, description="Nav weight:")
    comms_widget = widgets.FloatText(value=default_comms, description="Comms weight:")
    eo_widget = widgets.FloatText(value=default_eo, description="EO weight:")
    
    update_button = widgets.Button(description="Set Values")
    output = widgets.Output()
    
    ui = widgets.VBox([gva_widget, nav_widget, comms_widget, eo_widget, update_button, output])
    
    # --- Function ---
    result = {}
    
    def on_update_clicked(b):
        with output:
            clear_output()
            total_weight = nav_widget.value + comms_widget.value + eo_widget.value
            if abs(total_weight - 1.0) > 1e-6:
                print(f"Error: Weights sum to {total_weight:.3f}, but must equal 1. Please adjust.")
                return
            # Store values
            result['Space_GVA'] = gva_widget.value
            result['Wgva_Nav'] = nav_widget.value
            result['Wgva_Comms'] = comms_widget.value
            result['Wgva_EO'] = eo_widget.value
            print("Values set successfully!")
            #print(result)
    
    update_button.on_click(on_update_clicked)
    
    display(ui)
    
    return result