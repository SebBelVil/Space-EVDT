import pandas as pd
import requests

def load_discos_data(choice, username_widget=None, password_widget=None, token_widget=None):
    """
    Loads orbital information from ESA DISCOSweb through the API
    """
    if choice == 0:
        # CSV in the data folder
        file_path = "data/discos_objects_flat.csv"
        df = pd.read_csv(file_path)
        print(f"Extracted {len(df)} objects from local file.")
        return df
    
    elif choice == 1:
        # API mode
        username = username_widget.value
        password = password_widget.value
        token = token_widget.value
        
        URL = "https://discosweb.esoc.esa.int"
        headers = {"Authorization": f"Bearer {token}", "DiscosWeb-Api-Version": "2"}
        
        data = []
        url = f"{URL}/api/objects"
        
        while url:
            response = requests.get(url, headers=headers)
            if response.status_code != 200:
                print(f"Skipping page (error {response.status_code})")
                break
            try:
                doc = response.json()
            except:
                print("Skipping page (invalid JSON)")
                break
            data.extend(doc.get("data", []))
            next_link = doc.get("links", {}).get("next")
            url = URL + next_link if next_link and next_link.startswith("/") else next_link
            print(f"Fetched {len(data)} objects so far…")
            
        df = pd.DataFrame(data)
        return df