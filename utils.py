import os
import pickle


def load_asset(filename, asset_path: str = "./saved_models/"):
    print(f"Loading asset: {filename}")
    with open(os.path.join(asset_path, filename), "rb") as f:
        asset = pickle.load(f)
    return asset
