# demo_gui.py

# Suppress Tk deprecation warning on macOS
import os
os.environ["TK_SILENCE_DEPRECATION"] = "1"

import warnings
warnings.filterwarnings("ignore")

import tkinter as tk
import webbrowser
from pathlib import Path


# Import pipelines from main
from main import (
    geo_pipeline,
    grid_pipeline,
    ml_pipeline,
    analysis_pipeline,
    map_pipeline,
    error_check,
)

BASE_DIR = Path(__file__).resolve().parent


class DemoApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("E-scooter Prediction Demo")
        self.root.geometry("960x640")
        self.root.resizable(False, False)

        # Main layout
        self.main = tk.Frame(self.root, padx=40, pady=30)
        self.main.pack(fill="both", expand=True)

        self.header_label = tk.Label(
            self.main,
            text="E-scooter Parking Prediction",
            font=("Apple SD Gothic Neo", 24, "bold"),
        )
        self.header_label.pack(pady=(0, 30))

        self.body = tk.Frame(self.main)
        self.body.pack(expand=True)

        self.footer = tk.Frame(self.main)
        self.footer.pack(side="bottom", pady=(20, 0))

        # Pipeline buttons
        self._make_button("1. Geocoding Pipeline", geo_pipeline)
        self._make_button("2. Grid Pipeline", grid_pipeline)
        self._make_button("3. Model Testing (Train + Predict)", ml_pipeline)
        self._make_button("4. Result Analysis (Top-10)", analysis_pipeline)
        self._make_button("5. Visualization (Maps)", self.open_maps)
        self._make_button("6. Error Check (MAE / RMSE)", error_check)
        self._make_button(
            "7. All Pipelines",
            self.run_all_pipelines,
            pady=18,
        )

        tk.Button(
            self.footer,
            text="종료",
            font=("Apple SD Gothic Neo", 14, "bold"),
            width=12,
            command=self.root.destroy,
        ).pack()

    # UI helper
    def _make_button(self, text, command, pady=8):
        tk.Button(
            self.body,
            text=text,
            width=42,
            font=("Apple SD Gothic Neo", 14),
            command=lambda: self.run(command),
        ).pack(pady=pady)

    def run(self, fn):
        self.root.after(100, fn)

    # Open map HTML files
    def open_maps(self):
        real_map = BASE_DIR / "map" / "real_12.html"
        pred_map = BASE_DIR / "map" / "pred_12.html"
        error_map = BASE_DIR / "map" / "error_12.html"

        if real_map.exists():
            webbrowser.open(real_map.as_uri())
        if pred_map.exists():
            webbrowser.open(pred_map.as_uri())
        if error_map.exists():
            webbrowser.open(error_map.as_uri())

    # Run all pipelines sequentially
    def run_all_pipelines(self):
        self.root.after(100, geo_pipeline)
        self.root.after(200, grid_pipeline)
        self.root.after(300, ml_pipeline)
        self.root.after(400, analysis_pipeline)
        self.root.after(500, map_pipeline)
        self.root.after(600, error_check)


# Start Tk main loop
if __name__ == "__main__":
    root = tk.Tk()
    app = DemoApp(root)
    root.mainloop()