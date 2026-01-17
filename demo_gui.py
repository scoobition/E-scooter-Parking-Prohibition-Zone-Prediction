# demo_gui.py
# ======================================
# macOS Tk 경고 억제 (Tk import 이전)
# ======================================
import os
os.environ["TK_SILENCE_DEPRECATION"] = "1"

import warnings
warnings.filterwarnings("ignore")

import tkinter as tk

# =========================
# main.py 파이프라인 import
# =========================
from main import (
    geo_pipeline,
    grid_pipeline,
    ml_pipeline,
    analysis_pipeline,
    map_pipeline,
)


class DemoApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("E-scooter Prediction Demo")
        self.root.geometry("960x640")
        self.root.resizable(False, False)

        # -----------------
        # Layout
        # -----------------
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

        # -----------------
        # Buttons (main.py 메뉴 그대로)
        # -----------------
        self._make_button("1. Geocoding Pipeline", geo_pipeline)
        self._make_button("2. Grid Pipeline", grid_pipeline)
        self._make_button("3. Model Testing (Train + Predict)", ml_pipeline)
        self._make_button("4. Result Analysis (Top-10)", analysis_pipeline)
        self._make_button("5. Visualization (Maps)", map_pipeline)
        self._make_button(
            "6. All Pipelines",
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

    # =========================
    # UI Helper
    # =========================
    def _make_button(self, text, command, pady=8):
        tk.Button(
            self.body,
            text=text,
            width=36,
            font=("Apple SD Gothic Neo", 14),
            command=lambda: self.run(command),
        ).pack(pady=pady)

    def run(self, fn):
        """
        버튼 클릭 → 기존 main.py 함수 실행
        """
        self.root.after(100, fn)

    # =========================
    # main.py 메뉴 6번과 동일
    # =========================
    def run_all_pipelines(self):
        self.root.after(100, geo_pipeline)
        self.root.after(200, grid_pipeline)
        self.root.after(300, ml_pipeline)
        self.root.after(400, analysis_pipeline)
        self.root.after(500, map_pipeline)


# ============================
# mainloop (1회)
# ============================
if __name__ == "__main__":
    root = tk.Tk()
    app = DemoApp(root)
    root.mainloop()