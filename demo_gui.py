# demo_gui.py
# ======================================
# macOS Tk 경고 억제 (Tk import 이전)
# ======================================
import os
os.environ["TK_SILENCE_DEPRECATION"] = "1"

import warnings
warnings.filterwarnings("ignore")

import sys
import io
import tkinter as tk
import tkinter.messagebox as messagebox
import webbrowser
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# === 기존 코드 import (호출만) ===
from src.predict_rf import predict_rf


# ======================================
# stdout 억제용 컨텍스트
# ======================================
class SilentStdout:
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = io.StringIO()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._stdout


class DemoApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("대회 시연")
        self.root.geometry("960x640")
        self.root.resizable(False, False)

        self.step_index = 0

        # =========================
        # 레이아웃
        # =========================
        self.main = tk.Frame(self.root, padx=28, pady=22)
        self.main.pack(fill="both", expand=True)

        self.content = tk.Frame(self.main)
        self.content.pack(fill="both", expand=True)

        self.footer = tk.Frame(self.main)
        self.footer.pack(side="bottom", fill="x", pady=(14, 20))

        # 버튼은 하나만 유지
        self.next_button = tk.Button(
            self.footer,
            font=("Apple SD Gothic Neo", 14, "bold"),
            width=14,
        )
        self.next_button.pack()

        # =========================
        # STEP 목록
        # =========================
        self.STEPS = [
            self.step1_intro,        # 0
            self.step2_problem,      # 1
            self.step3_data,         # 2
            self.step4_predict,      # 3
            self.step5_result,       # 4
            self.step6_validation,   # 5
            self.step7_outro,        # 6
        ]

        self.go_step(0)

    # -----------------
    # 공통 UI
    # -----------------
    def clear(self):
        for w in self.content.winfo_children():
            w.destroy()

    def header(self, title: str):
        tk.Label(
            self.content,
            text=title,
            font=("Apple SD Gothic Neo", 22, "bold"),
            anchor="w",
        ).pack(fill="x", pady=(0, 14))

    def body(self, lines):
        tk.Label(
            self.content,
            text="\n".join(lines[:3]),
            font=("Apple SD Gothic Neo", 14),
            justify="left",
            anchor="w",
        ).pack(fill="x", pady=(0, 12))

    def set_button(self, text: str, command, state="normal"):
        self.next_button.config(text=text, command=command, state=state)

    # -----------------
    # STEP 이동
    # -----------------
    def go_step(self, idx: int):
        self.step_index = idx
        self.STEPS[idx]()

    # ============================
    # STEP 1. Intro
    # ============================
    def step1_intro(self):
        print("[STEP 1] Intro")
        self.clear()

        self.header("전동킥보드 불법주차 예측 시연")
        self.body(
            [
                "팀명: (여기에 팀명 입력)",
                "불법 주차가 집중될 지역을 사전에 예측합니다.",
                "",
            ]
        )

        self.set_button("시연 시작", lambda: self.go_step(1))

    # ============================
    # STEP 2. Problem Definition
    # ============================
    def step2_problem(self):
        print("[STEP 2] Problem Definition")
        self.clear()

        self.header("문제 정의")
        self.body(
            [
                "내용1",
                "내용2",
                "",
            ]
        )

        self.set_button("다음", lambda: self.go_step(2))

    # ============================
    # STEP 3. Data Overview
    # ============================
    def step3_data(self):
        print("[STEP 3] Data Overview")
        self.clear()

        self.header("데이터 개요")
        self.body(
            [
                "불법 주차 발생 위치를 200m 격자로 집계했습니다.",
                "각 격자는 한 달 동안의 견인 발생 횟수를 나타냅니다.",
                "",
            ]
        )

        df = pd.read_csv("data/predata_12.csv")

        fig = Figure(figsize=(7.5, 4.2), dpi=100)
        ax = fig.add_subplot(111)
        ax.hist(df["count"], bins=30)
        ax.set_title("격자별 불법 주차 발생량 분포")
        ax.set_xlabel("발생 횟수")
        ax.set_ylabel("격자 수")

        canvas = FigureCanvasTkAgg(fig, master=self.content)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, pady=(8, 0))

        self.set_button("다음", lambda: self.go_step(3))

    # ============================
    # STEP 4. Prediction
    # ============================
    def step4_predict(self):
        print("[STEP 4] Prediction")
        self.clear()

        self.header("예측 실행")
        self.body(
            [
                "사전 학습된 모델을 로드해 예측을 수행.",
                "모델 학습은 사전에 완료.",
                "",
            ]
        )

        status = tk.Label(
            self.content,
            text="대기 중...",
            font=("Apple SD Gothic Neo", 13),
        )
        status.pack(pady=(6, 12))

        def run_prediction():
            status.config(text="예측 중...")
            self.set_button("예측 중", lambda: None, state="disabled")

            def _do():
                with SilentStdout():
                    predict_rf(
                        data_path="data/features.csv",
                        model_path="model_rf.pkl",
                        out_path="data/pred_12.csv",
                        pred_month=11,
                    )

                status.config(text="예측 완료")
                self.set_button("다음", lambda: self.go_step(4))

            self.root.after(800, _do)

        self.set_button("예측 실행", run_prediction)

    # ============================
    # STEP 5. Prediction Result
    # ============================
    def step5_result(self):
        print("[STEP 5] Prediction Result")
        self.clear()

        self.header("예측 결과")
        self.body(
            [
                "내용1",
                "내용2.",
                "",
            ]
        )

        webbrowser.open("map/pred_12.html")
        self.set_button("다음", lambda: self.go_step(5))

    # ============================
    # STEP 6. Validation
    # ============================
    def step6_validation(self):
        print("[STEP 6] Validation")
        self.clear()

        self.header("검증")
        self.body(
            [
                "내용1",
                "내용2.",
                "",
            ]
        )

        try:
            real = pd.read_csv("data/predata_12.csv")
            pred = pd.read_csv("data/pred_12.csv")

            merged = real.merge(pred, on="grid_id", how="inner")

            fig = Figure(figsize=(7.5, 4.2), dpi=100)
            ax = fig.add_subplot(111)

            ax.scatter(
                merged["count_x"],  # 실제
                merged["count_y"],  # 예측
                alpha=0.4
            )

            ax.set_xlabel("실제 발생량")
            ax.set_ylabel("예측 발생량")
            ax.set_title("실제 vs 예측 비교")

            canvas = FigureCanvasTkAgg(fig, master=self.content)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True, pady=(8, 0))

        except Exception:
            self.body(
                [
                    "검증 데이터를 불러올 수 없습니다.",
                    "실제/예측 결과 파일을 확인하세요.",
                    "",
                ]
            )

        self.set_button("다음", lambda: self.go_step(6))

    # ============================
    # STEP 7. Outro
    # ============================
    def step7_outro(self):
        print("[STEP 7] Outro")
        self.clear()

        self.header("시연 종료")
        self.body(
            [
                "불법 주차를 사전에 예측해",
                "효율적인 관리 전략을 제시할 수 있습니다.",
                "감사합니다.",
            ]
        )

        self.set_button("종료", self.root.destroy)


# ============================
# mainloop (1회)
# ============================
if __name__ == "__main__":
    root = tk.Tk()
    app = DemoApp(root)
    root.mainloop()