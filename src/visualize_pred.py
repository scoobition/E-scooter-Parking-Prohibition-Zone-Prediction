import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

PRED_PATH = "data/pred_12.csv"
META_PATH = "data/grid_meta.csv"

GRID_SIZE_M = 200  # grid.py에서 사용한 격자 한 변 길이(미터)로 맞추기


def load_and_merge():
    pred = pd.read_csv(PRED_PATH)
    meta = pd.read_csv(META_PATH)
    return pred.merge(meta, on="grid_id", how="left")


def print_top10(df):
    top10 = df.sort_values("pred_12", ascending=False).head(10).copy()
    top10 = top10.reset_index(drop=True)
    top10.index += 1
    print("\n[TOP 10 HIGH-RISK GRIDS]")
    print(top10[["grid_id", "pred_12"]])

def plot_grid_heatmap(df, grid_size=GRID_SIZE_M, alpha=0.55):
    """
    격자별 예측값을 '붉은색 농도'로 표현
    """

    df = df.dropna(subset=["center_x_m", "center_y_m", "pred_12"]).copy()

    # ===== 값 정규화 (색 농도용) =====
    v_raw = df["pred_12"].values.astype(float)
    vmin, vmax = np.percentile(v_raw, 5), np.percentile(v_raw, 95)
    v = np.clip((v_raw - vmin) / (vmax - vmin + 1e-9), 0, 1)

    fig, ax = plt.subplots(figsize=(9, 9))

    half = grid_size / 2

    # ===== 핵심 1: axis 범위 계산 =====
    xmin = df["center_x_m"].min() - half
    xmax = df["center_x_m"].max() + half
    ymin = df["center_y_m"].min() - half
    ymax = df["center_y_m"].max() + half

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # ===== 격자 그리기 =====
    for (x, y, t) in zip(df["center_x_m"], df["center_y_m"], v):
        color = plt.cm.Reds(t)
        rect = Rectangle(
            (x - half, y - half),
            grid_size,
            grid_size,
            facecolor=(color[0], color[1], color[2], alpha),
            edgecolor="none",
        )
        ax.add_patch(rect)

    ax.set_title("December Scooter Towing Risk (Grid Heatmap)")
    ax.set_aspect("equal")
    ax.set_axis_off()

    # ===== 컬러바 =====
    sm = plt.cm.ScalarMappable(
        cmap="Reds",
        norm=plt.Normalize(vmin=vmin, vmax=vmax)
    )
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.036, pad=0.02)
    cbar.set_label("Predicted towing count (pred_12)")

    plt.show()


if __name__ == "__main__":
    df = load_and_merge()
    print(df.head())

    plot_grid_heatmap(df, grid_size=GRID_SIZE_M, alpha=0.55)
    print_top10(df)
