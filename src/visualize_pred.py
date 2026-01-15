import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from pathlib import Path
from typing import Optional

PRED_PATH = "data/pred_12.csv"
META_PATH = "data/grid_meta.csv"
GRID_SIZE_M = 200  # grid.py에서 사용한 격자 한 변 길이(미터)


def load_and_merge(pred_path: str = PRED_PATH, meta_path: str = META_PATH) -> pd.DataFrame:
    pred = pd.read_csv(pred_path)
    meta = pd.read_csv(meta_path)
    df = pred.merge(meta, on="grid_id", how="left")
    df = df.dropna(subset=["center_x_m", "center_y_m", "pred_12"]).copy()
    return df


def print_top10(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    top = df.sort_values("pred_12", ascending=False).head(n).copy()
    top = top.reset_index(drop=True)
    top.index += 1
    print("\n[TOP 10 HIGH-RISK GRIDS]")
    print(top[["grid_id", "pred_12"]])
    return top


def plot_grid_heatmap(
    df: pd.DataFrame,
    grid_size: int = GRID_SIZE_M,
    alpha: float = 0.55,
    save_path: Optional[str] = None,
    show: bool = True,
):
    """center_x_m/center_y_m 기반으로 예측값 heatmap 스타일로 그리기."""
    x = df["center_x_m"].to_numpy()
    y = df["center_y_m"].to_numpy()
    v = df["pred_12"].to_numpy()

    vmin, vmax = float(np.nanmin(v)), float(np.nanmax(v))

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect("equal", adjustable="box")

    # 값이 큰 격자를 위에 그리도록 정렬
    order = np.argsort(v)
    for idx in order:
        xi, yi, vi = x[idx], y[idx], v[idx]
        t = 1.0 if vmax <= vmin else (vi - vmin) / (vmax - vmin)
        t = float(np.clip(t, 0.0, 1.0))
        color = plt.cm.Reds(t)
        rect = Rectangle(
            (xi - grid_size / 2, yi - grid_size / 2),
            grid_size,
            grid_size,
            facecolor=color,
            edgecolor=None,
            alpha=alpha,
            linewidth=0.0,
        )
        ax.add_patch(rect)

    # ===== [핵심 추가] 축 범위를 데이터 기준으로 설정 =====
    xmin = x.min() - grid_size
    xmax = x.max() + grid_size
    ymin = y.min() - grid_size
    ymax = y.max() + grid_size

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    ax.set_title("Predicted towing count heatmap (pred_12)")
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")

    sm = plt.cm.ScalarMappable(cmap="Reds", norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.036, pad=0.02)
    cbar.set_label("Predicted towing count (pred_12)")

    if save_path:
        out_p = Path(save_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_p, dpi=200, bbox_inches="tight")
        print(f"[DONE] 시각화 저장: {out_p}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def visualize_pred(
    pred_path: str = PRED_PATH,
    meta_path: str = META_PATH,
    grid_size: int = GRID_SIZE_M,
    alpha: float = 0.55,
    save_path: Optional[str] = None,
    show: bool = True,
):
    df = load_and_merge(pred_path, meta_path)
    plot_grid_heatmap(df, grid_size=grid_size, alpha=alpha, save_path=save_path, show=show)
    print_top10(df)


def main():
    visualize_pred()


if __name__ == "__main__":
    main()
