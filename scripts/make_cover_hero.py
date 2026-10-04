"""Paper-specific title-page hero for the ARC Prize 2026 Paper Track writeup.

Same visual language as the promo card (isometric ARC grid, Times serif, blue accent), but
distinct from it: a different grid and the single-GPU / engine-speedup framing the paper argues,
so the title-page cover is specific to the paper rather than the reused promo image.

-> paper/cover_hero.pdf            (vector, embedded by the LaTeX title page)
   paper/cover_hero.svg
   paper/cover_hero_1120x560.png
"""
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

ROOT = r"D:\ARC_AGI-2,3"

# A dense, four-fold-symmetric ARC motif on an azure ground so every cell renders. Distinct from
# the promo card's grid.
GRID = [
    [1, 8, 3, 3, 3, 3, 8, 1],
    [8, 2, 2, 2, 2, 2, 2, 8],
    [3, 2, 7, 7, 7, 7, 2, 3],
    [3, 2, 7, 4, 4, 7, 2, 3],
    [3, 2, 7, 4, 4, 7, 2, 3],
    [3, 2, 7, 7, 7, 7, 2, 3],
    [8, 2, 2, 2, 2, 2, 2, 8],
    [1, 8, 3, 3, 3, 3, 8, 1],
]

ARC = ["#101014", "#1E6FD0", "#E23B36", "#37B24D", "#F0C419",
       "#9AA0A6", "#D6489B", "#E8820C", "#5FC6E8", "#8A1B2E"]
INK, BLUE, GREY = "#1b1b1b", "#1E5FA8", "#6b6b6b"

AX_RECT = [0.0, 0.10, 0.63, 0.86]
ELEV, AZIM = 30, -60
TEXT_X = 0.635


def blocks3d(ax, grid):
    h, w = len(grid), len(grid[0])
    xs, ys, zs, dx, dy, dz, cols = [], [], [], [], [], [], []
    for r in range(h):
        for c in range(w):
            k = grid[r][c]
            xs.append(c + 0.06); ys.append((h - 1 - r) + 0.06); zs.append(0)
            dx.append(0.88); dy.append(0.88); dz.append(1.0)
            cols.append(ARC[k])
    ax.bar3d(xs, ys, zs, dx, dy, dz, color=cols, edgecolor="white", linewidth=0.5, shade=True)
    ax.set_proj_type("ortho")
    ax.set_xlim(0, w); ax.set_ylim(0, h); ax.set_zlim(0, 1.6)
    ax.set_box_aspect((w, h, 2.2))
    ax.view_init(elev=ELEV, azim=AZIM)
    ax.set_axis_off()


def build():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "svg.fonttype": "path",
    })
    fig = plt.figure(figsize=(560 / 72, 280 / 72))
    fig.patch.set_facecolor("white")
    bg = fig.add_axes([0, 0, 1, 1]); bg.axis("off")
    bg.set_xlim(0, 1); bg.set_ylim(0, 1)

    ax = fig.add_axes(AX_RECT, projection="3d")
    blocks3d(ax, GRID)

    # The June cover read 16.28%, a public-evaluation blend later superseded by 18.19 on the hidden
    # set. These are the two numbers the paper and the writeup lead with.
    bg.text(TEXT_X, 0.60, "18.19%", fontsize=46, color=BLUE, weight="bold", va="center")
    bg.text(TEXT_X, 0.405, "on a single 16 GB GPU", fontsize=12.5, color=INK, va="center")
    bg.text(TEXT_X, 0.315, "29.90% on the contest's four L4s", fontsize=12.5, color=GREY,
            va="center")
    bg.text(TEXT_X, 0.10, "ARC Prize 2026  ·  Paper Track", fontsize=11.5, color=GREY, va="center")
    return fig


def main():
    fig = build()
    fig.savefig(os.path.join(ROOT, "paper", "cover_hero.pdf"))          # vector, for LaTeX
    svg = os.path.join(ROOT, "paper", "cover_hero.svg")
    fig.savefig(svg)
    s = open(svg, encoding="utf-8").read()
    s = re.sub(r'(<svg[^>]*?)\swidth="[^"]*"', r'\1 width="560"', s, count=1)
    s = re.sub(r'(<svg[^>]*?)\sheight="[^"]*"', r'\1 height="280"', s, count=1)
    open(svg, "w", encoding="utf-8").write(s)
    fig.savefig(os.path.join(ROOT, "paper", "cover_hero_1120x560.png"), dpi=144)
    print("saved cover_hero.pdf, cover_hero.svg, cover_hero_1120x560.png")


if __name__ == "__main__":
    main()
