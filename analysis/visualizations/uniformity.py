import numpy as np
import matplotlib.pyplot as plt
from itertools import product, combinations

# ============================================================
# Global style
# ============================================================

np.random.seed(7)

plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "figure.dpi": 160,
    "savefig.dpi": 320,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "DejaVu Sans",
})

# Macaron-ish palette
HUMAN = "#8EC5E8"      # light blue
REGULAR = "#F3C178"    # apricot
CLUSTER = "#E98B95"    # rose
EDGE = "#B8B8B8"       # light gray
TEXT = "#4A4A4A"


# ============================================================
# Utilities
# ============================================================

def clean_diagram_axes(ax, elev=20, azim=35):
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.35, 1.35)
    ax.set_zlim(-1.35, 1.35)


def draw_wire_cube(ax, color=EDGE, lw=1.5, alpha=0.9):
    r = [-1, 1]
    vertices = np.array(list(product(r, r, r)))
    for s, e in combinations(vertices, 2):
        if np.sum(np.abs(s - e)) == 2:
            ax.plot3D(*zip(s, e), color=color, linewidth=lw, alpha=alpha)


def human_points(n=85):
    return np.random.uniform(-0.95, 0.95, size=(n, 3))


def clustered_points():
    centers = np.array([
        [-0.78, -0.72, -0.76],
        [0.76, 0.74, 0.78],
        [-0.82, 0.72, 0.76],
    ])
    pts = []
    for c in centers:
        pts.append(c + 0.12 * np.random.randn(28, 3))
    return np.vstack(pts)


def regular_points():
    corners = np.array(list(product([-1, 1], repeat=3)))
    mids = np.array([
        [0, 0, 1], [0, 0, -1],
        [0, 1, 0], [0, -1, 0],
        [1, 0, 0], [-1, 0, 0],
    ])
    return np.vstack([corners, mids])


def add_soft_shadow_scatter(ax, pts, color, s_main, alpha_main, s_bg=None, alpha_bg=0.08):
    if s_bg is None:
        s_bg = s_main * 2.2
    ax.scatter(
        pts[:, 0], pts[:, 1], pts[:, 2],
        s=s_bg, c=color, alpha=alpha_bg, linewidths=0, depthshade=False
    )
    ax.scatter(
        pts[:, 0], pts[:, 1], pts[:, 2],
        s=s_main, c=color, alpha=alpha_main, linewidths=0, depthshade=False
    )


# ============================================================
# Panels
# ============================================================

def make_human_panel():
    pts = human_points()

    fig = plt.figure(figsize=(5.3, 5.0))
    ax = fig.add_subplot(111, projection="3d")

    add_soft_shadow_scatter(ax, pts, HUMAN, s_main=42, alpha_main=0.86, s_bg=105, alpha_bg=0.07)

    clean_diagram_axes(ax, elev=21, azim=34)

    return fig


def make_regular_panel():
    pts = regular_points()

    fig = plt.figure(figsize=(5.3, 5.0))
    ax = fig.add_subplot(111, projection="3d")

    draw_wire_cube(ax, color=EDGE, lw=1.6, alpha=0.95)
    add_soft_shadow_scatter(ax, pts, REGULAR, s_main=78, alpha_main=0.96, s_bg=170, alpha_bg=0.08)

    clean_diagram_axes(ax, elev=20, azim=35)

    return fig


def make_cluster_panel():
    pts = clustered_points()

    fig = plt.figure(figsize=(5.3, 5.0))
    ax = fig.add_subplot(111, projection="3d")

    add_soft_shadow_scatter(ax, pts, CLUSTER, s_main=48, alpha_main=0.90, s_bg=115, alpha_bg=0.09)

    clean_diagram_axes(ax, elev=18, azim=35)

    return fig


# ============================================================
# Optional combined figure
# ============================================================

def make_combined_figure():
    fig = plt.figure(figsize=(15.5, 5.2))

    ax1 = fig.add_subplot(131, projection="3d")
    ax2 = fig.add_subplot(132, projection="3d")
    ax3 = fig.add_subplot(133, projection="3d")

    # Human
    pts_h = human_points()
    add_soft_shadow_scatter(ax1, pts_h, HUMAN, s_main=42, alpha_main=0.86, s_bg=105, alpha_bg=0.07)
    clean_diagram_axes(ax1, elev=21, azim=34)

    # Regular
    pts_r = regular_points()
    draw_wire_cube(ax2, color=EDGE, lw=1.6, alpha=0.95)
    add_soft_shadow_scatter(ax2, pts_r, REGULAR, s_main=78, alpha_main=0.96, s_bg=170, alpha_bg=0.08)
    clean_diagram_axes(ax2, elev=20, azim=35)

    # Clustered
    pts_c = clustered_points()
    add_soft_shadow_scatter(ax3, pts_c, CLUSTER, s_main=48, alpha_main=0.90, s_bg=115, alpha_bg=0.09)
    clean_diagram_axes(ax3, elev=18, azim=35)

    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.90, wspace=0.02)
    return fig


# ============================================================
# Save
# ============================================================

if __name__ == "__main__":
    fig1 = make_human_panel()
    fig2 = make_regular_panel()
    fig3 = make_cluster_panel()
    fig_all = make_combined_figure()

    fig1.savefig("hopkins_human_polished.pdf", bbox_inches="tight", transparent=True)
    fig2.savefig("hopkins_regular_polished.pdf", bbox_inches="tight", transparent=True)
    fig3.savefig("hopkins_clustered_polished.pdf", bbox_inches="tight", transparent=True)
    fig_all.savefig("hopkins_combined_polished.pdf", bbox_inches="tight", transparent=True)

    fig1.savefig("hopkins_human_polished.png", bbox_inches="tight", transparent=True)
    fig2.savefig("hopkins_regular_polished.png", bbox_inches="tight", transparent=True)
    fig3.savefig("hopkins_clustered_polished.png", bbox_inches="tight", transparent=True)
    fig_all.savefig("hopkins_combined_polished.png", bbox_inches="tight", transparent=True)

    plt.show()
