import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.neighbors import NearestNeighbors

# ============================================================
# Global style
# ============================================================

np.random.seed(7)

plt.rcParams.update({
    "font.size": 36,
    'font.family': 'Aptos',
    "axes.titlesize": 22,
    "axes.labelsize": 22,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 20,
    "figure.dpi": 160,
    "savefig.dpi": 320,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# ------------------------------------------------------------
# Soft macaron-like palette
# ------------------------------------------------------------
HUMAN = "#8EC5E8"       # soft light blue
MODEL = "#EE8F95"       # soft pink-red
ACCENT = "#F6C177"      # apricot for missed regions
PROBE = "#D95F5F"       # darker coral red for probes
TEXT = "#4A4A4A"

CMAP = "viridis"


# ============================================================
# Utilities
# ============================================================

def clean_3d_axes(ax, elev=22, azim=38):
    ax.view_init(elev=elev, azim=azim)

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis._axinfo["grid"]["linewidth"] = 0.75
        axis._axinfo["grid"]["linestyle"] = "-"
        axis._axinfo["grid"]["color"] = (0.86, 0.86, 0.86, 1.0)
        axis._axinfo["axisline"]["linewidth"] = 0.9
        axis._axinfo["axisline"]["color"] = (0.68, 0.68, 0.68, 1.0)

    ax.set_xlabel("x", labelpad=8)
    ax.set_ylabel("y", labelpad=8)
    ax.set_zlabel("z", labelpad=12)

    ax.tick_params(colors="#6B7280", pad=2)
    ax.set_box_aspect((1, 1, 1))


def set_equal_limits(ax, pts_list, pad=0.4):
    pts = np.vstack(pts_list)
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    center = (mins + maxs) / 2
    radius = (maxs - mins).max() / 2 + pad

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def clustered_points_3d(centers, n_per_cluster=120, scale=0.15):
    pts = []
    for c in centers:
        pts.append(np.random.normal(loc=c, scale=scale, size=(n_per_cluster, 3)))
    return np.vstack(pts)


def estimate_lid(points, k=20):
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(points)
    dists, _ = nbrs.kneighbors(points)
    dists = dists[:, 1:]  # remove self-distance
    rk = dists[:, -1][:, None]
    eps = 1e-12
    lid = -1.0 / np.mean(np.log((dists + eps) / (rk + eps)), axis=1)
    return lid


def uniform_volume_points(n=600, low=-1.7, high=1.7):
    return np.random.uniform(low, high, size=(n, 3))


# ============================================================
# 1) Coverage
# ============================================================

def make_coverage_figure():
    human_centers = np.array([
        [-1.9, -1.3, -1.4],
        [-1.4,  1.6, -0.8],
        [ 0.0,  0.0,  0.0],
        [ 1.4, -1.5,  1.1],
        [ 2.0,  1.4,  1.7],
        [ 2.4, -0.2, -1.7],
        [-2.2,  0.1,  1.8],
    ])

    model_centers = np.array([
        [-1.4,  1.6, -0.8],
        [ 0.0,  0.0,  0.0],
        [ 1.4, -1.5,  1.1],
    ])

    human_pts = clustered_points_3d(human_centers, n_per_cluster=80, scale=0.22)
    model_pts = clustered_points_3d(model_centers, n_per_cluster=90, scale=0.22)

    missed = np.array([
        [-1.9, -1.3, -1.4],
        [ 2.0,  1.4,  1.7],
        [ 2.4, -0.2, -1.7],
        [-2.2,  0.1,  1.8],
    ])

    fig = plt.figure(figsize=(6.5, 6.0))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        human_pts[:, 0], human_pts[:, 1], human_pts[:, 2],
        s=18, c=HUMAN, alpha=0.22, depthshade=False, linewidths=0
    )
    ax.scatter(
        model_pts[:, 0], model_pts[:, 1], model_pts[:, 2],
        s=22, c=MODEL, alpha=0.88, depthshade=False, linewidths=0
    )
    ax.scatter(
        missed[:, 0], missed[:, 1], missed[:, 2],
        s=84, c=ACCENT, alpha=0.98, marker="D", depthshade=False, linewidths=0
    )

    clean_3d_axes(ax)
    set_equal_limits(ax, [human_pts, model_pts, missed], pad=0.55)
    ax.set_title("Low coverage", pad=12)

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Human population',
               markerfacecolor=HUMAN, markeredgecolor='none', markersize=10, alpha=0.9),
        Line2D([0], [0], marker='o', color='w', label='Model population',
               markerfacecolor=MODEL, markeredgecolor='none', markersize=10),
        Line2D([0], [0], marker='D', color='w', label='Missed regions',
               markerfacecolor=ACCENT, markeredgecolor='none', markersize=9),
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper left",
        bbox_to_anchor=(0.00, 0.98),
        frameon=False,
        borderaxespad=0.0,
        handletextpad=0.4
    )

    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.02, top=0.92)
    return fig


# ============================================================
# 2) Uniformity
# ============================================================

def make_uniformity_figure():
    human_pts = uniform_volume_points(n=560, low=-1.65, high=1.65)

    model_centers = np.array([
        [-1.0, -0.8,  0.9],
        [ 0.9,  1.0,  0.7],
        [ 0.8, -1.1, -0.8],
        [-0.7,  0.8, -1.0],
    ])
    model_pts = clustered_points_3d(model_centers, n_per_cluster=105, scale=0.18)

    probes = np.array([
        [0.00, 0.00, 0.00],
        [-0.10, 0.30, -0.10],
        [0.18, -0.25, 0.18],
    ])

    fig = plt.figure(figsize=(6.5, 6.0))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        human_pts[:, 0], human_pts[:, 1], human_pts[:, 2],
        s=16, c=HUMAN, alpha=0.15, depthshade=False, linewidths=0
    )
    ax.scatter(
        model_pts[:, 0], model_pts[:, 1], model_pts[:, 2],
        s=22, c=MODEL, alpha=0.88, depthshade=False, linewidths=0
    )
    ax.scatter(
        probes[:, 0], probes[:, 1], probes[:, 2],
        s=80, c=PROBE, marker="x", alpha=0.98, depthshade=False, linewidths=2.2
    )

    clean_3d_axes(ax)
    set_equal_limits(ax, [human_pts, model_pts, probes], pad=0.45)
    ax.set_title("Low uniformity", pad=12)

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Human population',
               markerfacecolor=HUMAN, markeredgecolor='none', markersize=10, alpha=0.9),
        Line2D([0], [0], marker='o', color='w', label='Model population',
               markerfacecolor=MODEL, markeredgecolor='none', markersize=10),
        Line2D([0], [0], marker='x', color=PROBE, label='Random probes',
               markersize=10, markeredgewidth=2),
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper left",
        bbox_to_anchor=(0.00, 0.98),
        frameon=False,
        borderaxespad=0.0,
        handletextpad=0.4
    )

    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.02, top=0.92)
    return fig


# ============================================================
# 3) Complexity
# ============================================================

def generate_line_points(n=420):
    t = np.linspace(-2.2, 2.2, n)
    pts = np.stack([
        t,
        0.035 * np.random.randn(n),
        0.035 * np.random.randn(n),
    ], axis=1)
    return pts


def make_complexity_figure():
    human_pts = uniform_volume_points(n=580, low=-1.7, high=1.7)
    model_pts = generate_line_points(n=420)

    lid = estimate_lid(model_pts, k=20)

    fig = plt.figure(figsize=(6.7, 6.0))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        human_pts[:, 0], human_pts[:, 1], human_pts[:, 2],
        s=16, c=HUMAN, alpha=0.14, depthshade=False, linewidths=0
    )
    sc = ax.scatter(
        model_pts[:, 0], model_pts[:, 1], model_pts[:, 2],
        c=lid,
        cmap=CMAP,
        s=24,
        alpha=0.92,
        depthshade=False,
        linewidths=0
    )

    clean_3d_axes(ax)
    set_equal_limits(ax, [human_pts, model_pts], pad=0.50)
    ax.set_title("Low complexity (line)", pad=12)

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Human population',
               markerfacecolor=HUMAN, markeredgecolor='none', markersize=10, alpha=0.9),
        Line2D([0], [0], marker='o', color='w', label='Model population',
               markerfacecolor="#5B5B5B", markeredgecolor='none', markersize=10),
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper left",
        bbox_to_anchor=(0.00, 0.98),
        frameon=False,
        borderaxespad=0.0,
        handletextpad=0.4
    )

    cbar = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.06)
    cbar.set_label("LID", labelpad=8)
    cbar.ax.tick_params(labelsize=11)

    fig.subplots_adjust(left=0.12, right=0.90, bottom=0.02, top=0.92)
    return fig


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    fig1 = make_coverage_figure()
    fig2 = make_uniformity_figure()
    fig3 = make_complexity_figure()

    savekw = dict(transparent=True)
    fig1.savefig("coverage_3d_macaron.pdf", **savekw)
    fig2.savefig("uniformity_3d_macaron.pdf", **savekw)
    fig3.savefig("complexity_3d_macaron.pdf", **savekw)

    fig1.savefig("coverage_3d_macaron.png", **savekw)
    fig2.savefig("uniformity_3d_macaron.png", **savekw)
    fig3.savefig("complexity_3d_macaron.png", **savekw)

    plt.show()
