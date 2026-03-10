import matplotlib.pyplot as plt


ALGO_COLORS = {
    "ppo": "#0072B2",
    "ppo_lag": "#D55E00",
    "caf_cone": "#009E73",
    "caf_cone_pareto": "#C00000",
    "cost_cone": "#7B61FF",
    "cost_cone_pareto": "#E61D1D",
}


ALGO_LABELS = {
    "ppo": "PPO",
    "ppo_lag": "PPO-Lag",
    "caf_cone": "CAF-CONE",
    "caf_cone_pareto": "CAF-CONE-Pareto",
    "cost_cone": "COST-CONE",
    "cost_cone_pareto": "COST-CONE-Pareto",
}


def apply_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "STIXGeneral"],
            "mathtext.fontset": "stix",
            "axes.titlesize": 28,
            "axes.labelsize": 32,
            "xtick.labelsize": 28,
            "ytick.labelsize": 28,
            "legend.fontsize": 24,
            "figure.titlesize": 32,
        }
    )
