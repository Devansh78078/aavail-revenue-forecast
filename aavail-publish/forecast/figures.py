"""Generate the training EDA, validation and production-replay charts."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import dates as mdates
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np
import pandas as pd

from .ingest import ALL
from .model import BASELINE, load_model

INK, SUB, GRID = "#162436", "#526477", "#e1e8ef"
TEAL, BLUE, AMBER, RED = "#007f78", "#3569a8", "#c07b19", "#b6414a"
REPLAY = [ALL, "United Kingdom", "EIRE", "Germany", "France", "Norway"]
LABELS = {
    "trailing_30": "Trailing 30 days", "seasonal_naive": "Prior year",
    "ridge": "Ridge", "ridge_activity": "Ridge + activity",
    "forest": "Random forest", "forest_activity": "Forest + activity",
}


def finish(fig, output):
    fig.savefig(output, dpi=160, bbox_inches="tight", pad_inches=0.25, facecolor="white")
    plt.close(fig)


def thousands(value, _):
    return f"{value / 1000:,.0f}k" if abs(value) >= 1000 else f"{value:,.0f}"


def heading(fig, title, subtitle):
    fig.text(.07, .97, title, fontsize=21, fontweight="bold", color=INK, va="top")
    fig.text(.07, .902, subtitle, fontsize=10, color=SUB, va="top")


def style(axis, direction="y"):
    axis.grid(axis=direction, color=GRID, linewidth=.75)
    axis.set_axisbelow(True)
    axis.spines[["top", "right", "left", "bottom"]].set_visible(False)
    axis.tick_params(length=0, labelcolor=SUB)
    if direction == "y":
        axis.yaxis.set_major_formatter(FuncFormatter(thousands))


def eda(bundle, output):
    daily = bundle["daily"]
    global_history = daily.xs(ALL, level="country")
    monthly = global_history["revenue"].resample("MS").sum().iloc[1:]
    countries = daily.groupby(level="country")["revenue"].sum().drop(ALL).nlargest(6)
    fig, (trend, markets) = plt.subplots(1, 2, figsize=(11, 4.5), gridspec_kw={"width_ratios":[1.65,1]})
    fig.subplots_adjust(left=.07, right=.97, top=.78, bottom=.15, wspace=.47)
    heading(fig, "UK revenue dominates the training data",
            "Invoice line values in source currency units  •  Training period only")
    style(trend)
    trend.set_title("Worldwide revenue by full month", loc="left", fontweight="bold", pad=12, color=INK)
    trend.bar(monthly.index, monthly, width=23, color=[TEAL if d.month in (11, 12) else BLUE for d in monthly.index])
    trend.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    trend.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    trend.set_ylim(0, 400000)
    style(markets, "x")
    markets.set_title("Share of total revenue by country", loc="left", fontweight="bold", pad=12, color=INK)
    shares = 100 * countries / global_history["revenue"].sum()
    bars = markets.barh(shares.index, shares, color=[TEAL] + [BLUE]*5)
    markets.set_xlim(0, 104)
    markets.invert_yaxis()
    markets.bar_label(bars, [f"{v:.1f}%" for v in shares], padding=5, color=INK, fontsize=9)
    marks = np.arange(0,101,25)
    markets.set_xticks(marks, [f"{v}%" for v in marks])
    finish(fig, output)


def comparison(bundle, validation, output):
    order = list(LABELS)
    val = validation.assign(error=(validation["prediction"]-validation["actual"]).abs())
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.4))
    fig.subplots_adjust(left=.14, right=.98, top=.83, bottom=.09, hspace=.64, wspace=1.05)
    heading(fig, "The best validation model varies by market",
            "Mean absolute error across 90 forecast dates per scope  •  Gold = trailing baseline")
    for ax, country in zip(axes.flat, REPLAY):
        values = val.loc[val["country"].eq(country)].groupby("model")["error"].mean().reindex(order).dropna()
        winner = bundle["models"][country]["name"]
        colors = [AMBER if name == BASELINE else TEAL if name == winner else "#8da6c0" for name in values.index]
        bars = ax.barh([LABELS[name] for name in values.index], values, color=colors)
        style(ax, "x")
        ax.xaxis.set_major_formatter(FuncFormatter(thousands))
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.set_title(country, loc="left", fontweight="bold", color=INK)
        ax.set_xlim(0, max(values) * 1.3)
        ax.invert_yaxis()
        ax.bar_label(bars, [f"{value:,.0f}" + (" *" if name == winner else "")
                            for name, value in values.items()], padding=3, color=INK, fontsize=8)
        ax.tick_params(axis="y", labelsize=8)
    fig.text(.14, .015, "* Selected at training time. Sparse scopes compare the two naïve baselines.",
             fontsize=9, color=SUB)
    finish(fig, output)


def production(scored, output):
    fig, axes = plt.subplots(3, 2, figsize=(11, 8))
    fig.subplots_adjust(left=.08, right=.98, top=.82, bottom=.08, hspace=.62, wspace=.2)
    heading(fig, "Production forecasts diverged as revenue shifted",
            "Next 30 days of revenue by forecast start  •  99 matured starts per scope")
    for ax, country in zip(axes.flat, REPLAY):
        group = scored.loc[scored["country"].eq(country)]
        dates = pd.to_datetime(group["date"])
        model = group["model"].iloc[0]
        same = np.allclose(group["baseline"], group["prediction"])
        ax.plot(dates, group["actual"], color=INK, linewidth=2.1, label="Actual")
        ax.plot(dates, group["baseline"], color=AMBER, linestyle="--", linewidth=1.9,
                label="Baseline" + (" (also selected)" if same else ""))
        if not same:
            ax.plot(dates, group["prediction"], color=TEAL, linewidth=1.6, label="Selected model")
        style(ax)
        ax.set_title(f"{country}  |  {LABELS[model]}", loc="left", color=INK, fontsize=10, fontweight="bold")
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        ax.legend(loc="best", frameon=False, fontsize=8)
    finish(fig, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("artifacts/model.joblib"))
    parser.add_argument("--reports", type=Path, default=Path("reports"))
    args = parser.parse_args()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    directory = args.reports
    bundle = load_model(args.model)
    eda(bundle, directory / "training_eda.png")
    comparison(bundle, pd.read_csv(directory / "validation_predictions.csv"), directory / "validation_comparison.png")
    production(pd.read_csv(directory / "scored_predictions.csv"), directory / "production_replay.png")


if __name__ == "__main__":
    main()
