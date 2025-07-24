import re
import sys
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

LOG_FILE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("train.log")


def parse_log(path: Path) -> pd.DataFrame:
    """
    Return a DataFrame with columns:
        split  (Train / Test)
        epoch  (int)
        metric (Acc / pLoss)
        value  (float)
    """
    pattern = re.compile(
        r'^(Train|Test) Epoch \[(\d+)/\d+\].*?(Acc|pLoss):\s*([0-9.]+)')

    records = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            m = pattern.search(line)
            if m:
                split, epoch, metric, value = m.groups()
                records.append(
                    dict(split=split, epoch=int(epoch),
                         metric=metric, value=float(value))
                )
    if not records:
        raise ValueError("No Train/Test Acc/Loss lines found!")
    return pd.DataFrame(records)


def plot(df: pd.DataFrame) -> None:
    """Per epoch means and dual axes chart"""
    df_mean = (
        df.groupby(['split', 'epoch', 'metric'])
          .value.mean()
          .reset_index()
    )

    train = (df_mean[df_mean.split == 'Train']
             .pivot(index='epoch', columns='metric', values='value'))
    test = (df_mean[df_mean.split == 'Test']
            .pivot(index='epoch', columns='metric', values='value'))

    sns.set_style("darkgrid")
    sns.set_theme(style="dark")
    # plt.style.use("dark_background")
    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Accuracy (left axis, blue)
    sns.lineplot(data=train['Acc'], ax=ax1, label="Train Acc",
                 color="tab:blue")
    sns.lineplot(data=test['Acc'],  ax=ax1, label="Test Acc",
                 color="tab:blue", linestyle="--")
    ax1.set_ylabel("Accuracy", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    # Loss (right axis, orange)
    ax2 = ax1.twinx()
    sns.lineplot(data=train['pLoss'], ax=ax2, label="Train Loss",
                 color="tab:orange")
    sns.lineplot(data=test['pLoss'],  ax=ax2, label="Test Loss",
                 color="tab:orange", linestyle="--")
    ax2.set_ylabel("Loss", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    ax1.grid(True, which="major", linestyle="-",
             linewidth=1.0, alpha=0.75)
    ax2.grid(False)

    # keep one legend that combines handles from both axes
    ax1.get_legend().remove()
    ax2.get_legend().remove()
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc="center right",            # anchor corner
            #    bbox_to_anchor=(1.02, 1.00),  # nudge it just outside the axes
               frameon=False)               # optional: no legend frame    

    ax1.set_xlabel("Epoch")
    from matplotlib.ticker import MaxNLocator
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.title("Training vs Test Accuracy & Loss")
    fig.tight_layout()
    # plt.legend()         # merge legends from both axes
    plt.savefig("log_plot.png", dpi=600)


if __name__ == "__main__":
    df = parse_log(LOG_FILE)
    plot(df)
