#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

try:
    import seaborn as sns
except ImportError:  # pragma: no cover
    sns = None


def load_cf_query_summaries(pattern: str, agent_label: str) -> pd.DataFrame:
    rows = []
    for path in sorted(glob.glob(pattern)):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("trace_kind") != "cf_query_summary":
                    continue
                rows.append(
                    {
                        "agent": agent_label,
                        "question_id": rec["question_id"],
                        "question_type": rec.get("question_type"),
                        "rollback_gini": rec.get("rollback_gini"),
                    }
                )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.dropna(subset=["rollback_gini"]).copy()


def summarize(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for agent, sub in df.groupby("agent"):
        vals = sub["rollback_gini"].astype(float)
        summary[agent] = {
            "n": int(len(vals)),
            "mean": float(vals.mean()),
            "median": float(vals.median()),
            "max": float(vals.max()),
        }
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot Gini distributions from CF query summaries.")
    ap.add_argument(
        "--memoryos-pattern",
        default="cf_compare_results/memoryos_cf_trace/*06_52.cf_queries.jsonl",
    )
    ap.add_argument(
        "--share-pattern",
        default="cf_compare_results/share_cf_trace/*.cf_queries.jsonl",
    )
    ap.add_argument(
        "--out",
        default="cf_compare_results/gini_comparison_memoryos_share.png",
    )
    args = ap.parse_args()

    df = pd.concat(
        [
            load_cf_query_summaries(args.memoryos_pattern, "MemoryOS"),
            load_cf_query_summaries(args.share_pattern, "SHARE"),
        ],
        ignore_index=True,
    )
    if df.empty:
        raise SystemExit("No cf_query_summary rows with rollback_gini were found.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    summary = summarize(df)
    order = ["MemoryOS", "SHARE"]
    colors = {"MemoryOS": "#4C78A8", "SHARE": "#F58518"}

    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(8.6, 5.6))

    if sns is not None:
        sns.boxplot(
            data=df,
            x="agent",
            y="rollback_gini",
            order=order,
            width=0.48,
            fliersize=0,
            linewidth=1.25,
            boxprops={"facecolor": "none", "edgecolor": "black", "linewidth": 1.6},
            whiskerprops={"color": "black", "linewidth": 1.25},
            capprops={"color": "black", "linewidth": 1.25},
            medianprops={"color": "black", "linewidth": 1.6},
            ax=ax,
        )
        sns.stripplot(
            data=df,
            x="agent",
            y="rollback_gini",
            order=order,
            palette=[colors[a] for a in order],
            size=5.5,
            jitter=0.18,
            alpha=0.75,
            edgecolor="white",
            linewidth=0.4,
            zorder=3,
            ax=ax,
        )
    else:
        data = [df.loc[df["agent"] == agent, "rollback_gini"].astype(float).tolist() for agent in order]
        bp = ax.boxplot(
            data,
            positions=range(len(order)),
            widths=0.48,
            patch_artist=True,
            showfliers=False,
        )
        for patch, agent in zip(bp["boxes"], order):
            patch.set_facecolor("none")
            patch.set_alpha(1.0)
            patch.set_edgecolor("black")
            patch.set_linewidth(1.6)
        for median in bp["medians"]:
            median.set_color("black")
            median.set_linewidth(1.5)
        rng = random.Random(7)
        for idx, agent in enumerate(order):
            vals = df.loc[df["agent"] == agent, "rollback_gini"].astype(float).tolist()
            xs = [idx + rng.uniform(-0.12, 0.12) for _ in vals]
            ax.scatter(
                xs,
                vals,
                s=28,
                alpha=0.75,
                color=colors[agent],
                edgecolors="white",
                linewidths=0.4,
                zorder=3,
            )
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order)

    ax.set_title("", fontsize=15, pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("Write-Ablation Gini Coefficient", fontsize=12)
    ax.set_ylim(0, max(0.4, float(df["rollback_gini"].max()) + 0.03))
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.45)
    ax.set_axisbelow(True)

    text_lines = []
    for agent in order:
        stats = summary[agent]
        text_lines.append(
            f"{agent}: n={stats['n']}, mean={stats['mean']:.3f}, "
            f"median={stats['median']:.3f}"
        )
    ax.text(
        0.02,
        0.98,
        "\n".join(text_lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.9, "edgecolor": "#bbbbbb"},
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    print(out_path)


if __name__ == "__main__":
    main()
