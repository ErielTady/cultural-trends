import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams["figure.figsize"]=(3.5, 2.6)

NAEL_ACC = 0.10418896051094526
NAEL_SOFT =  0.1303034008139594

# ---------------------theme map ---------------------------------------
THEME_MAP = {
    # Social Values
    2:"Social Values", 19:"Social Values", 21:"Social Values", 42:"Social Values",

    # Social Capital
    62: "Social Capital",  63:"Social Capital", 77:"Social Capital", 78:"Social Capital",
    83:"Social Capital", 84:"Social Capital", 87:"Social Capital", 88:"Social Capital",

    # Security
    142:"Security", 143:"Security", 149:"Security", 150:"Security",

    # Religious Values
    171:"Religious Values", 175:"Religious Values",

    # Migration
    124:"Migration", 126:"Migration", 127:"Migration",

    # Political Culture
    235:"Political Culture", 236:"Political Culture", 239:"Political Culture",

    # Political Interest
    199:"Political Interest", 209:"Political Interest", 210:"Political Interest",
    221:"Political Interest", 224:"Political Interest", 229:"Political Interest", 234:"Political Interest",

}


# --------------------------- helpers ---------------------------
def load_and_prepare_rows(path: Path)-> pd.DataFrame:
  df = pd.read_csv(path, on_bad_lines = "warn")
  df["Theme"] = df["question"].map(THEME_MAP).fillna("Other")
  return df

def per_theme(data: pd.DataFrame)-> pd.DataFrame:
  question_means=(
      data.groupby(["question","Theme"])[["mae-score","accuracy-score"]]
      .mean()
      .reset_index()
  )

  theme_summary=(
      question_means.groupby("Theme")[["mae-score", "accuracy-score"]]
      .mean()
      .reset_index()
      .sort_values("mae-score", ascending=False)
  )

  question_counts = (
      question_means.groupby("Theme")["question"]
      .nunique()
      .rename("num_questions")
      .reset_index()
  )
  return theme_summary.merge(question_counts, on="Theme", how="left")


def summarize_overall(df: pd.DataFrame) -> pd.DataFrame:
    persona_cols = [
        "persona.region", "persona.sex", "persona.age",
        "persona.marital_status", "persona.education", "persona.social_class"
    ]
    hard_acc = float(pd.to_numeric(df["accuracy-score"], errors="coerce").mean())
    soft_acc = float(pd.to_numeric(df["mae-score"], errors="coerce").mean())
    hard_acc_rand = float((hard_acc-NAEL_ACC)/(1-NAEL_ACC))
    soft_acc_rand = float((soft_acc - NAEL_SOFT)/(1-NAEL_SOFT))


    overall = {
        "total_rows":len(df),
        "total_questions": int(df.get("question", pd.Series(dtype=str)).nunique()),
        "total_persona": int(df[persona_cols].dropna().shape[0]),
        "overall_hard_acc": hard_acc,
        "overall_soft_acc": soft_acc,
        "overall_hard_acc_rand": hard_acc_rand,
        "overall_soft_acc_rand": soft_acc_rand,
    }
    return pd.DataFrame([overall])

def agent_random_avg_comparison(y_random:float, y_model:float, y_label:str, title: str):
  labels = ["Random Guesser", "Llama-2-13b-chat-hf"]
  values = [y_random, y_model]

  fig, ax = plt.subplots(figsize=(4.2, 3.2), dpi=150)

  bars = ax.bar(labels, values, width=0.55, color=["#9aa0a6", "#1F77B4"])

  ax.set_ylim(0, 1)
  ax.set_yticks(np.linspace(0, 1, 6))
  ax.set_axisbelow(True)

  ax.set_ylabel(y_label, labelpad=6)
 # ax.set_title(title, pad=10, fontsize=11, fontweight="semibold")

  ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.5)

  for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)

  ax.bar_label(bars, labels=[f"{v:.4f}" for v in values], padding=3, fontsize=9)

  fig.tight_layout()


  return fig

def plot_per_theme(theme_summary: pd.DataFrame, overall_soft: float, overall_hard:float ):
  # Soft
  fig1, ax1 = plt.subplots(figsize=(6.5, 3.5),dpi=150)

  #ax1.axhline(overall_soft, linestyle="--", linewidth=0.7)
  ax1.axhline(overall_soft,linestyle="--",linewidth=0.5,color="k",label="overall",zorder=0)
  ax1.legend(frameon=False,loc="upper right")

  bars = ax1.bar(theme_summary["Theme"],theme_summary["mae-score"],width=0.6)
  ax1.bar_label(bars,fmt="%.4f",padding=2,rotation=90)


  ax1.set_ylim(0,1)
  ax1.set_ylabel("Soft Similarity")
  ax1.set_axisbelow(True)
  ax1.grid(axis="y",linestyle=":",linewidth=0.5,alpha=0.7)
  for spine in ("top", "right"):
    ax1.spines[spine].set_visible(False)

  ax1.tick_params(axis="x",labelrotation=30)
  plt.setp(ax1.get_xticklabels(), ha="right")

  #ax1.set_xticklabels(theme_summary["Theme"], rotation=30, ha="right")
  #for index_bar, value in enumerate(theme_summary["mae-score"]):
   # ax1.text(index_bar, min(value  + 0.02, 0.98),f"{value:.4f}",ha="center",rotation=90,va="bottom")
  fig1.tight_layout()


  # Hard
  fig2, ax2 = plt.subplots(figsize=(6.5, 3.5),dpi=150)

  ax2.axhline(overall_hard,linestyle="--",linewidth=0.5,color="k",label="overall",zorder=0)
  ax2.legend(frameon=False,loc="upper right")

  bars = ax2.bar(theme_summary["Theme"],theme_summary["accuracy-score"],width=0.6)
  ax2.bar_label(bars,fmt="%.4f",padding=2,rotation=90)

  ax2.set_ylim(0,1)
  ax2.set_ylabel("Hard Similarity")
  ax2.set_axisbelow(True)
  ax2.grid(axis="y",linestyle=":",linewidth=0.5,alpha=0.7)
  for spine in ("top", "right"):
    ax2.spines[spine].set_visible(False)

  ax2.tick_params(axis="x",labelrotation=30)
  plt.setp(ax2.get_xticklabels(), ha="right")

  fig2.tight_layout()



  return fig1, fig2


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--input", required=True, help="Path to results CSV")
    arg_parser.add_argument("--outdir", default=None, help="Output directory (default: alongside input)")
    arg_parser.add_argument("--save", action="store_true", default=False, help="save_pngs")
    parsed_args = arg_parser.parse_args()

    input_path = Path(parsed_args.input)
    assert input_path.exists(), f"File not found:{input_path}"

    out_path = Path(parsed_args.outdir) if parsed_args.outdir else input_path.parent

    # Load
    df = pd.read_csv(input_path, on_bad_lines="warn")
    print("Dataset shape:", df.shape)
    print(df["accuracy"].dtype)
    df["accuracy"]=df["accuracy"].astype(float)


    # Summaries
    overall = summarize_overall(df)
    # per_question = question_summary(df)
    hard_acc = overall.loc[0, "overall_hard_acc"]
    soft_acc  = overall.loc[0, "overall_soft_acc"]
    hard_acc_rand = overall.loc[0, "overall_hard_acc_rand"]
    soft_acc_rand = overall.loc[0, "overall_soft_acc_rand"]

    # plots
    fig1 = agent_random_avg_comparison(hard_acc_rand, hard_acc, y_label= "Hard Similarity", title= "Model vs Random Guesser")
    fig2 = agent_random_avg_comparison(soft_acc_rand, soft_acc, y_label= "Soft Similarity", title="")

    # per_theme_bars
    data = load_and_prepare_rows(input_path)
    theme_summary = per_theme(data)
    fig3 ,fig4 = plot_per_theme(theme_summary, soft_acc, hard_acc)


    plt.show()

    if parsed_args.save:
      fig1.savefig(out_path/"acc_agent_vs_random.png", dpi=200, bbox_inches="tight")
      fig2.savefig(out_path/"soft_acc_agent_vs_random.png",dpi=200, bbox_inches="tight")
      fig3.savefig(out_path/"per_theme_soft.png", dpi=200, bbox_inches="tight")
      fig4.savefig(out_path/"per_theme_hard.png", dpi=200, bbox_inches="tight")
      print(f"Saved to:{out_path}")




if __name__ == "__main__":
    main()
