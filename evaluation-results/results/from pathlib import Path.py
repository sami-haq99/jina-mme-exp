from pathlib import Path
from statistics import mean, stdev
import re

import matplotlib
matplotlib.use("TkAgg")  # or Qt5Agg
import matplotlib.pyplot as plt
import math

languages = [ "fr", "de", "cs"]
systems = ["aya", "opusmt", "gemma", "zeromt"]

base_dir = Path("/home/sami/mmt-eval/doc-mte/mmss/evaluation-results/results")

metric_keys = [
    "Final_Score",
    "Text_Fidelity (Src-Tgt)",
    "Visual_Grounding (Tgt-Img)",
    "Image_Relevance (Src-Img)",
    "Fusion_Weight",
]

pattern = re.compile(r"(-?\d+(?:\.\d+)?)")

results = {}


def compute_graphs():
    system = "aya"
    language = "fr"
    rows = results.get(language, {}).get(system, [])

    final_scores = [row["Visual_Grounding (Tgt-Img)"] for row in rows]
    text_fidelity = [row["Image_Relevance (Src-Img)"] for row in rows]

    if final_scores and text_fidelity:
        final_mean = mean(final_scores)
        final_std = stdev(final_scores) if len(final_scores) > 1 else 0.0
        text_mean = mean(text_fidelity)
        text_std = stdev(text_fidelity) if len(text_fidelity) > 1 else 0.0

        plt.figure(figsize=(8, 4.5))
        plt.hist(final_scores, bins=30, alpha=0.6, label="Visual_Grounding (Tgt-Img)")
        plt.hist(text_fidelity, bins=30, alpha=0.6, label="Image_Relevance (Src-Img)")

        plt.axvline(final_mean, color="C0", linestyle="--", linewidth=1.5)
        plt.axvline(text_mean, color="C1", linestyle="--", linewidth=1.5)

        plt.title(f"Distribution for {system} ({language})")
        plt.xlabel("Score")
        plt.ylabel("Count")
        plt.legend(
            title=(
                f"Visual_Grounding (Tgt-Img): mean={final_mean:.4f}, std={final_std:.4f}\n"
                f"Image_Relevance (Src-Img): mean={text_mean:.4f}, std={text_std:.4f}"
            )
        )
        plt.tight_layout()
        plt.show()
    else:
        print(f"No data found for {system} ({language}).")

for system in systems:
    for language in languages:
        csv_path = base_dir / f"{system}_{language}_results.csv"
        rows = []

        if not csv_path.exists():
            continue
        if results.get(language) is None:
            results[language] = {}
            
        with csv_path.open("r", encoding="utf-8") as f:
            header = next(f, "")
            for line in f:
                nums = pattern.findall(line)
                if len(nums) >= 5:
                    metric_values = [float(x) for x in nums[-5:]]
                    rows.append(dict(zip(metric_keys, metric_values)))

        results[language][system] = rows


#compute_graphs()

#for above results calcualte new fusion weight using the formula: lambda_weight = sqrt(max(0, score_src_img)) and add it to the results dictionary for each row

import math

for language in languages:
    for system in systems:
        rows = results[language][system]
        for row in rows:
            score_src_img = row["Image_Relevance (Src-Img)"]
            k = 2.5# Sensitivity (square root)
            lambda_weight = max(0, score_src_img) ** k
            row["Lambda_Weight"] = lambda_weight
            score_src_tgt = row["Text_Fidelity (Src-Tgt)"]
            score_tgt_img = row["Visual_Grounding (Tgt-Img)"]
            final_score = (score_src_tgt + (lambda_weight * score_tgt_img)) / (1 + lambda_weight)
            row["Final_Score_sqrt"] = final_score

#save the updated results to a new CSV file
import csv
output_dir = base_dir
output_dir.mkdir(exist_ok=True)
for language in languages:
    for system in systems:
        rows = results[language][system]
        output_path = output_dir / f"{system}_{language}_updated_results.csv"
        with output_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

