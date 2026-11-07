"""Static paper-to-result consistency check; does not rerun model training.

Run from any directory with:
    python Semester/verify_paper_v4_consistency.py
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "Semester/paper_CLIP_Gemma4_revised_v4.md"
SOURCE = ROOT / "Semester/results/verified_stanford_dogs_official_split.json"
OUTPUT = ROOT / "Semester/results/paper_v4_consistency_check.json"

paper = PAPER.read_text(encoding="utf-8")
data = json.loads(SOURCE.read_text(encoding="utf-8"))
section = paper.split("### 5.7. Evaluation on a Second Fine-Grained Benchmark (Stanford Dogs)", 1)[1].split(
    "### 5.8.", 1
)[0]
table = section.split("*Table 8.", 1)[1].split("\n\n>", 1)[0]

models = {
    "CLIP Raw Class Names": data["zero_shot_baselines"]["clip_raw"],
    "CLIP Handcrafted Prompt": data["zero_shot_baselines"]["clip_handcrafted"],
    "CuPL Official Prompts [17]": {"correct_count": 5391, "top1_acc": 62.83},
    "CuPL-style (Gemma Proxy)": data["zero_shot_baselines"]["cupl"],
    "LaBo-style [15]": data["zero_shot_baselines"]["labo_style"],
    "Gemma-CBM (Ours)": data["zero_shot_baselines"]["gemma_cbm"],
    "Gemma-Centroid (Ours)": data["zero_shot_baselines"]["gemma_centroid"],
    "Gemma-ZeroShot (Ours)": data["zero_shot_baselines"]["gemma_zeroshot"],
    "Baseline CoOp": data["peft_models"]["coop_baseline"]["test_results"],
    "Gemma-VisAdapter (Ours)": data["peft_models"]["gemma_visadapter"]["test_results"],
    "Gemma-DualStream (Ours)": data["peft_models"]["gemma_dualstream"]["test_results"],
    "Gemma-CoOp (Ours)": data["peft_models"]["gemma_coop"]["test_results"],
}

checks: list[dict[str, object]] = []


def check(name: str, passed: bool, detail: str) -> None:
    checks.append({"check": name, "passed": bool(passed), "detail": detail})


total = data["partition_details"]["official_test"]
check(
    "official_split_counts",
    data["partition_details"]["official_train"] == 12000
    and total == 8580
    and data["partition_details"]["internal_train_split"] == 10800
    and data["partition_details"]["internal_val_split"] == 1200,
    "Result JSON: 12,000 train / 8,580 test; internal 10,800 train / 1,200 val.",
)

for label, result in models.items():
    matching = [line for line in table.splitlines() if label in line]
    if len(matching) != 1:
        check(label, False, f"Expected one Table 8 row; found {len(matching)}.")
        continue
    cells = [cell.strip() for cell in matching[0].strip().strip("|").split("|")]
    dogs = cells[4] if len(cells) == 6 else ""
    found = re.search(r"(\d+\.\d+)%\s*\(([\d,]+)/8,580\)", dogs)
    expected_count = result["correct_count"]
    expected_rate = round(100 * expected_count / total, 2)
    passed = bool(
        found
        and int(found.group(2).replace(",", "")) == expected_count
        and abs(float(found.group(1)) - expected_rate) < 0.005
        and abs(result["top1_acc"] - expected_rate) < 0.01
    )
    check(label, passed, f"Paper: {dogs}; JSON: {expected_count}/{total} = {expected_rate:.2f}%.")

beta = data["peft_models"]["gemma_dualstream"]["best_beta"]
check("dualstream_beta", beta == 0.1 and r"\beta=0.1" in section, f"JSON best_beta={beta}; paper footnote checked.")
check("no_old_dogs_split", "6,174" not in section and "14,406" not in section, "Old custom-split counts absent from Section 5.7.")
check("no_old_dogs_accuracy", "72.47%" not in section and "59.57%" not in section, "Old custom-split accuracies absent from Section 5.7.")
check("fewshot_protocol_note", "single-view raw" in paper.split("### 5.6.", 1)[1].split("### 5.7.", 1)[0], "Table 7 explicitly labels the raw K=0 fallback.")
check("mcnemar_caveat", "does **not** reach the 0.05 significance threshold" in paper, "CoOp McNemar p=0.0696 is not described as significant.")
check("alpha_caveat", "not** the setting with the highest validation accuracy" in paper, "The alpha=0.2 selection limitation is disclosed.")
check("no_sota_claim", not re.search(r"\bSOTA\b|state[- ]of[- ]the[- ]art", paper, flags=re.I), "No SOTA assertion in this paper version.")

report = {
    "scope": "Static consistency check only; no model training or feature extraction was run.",
    "paper": PAPER.relative_to(ROOT).as_posix(),
    "paper_sha256": hashlib.sha256(PAPER.read_bytes()).hexdigest(),
    "result_source": SOURCE.relative_to(ROOT).as_posix(),
    "result_source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    "all_passed": all(item["passed"] for item in checks),
    "checks": checks,
}
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"{sum(item['passed'] for item in checks)}/{len(checks)} static checks passed; output: {OUTPUT}")
raise SystemExit(0 if report["all_passed"] else 1)
