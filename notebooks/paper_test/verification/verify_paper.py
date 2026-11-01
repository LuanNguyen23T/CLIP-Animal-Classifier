from __future__ import annotations
import json, math, re, statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PAPER = ROOT / "paper/paper_CLIP_Gemma4_revised_v2.md"
RESULTS = ROOT / "Semester/results"
LOCAL = ROOT / "notebooks/paper_test/results"
STATUSES = ("match", "conflict", "missing_evidence", "protocol_mismatch", "manual_review")

def read(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))

def paper_lines():
    return PAPER.read_text(encoding="utf-8").splitlines()

def rows(start, end):
    lines = paper_lines()[start-1:end]
    return [line for line in lines if line.startswith("|") and not line.startswith("|---") and not line.startswith("|:---")]

def add(out, table, claim, status, evidence, detail):
    assert status in STATUSES
    out.append(dict(table=table, claim=claim, status=status, evidence=evidence, detail=detail))

def pct(correct, total):
    return 100 * correct / total

def near(a, b, tol=.015):
    return abs(a-b) <= tol

def torch_data(path):
    try:
        import torch
    except ImportError:
        return None
    p = ROOT / path
    return torch.load(p, map_location="cpu", weights_only=False) if p.exists() else None

def verify(table=None):
    out=[]
    lines=paper_lines()
    if table in (None,1):
        bank = ROOT / "Semester/attributes"
        files=list(bank.rglob("*.json")) if bank.exists() else []
        add(out,1,"Descriptor source for Table 1","manual_review","Semester/attributes; paper lines 87-94",
            f"{len(files)} JSON attribute files found. The anatomical statements need row-by-row comparison to generated descriptors and registry references; a generated descriptor is not automatically a verified biological fact.")
        add(out,1,"Birman/Ragdoll white gloves statement","conflict","paper lines 91-92; Semester/attributes",
            "Paper asserts absence of white gloves in Ragdoll; mitted Ragdolls exist. Verify the attribute bank and registry standard before using this as a decisive exclusive marker.")
        add(out,1,"18/19 manual attribute spot-check","missing_evidence","paper narrative",
            "Need sampled attribute IDs, human annotation rubric and per-item decisions. A reported count alone cannot be independently checked.")
    if table in (None,2):
        d=read("Semester/results/verified_oxford_pets_benchmark.json")
        claims={"CLIP Raw Class Names":(80.16,80.38,"clip_raw"),
                "CLIP Handcrafted Prompt":(87.52,87.76,"clip_handcrafted"),
                "CoOp Baseline":(91.20,91.52,"coop_baseline"),
                "ResNet-50 Cosine Head":(92.78,93.08,"resnet50_cosine"),
                "Gemma-CBM":(80.40,80.02,"gemma_cbm"),
                "Gemma-Centroid":(88.20,88.25,"gemma_centroid"),
                "Gemma-ZeroShot":(90.41,90.85,"gemma_zeroshot"),
                "Gemma-VisAdapter":(91.03,91.41,"gemma_visadapter"),
                "Gemma-CoOp":(91.11,91.66,"gemma_coop"),
                "Gemma-DualStream":(91.74,92.15,"gemma_dualstream")}
        for name,(raw,tta,key) in claims.items():
            actual=d[key]
            for metric,claimed in [("raw",raw),("tta",tta)]:
                observed=actual[metric]
                if observed is None:
                    status="missing_evidence"; detail="Artifact has no raw metric."
                else:
                    status="match" if near(observed,claimed) else "conflict"
                    detail=f"paper={claimed:.2f}%, artifact={observed:.2f}% ({actual.get('correct')} TTA correct). Source may be a distinct run; label each run explicitly."
                add(out,2,f"{name} {metric}",status,"Semester/results/verified_oxford_pets_benchmark.json",detail)
        names=["CLIP 80-Prompt Ensemble","Tip-Adapter","Tip-Adapter-F","CLIP-Adapter","ResNet-50 Prototype","ResNet-50 Linear Probe","ResNet-50 Residual Adapter"]
        for name in names:
            add(out,2,name,"missing_evidence","paper Table 2; training notebooks",
                "No per-image predictions or run manifest tied to this paper row identified in committed results. Locate run-specific artifact.")
        a=torch_data("Semester/results/verified_test_predictions.pt")
        if a:
            y=a["test_labels"]; pred=a["pred_dual"]; correct=int((pred==y).sum())
            add(out,2,"DualStream prediction alignment","conflict","Semester/results/verified_test_predictions.pt",
                f"Saved vector is {correct}/{len(y)}={pct(correct,len(y)):.2f}%, not paper Table 2 TTA 92.15%; cannot certify that row or its McNemar result with this vector.")
    if table in (None,3):
        for name in ["Baseline CoOp","CoOp + Verbose Mark","CoOp + Verbose Body & Head","Gemma-CoOp Compact Trait","Gemma-DualStream"]:
            add(out,3,name,"missing_evidence","paper Table 3; notebooks/training",
                "Need exact prompt strings/tokenizer output by class, run checkpoint, per-image predictions and metric script for this row; notebook output alone is not independent evidence.")
        add(out,3,"24-token claim","protocol_mismatch","paper Table 3; Stanford notebook",
            "Stanford prompt token counts in the executed notebook were 27–52; the paper must state which dataset/run has 24 and provide tokenizer output.")
    if table in (None,4):
        d=read("Semester/results/table5_alpha_verification.json")
        claimed={0.0:(83.24,86.45,86.81),0.1:(91.89,89.70,90.57),0.2:(92.43,90.41,91.01),0.4:(92.43,89.72,90.92),0.6:(92.16,89.23,90.62)}
        for alpha,(val,raw,tta) in claimed.items():
            run=d[str(alpha)]
            for label,expected,key in [("val",val,"val_acc"),("raw",raw,"test_raw_acc")]:
                actual=run[key]; add(out,4,f"alpha={alpha} {label}",
                    "match" if near(expected,actual) else "conflict",
                    "Semester/results/table5_alpha_verification.json",
                    f"paper={expected:.2f}%, artifact={actual:.2f}%")
            add(out,4,f"alpha={alpha} TTA","missing_evidence","Semester/results/table5_alpha_verification.json",
                f"paper={tta:.2f}%; this artifact does not contain TTA or per-image predictions.")
        add(out,4,"alpha=0.2 uniquely optimal","conflict","Semester/results/table5_alpha_verification.json",
            "Validation accuracy ties at alpha 0.2 and 0.4. State the validation-only tie-break rule; test accuracy cannot select alpha.")
    if table in (None,5):
        for axis in ["head_and_ears","eyes","coat","body","distinguishing_mark"]:
            add(out,5,axis,"missing_evidence","paper Table 5",
                "Need case image ID, concept/axis scoring code, saved vectors for both breeds and normalization rule. Current table values cannot be recalculated from committed artifacts.")
    if table in (None,6):
        data=read("Semester/results/verified_confusable_pairs.json")
        for r in data:
            expected=pct(r["correct_n"],r["test_n"])
            reported=r["correct_pct"]
            add(out,6,r["ground_truth"]+" count","match" if near(expected,reported) else "conflict",
                "Semester/results/verified_confusable_pairs.json",
                f"{r['correct_n']}/{r['test_n']}={expected:.2f}%; counterpart errors {r['errors_to_counterpart']}/{r['total_errors']}.")
            if r["ground_truth"]=="Beagle":
                add(out,6,"Beagle most-confusable label","conflict","Semester/results/verified_confusable_pairs.json",
                    "0 of 1 error goes to Basset Hound, so Basset cannot be the most confused class for this run.")
        add(out,6,"photographic ambiguity explanations","manual_review","paper Table 6",
            "Paw occlusion, low light and framing are causal explanations; require case image IDs and reviewed examples.")
    if table in (None,7):
        d=read("Semester/results/table7_full_benchmark_results.json")
        for model,values in d["models"].items():
            for budget,entry in values.items():
                runs=entry["runs"]; mean=statistics.mean(runs); std=statistics.stdev(runs)
                good=near(mean,entry["mean"]) and near(std,entry["std"])
                add(out,7,f"{model} {budget} summary","match" if good else "conflict",
                    "Semester/results/table7_full_benchmark_results.json",
                    f"runs={runs}; recomputed {mean:.2f}±{std:.2f}; JSON {entry['mean']:.2f}±{entry['std']:.2f}. This validates aggregation only, not training provenance.")
        for budget in ["1_shot","2_shot","4_shot","8_shot","16_shot","full_trainval"]:
            delta=d["models"]["Gemma-CLIP Full"][budget]["mean"]-d["models"]["CLIP + CoOp"][budget]["mean"]
            add(out,7,f"Full vs CoOp {budget}","conflict" if delta<0 else "match",
                "Semester/results/table7_full_benchmark_results.json",
                f"Gemma-CLIP Full minus CoOp = {delta:+.2f} percentage points. A blanket gain claim is unsupported.")
        add(out,7,"K=0 zero-shot values","conflict","paper Table 7; Semester/results/table7_full_benchmark_results.json",
            f"Paper uses 88.90/90.85%; JSON's handcrafted zero-shot is {d['clip_handcrafted_zero_shot']:.2f}%. No K=0 per-image artifact tied to the paper row.")
        add(out,7,"Table 7 Gemma-CLIP Full model identity","missing_evidence","Semester/results/table7_full_benchmark_results.json; notebooks/training/few_shot_multiseed_scaling_benchmark.ipynb",
            "Table 2 evaluates DualStream. A few-shot notebook contains a SimpleAdapter experiment, but the three-seed JSON does not identify the architecture/checkpoints of its Gemma-CLIP Full row. Do not equate the rows or rename the latter until its run manifest confirms this.")
        add(out,7,"three-seed training provenance","missing_evidence","notebooks/training/run_full_table7_benchmark.ipynb",
            "JSON has three aggregate runs; require per-seed run IDs, sampled train image IDs, checkpoints and per-image predictions to independently re-evaluate.")
    if table in (None,8):
        new=read("notebooks/paper_test/results/stanford_dualstream_recheck.json")
        old=read("Semester/results/verified_stanford_dogs_dualstream.json")
        for name,claimed,key in [("CoOp Stanford",57.55,"coop"),("DualStream Stanford",59.62,"dualstream")]:
            r=new["test"][key]; observed=pct(r["correct"],r["total"])
            add(out,8,name,"conflict","notebooks/paper_test/results/stanford_dualstream_recheck.json",
                f"paper={claimed:.2f}%, corrected run={r['correct']}/{r['total']}={observed:.2f}%; old run={old['models']['coop_baseline' if key=='coop' else 'gemma_dualstream']['test_top1']:.2f}%. Keep run labels separate.")
        add(out,8,"Stanford beta","conflict","notebooks/paper_test/results/stanford_dualstream_recheck.json",
            f"paper beta=0.7; corrected run selects beta={new['selected_beta']} using validation.")
        cupl=read("notebooks/paper_test/results/cupl_official_prompts_oxford.json")
        r=cupl["results"]["tta"]
        add(out,8,"CuPL Oxford TTA","protocol_mismatch","notebooks/paper_test/results/cupl_official_prompts_oxford.json",
            f"paper=86.02% from descriptor-based proxy; author prompt bank run={r['correct']}/{r['total']}={pct(r['correct'],r['total']):.2f}%. Distinct protocols.")
        add(out,8,"CuPL Stanford","protocol_mismatch","Semester/results/verified_cupl_labo_baselines.json",
            "Reported 56.85% uses Gemma descriptor proxy; no official CuPL prompt run saved for Stanford.")
        add(out,8,"LaBo Oxford and Stanford","protocol_mismatch","Semester/results/verified_cupl_labo_baselines.json",
            "Reported rows use pooled Gemma descriptors; original LaBo concept selection and classifier not evidenced.")
        add(out,8,"Stanford split designation","conflict","notebooks/paper_test/results/stanford_dualstream_recheck.json",
            "14,406/6,174 is a custom cached split, not the official 12,000/8,580 split.")
        pt=torch_data("notebooks/paper_test/results/stanford_dualstream_recheck_predictions.pt")
        if pt:
            y=pt["test_labels"]
            for key in ["pred_coop","pred_gemma_coop","pred_dualstream"]:
                c=int((pt[key]==y).sum())
                expected=new["test"][{"pred_coop":"coop","pred_gemma_coop":"gemma_coop","pred_dualstream":"dualstream"}[key]]["correct"]
                add(out,8,key+" tensor/JSON","match" if c==expected else "conflict",
                    "notebooks/paper_test/results/stanford_dualstream_recheck_predictions.pt",
                    f"tensor correct={c}, JSON correct={expected}, N={len(y)}.")
        cupt=torch_data("notebooks/paper_test/results/cupl_official_prompts_oxford_predictions.pt")
        if cupt:
            y=cupt["test_labels"]
            for key,metric in [("pred_raw","raw"),("pred_tta","tta")]:
                c=int((cupt[key]==y).sum()); expected=cupl["results"][metric]["correct"]
                add(out,8,"CuPL Oxford "+metric+" tensor/JSON","match" if c==expected else "conflict",
                    "notebooks/paper_test/results/cupl_official_prompts_oxford_predictions.pt",
                    f"tensor correct={c}, JSON correct={expected}, N={len(y)}.")
    if table in (None,9):
        for label,lat,fps in [("CLIP full",11.85,84.4),("DualStream 5-view",58.14,17.2)]:
            actual=1000/lat
            add(out,9,label+" latency/FPS arithmetic","match" if abs(actual-fps)<.2 else "conflict",
                "paper Table 9",f"1000/{lat} ms={actual:.2f} FPS; paper={fps}. Arithmetic only.")
        add(out,9,"hardware timing provenance","missing_evidence","paper Table 9",
            "Need device, batch size, warm-up, repetitions, CUDA sync, cache status and per-trial timing. No run-linked latency log identified.")
        add(out,9,"trainable parameter counts","conflict","paper Table 9; Semester/results/verified_oxford_pets_benchmark.json",
            "ResNet cosine paper=75,815 whereas verified Oxford JSON records 75,777. Count parameters from the actual saved model state.")
    if table is None:
        a=torch_data("Semester/results/verified_test_predictions.pt")
        cited=torch_data("Semester/results/test_predictions_verified.pt")
        if a:
            y=a["test_labels"]; dual=a["pred_dual"]; coop=a["pred_coop"]; res=a["pred_res"]
            b=int(((dual==y)&(coop!=y)).sum()); c=int(((dual!=y)&(coop==y)).sum())
            br=int(((dual==y)&(res!=y)).sum()); cr=int(((dual!=y)&(res==y)).sum())
            add(out,0,"Oxford McNemar DualStream vs CoOp","conflict","Semester/results/verified_test_predictions.pt",
                f"b/c={b}/{c}; DualStream={int((dual==y).sum())}/{len(y)}=91.71%, paper cites 92.15% for the row. Pair valid for this run only.")
            add(out,0,"Oxford McNemar DualStream vs ResNet alternative run","protocol_mismatch","Semester/results/verified_test_predictions.pt",
                f"b/c={br}/{cr} matches paper stats, but paper cites a different prediction file. This run gives DualStream={int((dual==y).sum())}/{len(y)} and ResNet={int((res==y).sum())}/{len(y)}.")
        if cited:
            y=cited["test_labels"]; dual=cited["pred_dual"]; res=cited["pred_res"]
            br=int(((dual==y)&(res!=y)).sum()); cr=int(((dual!=y)&(res==y)).sum())
            add(out,0,"Oxford McNemar DualStream vs ResNet cited file","conflict","Semester/results/test_predictions_verified.pt",
                f"paper cites this file and b/c=352/161; actual b/c={br}/{cr}; ResNet={int((res==y).sum())}/{len(y)}.")
        add(out,0,"No SOTA framing","conflict" if re.search(r"\bSOTA\b|state.of.the.art",PAPER.read_text(encoding="utf-8"),re.I) else "match",
            "paper/paper_CLIP_Gemma4_revised_v2.md","Search for SOTA or state-of-the-art framing; avoid benchmark leadership claims.")
    return out

def markdown(items):
    lines=["| Table | Claim | Status | Evidence | Detail |",
           "|---|---|---|---|---|"]
    for r in items:
        esc=lambda value:str(value).replace("|","/").replace(chr(10)," ")
        lines.append("| "+" | ".join(esc(r[k]) for k in ["table","claim","status","evidence","detail"])+" |")
    return chr(10).join(lines)
