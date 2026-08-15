#!/usr/bin/env python3
"""Finalize the preregistered C46 contract-aligned online ranker canary."""

from __future__ import annotations

import argparse, json, os
from collections import defaultdict
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite C46 report")
    if not all((args.root / f"shard{i}.COMPLETED").is_file() for i in range(4)):
        raise RuntimeError("C46 shards are incomplete")
    rows = {}; ordinal = 0
    for suite in ("libero_goal", "libero_object", "libero_spatial", "libero_10"):
        for task in range(5):
            for arm in ("control", "bestof4"):
                path = args.root / "runs" / f"{ordinal}_{arm}_{suite[7:]}_task{task}_trial23" / "results.json"
                rows[(suite, task, arm)] = json.loads(path.read_text())["tasks"][0]["episodes"][0]
                ordinal += 1
    control_success = candidate_success = wins = losses = 0
    per_suite = defaultdict(lambda: {"control": 0, "bestof4": 0})
    eligible = selected_nonzero = 0
    exact_prestate = exact_candidate0 = seeds_valid = True
    score_ranges = []; pairs = []
    offsets = [0, 1_000_000, 2_000_000, 3_000_000]
    for suite in ("libero_goal", "libero_object", "libero_spatial", "libero_10"):
        for task in range(5):
            control = rows[(suite, task, "control")]; candidate = rows[(suite, task, "bestof4")]
            c0, c1 = bool(control["success"]), bool(candidate["success"])
            control_success += c0; candidate_success += c1
            wins += c1 and not c0; losses += c0 and not c1
            per_suite[suite]["control"] += c0; per_suite[suite]["bestof4"] += c1
            indices = candidate["consequence_selected_indices"]
            if indices:
                eligible += 1
                selected_nonzero += int(indices[0] != 0)
                if len(indices) != 1:
                    raise RuntimeError("C46 candidate ranked more than once")
                score_ranges.extend(candidate["consequence_score_ranges"])
                cp = np.load(control["trajectory"]); bp = np.load(candidate["trajectory"])
                ci = int(np.flatnonzero(cp["step"] == 80)[0]); bi = int(np.flatnonzero(bp["step"] == 80)[0])
                exact_prestate &= np.array_equal(cp["sim_state"][ci], bp["sim_state"][bi])
                exact_candidate0 &= np.array_equal(
                    cp["policy_actions"][ci],
                    np.asarray(candidate["first_consequence_candidate0_chunk"], dtype=np.float32),
                )
                base = int(candidate["replan_noise_seeds"][bi])
                seeds_valid &= candidate["consequence_candidate_seeds"][0] == [base + x for x in offsets]
            else:
                seeds_valid &= int(candidate["steps"]) < 80
                exact_prestate &= c0 == c1
            pairs.append({"suite": suite, "task": task, "control_success": c0, "bestof4_success": c1})
    gate = {
        "all_step80_preintervention_states_exact": exact_prestate,
        "all_candidate0_step80_chunks_exact_control": exact_candidate0,
        "all_seed_offsets_match_c43": seeds_valid,
        "exactly_one_ranking_for_each_eligible_episode": eligible > 0,
        "all_rankings_have_score_variation": bool(score_ranges) and min(score_ranges) > 1e-8,
        "nonzero_candidate_selected_at_least_20_percent": eligible > 0 and selected_nonzero / eligible >= .20,
        "candidate_success_at_least_control_plus_2": candidate_success >= control_success + 2,
        "paired_wins_at_least_3": wins >= 3,
        "paired_net_wins_at_least_2": wins - losses >= 2,
        "no_suite_regresses_by_more_than_1_of_5": all(v["bestof4"] >= v["control"] - 1 for v in per_suite.values()),
    }
    gate["passed"] = all(gate.values())
    report = {
        "format": "h3wam-c46-contract-aligned-ranker-v1",
        "status": "PASS_C46_CONTRACT_ALIGNED_RANKER" if gate["passed"] else "FAIL_C46_CONTRACT_ALIGNED_RANKER",
        "permission": "GO_POWERED_CONTRACT_ALIGNED_RANKER" if gate["passed"] else "NO_GO_C44_ONLINE_RANKER",
        "claim_boundary": "Twenty paired new-trial episodes test one C43-aligned intervention, not full benchmark superiority.",
        "results": {"control_successes": control_success, "candidate_successes": candidate_success,
                    "wins": wins, "losses": losses, "ties": 20-wins-losses,
                    "eligible_step80_pairs": eligible, "selected_nonzero": selected_nonzero,
                    "per_suite": dict(per_suite), "pairs": pairs},
        "gate": gate,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_name(f".{args.output.name}.{os.getpid()}.partial")
    tmp.write_text(json.dumps(report, indent=2) + "\n"); os.replace(tmp, args.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
