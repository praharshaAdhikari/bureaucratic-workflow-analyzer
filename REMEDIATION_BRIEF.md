# Remediation Brief: DigitalTwinsRL / Managerial-RL for BPM

**Purpose.** This document is an instruction set for an AI coding agent working in the project repository. It covers (A) code and experiments that must be run, (B) the rewrite of the project report, and (C) the rewrite of the IEEE NEPCON conference paper.

**Two deliverables exist and must stay consistent:**
1. `conferencePaper` — 6-page IEEE two-column, NEPCON 2026, deadline 15 September 2026 (verify: the CFP page says 15 August, the homepage and Important Dates page say 15 September).
2. `digitalTwins` — the full B.E. project report.

Every number in the conference paper must be traceable to a number in the report, and every number in the report must be traceable to an artefact on disk.

---

## 0. Ground rules for the agent

- **Do not fabricate results.** If an experiment has not been run, write `TODO(experiment-name)` and stop. Never write a plausible number.
- **Do not delete negative results.** BPIC2015 underperforming Reward-Greedy is a finding, not an embarrassment.
- Every table and figure must be regenerated from a script that is committed. No hand-edited numbers.
- Emit one JSON/CSV artefact per experiment under `results/<experiment>/`, and have the report build read from those files.
- After every experiment below, append a row to `results/EXPERIMENT_LOG.md` with date, commit hash, seed(s), and output path.

---

# PART A — CODE AND EXPERIMENTS

Ordered by priority. A1–A4 are blocking: the paper should not be submitted without them.

## A1. Constrain the routing mask to log-observed transitions [BLOCKING, HIGHEST VALUE]

**Problem.** Report Table 5.6 shows that 30 of 96 transitions taken by the RL agent (31.2%) do **not exist in the real event log**. They are labelled "impossible: does not exist in real log (simulator limitation)." Section 6.2 currently mis-describes this as masking *eliminating* impossible transitions; it does the opposite. Combined with mean episode lengths of 1.4 steps (BPIC2015, empirical median 45) and 7.0 steps (BPIC2017, empirical median 35), and heuristic baselines terminating in 1–2 steps on a process with median trace length 11, the conclusion is that **the digital shadow permits trajectories that cannot occur in reality, including near-immediate jumps to terminal states.**

This is the root cause behind several separately-reported symptoms. Fix it first; several other results may change.

**Tasks.**
1. In the environment's routing mask construction, restrict valid successors of activity `a` to `{b : count(a→b) > 0}` from the training log's directly-follows relation. Add config flag `strict_transition_mask: true|false` so the permissive behaviour remains reproducible for comparison.
2. Audit the fallback distribution (`P_fb` over non-terminal activities, Methodology 3.3.1). This fallback is a likely source of impossible transitions. When `strict_transition_mask` is on, an activity with no outgoing transitions should terminate the episode, not sample globally.
3. Verify the terminal-activity detection rule is applied consistently. Section 3.3.1 says `end_frac > 0.15 AND terminal_rate > 0.40`; Section 5.4 says `≥ 5% rate = good`. **Pick one, document it, apply it everywhere.**
4. Re-run the full pipeline (calibration → validation → training → evaluation) under both flags for all three datasets.
5. Produce `results/strict_mask_ablation/comparison.csv` with, per dataset per flag: mean reward, bad-terminal %, mean episode length, empirical median length, and % of transitions classified `impossible`.

**Expected outcome and how to report it.** Under `strict_transition_mask: true` the impossible-transition rate must be 0% by construction. Mean episode lengths should move toward the empirical median. Reward and bad-terminal numbers may degrade. **If they degrade, that is the paper's main finding, not a failure:** aggregate distributional fidelity metrics are insufficient to validate an RL environment for prescriptive process monitoring. Report it as such.

## A2. Add trajectory-level fidelity metrics [BLOCKING]

**Problem.** All seven current metrics (Table 5.1) are aggregate or distributional: activity-frequency JSD, transition-matrix L1, trace-length Wasserstein, duration KS, resource-utilisation MAE. Variant coverage is measured **one-directionally** (do the top-N real bigrams appear in sim?), so it structurally cannot detect simulated bigrams that never occur in reality. This is why the shadow passes every threshold while generating impossible trajectories.

**Tasks.**
1. Implement **bidirectional variant coverage**: report both `|B_real ∩ B_sim| / |B_real|` (existing) and `|B_sim \ B_real| / |B_sim|` (new: fraction of simulated bigrams absent from the real log). Threshold the new metric at < 0.01.
2. Implement **trajectory validity rate**: fraction of complete simulated traces whose every consecutive pair exists in the real directly-follows relation. Threshold > 0.95.
3. Implement **terminal-reachability sanity check**: distribution of steps-to-terminal in sim vs real, reported as a Wasserstein distance and as a `P(length ≤ 2)` comparison. The current heuristic baselines terminating in 1–2 steps should fail this loudly.
4. Add all three to the validation report and to Table 5.1. Re-run validation for all datasets under both mask settings.

**Note for the writeup.** These three metrics are a genuine small methodological contribution. Frame them that way.

## A3. Routing-only ablation [BLOCKING]

**Problem.** The entire novelty claim of the conference paper is the 15-action managerial intervention space. No experiment isolates its contribution. Since management actions carry no intrinsic reward, a routing-only agent may match performance exactly.

**Tasks.**
1. Add config flag `enable_management_actions: true|false`. When false, the action space collapses to `Discrete(K_t)` and only action 0 (`assign_to_primary_team`) is applied internally.
2. Train and evaluate on all three datasets, 5 seeds each (see A4), under `strict_transition_mask: true`.
3. Output `results/mgmt_ablation/comparison.csv`: mean reward, bad-terminal %, episode length, cycle time, with per-seed values and across-seed mean ± std.
4. Run a two-sample test (Welch's t-test or Mann–Whitney) on across-seed reward, report the p-value.

**Report the result whatever it is.** If management actions do not help, say so; that is a publishable negative result about the managerial framing and is far better than an unsupported claim.

## A4. Multiple seeds [BLOCKING]

**Problem.** All reported `± std` values (e.g. `+6.59 ± 0.15`) are over 500 evaluation episodes of **one trained policy**. This measures environment stochasticity, not policy reliability. RL results without training-seed variance are not credible. BPIC2015 RL is `+0.31 ± 0.53`, so the mean is well inside one evaluation std of zero already.

**Tasks.**
1. Parameterise the training seed. Run seeds `{0, 1, 2, 3, 4}` per dataset per configuration.
2. Report **two** variance figures everywhere and label them distinctly:
   - *across-seed std* of the per-seed mean reward (this is the one that matters);
   - *within-seed std* over 500 evaluation episodes (keep, but subordinate).
3. Update every results table to `mean ± std_across_seeds (n=5)`.
4. If across-seed std exceeds the RL-vs-best-heuristic gap on any dataset, state plainly that the difference is not resolvable at this sample size.

## A5. Report process KPIs, not just reward [BLOCKING]

**Problem.** Report Section 3.6.3 defines cycle time, terminal rate, and KPI signals as evaluation metrics. **Cycle time is never reported anywhere in the results.** The evaluation reports only mean episode reward — the exact quantity PPO was trained to maximise and that no baseline optimises. This is circular and a reviewer will say so in one sentence.

**Tasks.**
1. In `PolicyEvaluator`, actually emit and persist per-episode: cycle time (seconds, first to last event), completion rate, rework count, loop rate, mean resource utilisation, and episode length.
2. Produce a new results table, **RL vs all baselines, per dataset, on these KPIs** — reward gets one column among many, not its own table.
3. Add real-log reference values as a row so the reader can see how far each policy sits from observed practice.
4. This table becomes the primary results table in both documents. The reward table becomes secondary.

## A6. Reconcile training reward with evaluation reward [BLOCKING]

**Problem.** Training curves plateau at roughly +37 (BPIC2012), +35 (BPIC2015), +19 (BPIC2017). Evaluation Table 5.2 reports +6.59, +0.31, +2.58. That is a 5x to 100x discrepancy and nothing in 72 pages explains it.

**Tasks.**
1. Diff the reward configuration, episode length cap, and normalisation used by the training env vs the evaluation env. Likely culprits: reward weights re-tuned or defaulted between phases; different `max_steps`; evaluation wrapper applying different termination.
2. Fix so both use identical configuration, or, if the difference is intentional, document it explicitly in one paragraph.
3. Re-run and confirm training plateau and evaluation mean agree to within the across-seed std.

## A7. Fix the reward function definition

1. **Dead term.** `R_mgmt = -0.1` for an invalid management action can never fire, because MaskablePPO sets invalid logits to `-∞`. Either delete the term from the formalism, or document a code path where masking is bypassed. Currently the paper defines a term that is identically zero.
2. **Inconsistent routing reward.** Methodology 3.5.3 writes the good-terminal branch as `-w_step + w_terminal + b(L_t)`; System Design 4.6.3 writes it as `w_terminal + b(L_t)`, and the "otherwise" branch differs between the two sections (`w_loop` appears with opposite sign conventions, and 4.6.3 adds `w_length` and `w_loop_rate` terms that 3.5.3 does not define). **Write the reward once, in one place, matching the code, and reference it from everywhere else.**
3. **No intervention cost.** `add_temporary_staff`, `outsource_to_volunteer_pool`, and `trigger_high_cost_escalation` are free in the reward. The agent has no incentive to be parsimonious with expensive interventions. Add a small per-intervention cost term (config-driven, default nonzero) and re-tune. If you keep them free, defend it explicitly.
4. **No compliance term.** See A8.

## A8. Address the governance-relaxation finding honestly

**Problem.** The two dominant learned interventions on BPIC2012 are `relax_rules_for_low_risk` (34–41% of steps) and `skip_optional_subprocess` (23–25%). Both documents present this as evidence of interpretable managerial logic. An equally consistent reading: with a per-step cost and a length-peaked terminal bonus, the agent learned that skipping work and waiving rules is the cheapest route to a good terminal. In a permit-issuing public-administration context, "waive the rules and skip the steps" is a compliance failure, not an optimisation.

**Tasks.**
1. Add a configurable compliance penalty triggered when `relax_rules_for_low_risk` or `skip_optional_subprocess` fire on cases that later show objection/suspension/refusal flags, or that skip activities appearing in >90% of real traces.
2. Re-run with the penalty on and off. Report both.
3. Add an explicit analysis: what fraction of episodes where `relax_rules` fired would, in the real log, have required the skipped step?
4. In the writeup, present this as a limitation and a reward-design finding. Do not describe it as "governance innovation" (Report 5.3.6 claim 4, 6.4 benefit 2). Delete that framing.

## A9. Fix the percentage-improvement mathematics

**Problem.** `Δ(%) = (R_RL − R_base)/|R_base| × 100` is applied across a sign change: −14.02 → +6.59 yields "+146.96%". A percentage improvement over a negative baseline is not interpretable, and "108–147%" in the abstract makes it look systematic.

**Tasks.**
1. Stop reporting percentage gain vs Random. Report absolute rewards and the absolute difference.
2. Where a normalised figure is wanted, define a bounded range (e.g. `[R_worst_observed, R_best_achievable]`) and report position within it, stating the range.
3. Percentage vs the best heuristic is acceptable **only where both values are positive** (BPIC2012, BPIC2017). It is not valid where signs differ.
4. Add effect sizes (Cohen's d) computed across seeds.
5. Report Table 5.3 is truncated mid-column in the current PDF. Regenerate it.

## A10. Fix dataset facts

1. **Report Section 5.2.2 labels BPIC2012 as "Manufacturing" and BPIC2017 as "Permit Workflows." Both are wrong.** BPIC2012 and BPIC2017 are both Dutch financial-institution loan applications. BPIC2015 is the Dutch municipal building-permit log. Fix every occurrence. If this reaches a BPM reviewer the paper is finished.
2. **Acknowledge BPIC2012 and BPIC2017 are the same underlying process** (2017 is a re-recording of the loan process from a newer system). The claim of "three structurally different processes" must be softened to two domains, one of them observed twice. Any BPM reviewer knows this.
3. **Activity counts are inconsistent.** Abstract and evaluation say "24 to 356 activities"; Figure 5.5 says BPIC2012=24, BPIC2015=356, BPIC2017=26; the datasets section gives none of these. Compute them from the logs, put them in the dataset table, use them consistently.
4. **State BPIC2015 sublog handling explicitly.** Five municipalities exist. Figure 3.2 shows five. Section 6.5 mentions M1–M5. The ProM screenshot reports 1,199 traces, consistent with one municipality. Say exactly which logs were used, whether they were merged or trained separately, and give trace/event counts per municipality.
5. Give trace and event counts for BPIC2015 alongside those already given for 2012 and 2017.

## A11. Reproducibility

1. `MaskablePPO Training Configuration` (Table 4.5) says "Default SB3 hyperparameters are used for learning rate, batch size, n_epochs, and clip range with dataset-specific tuning." These are contradictory. State the actual values used, and state plainly whether any per-dataset tuning occurred — the conference paper's central claim is "identical hyperparameters across all three datasets," so any dataset-specific tuning directly contradicts it.
2. Document: learning rate, n_steps, batch size, n_epochs, clip range, entropy coefficient, GAE lambda, gamma, network architecture, embedding dimension (32 per Table 4.3, confirm), `max_steps` per episode per dataset, and total wall-clock/compute.
3. Emit a `config_used.json` with every effective hyperparameter next to each result.
4. Add a repository link and a `REPRODUCE.md` with exact commands.
5. Add a sensitivity analysis over the KPI thresholds in Table 4.4 (`DELAY_HIGH=1.0`, `VOLUME_HIGH=0.5`, etc.). Perturb each by ±25%, report whether conclusions hold. These constants are currently unjustified.

## A12. Complete or cut RQ2

**Problem.** Report Section 5.4.1 contains literal placeholders: `[To be populated after running drift detection...]` and `[Policy performance comparison pre/post refit would show...]`. Table 5.4 shows TL Wasserstein degrading from 0.782 to 1.717 after refit — a 2.2x worsening — labelled `✓ recovered` because it stayed under threshold.

**Tasks (choose one path):**
- **Complete it.** Inject controlled synthetic drift (e.g. shift transition probabilities for a subset of activities, or split BPIC2015 temporally by year). Measure fidelity and *downstream RL policy reward* before drift, after drift without refit, and after refit. Without the policy-performance arm, RQ2 shows only that refitting a simulator on new data fits the new data, which is trivially true.
- **Cut it.** Remove RQ2 from the evaluation, retain drift detection as a described system capability in System Design, and move validation to Future Work.
- **Either way:** relabel the Table 5.4 status column. A 2.2x degradation is not "recovered."

## A13. Consistency sweep

Fix all of these:
- Section 3.6.3 says evaluation uses **200** episodes; every table says **500**.
- Section 5.1 announces **"three research questions"** and lists **two**.
- Section 5.3.6 claim 4 asserts multi-action strategies on BPIC2017; Section 5.2.2 says BPIC2017 action-adoption rates "require evaluation-phase measurement," i.e. were never measured. Either measure them or delete the claim.
- **All citations render as `?`** — the BibTeX is broken. Fix before anything else is judged.
- Table 4.1 and Table 4.4 have run-together text (`unique_resourcesCardinality`, `has_suspensionBoolean`, `outsource to volunteer poolvolume pressure`). Fix column widths.
- Typo: `voume pressure` → `volume pressure` (conference paper Table I, row 13).
- Report Section 6.2 references "Section 3.5.5" which does not exist.
- Report Section 5.2.1 references "Section 4.3" for the seven-metric framework; it is Section 3.3.6.
- Repeated `i.e.` where `e.g.` is meant ("interventions i.e. task prioritization", "future process states i.e. next activity", "managerial-level interventions i.e. prioritization"). At least four occurrences across both documents.
- Terminal classification threshold: `end_frac > 0.15 AND terminal_rate > 0.40` (3.3.1) vs `≥ 5% rate = good` (5.4). Pick one.

## A14. Literature additions

Add and engage with (not just cite):
- **Metzger et al.**, online RL for triggering proactive business process adaptations — the closest prior work to a process-level intervention agent. Its absence undermines the novelty claim.
- **Fahrenkrog-Petersen et al.**, alarm-based prescriptive monitoring with cost models — directly relevant to when-to-intervene and to intervention cost (A7.3).
- **Bozorgi et al.**, causal inference for prescriptive process monitoring.
- **Weinzierl et al.**, next-best-action prescription.
- **Branchi et al.**, prescriptive monitoring under uncertainty.

Also fix: the conference paper supports the claim that managerial decision spaces are "comparatively underexplored" with citations [18]–[20] = Mnih (DQN), Schulman (PPO), Williams (REINFORCE). These are general RL algorithm papers and say nothing about BPM decision granularity. **This is the most damaging citation error in the paper because it sits exactly where novelty is staked.** Replace with real BPM evidence or drop the claim.

## A15. Terminology: shadow vs twin

The report correctly cites Kritzinger's taxonomy (model / shadow / twin, distinguished by data-flow directionality) and correctly calls the artefact a **digital shadow** throughout Section 3.3. But the project title, abstract, and conference framing say **Digital Twin**. The artefact is unidirectional and offline; it is a shadow. Retitle and reframe, or explain precisely which bidirectional mechanism makes it a twin (the drift-refit loop is a candidate argument, but only if A12 is completed). Being precise here is a credibility gain, not a loss.

---

# PART B — REPORT REWRITE

Keep the existing chapter structure. Apply the following.

### Chapter 3 (Methodology)
- Single canonical reward definition (A7.2). Delete the duplicate in 4.6.3, replace with a cross-reference.
- Add the three new fidelity metrics to 3.3.6 (A2).
- Add the strict-transition-mask formulation to 3.5.2 (A1).
- Fix dataset facts and add BPIC2015 counts and sublog handling (A10).
- State evaluation episode count once (A13).

### Chapter 4 (System Design)
- Fill in real hyperparameters (A11).
- Add compliance penalty and intervention cost to 4.6.3 (A7.3, A8).
- Fix Table 4.1 and 4.4 formatting.

### Chapter 5 (Evaluation) — heaviest changes
Restructure results in this order:
1. **Simulation fidelity** — Table 5.1 extended with the three new metrics, both mask settings. Lead with the finding that the permissive shadow passes all seven original metrics while producing a 31.2% impossible-transition rate. This motivates everything after.
2. **Process KPI comparison** (A5) — the new primary table. RL vs all baselines vs real-log reference.
3. **Episode-length fidelity, all three datasets** — not a BPIC2015 footnote. 10.0/11, 7.0/35, 1.4/45. Note that heuristics terminate in 1–2 steps, which is itself evidence of a permissive environment rather than of good heuristics.
4. **Reward comparison** — now secondary, with across-seed variance (A4).
5. **Ablations** — management actions on/off (A3), strict mask on/off (A1).
6. **Interpretability layer** — Tables 5.5, 5.6, Figures 5.8, 5.9. Keep. Report the impossible-transition row prominently rather than burying it.
7. **RQ2** — completed or cut (A12).

Delete or rewrite: Section 5.3.6 claim 4; the "Manufacturing"/"Permit Workflows" complexity labels; Section 6.4's "quantifiable business benefits" (all measured against Random, which is not a business-relevant comparator).

### Chapter 6 (Discussion)
- Rewrite 6.1 around the environment-validity finding as the central result.
- Fix 6.2's mis-description of the 31.2% figure (it currently claims masking *eliminated* impossible transitions; the table says it permitted them).
- Keep 6.3 (RIMS scope argument) essentially as-is; it is sound.
- Expand 6.5 Limitations with: single-log-family overlap (BPIC2012/2017), no sim-to-real validation, reward-shaping interaction with process structure, compliance risk of learned interventions.

---

# PART C — CONFERENCE PAPER REWRITE

**Constraints.** IEEE two-column, max 10 pages, registration covers 6, USD 5/page for 7–10. Target 6–7 pages. Verify blind-review policy on the Author Guidelines page before submitting; the site says single/double-blind is TBC and the current PDF has full names and affiliation.

### Reframe the contribution

The current framing ("managerial RL eliminates bad terminals and improves reward 108–147%") does not survive its own Table II: Greedy, Empirical Markov, and Reward-Greedy also achieve 0% bad terminals on BPIC2012 and BPIC2017. Multiple cheap heuristics already achieve the headline result.

**Proposed new framing:**

> We formulate prescriptive BPM as a managerial intervention problem and train masked PPO over a 15-action intervention catalogue on three BPIC logs. In doing so we find that discrete-event simulators calibrated to standard aggregate fidelity metrics (activity-frequency divergence, transition-matrix distance, trace-length Wasserstein) admit trajectories that cannot occur in the source log — 31.2% of learned transitions on BPIC2012 — producing policies that appear to eliminate bad outcomes while terminating in a small fraction of the empirical trace length. We introduce three trajectory-level fidelity metrics that detect this, show that constraining the routing mask to log-observed transitions changes the picture, and report a managerial-action ablation isolating the contribution of the intervention catalogue.

This is defensible, novel, and reviewer-proof in a way the current framing is not. It also makes every awkward number an asset rather than a liability.

### Section plan (6–7 pages)

| Section | Content | Approx. |
|---|---|---|
| I. Introduction | Managerial vs worker-level granularity. New RQ per above. | 0.75 col |
| II. Related Work | Existing content + A14 additions. Fix [18]–[20] misuse. | 1 col |
| III. Method | MDP, managerial action space (Table I), single canonical reward, strict mask, three new fidelity metrics | 1.5 col |
| IV. Setup | Datasets (corrected facts, A10), baselines, hyperparameters, seeds | 0.75 col |
| V. Results | Fidelity table (both mask settings, new metrics); KPI table; episode-length table; reward table w/ across-seed std; management ablation | 2.5 col |
| VI. Discussion | Environment-validity finding; compliance risk of learned interventions; RIMS scope | 1 col |
| VII. Limitations & Conclusion | Sim-to-real, log-family overlap, single institution | 0.5 col |

### Port from the report

1. Table 5.1 (fidelity, compressed, extended with new metrics) — highest value.
2. Section 6.3 RIMS scope paragraph, roughly verbatim.
3. Episode-length vs empirical median, all three datasets, as a compact table.
4. Table 5.6 (transition classification incl. the 31.2% row) and Figure 5.9 (top divergences).
5. One training-convergence figure (BPIC2012).
6. Richer related work incl. Kritzinger taxonomy.
7. Documented KPI thresholds as a footnote to Table I.

### Do NOT port
- RQ2 / drift adaptation beyond one future-work sentence.
- Section 6.4 "quantifiable business benefits."
- The "Manufacturing" / complexity labels.
- Section 5.3.6 claim 4.
- Any percentage-vs-Random figure.

### Abstract and title
- Rewrite the abstract around the new framing. Remove "eliminates bad terminal outcomes entirely" and "108–147%."
- Retitle to reflect either the shadow terminology (A15) or the environment-validity finding, e.g. *"Managerial-Level Reinforcement Learning for Business Process Optimization: On the Validity of Log-Calibrated Simulators as Training Environments."*
- Add a short paragraph on accountability and worker impact of automating managerial decisions — NEPCON has a Societal & Ethical Aspects track and this costs one paragraph.

---

# EXECUTION ORDER

Given roughly three weeks to the deadline, and that A1 may change downstream numbers:

**Week 1** — A13 (fix BibTeX first; nothing is assessable while citations render as `?`), A10, A7, A9, A11. Then A1 and A2 implemented and validation re-run. Launch A4 seeds in background.

**Week 2** — A3 ablation (5 seeds), A5 KPI evaluation, A6 reconciliation, A8 compliance analysis. A12 decision (complete or cut — cut if time-constrained).

**Week 3** — Report rewrite (Part B), conference paper rewrite (Part C), A14 literature, A15 terminology. Freeze results by day 5 of week 3.

**Definition of done:** every number in both documents regenerable by a committed script; no `TODO` markers remain in the submitted paper; the conference paper's claims are each supported by a table in the report; and the abstract contains no claim that Table II contradicts.
