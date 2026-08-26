# Fix log

Plain-language record of what was changed, why, and what it produced.
One entry per fix. Newest work at the bottom.

**How we're working:** small fix → cheap check → you review → next fix.
The expensive step (retraining the three agents) is held back until Fix 1, 2
and 3 are all in, so we only pay for it once.

---

## Fix 1 — The simulator was learning steps that never happened

**Status:** ✅ Done and verified end-to-end.
**Date:** 2026-08-23

### What was wrong

To learn the process, the code reads each case in order and notes which step
follows which. Before doing that it converted every timestamp into text and
back into a date again.

Timestamps that land exactly on a whole second — no fraction of a second —
fail that conversion. The code was set to silently replace anything it could
not read with a blank. Blank dates get sorted to the **end** of the case, so a
step from the middle of a case was moved to the last position. The simulator
then recorded a link between two steps that never actually follow each other.

### What was changed

| File | Change |
|---|---|
| `src/timeutils.py` | **New.** One safe way to convert timestamps, used everywhere. Never converts a date to text and back. Raises a clear error instead of silently producing blanks. Also provides `sort_events()`, which uses a stable sort so events sharing a timestamp keep their recorded order. |
| `src/digital_twin.py` | Two places (`fit`, `_fit_durations`) now use the safe helper. |
| `src/validation.py` | Four places now use the safe helper. |
| `src/feature_engineering.py` | `_ensure_timestamps` now delegates to the helper instead of keeping its own copy of the logic. |
| `src/data_ingestion.py` | Uses the helper, and the stable sort. |
| `checks/fix1_transition_integrity.py` | **New.** Re-measures the problem so the numbers are reproducible. |

### What it solves

Every "impossible transition" reported in the project report. These were not a
design flaw in the simulator and not a limitation of the approach — they were
a data-loading mistake.

### Result

`python checks/fix1_transition_integrity.py`

| Dataset | Broken timestamps (before) | Broken timestamps (after) | Real step-links | Simulator step-links | Invented |
|---|---|---|---|---|---|
| BPIC2012 | 286 | **0** | 125 | 125 | **0** (was 48, 27.7%) |
| BPIC2015 | 0 | **0** | 9,064 | 9,064 | **0** (was 0) |
| BPIC2017 | 1,177 | **0** | 178 | 178 | **0** (was 65, 26.7%) |

In plain terms: the simulator now believes in exactly the step-to-step links
that appear in the real records, and nothing else. On the two loan datasets
that removes just over a quarter of what it previously believed.

Artefact: `results/fix1_transition_integrity/comparison.csv`

### Side effects and things to know

- A smoke run (fit → simulate → validate on 400 BPIC2012 cases) still passes
  all seven quality checks, so the fix does not break the existing pipeline.
- The test suite: **63 pass, 1 fail.** The failing test
  (`test_integration.py::test_round_trip_fidelity`) **already failed before
  this change** — it is a pandas 3 text-column type comparison inside the test
  itself, not a problem with the pipeline. Left alone for now.
- The agent's current trained policy on BPIC2012 used three of the invented
  links in every one of its 500 test cases. That policy is now invalid and
  must be retrained — held until Fix 2 and Fix 3 are in.

### Addendum — one file was missed, found later

`src/insights.py` built its own copy of the real transition set and was not
patched with the others. It sorted by timestamp alone, and pandas' default
single-column sort is **not stable**. BPIC2015 has **35,504** events sharing a
timestamp with another event in the same case, so the shuffle produced 11,381
directly-follows edges instead of the true 9,064, with 1,582 genuine edges
missing. Those missing edges were then reported as "impossible transitions".

That is why the first post-fix run still showed 1,266 impossible transitions on
BPIC2015 while BPIC2012 and BPIC2017 showed zero — the two loan logs have few
tied timestamps, so an unstable sort happens to be harmless there.

Now fixed by routing through `timeutils.sort_events`. Recomputed against the
existing trajectories:

| Dataset | Impossible transitions before | After |
|---|---|---|
| BPIC2012 | 0 | **0** |
| BPIC2015 | 1,266 | **0** |
| BPIC2017 | 0 | **0** |

The report's headline "31.2% of transitions are impossible" is now zero across
all three datasets, and the two causes were both measurement bugs: a lossy
timestamp round-trip when fitting, and an unstable sort when checking.

### Confirmed after your pipeline run (notebook 3, all three datasets)

The rebuilt simulators on disk contain **zero** invented links:

| Dataset | Simulator step-links | Real step-links | Invented |
|---|---|---|---|
| BPIC2012 | 125 | 125 | 0 |
| BPIC2015 | 9,064 | 9,064 | 0 |
| BPIC2017 | 178 | 178 | 0 |

Quality checks, before → after:

| Metric | BPIC2012 | BPIC2015 | BPIC2017 |
|---|---|---|---|
| Trace length (Wasserstein) | 0.7828 → 0.7306 | 0.4203 → 0.4203 | 0.4599 → 0.4736 |
| Case duration | 0.0349 → 0.0346 | 0.0647 → 0.0647 | 0.0498 → 0.0467 |
| Activity frequency (JSD) | 0.0014 → 0.0013 | 0.0097 → 0.0097 | 0.0009 → 0.0008 |
| **Transition matrix (L1)** | **0.0017 → 0.0009** | 0.0018 → 0.0018 | 0.0004 → 0.0005 |
| Duration distribution (KS) | 0.0112 → 0.0083 | 0.0500 → 0.0500 | 0.0082 → 0.0073 |
| Variant coverage | 1.0000 → 1.0000 | 0.9859 → 0.9859 | 1.0000 → 1.0000 |
| Resource utilisation | 0.0002 → 0.0002 | 0.0014 → 0.0014 | 0.0002 → 0.0003 |

Two things to notice, and both matter for the write-up:

1. **BPIC2015 is unchanged to four decimal places.** It had no broken
   timestamps, so nothing should have moved, and nothing did. That is the
   control case proving the fix touched only what it should.
2. **The quality checks barely reacted.** On BPIC2012 more than a quarter of
   the simulator's step-links were wrong, and the strictest relevant metric
   still read 0.0017 against a pass threshold of 0.1 — about sixty times
   better than required. After the fix it reads 0.0009. **No reader of the
   original results table could have detected the problem.** This is the
   paper's real finding, now measured rather than argued.

### What this does *not* fix

Nothing about the "successful ending" labels (Fix 2) or the two different
scoreboards (Fix 3). Expect the headline numbers to stay wrong until those
land.

---

## Fix 2 — The "successful endings" were not endings

**Status:** code changed, checks passing. Waiting on your notebook 02 run.
**Date:** 2026-08-23

### What was wrong

The agent needs to know which activities count as finishing a case. The old
rule kept any activity that ended at least 5% of real cases, threw out
anything starting with `W_`, and — if that left nothing — quietly fell back to
matching words like "accept", "approve" or "complete" anywhere in the name.

On both loan datasets the first rule always left nothing, so the word-matching
fallback always fired. It produced endings that end **zero** real cases:

| Dataset | Old "good endings" | How many real cases they end |
|---|---|---|
| BPIC2012 | A_ACCEPTED, A_APPROVED, A_FINALIZED, A_PREACCEPTED, O_ACCEPTED | **0** |
| BPIC2017 | A_Accepted, A_Complete, O_Accepted | **0** |

A_PREACCEPTED happens about 17% of the way through a case. A_ACCEPTED about
28%. The agent was "winning" by reaching the early middle.

### What was changed

The test is no longer *"does this activity end a lot of cases?"* but
**"whenever this activity happens, does it happen near the end?"** Concretely,
an activity qualifies when 75% of its occurrences fall in the last 20% of the
trace, and it appears in at least 2% of cases. Good versus bad is then decided
from the activity name as before.

| File | Change |
|---|---|
| `src/feature_engineering.py` | New `terminal_diagnostics()` (the evidence table) and `classify_terminals()` (the new rule). **The silent fallback is gone** — if no defensible ending exists, it raises `TerminalClassificationError` instead of inventing one. `classify_good_terminals()` kept as a thin wrapper so nothing else breaks. |
| `notebooks/02_feature_engineering.ipynb` | Uses the new function, and now saves the thresholds and the supporting evidence into `terminal_classification.json` so the labels are auditable. |
| `checks/fix2_terminal_labels.py` | **New.** Before/after comparison. |

### What it solves

The paper's headline claim. "0% bad terminal outcomes" was not a result — it
was the agent stopping at a mid-process step that had been mislabelled a
success. It also explains the short episodes.

### Result

`python checks/fix2_terminal_labels.py`

The new labels are the actual outcome markers of each process:

| Dataset | Good endings | Bad endings |
|---|---|---|
| BPIC2012 | A_ACTIVATED, A_APPROVED, A_REGISTERED, O_ACCEPTED | A_CANCELLED, A_DECLINED, O_DECLINED |
| BPIC2017 | A_Pending, O_Accepted | A_Cancelled, A_Denied, O_Cancelled, O_Refused |
| BPIC2015 | see hand-labelling below | see hand-labelling below |

These are the right ones. On a loan process, "approved / activated /
registered" is the successful outcome and "declined / cancelled" is not.

| Check | Before | After |
|---|---|---|
| Earliest point a labelled ending can occur (BPIC2012) | 8% into the trace | **86%** |
| Earliest point a labelled ending can occur (BPIC2017) | 16% into the trace | **89%** |
| Earliest point a labelled ending can occur (BPIC2015) | 82% into the trace | 80% |

**The task is now genuinely hard.** With a purely random policy:

| Dataset | Episodes reaching a good outcome |
|---|---|
| BPIC2012 | 11% |
| BPIC2017 | 14% |
| BPIC2015 | 83% (after hand-labelling; was 96%) |

Before this fix, "reach a good outcome" was nearly free. There is now a real
gap for the agent to close, which is what makes the comparison against the
heuristics meaningful at all.

### BPIC2015 hand-labelling (your call: option 2 — done)

All 24 BPIC2015 endings are now labelled by hand in
`config/terminal_labels/BPIC2015.json`, one entry each with a stated reason.
`classify_terminals()` accepts these as overrides; the loan datasets need none.

Three labels are allowed: `good`, `bad`, and **`exclude`** — an activity that
happens near the end but does not settle anything. Excluded activities are
dropped from the terminal set entirely, rather than counted as successes.

**The log corrected our first attempt.** We initially marked "objection lodged
against decision", "appeal logded", "contested decision affected" and "set
phase decision revoked" as bad outcomes. Then we checked whether such cases
recover:

| Activity | Cases | Still reach a positive settlement afterwards |
|---|---|---|
| objection lodged against decision | 2,035 | **79%** |
| appeal logded | 327 | **69%** |
| contested decision affected | 150 | **65%** |
| set phase decision revoked | 258 | **48%** |
| phase procedure aborted | 335 | 0% |
| decision permanent suspension irrevocable | 146 | 0% |
| copy decision permanent suspension to stakeholders | 119 | 0% |

An objection or an appeal is a negative **event during handling**, not a
negative **outcome** — four out of five of those cases still end with a permit.
So the top four became `exclude` and only the three clean ones stayed `bad`.
Labelling by name alone would have got this wrong in the opposite direction
from the original bug.

"close case", "phase archived case" and "phase case handled" are excluded for
the opposite reason: they occur on granted and refused cases alike, so they
record that handling stopped, not how it ended. "close case" is the single
strongest raw ending in the log (18.1% of traces), which makes it the most
tempting wrong answer.

**Final BPIC2015 set** — 2 good, 3 bad, 19 excluded:

- good: `set phase: phase permitting irrevocable`, `phase decision irrevocable`
- bad: `phase procedure aborted`, `decision permanent suspension irrevocable`,
  `copy decision permanent suspension to stakeholders`

Effect on the random-policy good rate: **96% → 55% → 83%** across the three
labellings we tried. The final 83% is the honest one, and it matches the real
log: of BPIC2015 cases that reach a settled outcome, 82.7% are positive.

**The remaining limitation, which must go in the write-up:** only **43.1%** of
BPIC2015 cases ever record a settled outcome. For the other 57% this log never
says whether the applicant got their permit. BPIC2015 outcome results describe
a minority of cases and have to be reported that way. That is a property of
the log, not of the fix.

### Real-log reference numbers (new — the report has none of these)

Now emitted by notebook 02 and by the check script. These are what any policy
should be compared against:

| Dataset | Cases reaching a settled outcome | Of those, positive | Median steps to outcome | Median trace length |
|---|---|---|---|---|
| BPIC2012 | 97.0% | **17.7%** | 9 | 11 |
| BPIC2015 | 43.1% | **82.7%** | 48 | 45 |
| BPIC2017 | 99.7% | **52.0%** | 31 | 35 |

BPIC2012's 17.7% and BPIC2017's 52.0% line up with the published acceptance
rates for those two logs, which is a good independent sign the labels are
right.

### Side effects and things to know

- Every episode still terminates on all three datasets (100% of 200 random
  rollouts), so nothing hangs or runs to the step cap.
- The length bonus still targets the median trace length, and that remains
  correct: the new outcome markers sit at roughly 90–97% of the way through a
  trace, so aiming for the full median length is right.
- Test suite unchanged: **63 pass, 1 fail** — the same pre-existing
  `test_round_trip_fidelity` failure described under Fix 1.
- `terminal_classification.json` gains two new keys (`thresholds`,
  `diagnostics`). Existing readers only use `bad_terminals` / `good_terminals`
  and are unaffected.

### Confirmed after your notebook 02 run

All three datasets wrote the expected labels, BPIC2015 applied all 24 manual
labels with no unmatched names, and 19 activities were excluded as
outcome-neutral.

---

## Fix 3 — The agent was trained on one scoreboard and graded on another

**Status:** code changed, checks passing. Waiting on your retrain.
**Date:** 2026-08-23

### What was wrong

Notebook 04 loaded `reward_weights.json` and set the reward for reaching a
good outcome to 30 points. Notebooks 05 and 06 built the same environment and
never loaded that file, so during grading a good outcome was worth 0.

Priced on the same episode, that is:

| | Good outcome worth |
|---|---|
| During training | **+34.95** |
| During grading | **+4.95** |

Which is the whole of the reported gap — training curves plateau near +36, the
results table says +6.59. There was never anything else to explain.

Two more problems came with it:

- **The key names did not match what they set.** `w_delay` set the *progress*
  bonus; `w_throughput` set the *per-step cost*. You could not check the
  mapping by reading it.
- **The weights differed per dataset.** The per-step cost was 0.05 / 0.2 /
  0.02 on BPIC2012 / BPIC2015 / BPIC2017, while the conference paper says in
  three places that identical reward weights were used everywhere.

### The real cause, and why fixing the symptom wasn't enough

The three notebooks each built the environment by hand — the same dozen lines
copied three times. They drifted because they were copies. Editing the two
that were wrong would have left the same trap for the next change, so the
duplication itself had to go.

| File | Change |
|---|---|
| `src/reward_config.py` | **New.** One `RewardConfig` holding every weight, under the names the environment actually uses. Saves and loads as JSON. Refuses a positive bad-outcome penalty. |
| `src/env_factory.py` | **New.** `build_process_env()` — the single way to construct the environment. Also `save_run_config()`, which writes `config_used.json` next to the results. |
| `src/rl_env.py` | Takes a `RewardConfig` instead of hard-coding defaults that were then overwritten elsewhere. |
| `notebooks/04, 05, 06` | All three now call the factory. None of them can set a weight the others don't see. |
| `checks/fix3_reward_parity.py` | **New.** Builds the environment the way each notebook does and asserts the reward and the episode cap match. |

### Decisions taken

1. **One reward for all three datasets, with the per-step terms expressed as
   shares rather than constants.** Per-dataset tuning is off. The tuned
   weights are still on disk and reproducible via
   `RewardConfig.from_legacy_file(...)` as an ablation — nothing uses them by
   default. See the revision note below: the first version of this decision
   was justified badly and only half-fixed the problem.
2. **A good outcome is +30 and a bad outcome is −30.** Symmetric, and the same
   magnitude training already used, so the scale of the training curves is
   unchanged.
3. **Early stopping is no longer dataset-adaptive.** `min_delta` used to be
   derived from `w_terminal` and `patience` from the activity count — both
   contradict the identical-hyperparameters claim. Now fixed at `min_delta=0.3`
   and `patience=8`. Patience is set to the largest value the old formula ever
   produced, so no dataset gets less exploration than before.
4. **The training seed is now set and recorded** (`SEED = 0`). Training was
   previously unseeded, so the retrain would not have been reproducible. This
   is not the multiple-seeds work — that comes later — but it makes one run
   repeatable.
5. **`config_used.json` is written next to every model**, holding the reward,
   the terminals, the episode cap, the seed and every PPO hyperparameter.

### Revision: identical numbers are not an identical reward

The first version of decision 1 was argued from the wrong place — that one
shared reward makes the paper's claim true. On merit that is not the point,
and the fix was incomplete.

`w_step` and `w_progress` are charged **per step**, so their total over an
episode grows with the length of the process while the outcome term does not.
Fixed constants therefore mean different things on different logs. Measured
under a random policy, the share of the reward at stake that came from shaping
rather than from the outcome:

| Dataset | Median trace | Old fixed constants | After the change |
|---|---|---|---|
| BPIC2012 | 11 | 2.9% | 2.5% |
| BPIC2017 | 35 | 3.6% | 1.0% |
| **BPIC2015** | 45 | **16.1%** | **9.7%** |

BPIC2015 has 356 activities and ~48-step episodes, so nearly every step earned
the "new activity" bonus. The agent was substantially being paid to wander
rather than to reach a good outcome. And the old constants (0.3 and 0.05) turn
out to be almost exactly what the new rule produces for BPIC2012 (0.2727 and
0.0545) — they had been fitted to that one log's scale and then applied to the
other two.

**The rule now:** each per-step weight is a fixed fraction of the outcome
magnitude, spent over one median-length episode, divided by that dataset's
median trace length.

    w_progress = progress_share (0.10) x w_terminal / median_trace_length
    w_step     = step_share     (0.02) x w_terminal / median_trace_length

The shares are identical everywhere, so this is *more* uniform than fixed
constants, not less — it makes the reward **structure** uniform rather than
just the numbers. The loop penalty stays absolute because it is already a rate
per step and does not accumulate with length.

The residual spread in the table above is not a configuration difference: the
shaping *budget* is now exactly 10.7% of the reward on all three. A random
policy on the loan logs terminates long before the median (5.8 steps against
11, 7.3 against 35) so it never collects its full budget, while BPIC2015 runs
48 steps against a median of 45 and collects nearly all of it. A trained agent
aiming for median-length episodes will land close to the budget on all three.

Also worth recording: **the old per-dataset tuned weights were never really an
option.** They came from a 60-trial random search scored on *random-policy*
rollouts against an environment that still had fabricated transitions and
mislabelled outcomes. Tuning a reward so that a random policy matches real KPIs
says nothing about what reward makes a *learned* policy good.

### Result

`python checks/fix3_reward_parity.py`

| Dataset | Reward identical across notebooks 04/05/06 | Episode cap identical | Good outcome worth | w_progress | w_step |
|---|---|---|---|---|---|
| BPIC2012 | yes | yes (150) | +34.95 | 0.2727 | 0.0545 |
| BPIC2015 | yes | yes (180) | +34.99 | 0.0667 | 0.0133 |
| BPIC2017 | yes | yes (150) | +34.98 | 0.0857 | 0.0171 |

The check now tests two things, and the second is the stronger one:

- the reward config is identical across datasets (no per-dataset tuning);
- **shaping is 10.7% of the reward on every dataset** — so the same config is
  training for the same goal everywhere. Fixed per-step constants pass the
  first test while failing this one.

### On the other three decisions

- **+30 / −30 symmetry** is a neutral default, not a derived result. Whether
  declining a good applicant costs the same as approving a bad one is a
  business question and we have no cost data. Worth one sentence in the
  limitations.
- **Fixed early stopping** (`min_delta=0.3`, `patience=8`) holds up on merit:
  0.3 is 1% of the outcome scale, and that scale is now the same everywhere,
  so the criterion means the same thing on each dataset. More patience is
  never worse, only slower. One unrelated weakness: early stopping watches
  *training* reward, not held-out performance.
- **Setting the seed** is an unambiguous win with no trade-off.

### Side effects and things to know

- Test suite: **64 pass, 1 fail** — the same pre-existing
  `test_round_trip_fidelity` failure. Two tests in
  `test_fix2_episode_length.py` were asserting the hard-coded `w_step == 0.05`
  and now assert the invariant that replaced it: total step cost over a
  median-length episode is a constant share of the outcome, whatever the
  process length. A new test covers the `w_step_abs` override.
- The reward-weight search in `reward_tuning.py` is no longer run by notebook
  04. The module is untouched and still works if called directly.
- **The existing trained models are now invalid** — they were trained against
  fabricated transitions, mislabelled outcomes and a different reward. All
  three must be retrained before any number is quoted.
- Expect the headline numbers to move a long way. The old "+6.59, 0% bad
  outcomes" cannot survive: reaching a good outcome is now genuinely hard
  (11% for a random policy on BPIC2012), so a real result will look far more
  modest and far more believable.

---

## Fix 4 — The agent could finish a case faster than any real case ever has

**Status:** code changed, checks passing. Waiting on your re-run.
**Date:** 2026-08-23

### How this was found

After the first retrain, BPIC2015's numbers looked far better than BPIC2012's,
and that was the clue. Its results were *too* clean:

| Dataset | Reward | Good outcomes | Episode length | Real steps to outcome |
|---|---|---|---|---|
| BPIC2012 | +32.54 ± **14.39** | 94.8% | **9.5** | 9 (median trace 11) |
| BPIC2015 | +30.41 ± **0.94** | **100%** | **2.3** | 48 (median trace 45) |
| BPIC2017 | +30.36 ± 4.97 | 99.3% | **3.4** | 31 (median trace 35) |

Near-zero variance and a perfect outcome rate are not a good policy. They are
the fingerprint of a loophole taken every single time.

### What was wrong

The transition graph permitted routes far shorter than anything real:

```
register submission date request -> create publication document
                                 -> set phase: phase permitting irrevocable
```

Two steps, from the activity that starts 93% of BPIC2015 cases, to "permit
granted, irrevocable". **Both edges exist in the real log.** No real case has
ever composed them — the fastest real route is 11 steps and the median is 48.

| Dataset | Shortest simulated route | Real minimum | Real median |
|---|---|---|---|
| BPIC2012 | 4 | 3 | 9 |
| BPIC2015 | **2** | 11 | 48 |
| BPIC2017 | **3** | 13 | 31 |

BPIC2012 had no exploitable shortcut — its fastest simulated route sits inside
the real range — which is exactly why it was the one behaving sensibly, and
why its variance was healthy rather than suspicious.

This is the same class of failure as Fix 1, one level up, and this time it is
not a coding mistake. Every individual step is valid; the **composed path** is
not. A per-edge mask cannot see it, and neither can any aggregate fidelity
metric. It is a genuine limitation of a first-order Markov simulator.

### What was changed

An episode may not reach an outcome sooner than the fastest real case does.
The floor is the 1st percentile of the real steps-to-outcome distribution, and
below it terminal activities are removed from the routing choices.

| File | Change |
|---|---|
| `src/feature_engineering.py` | New `steps_to_outcome()` (the real distribution) and `OUTCOME_FLOOR_PERCENTILE`. `classify_terminals()` now returns `min_steps_to_outcome`. |
| `src/rl_env.py` | `min_steps_to_outcome` parameter. Terminal successors are masked below the floor, and a terminal reached below it does not end the episode. |
| `src/env_factory.py` | Reads the floor and **refuses to build** an environment from a label file written before the floor existed. |
| `notebooks/02` | Saves the floor and the real distribution into `terminal_classification.json`. |
| `checks/fix4_outcome_floor.py` | **New.** Asserts the floor binds and does not strand episodes. |
| `src/tests/test_fix4_outcome_floor.py` | **New.** Six unit tests. |

The floor is waived at any state whose successors are *all* terminal, so an
episode is never left without a legal move.

### Result

Under a random policy, before and after:

| Dataset | Floor | Shortest episode | Mean length | Still reaches an outcome |
|---|---|---|---|---|
| BPIC2012 | 3 | 2 → **3** | 5.7 → 7.1 | 100% → 100% |
| BPIC2015 | 11 | 2 → **11** | 49.1 → 57.0 | 97% → 97% |
| BPIC2017 | 13 | 2 → **13** | 7.3 → 16.7 | 100% → 100% |

The floor binds on all three and traps nothing. BPIC2012 barely moves, which
is correct — it was never the broken one.

Tests: **70 pass, 1 fail** — the same pre-existing `test_round_trip_fidelity`.

### Why not just raise the length bonus

Making long episodes more rewarding would have hidden the problem rather than
fixed it. The agent's short route was genuinely optimal under the old rules —
a certain +30 in two steps beats a 45-step path carrying a real risk of a −30
outcome. The reward was working; the environment was wrong. Tuning the bonus
until the arithmetic came out differently would also have meant per-dataset
constants again, which Fix 3 had just removed.

### Choice worth knowing about

The floor uses the **1st percentile**, not the observed minimum, because a
single minimum is easily a truncated or malformed trace. The cost is that the
floor sits slightly above the genuine fastest case (BPIC2015 min 7 vs p1 11;
BPIC2017 min 8 vs p1 13), so a small number of real trajectory lengths become
unreachable. Set `OUTCOME_FLOOR_PERCENTILE = 0` to use the minimum instead.

### Still open — not fixed by this

On BPIC2012 the agent reaches a good outcome 94.8% of the time against a
real-log rate of **17.7%**. Its episode length is honest, so this is not the
same exploit, but **nothing in the environment models whether an applicant
qualifies for a loan.** The agent is free to route every case to "approved".
No mask fixes that; it is a modelling limitation that caps what the results
can claim, and it needs a decision before the write-up.

**Resolved by Fix 5 below.**

---

## Fix 5 — The agent could decide whether a case succeeded

**Status:** code changed, checks and tests passing. Waiting on your re-run.
**Date:** 2026-08-23

### What was wrong

"Route to `A_APPROVED`" was an ordinary routing action: always available,
always worth the outcome bonus. So the agent reached a good outcome 94.8% of
the time on BPIC2012 against a real rate of 17.7%. It had not learned anything
about handling cases well — it had learned that the approve node was one click
away.

Look at what the 15 management actions actually are: prioritise, escalate, add
staff, cross-train, rebalance the queue, reroute. **None of them plausibly
changes whether an applicant is creditworthy.** They are efficiency levers.
The framing said "managerial interventions"; the mechanics let the manager do
underwriting.

### Is the information even in the log?

Partly. Case attributes known at the start give a weak signal:

| Dataset | Attributes | Real good rate | AUC predicting the outcome |
|---|---|---|---|
| BPIC2012 | `AMOUNT_REQ` | 17.7% | 0.631 |
| BPIC2017 | `LoanGoal`, `ApplicationType`, `RequestedAmount` | 52.0% | 0.600 |
| BPIC2015 | `SUMleges`, `case_type`, `parts`, `termName`, … | 82.7% | 0.865 |

So the verdict is not determinable from the log — the creditworthiness data
simply is not there. But that was never the real problem. The problem was that
the agent got to choose.

### Why the verdict is drawn once per episode

The obvious fix — let the environment sample the verdict, conditioned on the
current activity — does not work. The agent can park at whichever activity has
the friendliest terminal distribution and wait:

| Dataset | Best steerable P(good \| conclude here) | at |
|---|---|---|
| BPIC2012 | 0.751 | `W_Valideren aanvraag` |
| BPIC2015 | **1.000** | `create publication document` |
| BPIC2017 | 0.940 | `O_Returned` |

So the verdict is drawn **once at reset**, before the agent acts, from the
log's base rate. That is the only version it cannot steer, and it matches the
domain: creditworthiness is a property of the applicant, not of which queue
the case sits in.

### What was changed

| File | Change |
|---|---|
| `src/feature_engineering.py` | New `outcome_base_rates()` — the real success rate and which terminal activity each class ends at. Returned by `classify_terminals()`. |
| `src/rl_env.py` | New `verdict_mode`. In `"environment"` (default) terminals are never offered for routing, the verdict is drawn at `reset()`, and the environment decides *when* a case concludes from the log's own chance of concluding at the current activity. `"agent"` keeps the old behaviour as an ablation. |
| `src/env_factory.py` | Passes the mode and base rates; **refuses to build** without them. |
| `notebooks/02` | Saves the base rates. |
| `checks/fix5_verdict_control.py` | **New.** Policy-invariance test. |
| `src/tests/test_fix5_environment_verdict.py` | **New.** Six unit tests. |

The reward no longer distinguishes good from bad at a conclusion. Scoring the
agent on something it cannot influence is pure noise. What it *can* do is
conclude the case in a plausible number of steps, and that is what earns the
bonus. The good/bad split is still recorded, and becomes a **fidelity check**
— does the simulation reproduce the real base rate? — rather than a
performance claim.

### Result

The test is policy-invariance: run two deliberately different policies and see
whether the outcome rate moves.

| Dataset | Real p(good) | **agent mode** random → fixed | **environment mode** random → fixed |
|---|---|---|---|
| BPIC2012 | 17.7% | 17.0% → **0.0%** | 18.0% → 19.0% |
| BPIC2015 | 82.7% | 49.0% → **0.0%** | 84.9% → 86.7% |
| BPIC2017 | 52.0% | 15.5% → **0.0%** | 50.9% → 52.1% |

In the old mode the outcome is whatever the policy wants. In the new mode it
sits at the real base rate whatever the policy does. That is the fix, measured
rather than asserted.

Episode lengths also land in the right range, and the floor holds:

| Dataset | Floor | Shortest concluded episode | Mean | Abandoned |
|---|---|---|---|---|
| BPIC2012 | 3 | 3 | 19.4 | 0.2% |
| BPIC2015 | 11 | 11 | 50.8 | 3.8% |
| BPIC2017 | 13 | 13 | 45.3 | 1.0% |

Tests: **77 pass, 1 fail** — the same pre-existing `test_round_trip_fidelity`.

### A bug this surfaced in Fix 4

The dead-end waiver let a state whose only successors are terminals conclude
*below* the floor, so BPIC2015 was still recording 5-step outcomes against a
floor of 11. Such a state is a dead end the model cannot continue from, and
recording an outcome there is exactly the implausibly short trajectory the
floor exists to prevent. The episode is now **abandoned without an outcome**
instead. Both check scripts had the matching measurement flaw — they took the
minimum over all episodes including truncated ones, which made a legitimate
short truncation look like a violation. Both now measure concluded episodes
only.

### What this costs the write-up

Outcome rate is no longer a performance metric — by construction the agent
cannot move it. The claims that survive are about **efficiency**: cycle time,
episode length against the real distribution, rework, loop rate, and (once
added) compliance. That is a narrower paper, and a true one.

The good/bad rate becomes a validation number instead: the simulation
reproduces the real base rate, which is worth one line in the fidelity table.

### Confirmed after your re-run — the first honest numbers in this project

| Dataset | Real p(good) | Trained agent | Random policy | Agent episode length | Random | Real median steps |
|---|---|---|---|---|---|---|
| BPIC2012 | 17.7% | **17.7%** | 18.0% | 9.9 | 18.4 | 9 |
| BPIC2015 | 82.7% | **84.3%** | 85.2% | 29.6 | 50.4 | 48 |
| BPIC2017 | 52.0% | **53.7%** | 50.9% | 24.4 | 45.6 | 31 |

The outcome rate sits on the real base rate for every policy including the
trained one — the agent cannot move it, by construction. What it *can* move is
how long a case takes, and it concludes cases roughly **45% faster than a
random policy** on all three datasets while leaving outcomes untouched.

That is the result the paper should be built on. It is modest, it is real, and
nothing about it is an artefact.

Both checks pass:

- `checks/fix4_outcome_floor.py` — no episode reaches an outcome faster than
  the fastest real case (shortest concluded episode equals the floor exactly
  on all three: 3, 11, 13).
- `checks/fix5_verdict_control.py` — the good-outcome rate moves by at most
  1.9% between two deliberately different policies, against 17.0% and 15.5%
  swings in the old agent-controlled mode.

Training reward is now +34.25 / +31.98 / +32.79, all close to the ceiling of
w_terminal (30) plus the length bonus (5), and the training plateau and the
evaluation mean agree — Fix 3 holding up.

---

## Fix 7 — Cycle time was not measured from the agent's episode

**Status:** code changed, check and tests passing. Waiting on your re-run.
**Date:** 2026-08-23

### What was wrong — two separate problems

**1. The number never touched the policy.** For each episode,
`PolicyEvaluator` re-simulated a *fresh* case with `twin.simulate_case()`,
drew its length from the short half of the empirical distribution if the
episode had terminated and the long half if it had not, then multiplied by a
rework-based factor clipped to [0.70, 1.50]. The only things from the actual
episode that reached the result were a boolean — did it end? — and mean
rework. Which activities the policy chose, and how many, were discarded.

Any cycle-time advantage this produced would have been an artefact of the
terminated/truncated split. It would have looked like a result while measuring
almost nothing.

**2. The duration model threw away most of the data.** `_fit_durations` kept
only inter-event gaps longer than 60 seconds, with the comment that this
"matches what the KS metric sees". It does — and that is the problem. The KS
metric filters both the real and simulated sides to > 60s, and
`case_duration_wasserstein` divides each distribution by *its own* 99th
percentile. Both are blind to whatever the fit discards, so the fit and the
metric agreed with each other while disagreeing with the log.

BPIC2012's `A_SUBMITTED` is followed within a second by `A_PARTLYSUBMITTED` in
essentially every case, so its real median gap is **0 seconds**. With
everything under a minute dropped, only the rare long tail survived:

| Activity | Real median gap | Real mean | Sampled median | Sampled mean |
|---|---|---|---|---|
| `A_SUBMITTED` | 0s | 1s | **1,035s** | **84,118s** |
| `W_Completeren aanvraag` | 357s | 60,189s | 1,141s | 82,694s |
| `W_Valideren aanvraag` | 360s | 8,008s | 1,144s | 14,435s |

Summed over a trace, simulated cycle times came out 3–11x too long, and all
seven original fidelity metrics passed comfortably throughout.

### What was changed

| File | Change |
|---|---|
| `src/rl_env.py` | Accumulates `cycle_time_s` from the activities actually visited and reports it in `info`. Durations are fitted as the gap to the following event, so the running total reconstructs first-to-last elapsed time on the real log's scale. |
| `src/digital_twin.py` | `_fit_durations` keeps **every** observed gap, including zeros and sub-minute ones. `_sample_duration` no longer short-circuits "zero-gap" activities to a synthetic ~10s — zeros are sampled naturally now. |
| `src/evaluation.py` | Reads `info["cycle_time_s"]`. The 66 lines of `_simulate_case_duration_s` and `_kpi_multiplier` are **deleted** rather than left dormant — code that silently returns plausible numbers is exactly what caused this. |
| `src/validation.py` | New `cycle_time_scale_error` metric and threshold. |
| `checks/fix7_cycle_time.py` | **New.** Replays real traces through the sampler. |

### The new metric, and why it exists

`cycle_time_scale_error` = `|log(median sim / median real)|`. Every other
duration metric in `validation.py` is scale-invariant by construction, so a
simulator ten times too slow scored identically to a correct one. This one
does not normalise. Threshold 0.40, i.e. a median within roughly [0.67x,
1.49x].

### Result

`python checks/fix7_cycle_time.py` — replaying the **real** activity sequences
through the sampler, so any difference is the duration model rather than the
policy:

| Dataset | Real median | Sim median | Real mean | Sim mean | Scale error |
|---|---|---|---|---|---|
| BPIC2012 | 1.70 d | 1.91 d | 9.63 d | 10.23 d | 0.118 |
| BPIC2015 | 60.75 d | 63.49 d | 93.69 d | 98.41 d | 0.044 |
| BPIC2017 | 18.95 d | 17.14 d | 22.05 d | 21.74 d | 0.100 |

Means now agree within 6%. Before the fix, BPIC2012's replayed median was
8.67 days against a real 1.70 — a 5x error that no existing metric detected.

Tests: **77 pass, 1 fail** — the same pre-existing `test_round_trip_fidelity`.

### What this unblocks

Cycle time can now be reported as a real result, which is what the brief's A5
asked for and what the paper needs, since reward has stopped discriminating
between policies (see the Fix 6 audit below).

---

## Fix 6 — Rebuild the notebook figures

**Status:** implemented. Waiting on your re-run to regenerate the images.
**Date:** 2026-08-23

Fixes 1–7 changed what the numbers mean. This audit asks two questions of each
of the 21 saved panels: **does it show the right thing**, and **does it carry
any signal at all**. Several fail the second test even though nobody changed
them — they were always flat, always redundant, or always unreadable.

Everything below is measured, not eyeballed.

---

### A. Panels that are plain duplicates

**`data_overview.png` (notebook 01) is half redundant.** It has four panels
holding two distinct series:

| Panel | Shows | Verdict |
|---|---|---|
| [0,0] Activity Distribution | horizontal bar of **all** activity counts | same series as [1,0] |
| [1,0] Top 15 Activities | vertical bar of the **top 15 of the same counts** | duplicate |
| [0,1] Events per Case Distribution | histogram | keep |
| [1,1] Events per Case — Percentiles | bar chart of quantiles **of that histogram** | strictly less informative |

On BPIC2015 panel [0,0] is a 356-row horizontal bar chart, which is unreadable
at any print size.

**Fix:** two panels, not four. A top-15 activity bar (drop the all-activities
version, state the total in the caption) and an events-per-case histogram with
the median, p90 and p99 marked on it — which is everything the percentile bar
chart said, in the panel that already exists.

**`training_progress.png` plots the same series twice.** `action_entropy` is
byte-identical to `routing_entropy` on all three datasets (verified with an
equality check on every row). One of the two lines is redundant.

---

### B. Panels that are dead flat

Ranges are min..max across the whole training run.

| Series | BPIC2012 | BPIC2015 | BPIC2017 | Verdict |
|---|---|---|---|---|
| `terminal_rate` | 1.000..1.000 | 0.944..1.000 | 0.988..1.000 | **dead flat** |
| `truncated_rate` | 0.000..0.000 | 0.000..0.056 | 0.000..0.012 | **dead flat** |
| `ep_reward_mean` | 33.35..34.25 | 26.37..32.30 | 31.24..32.79 | **nearly flat** |
| `ep_len_mean` | 9.3..17.2 | 29.6..55.9 | 24.4..46.1 | real signal |
| `good_terminal_rate` | 0.157..0.201 | 0.749..0.864 | 0.506..0.573 | flat *by design* |
| `delay/rework/risk_mean` | wide | wide | wide | real signal |
| `value_loss`, `explained_variance` | wide | wide | wide | real signal |

**The headline "Episode Reward" curve is now the misleading one.** It moves
about one point on a scale of 34, because concluding a case is worth +30 and
almost every episode concludes. All the learning lives in the length bonus
(max 5) and the shaping terms, and the y-axis hides it. A reader sees a flat
line and concludes the agent learned nothing, which is the opposite of what
happened.

---

### C. Reward has stopped separating policies; length has not

From `evaluation_full.csv`, how far RL sits ahead of the best baseline:

| Dataset | On reward | On episode length |
|---|---|---|
| BPIC2012 | **+1.16** | 9.49 vs 15.44 steps |
| BPIC2015 | **+0.86** | 18.17 vs 23.45 steps |
| BPIC2017 | **+2.60** | 22.56 vs 48.18 steps |

On BPIC2017 the RL policy finishes in **less than half** the steps of the best
baseline, and reward records that as +2.60 on a scale of 33. Every
reward-based figure understates the result; every length-based one shows it.

**Fix:** demote reward to a supporting panel, lead with length and cycle time.

---

### D. Panels that are wrong, not just weak

**The routing heatmap computes the wrong quantity** (notebook 06, cell 17):

```python
ratio_mat[i, j] = np.log2(max(row['log2_ratio'], 0.01))
```

`log2_ratio` is already a log2. This takes log2 of a log2. Worse, the
`max(..., 0.01)` clamps **every negative value** — that is, every route the
agent avoids — to the same number, −6.64:

| Dataset | Rows collapsed to one colour |
|---|---|
| BPIC2012 | 47 of 63 (**74.6%**) |
| BPIC2017 | 53 of 106 (**50.0%**) |
| BPIC2015 | 1,401 of 7,277 (19.3%) |

So on BPIC2012 three quarters of the plotted cells are the same shade, the
entire "RL avoids" half of the legend is meaningless, and the axis label
"log2 scale: +1 = 2x more likely" describes a quantity that is not being
plotted. **Fix:** plot `log2_ratio` directly, with a diverging scale centred
on zero.

**The heatmap is also mostly empty**, and empty means the wrong thing:

| Dataset | Cells | Non-empty |
|---|---|---|
| BPIC2012 | 24×24 = 576 | 63 (**10.9%**) |
| BPIC2017 | 26×26 = 676 | 106 (**15.7%**) |

A transition neither policy ever takes renders as 0 — visually identical to
one both policies take equally often. **Fix:** mask never-visited cells to a
distinct neutral colour, or drop the heatmap for the divergence bar chart,
which shows the same information without the 85–89% of blank cells. Note the
heatmap is already skipped entirely on BPIC2015 (356 activities), so it
appears for only two of the three datasets.

**`eval_relative_improvement.png` (notebook 05) has three panels and all three
are now invalid:**

- *Δ reward vs best heuristic %* — 3.49 / 2.94 / 8.62%, computed on a reward
  dominated by a constant. This is the brief's A9 percentage problem attached
  to a quantity that no longer varies.
- *Δ bad terminal rate vs best heuristic* — the outcome rate is now set by the
  environment, so this measures nothing about the policy.
- *Δ good terminal rate vs best heuristic* — same.

**Fix:** delete the figure. Replace with absolute differences in episode length
and cycle time against the real-log reference.

**"Bad Terminal Rate (lower = better)"** (notebook 05, `eval_outcome_metrics.png`
row 0) — same problem, plus the subtitle actively misleads: no policy can lower
it. **Fix:** rebuild as a fidelity panel showing every policy against the real
base rate, captioned as validation rather than performance.

**"Final Activity Distribution"** (notebook 06, `episode_outcomes.png` panel 3)
— the final activity is drawn by the environment, so RL and Random are
identical by construction. **Fix:** replace with steps-to-outcome, RL vs Random
vs the real log.

**"Trace-Ending Activity Rates"** (notebook 02) — colours the top-15
trace-enders green/red by good/bad, but the new outcome markers end ~0% of
traces, so nearly every bar renders grey and the figure visually contradicts
the labelling. **Fix:** plot the evidence that actually decides the labels —
`rel_pos_p25` against `case_coverage` for every activity, qualifying region
shaded, good/bad/excluded coloured.

**`[GOOD TERMINAL]` annotation** in `print_decision_rules` — dead branch. The
agent cannot route to a terminal, so no rule ends at one (verified: 0 of 14,
25 and 11 rules).

---

### E. Panels that are readable-in-principle but not in practice

**Skewed histograms with no log axis** (notebooks 01 and 02). Share of cases
falling in the first of 30 bins:

| Dataset | Case age | Trace length |
|---|---|---|
| BPIC2012 | **57.1%** | **45.3%** |
| BPIC2015 | 34.7% | 1.5% |
| BPIC2017 | 16.9% | 0.3% |

BPIC2012's throughput histogram puts 57% of cases in one bar with a tail
running to 170× the median. **Fix:** log-scale x-axis, or an ECDF, for the
duration panels.

**The validation figure** (notebook 03) plots metric value against threshold on
a linear axis, but the values span **5,224×** on BPIC2012 (0.0002 to 1.0000)
and 3,897× on BPIC2017. Five of the seven bars render as invisible slivers.
And because `overall_pass` is True everywhere, the second panel — a pass/fail
scorecard — is seven green ticks carrying no information, duplicating the
colour coding already in the first panel. **Fix:** one panel, plotting
*value ÷ threshold* on a log axis, so every metric is comparable and the pass
line sits at 1.0.

---

### F. Keep unchanged

Management action usage (04, 05 and 06 — core to the paper's claim); routing
divergences bar chart (06); episode reward *distribution* violins (05), though
they need a log or clipped axis because BPIC2015 spans −137 to +30; the
episode-length panel of `eval_outcome_metrics.png` (05), which should be
promoted to lead; `delay/rework/risk`, entropy, value-loss and
explained-variance panels (04).

**`eval_outcome_breakdown.png` (05) should be kept but reframed.** Its stacked
good/bad/truncated bars now carry one strong signal and one dead one: the
good/bad split is fixed by the environment, but the **truncated** fraction is a
genuine and large policy difference — 94.6% for FIFO on BPIC2015 and 93.0% for
Empirical Markov, against 0.6% for RL. Retitle it around conclusion rate and
de-emphasise the good/bad split.

---

### Summary of the work

| Action | Count |
|---|---|
| Delete outright | 4 panels (`eval_relative_improvement` ×3, completion rate) |
| Fix a computation bug | 1 (heatmap double-log and clamp) |
| Replace with a different chart | 5 |
| Merge or de-duplicate | 3 |
| Rescale axis only | 3 |
| Keep as-is | 9 |
| **New charts to add** | **4** |

New charts, in priority order:

1. **Episode length vs the real steps-to-outcome distribution** — RL, every
   baseline, and the real log. The headline result, and no chart shows it today.
2. **Cycle time**, same shape. Unblocked by Fix 7; it was meaningless before.
3. **Outcome-rate invariance** — every policy against the real base rate. One
   figure turning the project's biggest weakness into a demonstrated property.
4. **Reward decomposition** — conclusion bonus against shaping, which explains
   *why* the total reward is flat instead of leaving a reader to misread it.

### Ordering note

Nothing here should be built until the current run finishes, because the Fix 7
duration change moves every cycle-time number and the notebooks are being
re-run now.

### Implemented

A new `src/figures.py` holds every saved figure. The six notebooks each
hand-rolled their own matplotlib, which is how the same faults ended up
repeated across them; now each plotting cell is a few lines calling one
tested function.

| File | Change |
|---|---|
| `src/figures.py` | **New.** All rebuilt figures, one consistent palette. |
| `notebooks/01` | 4 panels → 2. Top-15 activities (total stated in the caption) and a log-scaled case-length histogram with p50/p90/p99 marked on it. |
| `notebooks/02` | Trace-ending bar chart → outcome-marker evidence: trace position against case coverage, qualifying region shaded, good/bad/excluded coloured and labelled. |
| `notebooks/03` | 2 panels → 1. Value ÷ threshold on a log axis, pass line at 1.0. |
| `notebooks/04` | Rebuilt: episode length promoted to lead, reward decomposed on a twin axis, outcome rate with a fixed window and a ±5pp band, completion-rate panel deleted, duplicate entropy series deleted. |
| `notebooks/05` | `run_policy` now records `cycle_time_s`. `eval_outcome_metrics` → `eval_policy_comparison` (length, cycle time, conclusion rate, outcome validation). `eval_relative_improvement` **deleted**. Outcome breakdown retitled around conclusion rate. |
| `notebooks/06` | Heatmap bug fixed and unvisited cells masked. "Final Activity Distribution" → case length vs the real log, plus an outcome-mix validation bar. |
| `src/insights.py` | The `[GOOD TERMINAL]` branch documented as reachable only in the `agent` ablation. |

Three judgement calls worth recording:

- **The reward decomposition needed a second axis.** On BPIC2012 the shaped
  component runs 3.3..4.2 while the total runs 33.4..34.2; on a shared axis the
  only part that moves is an invisible wiggle. It now has its own scale, so the
  figure shows *both* that the total is flat and why.
- **The outcome-rate panel is on a fixed window with a ±5pp band.** Left to
  auto-scale it rendered ordinary sampling noise (15.6–20.1%) as a dramatic
  zig-zag on a metric that is supposed to be constant — inviting exactly the
  misreading the panel exists to prevent.
- **The heatmap colour scale clips at the 90th percentile of |value|.** The
  distribution is lopsided (BPIC2012 runs to −8.8 but only +3.0), so scaling to
  the extremes washed every mid-range cell out to near-white.

Verified by rendering all figures for all three datasets from the committed
artefacts. The heatmap is still skipped on BPIC2015, where 356 activities make
a 356×356 matrix pointless; the divergence bar chart covers that case.

Tests: **77 pass, 1 fail** — the same pre-existing `test_round_trip_fidelity`.

---

## Fix 7 addendum — the new metric immediately caught something

`cycle_time_scale_error` **fails on BPIC2012** (0.712 against a threshold of
0.40) while all seven original metrics pass with 3–300x headroom. This is the
metric doing its job on its first run, so the threshold stays as it is —
loosening it to get a green tick would repeat the exact mistake that caused the
bug it was written to catch.

The cause is not the duration model. Replaying real traces through the sampler
reproduces the real scale to within 12% (Fix 7 above), and simulated trace
lengths and per-activity time shares both match the log closely (57.9% → 54.2%
for the dominant activity). The failure is that durations are sampled
**independently**, which destroys the within-case correlation. In reality a
fast case has short gaps throughout; sampling each gap independently, a
10-step trace almost always draws at least one long one, so the simulated
*median* rises toward the mean while the mean stays correct.

The size of the error tracks each log's skew exactly:

| Dataset | Real mean ÷ median | Scale error |
|---|---|---|
| BPIC2012 | **10.7x** | **0.712** |
| BPIC2015 | 1.5x | 0.141 |
| BPIC2017 | 1.1x | 0.121 |

So it is a known and explainable limitation of a first-order model with
independent durations, of the same family as the trajectory-validity problem
in Fix 4: each piece is right, the composition is not. It belongs in the
write-up as a limitation with this table attached, not as something to tune
away. Fixing it properly would mean modelling gap correlation within a case,
which is a larger piece of work than anything done so far.

---

## Fix 8 — The one failing test was hiding a broken test, not broken code

**Status:** ✅ Done. **78 pass, 0 fail.**
**Date:** 2026-08-23

### What it looked like

`test_integration.py::test_round_trip_fidelity` failed from the very first run
of this session, and I repeatedly described it as "a pandas 3 text-column type
comparison inside the test itself, not a problem with the pipeline". That was
based on the assertion message and was **not verified**. It was half right and
half wrong.

### What was actually wrong

The test means to check that events survive the trip: synthetic events →
stream pipeline → buffer → Parquet flush → re-read. That is exactly the class
of bug Fix 1 turned out to be, so it earns its place.

But its setup compared two different simulations:

```python
gen.start(n_cases=30)             # _seed is None -> unseeded parallel path
original_events = list(gen._events)
...
gen.reset(seed=42)                # seeded single-threaded path
gen.start(n_cases=30)
gen.replay_into(pipeline, ...)    # replays the *seeded* events
```

`SyntheticStreamGenerator.__init__` leaves `_seed = None`, and `_generate`
branches on it — unseeded uses the parallel `simulate()`, seeded uses the
single-threaded path with `twin.rng` reseeded. So the captured events and the
replayed events came from two unrelated runs.

Measured: **87 of 90 timestamps differed**, by between 2.4 and 33 hours.
`case_id` and `activity` matched only because `make_twin()` builds a
deterministic A → B → C chain with sequential case ids, so those columns look
identical whatever the seed. A dtype mismatch was raised before the values
were ever compared, which is why the real problem stayed hidden.

So the test could never have passed on timestamps, and the "just a dtype
thing" reading was wrong.

### What was changed

- Generate **once**, deterministically, and compare against exactly those
  events: `start()`, then `reset(seed=42)`, then capture, then replay.
- `check_dtype=False` on the frame comparison, with a comment. A Parquet round
  trip legitimately changes the storage dtype of text columns under pandas 3
  (`str` out, `object` back); the values are still compared exactly. This part
  of the original diagnosis was right.

### Verified it still has teeth

A vacuous test that passes is worse than a failing one. Mutation check: patch
`StreamIngestionPipeline.push_batch` to shift every ingested timestamp by one
second, then re-run.

```
baseline: PASS
mutated:  FAIL  <-- the test detects a 1-second corruption
```

So it is genuinely checking round-trip fidelity to the microsecond.

### Lesson worth keeping

I carried "this failure is insignificant" through six fixes on the strength of
an error message. It took two minutes to check and the check changed the
answer. A test that has always failed is not evidence that it does not matter.

---

## Findings 9-12 — from reading the regenerated BPIC2015 and BPIC2017 insights

**Status:** diagnosed and measured, not yet fixed.
**Date:** 2026-08-23

### 9. The 15 management actions do not affect anything the agent is scored on

Holding routing decisions identical and swapping the entire management policy:

| Management policy | BPIC2015 reward | length | good rate | BPIC2012 reward | length |
|---|---|---|---|---|---|
| always no-op | 28.590 | 56.29 | 83.3% | 33.191 | 19.96 |
| random valid | 28.585 | 56.29 | 83.3% | 33.189 | 19.96 |
| always `skip_optional_subprocess` | 28.590 | 56.29 | 83.3% | 33.191 | 19.96 |
| always last valid | 28.585 | 56.29 | 83.3% | 33.190 | 19.96 |

Reward moves by 0.02%; episode length and outcome rate do not move at all.

**Why.** `_compute_reward` reads only `self._trace`, `self._step` and
`self._median_len` — never `_episode_state`. And `_build_kpi_vec` recomputes
delay and rework *from the trace* via `self._delay_norm()` /
`self._rework_norm()`, not from the `_episode_state` values the actions write.
So `skip_optional_subprocess` doing `state["_delay_norm"] -= 0.2` writes to a
dict nothing downstream reads. Only `_volume_pressure` reaches the
observation, which accounts for the 0.02% residual.

**The trained policies corroborate it, in the strongest possible way — they
disagree completely.** Same reward, same seed, same algorithm:

| | `assign_to_primary_team` (the no-op) |
|---|---|
| BPIC2015 | **0.05%** of steps |
| BPIC2017 | **91.9%** of steps |

One policy almost never idles, the other almost always does. If the catalogue
carried signal you would expect consistent logic; opposite extremes are what
arbitrary drift looks like.

The apparent BPIC2015 preference (`skip_optional_subprocess` 56%,
`rebalance_overloaded_queue` 33%) is explained by validity coverage, not
learning: the first is valid whenever risk is low, the second whenever delay or
rework is high. They are near-complementary and together cover nearly every
state, so "pick those two" is indistinguishable from "pick whatever is legal".

**This matters because the 15-action catalogue is the paper's central novelty
claim.** The A3 ablation would return zero by construction.

### 10. On BPIC2017 the agent camps on the highest-conclusion-probability activity

| Self-loop share of transitions | RL | Random | Real log |
|---|---|---|---|
| BPIC2017 | **68.6%** | 11.2% | 38.3% |
| BPIC2015 | 0.0% | 0.7% | 0.8% |

BPIC2017's loop sits on `W_Validate application`, 7,759 recorded steps — and
that activity has `P(case concludes here) = 0.092`, **the highest of any
activity in the process**.

Since the environment now decides *when* a case concludes from the current
activity, the optimal policy is to stand on the activity most likely to end the
case and wait. That is exactly the Fix 4 exploit in a new shape: previously a
2-step dash to a terminal, now a self-loop on the best waiting room. The
outcome floor does not catch it because the episode is long, not short.

The loop penalty is too weak to prevent it: `excess = rework/step -
baseline_loop_rate`, and the baseline is derived from the log's own mean
rework, which on BPIC2017 is high enough that the penalty rarely fires.

**The visible consequence** is that the top "process improvement
recommendation" the system emits is:

> PREFER `W_Validate application` → `W_Validate application`, RL 99.7% vs Random 11.3%

i.e. "keep repeating the validation step". That is not advice anyone can act on.

### 11. The interpretability layer has almost no KPI diversity

BPIC2017's extracted decision rules: **11 rules, 2 distinct KPI condition
combinations between them.** Nearly every rule carries the same
delay/rework/case-age/volume bucket, so the "IF at X AND low delay AND low
rework AND early in case THEN route to Y" framing is a template, not a learned
conditional. The report's claim that the rules capture "interpretable,
KPI-conditioned managerial logic" is not supported by its own output.

### 12. `collect_trajectories` mis-records the concluding step of every episode

`insights.collect_trajectories` derives the destination from the agent's
routing choice:

```python
to_act = successors[routing_idx] if routing_idx < len(successors) else "UNKNOWN"
```

In `verdict_mode="environment"` the agent never routes to a terminal — the
environment draws one. So on the step where the case concludes, the recorded
destination is the agent's non-terminal pick rather than what actually
happened. Measured: **146 of 150 episodes on BPIC2015 and 149 of 150 on
BPIC2017** (the remainder are truncations, which have no concluding step),
about 2% of all recorded steps.

Everything in notebook 06 inherits it: routing preferences, the heatmap, the
divergence chart, decision rules, recommendations. It is also the source of the
lone `UNKNOWN` row that shows up as an "impossible" transition on BPIC2015.

**Fix:** record `env._current_activity` after `step()` rather than
reconstructing the destination from the action.

### Suggested order

12 first (small, and it contaminates the evidence for everything else), then 10
(a genuine environment-validity problem of the same family as Fix 4), then 9
(the largest, since it decides what the paper can claim), then 11 (partly falls
out of 9).

### Corrections and additions after BPIC2012 regenerated

**Finding 9 stands, but my framing of it was too neat.** It is not a clean
split — two datasets converge on the same action, one on the no-op:

| Dataset | Top action | Its share | No-op share |
|---|---|---|---|
| BPIC2012 | `skip_optional_subprocess` | 77.9% | 0.8% |
| BPIC2015 | `skip_optional_subprocess` | 56.1% | 0.1% |
| BPIC2017 | `assign_to_primary_team` (no-op) | 91.9% | **91.9%** |

A 100x spread in no-op rate across provably inert actions still says the
choice carries no signal, but "opposite extremes" overstated a 2-vs-1 split.

**Finding 10 was right about BPIC2017 and wrong as a general claim. The real
finding is larger.** BPIC2012 fails in the *opposite* direction:

| Dataset | Self-loops: agent | Random | Real log | Real share of steps to a *new* activity |
|---|---|---|---|---|
| BPIC2012 | **0.0%** | 10.2% | **42.5%** | 37.7% |
| BPIC2015 | 0.0% | 0.7% | 0.8% | 92.4% |
| BPIC2017 | **68.6%** | 11.2% | 38.3% | 39.1% |

So the agent's repetition rate is uncalibrated in **both** directions, and
BPIC2015 only looks right by coincidence — its real process genuinely almost
never repeats (92.4% of steps go somewhere new).

The cause is not the loop penalty. BPIC2012's `baseline_loop_rate` is 1.078
against BPIC2017's 0.646, so the penalty bites *less* on BPIC2012 — the
opposite of what the numbers would need. It is the **progress bonus**:
`+w_progress` is paid only for an activity not yet in the trace, so a repeat is
strictly dominated whenever any new activity is available. On BPIC2012, with 24
activities and ~10-step episodes, one always is — so repeats go to zero against
a real rate of 62%. On BPIC2017 the new activities run out and the +30
conclusion bonus takes over, making it optimal to camp on
`W_Validate application`, the activity with the highest `P(conclude) = 0.092`
in the process.

Two reward terms, pulling in opposite directions, neither calibrated to how
often the real process repeats.

**Finding 11 does not generalise.** Distinct KPI condition combinations among
the extracted rules:

| Dataset | Rules | Distinct conditions |
|---|---|---|
| BPIC2012 | 14 | 7 |
| BPIC2015 | 25 | 11 |
| BPIC2017 | 11 | **2** |

Only BPIC2017 is degenerate, and that follows from finding 10 — an agent
sitting in one self-loop is in one KPI state, so it can only produce one
condition bucket. Downgrade this from a finding to a symptom.

### 13. Cycle times are far below anything in the log

| Dataset | Real median | RL median | Ratio |
|---|---|---|---|
| BPIC2012 | 0.81 d | 0.10 d | **0.12x** |
| BPIC2015 | 68.44 d | 14.60 d | 0.21x |
| BPIC2017 | 19.09 d | 8.80 d | 0.46x |

On BPIC2012 the agent finishes cases in an eighth of the median real time,
using 9.5 steps against a real 11 — so this is not merely "fewer steps", it is
routing through short-duration activities.

Worth being precise about what this is and is not. **Cycle time is not in the
reward**, so the agent did not optimise it; the speed-up is incidental to
chasing the conclusion and length bonuses. But a policy whose cases complete
8x faster than any real case is not describing an achievable improvement, it is
another instance of the recurring pattern: each edge is plausible, the composed
trajectory is not. Reporting this as "the agent reduces cycle time" would not
survive review.

### 14. Two gaps in my own Fix 6 work

- `run_policy` collects `median_cycle_days`, but the summary-table builder in
  notebook 05 was never updated, so **`evaluation_full.csv` has no cycle-time
  column**. The figure works (it reads the in-memory dict); the CSV does not.
- The cycle-time panel's **real-log reference line never renders**: I look up
  `real_cycle_days_median` in `terminal_classification.json`, and notebook 02
  does not write that key. Row 2 of `eval_policy_comparison.png` has no red
  line as a result.
- `rel_reward_vs_random_pct` and `rel_reward_vs_best_heur_pct` are still
  columns in `evaluation_full.csv`. I deleted the figure that plotted them but
  not the columns that feed it.

---

## Fix 13 — Insight-layer recording, and three gaps in Fix 6

**Status:** implemented, verified. Waiting on a re-run of notebooks 2, 5 and 6.
**Date:** 2026-08-23

### 1. `collect_trajectories` recorded the agent's intention, not the outcome

It derived the destination from the routing action:

```python
to_act = successors[routing_idx] if routing_idx < len(successors) else "UNKNOWN"
```

Under `verdict_mode="environment"` the agent never routes to a terminal — the
environment draws one when the case concludes. So on the concluding step the
recorded destination was the agent's non-terminal pick.

Measured after the fix: on **every** concluding step (114 of 114 on BPIC2015,
119 of 119 on BPIC2017) the agent had routed somewhere other than where the
case actually went. The old code was wrong 100% of the time on that step, about
2% of all recorded steps, and the `"UNKNOWN"` sentinel it produced was the
stray "impossible transition" on BPIC2015.

Now read from `env._current_activity` after `step()`. Two new fields are
recorded alongside: `agent_routed_to` (what the policy chose) and
`env_concluded` (whether this step was the environment ending the case), so the
two are never conflated again.

### 2. Concluding steps are excluded from the agent-behaviour analyses

A consequence of fixing (1): terminals now appear as destinations, and both
`routing_preference_table` and `extract_decision_rules` would have started
emitting rows like

> PREFER `W_Valideren aanvraag` → `A_APPROVED`

which reads as the agent choosing to approve the application. It does not — the
verdict is drawn at reset and is policy-invariant by construction. Both
functions now skip steps flagged `env_concluded`, because both are statements
about what the agent *decides*.

### 3. Cycle time never reached the results table

`run_policy` collected `median_cycle_days`, but the table builder in notebook
05 was never updated, so `evaluation_full.csv` had no cycle-time column while
the figure beside it plotted one. Added, along with `mean_cycle_days` and the
real-log values for comparison.

Also added `good_pct_of_concluded`. The existing `good_term_pct` is a share of
*all* episodes, so a policy that rarely finishes looks like one that rarely
reaches a good outcome — FIFO shows 3.0% on BPIC2015 purely because it
truncates 95.8% of the time.

Removed `rel_reward_vs_random_pct` and `rel_reward_vs_best_heur_pct`. Fix 6
deleted the figure that plotted them but left the columns feeding it.

### 4. The cycle-time reference line never rendered

`figures.policy_comparison` looks up `real_cycle_days_median`, and notebook 02
never wrote it, so row 2 of `eval_policy_comparison.png` had bars with nothing
to compare them against. Notebook 02 now computes and saves the real median and
mean case cycle time (68.44 d for BPIC2015, 19.09 d for BPIC2017, 0.81 d for
BPIC2012).

### Verified

- `UNKNOWN` destinations: **0** (was 1 on BPIC2015).
- Every concluding step now records a real terminal.
- Notebooks 02 and 05 parse cleanly.
- Test suite: **78 pass, 0 fail.**

Re-run notebooks 2, 5 and 6 to regenerate. Notebook 2 must go first — it writes
the cycle-time reference the others read.

### Fix 13 addendum — the last impossible transition

The rerun left exactly one on BPIC2015:
`environmental permit decision suspended -> create publication document`,
1 occurrence in ~7,000 steps.

That activity has **no outgoing edges at all**. When it was reached below the
outcome floor, the code fell through to `twin._sample_next_activity`, whose
global fallback distribution teleports the case to a frequently-seen activity
that is not a successor. This is A1.2 in the remediation brief — noted during
Fix 1 as a second-order leak and never closed.

Now: a dead end below the floor ends the episode where it stands rather than
inventing a move. The step is flagged `env_no_move` and both
`routing_preference_table` and `extract_decision_rules` skip it, alongside the
`env_concluded` steps.

Test suite: 78 pass, 0 fail. Needs notebook 6 re-run to show 0 impossible.

### Confirmed after the rerun

`evaluation_full.csv` now carries cycle time and its real-log reference, and
`good_pct_of_concluded` — which is flat at the base rate for every policy,
confirming verdict invariance holds at evaluation as well as in training:

| Dataset | Real | Random | FIFO | Greedy | EmpMk | RwdG | RL |
|---|---|---|---|---|---|---|---|
| BPIC2012 | 17.7% | 14.4 | 18.4 | 18.0 | 18.0 | 15.5 | 17.6 |
| BPIC2015 | 82.7% | 82.1 | 71.4 | 80.3 | 80.6 | 85.5 | 80.0 |
| BPIC2017 | 52.0% | 52.4 | 54.8 | 55.0 | — | 50.0 | 50.2 |

`UNKNOWN` destinations: 0 on all three.

---

## Fix 14 — Finding 10, plus a masking bug found while measuring it

**Status:** implemented, unit-verified. Needs a retrain to confirm the effect.
**Date:** 2026-08-23

### Finding 10: the progress bonus is gone (option a)

`progress_share` is now 0. The bonus paid only for reaching an activity not
yet in the trace, so repeating was strictly worse than any alternative
regardless of what the process actually does. The per-step cost and the length
bonus already push toward finishing efficiently, and neither cares whether a
step is novel.

Unit check — reward for a repeat vs a fresh activity, everything else equal:

| Dataset | Config | repeat | new | gap |
|---|---|---|---|---|
| BPIC2012 | old (0.10) | −0.0545 | +0.2182 | **+0.2727** |
| BPIC2012 | new (0.0) | −0.0545 | −0.0545 | 0.0000 |
| BPIC2017 | old (0.10) | −0.0171 | +0.0686 | **+0.0857** |
| BPIC2017 | new (0.0) | −0.0171 | −0.0171 | 0.0000 |

The environment is now indifferent between repeating and advancing, so
whatever repetition rate the agent settles on is driven by the process
structure rather than by a term that had no business encoding it.

**This cannot be verified without retraining** — the 0% / 68.6% self-loop rates
are learned behaviour, and a random policy does not respond to reward. The
target is the log's own rate: 42.5% on BPIC2012, 0.8% on BPIC2015, 38.3% on
BPIC2017. `progress_share=0.10` reproduces the old behaviour as an ablation.

### The invalid-action penalty does fire — A7.1 in the brief is wrong

Found while measuring which management actions change anything.
`_build_kpi_vec()` advanced the workload random walk on **every call**, and it
is called three times per step: from `action_masks()`, from `_get_obs()` and
from `step()`. Two consequences:

1. The walk moved three times faster than intended.
2. The action mask was computed from a different KPI vector than the validity
   check applied moments later inside `apply_management_action`. Near a
   threshold the two disagreed, and the agent was charged the −0.1
   invalid-action penalty for choosing an action the mask had just declared
   legal.

| Dataset | Steps | Mask disagrees with validity | Penalty charged |
|---|---|---|---|
| BPIC2012 | 3,964 | 9 (0.23%) | 9 |
| BPIC2015 | 10,128 | 8 (0.08%) | 8 |
| BPIC2017 | 9,131 | 15 (0.16%) | 15 |

Every disagreement produced the penalty. The remediation brief's A7.1 says the
−0.1 term "can never fire, because MaskablePPO sets invalid logits to −∞" and
recommends deleting it from the formalism. That is wrong: masking is only as
good as the state it was computed from. The term fires, rarely, and it charges
the agent for something it could not have avoided.

**Fix:** the random walk is now taken once per step, in `step()`, before
anything reads it. `_build_kpi_vec` is pure. Re-measured: **0 disagreements on
all three datasets.**

Test suite: 78 pass, 0 fail.

### Fix 14 addendum — the assertion fired, correctly

Notebook 04 failed on BPIC2015 with:

```
AssertionError: Environment reward does not match the shared config — rebuild the env.
```

That guard was added in Fix 3 to stop the agent ever being trained under a
reward other than the intended one, and it did its job. The cause was an
ordering problem it exposed rather than a fault in the guard.

`build_process_env` prefers `reward_config.json` on disk, deliberately: it is
what keeps notebooks 05 and 06 grading with exactly what training used. But
notebook 04 wrote that file in the reward cell, *after* the environment cell
had already built the env from whatever was there before. So a config left
behind by an earlier run always won, and changing the default in code could
never take effect — the assertion caught the first time that mattered.

**Fix:** notebook 04 now writes the reward config in the environment cell,
before `build_process_env` is called. Verified by restoring the stale
`progress_share=0.1` file and executing cells 7 and 9 verbatim for all three
datasets: all pass, env resolves to `progress_share=0.0`, `w_progress=0.0000`.

**The on-disk configs were deliberately left stale** (BPIC2012 and BPIC2017
still read 0.1). Refreshing them by hand would let a `--notebooks 5 6` run
evaluate the *existing* models — trained at 0.1 — under a reward of 0.0, which
is precisely the training/evaluation split Fix 3 removed. Notebook 04 rewrites
each one as part of retraining, keeping model and config in step.

---

## Fix 15 — Connecting the managerial catalogue

**Status:** implemented. Needs a retrain to see whether the agent uses it.
**Date:** 2026-08-23

### Finding 10, option (a): removing the progress bonus did not work

Reported first because it is a negative result and it shaped what came next.

| Dataset | Before | After removing `w_progress` | Real log |
|---|---|---|---|
| BPIC2012 | 0.0% | **0.0%** | 42.5% |
| BPIC2015 | 0.0% | 3.8% | 0.8% |
| BPIC2017 | 68.6% | **81.0%** | 38.3% |

BPIC2017 got worse. Removing the progress bonus took away the only thing
counteracting the camping behaviour, so the agent camped harder. That confirms
the driver is the **conclusion bonus**: the agent positions itself where
`P(conclude)` is highest and waits. `progress_share` stays at 0 — it was
distorting repetition for no good reason — but repetition calibration is still
open and needs a different lever.

### What the logs can and cannot support

Two things were checked before any mechanism was designed, because assigning
effect sizes without checking is how the "unjustified constants" problem
starts.

**Does congestion slow work down?** If busier resources were slower, staffing
and rebalancing would have a measurable basis. Spearman correlation between a
resource's daily workload and its step durations, within activity:

| Dataset | Median rho | Significant at p<0.01 |
|---|---|---|
| BPIC2012 | −0.014 | 5 of 23 (2 positive, 3 negative) |
| BPIC2017 | −0.053 | 22 of 24 (9 positive, 13 negative) |
| BPIC2015 | −0.082 | 53 of 90 (6 positive, **47 negative**) |

Mostly *negative* — busier resources are faster. Duration in these logs is
dominated by waiting on applicants, objections and external parties, not by
staff availability. **There is no congestion effect to calibrate against.**

**Are the interventions recorded at all?** Only on BPIC2015, and only 4 of the
15 concepts (defer, escalate, skip, close). BPIC2012 and BPIC2017 record
**none**. So an intervention's effect cannot be estimated from what followed
it, because it was never logged as happening.

**Conclusion: the effect sizes have to be assumptions.** That is normal for
simulation-based BPM — RIMS and Simod assume capacity effects too — but it has
to be declared, and results have to be reported across a sweep rather than at
one arbitrary point.

### The mechanism

`src/intervention_effects.py` gives every action three declared properties and
a written rationale for each: a **duration multiplier** (what it does to the
case), a **cost**, and a **compliance risk** for the two actions that waive
work. `EFFECT_SCALE` scales all of them together for sensitivity analysis, and
`0.0` disables the mechanism entirely — which is the management ablation, for
free and exactly reproducible.

For any of it to register, cycle time had to enter the reward. It now does:

    reward -= w_time * max(0, log2(elapsed / real_median_cycle))
    reward -= w_intervention * (accumulated cost + compliance risk)

Both charges are shares of `w_terminal`, and the intervention charge is divided
by median trace length exactly as `w_step` is — a per-step charge that is not
normalised costs eight times more on a 45-step process than an 11-step one,
which is the scale bug from Fix 3.

**Two design choices worth recording:**

*The time charge is on log2 of the ratio, not the ratio.* A linear charge let
one slow case dominate the return. A hard cap was tried first and was worse: a
random policy already sits at 3.6x the real median on BPIC2012, so with a cap
at 3.0 every improvement inside that range earned nothing, and **no policy
could beat doing nothing.** log2 always has gradient and never explodes.

*Floored at zero.* Finishing faster than the real median earns nothing extra.
The agent already produces cycle times well below anything in the log (finding
13: 0.12x the real median on BPIC2012), and paying it to go further would
reward implausibility.

### Result

| | Before | After |
|---|---|---|
| Reward spread across management policies | 0.02% | policies differ by up to 60% of cycle time |
| Cycle time response | none | e.g. BPIC2015 110.7 → 103.2 days from an intervention firing on 4.5% of steps |
| Ablation (`effect_scale=0`) | n/a | every policy byte-identical — a clean on/off experiment |
| Can good management beat none? | no (all identical) | yes — always-prioritise beats no-op on all three |

### Honest limits

- **The effect sizes are assumptions.** Every number in `intervention_effects.py`
  carries a rationale, and none of them is measured. Any result must be
  reported with the sensitivity sweep, not at `scale = 1.0` alone.
- **The margins are small.** Always-prioritise beats no-op by +0.118, +0.028
  and +0.112 on rewards of 15-22. That is a real but weak learning signal, and
  whether PPO finds it is an empirical question the retrain will answer.
- **The `selective` heuristic still loses on BPIC2012** (−2.1), because it
  leans on escalation, which is priced expensively. Whether that is the right
  price is exactly what the sweep is for.
- **Repetition calibration (finding 10) is still open.** It needs a lever on
  the conclusion bonus, not on shaping.

---

## Fix 16 — The cycle-time charge only punished slowness, so the agent raced

**Status:** implemented, retrained, verified across 5 seeds and 5 effect sizes.
**Date:** 2026-08-25

### The bug

Fix 15 put cycle time into the reward so the managerial interventions would
have something to bite on. It was floored at zero below the real median:

```python
reward -= w_time * max(0.0, log2(elapsed / real_median_cycle))
```

The reasoning recorded at the time was that paying the agent to finish faster
than any real case would reward implausibility. That reasoning was wrong in a
specific way: flooring does not *pay* the agent to go faster, but it makes
going faster **free**, and the per-step cost then breaks the tie. Finishing in
an hour a case that really takes a day scored exactly the same as finishing it
in a day, minus one step's cost.

The agent duly raced. Measured after Fix 15:

| Dataset | Simulated median | Real median | Ratio | \|log2\| |
|---|---|---|---|---|
| BPIC2012 | 0.05 d | 0.81 d | 0.06 | 4.02 |
| BPIC2015 | 13.78 d | 68.44 d | 0.20 | 2.31 |
| BPIC2017 | 4.90 d | 19.09 d | 0.26 | 1.96 |

Finding 13 was worse *after* the term than before it existed.

### The change

`RewardConfig.time_penalty_mode` selects the shape; `"two_sided"` is the
default and `"slow_only"` reproduces the old behaviour as an ablation.

```python
deviation = |log2(elapsed / real_median)|        # two_sided
deviation = max(0, log2(elapsed / real_median))  # slow_only
reward   -= w_time * min(deviation, time_penalty_cap)
```

`time_penalty_cap` is 6.0 log2 units — a factor of 64 either way. It is **not**
the ratio cap that failed in Fix 15, which sat at 3.0 and killed the gradient
where the agent actually operates. This one sits far outside the operating
range (the worst deviation ever measured is 4.0) and exists only so a
degenerate episode with near-zero elapsed time cannot swamp a batch.

The charge moved out of `_compute_reward` into `ProcessEnv._cycle_time_deviation()`
so it is testable without constructing a terminal step.

### Result — this is the one claim that survives everything

Median across 5 training seeds at `effect_scale = 1.0`, 300 evaluation
episodes each:

| Dataset | \|log2\| before | \|log2\| after (median) | across-seed range |
|---|---|---|---|
| BPIC2012 | 4.02 | **0.80** | 0.61 – 1.30 |
| BPIC2015 | 2.31 | **0.35** | 0.20 – 2.27 |
| BPIC2017 | 1.96 | **0.77** | 0.29 – 1.11 |

The improvement is far larger than the seed spread on every dataset, and it
holds at every `effect_scale` in {0, 0.5, 1.0, 1.5, 2.0} — mean cycle ratio
stays inside [1.20, 1.91] on BPIC2012, [1.04, 1.22] on BPIC2015 and
[1.04, 1.69] on BPIC2017. It does not depend on the assumed effect sizes.

Two other things improved as side effects: BPIC2015 truncation went 6.0% → 0%,
and BPIC2017 episode length went 22.8 → 27.4 against a real 31.

### What got worse

**Episode-length fidelity, on two of three datasets.** In \|log2(steps/real)\|:
BPIC2012 0.17 → 0.44, BPIC2015 0.52 → 0.96, BPIC2017 0.44 → 0.18. The agent
is hitting the duration target by choosing slow activities and repeating them,
not by doing the real amount of work. Duration and step count are now pulled
by two separate terms that can be satisfied independently, and nothing couples
them.

**BPIC2017 repetition.** Self-loops 65.1% → 74.9% ± 6.5 against a real 38.3%.
Repeating is the cheapest way to spend wall-clock time, so pricing slowness
made the camping behaviour of finding 10 *more* attractive on the dataset that
was already over-repeating.

### A retraction

On the single seed-0 run I recorded that BPIC2012 self-loops had gone
0.0% → 51.7% against a real 42.5%, and read that as finding 10 largely closing
on its own. **That was a seed artefact.** Across five seeds:

| Seed | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| BPIC2012 self-loops | 53.7% | **0.0%** | 14.8% | 32.6% | 8.8% |

Mean 22.0% ± 21.4 against a real 42.5%. Seed 0 was the best draw of five and
seed 1 reproduces the original 0.0% exactly. Finding 10 is **not** closed on
BPIC2012; it is uncalibrated with enormous variance. The lever it needs is
still the one Fix 15 identified — the conclusion bonus — and nothing here
touched that.

---

## Fix 17 — A4: every number in this project was one run, and the spread is large

**Status:** done. 51 runs, `run_experiments.py`, results in `results/sweep/`.
**Date:** 2026-08-25

Five seeds at `effect_scale = 1.0`, mean ± sd across seeds, 300 evaluation
episodes each:

| | BPIC2012 | BPIC2015 | BPIC2017 |
|---|---|---|---|
| Episode length | 13.03 ± 0.87 | 33.88 ± **14.97** | 25.62 ± 2.41 |
| Cycle ratio | 1.91 ± 0.41 | 1.06 ± 0.48 | 1.41 ± 0.60 |
| Good of concluded | 0.188 ± 0.014 | 0.838 ± 0.021 | 0.553 ± 0.018 |
| No-op rate | 0.897 ± 0.094 | 0.639 ± **0.345** | 0.905 ± 0.046 |
| Rule-waiving | 0.000 ± 0.000 | 0.075 ± 0.106 | 0.016 ± 0.035 |
| Self-loops | 0.220 ± **0.214** | 0.038 ± 0.064 | 0.749 ± 0.065 |
| Mean reward | 13.56 ± 1.25 | 14.24 ± **12.26** | 22.74 ± 2.09 |

**The spread is not a footnote.** On BPIC2015 the mean reward is 14.24 with a
standard deviation of 12.26 — the metric varies by nearly its own magnitude
between seeds. Seed 2 diverged outright (reward −4.37, cycle ratio 0.21,
rule-waiving 23.5%) while seed 3 reached +25.47. One of the five BPIC2015 runs
is a failure and the other four are not, from identical configuration.

**What this invalidates.** Any previously reported difference smaller than
these bars was not a result. That includes the Fix 15 margins — always-prioritise
beating no-op by +0.118 / +0.028 / +0.112 — which are one to two orders of
magnitude below the seed spread on the same datasets. Those margins were
measured under fixed heuristic policies rather than trained ones, so they are
not strictly the same quantity, but they cannot be used to argue that a trained
agent will find the difference.

**What survives.** Three things clear the bars comfortably:

- the cycle-time improvement (Fix 16), by a factor of 3–6 in \|log2\|;
- outcome-rate invariance — 0.014 to 0.021 sd against base rates of 0.177,
  0.827 and 0.520, so the verdict is not steerable by seed either;
- BPIC2017's excess repetition, 0.749 ± 0.065, which is consistently wrong
  rather than noisily wrong.

**What this costs.** Every headline number in the paper needs a seed bar or a
median-of-N, and single-run comparisons have to go. `run_experiments.py seeds`
regenerates the table.

---

## Fix 18 — The effect-size sweep, and what it says about the catalogue

**Status:** done, same 51 runs. `results/sweep/effect_scale.csv`.
**Date:** 2026-08-25

Every effect size in `intervention_effects.py` is an assumption the logs cannot
support (Fix 15 established this: congestion correlations are ~0 and mostly
negative, and BPIC2012/BPIC2017 record no interventions at all). So the sweep
over `effect_scale ∈ {0, 0.5, 1.0, 1.5, 2.0}` is the only thing that licenses
any claim about the catalogue. Three seeds per value, five at 1.0.

### 1. The cycle-time result does not depend on the assumptions

Mean cycle ratio by scale:

| Scale | 0.0 | 0.5 | 1.0 | 1.5 | 2.0 |
|---|---|---|---|---|---|
| BPIC2012 | 1.44 | 1.32 | 1.91 | 1.74 | 1.20 |
| BPIC2015 | 1.14 | 1.22 | 1.06 | 1.07 | 1.04 |
| BPIC2017 | 1.04 | 1.57 | 1.41 | 1.16 | 1.69 |

No trend, and every cell is within a factor of two of the real median — against
0.06 / 0.20 / 0.26 before Fix 16. This holds even at `effect_scale = 0`, where
the interventions do nothing at all, which is the strongest possible statement:
**the cycle-time fix is a property of the reward shape, not of the assumed
intervention effects.**

### 2. The compliance charge is causal — demonstrated, not just observed

Rule-waiving share (`skip_optional_subprocess` + `relax_rules_for_low_risk`):

| Scale | 0.0 | 0.5 | 1.0 | 1.5 | 2.0 |
|---|---|---|---|---|---|
| BPIC2012 | **48.1%** | 0.0% | 0.0% | 0.0% | 0.0% |
| BPIC2015 | **28.9%** | 16.8% | 7.5% | 0.0% | 0.0% |
| BPIC2017 | **59.2%** | 0.0% | 1.6% | 0.0% | 0.0% |

Turn the charge off and the behaviour comes straight back, at the same order of
magnitude as the 77.9% / 56.1% / 70.5% recorded before the charge existed. Turn
it on at any strength and it collapses. This is a dose-response curve on the
mechanism, which is a considerably stronger claim than the before/after pair
the handoff recorded.

**With one caveat that has to be stated: it is not clean on BPIC2015.** At
`effect_scale = 1.0`, two of five seeds show 13.2% and 23.5% rule-waiving while
three show ~0. The mean of 7.5% is not a policy that mostly complies; it is a
mixture of policies that comply and policies that do not.

### 3. `effect_scale = 0` is not the clean ablation it was described as

`scaled_compliance_risk()` multiplies by `scale`, so `effect_scale = 0` disables
the **compliance charge as well as** the duration effects. The A3 management
ablation and the compliance experiment are therefore entangled: the scale-0
column above is "actions inert *and* rule-waiving free", not "actions inert".

That is arguably the right semantics — charging compliance risk for an action
that provably does nothing would be odd — but it means scale 0 cannot be cited
as an ablation of the management actions alone. Separating them needs a second
knob.

### 4. The no-op dominance is an artefact of the compliance charge

No-op rate by scale:

| Scale | 0.0 | 0.5 | 1.0 | 1.5 | 2.0 |
|---|---|---|---|---|---|
| BPIC2012 | 44.6% | 98.1% | 89.7% | 94.6% | 96.5% |
| BPIC2015 | 41.0% | 33.7% | 63.9% | 93.2% | 72.3% |
| BPIC2017 | 33.2% | 95.6% | 90.5% | 91.3% | 95.9% |

The high no-op rates that finding 9 treated as "the action choice carries no
signal" appear only once the compliance charge is on. With it off the agent
spreads across the catalogue. So the agent is not ignoring the actions — it is
declining the two cheap ones it is being charged for and finding little reason
to prefer the rest. That is a different, and more defensible, statement.

### 5. Repetition is uncalibrated at every scale

Self-loop rate against the real log (0.425 / 0.008 / 0.383):

| Scale | 0.0 | 0.5 | 1.0 | 1.5 | 2.0 |
|---|---|---|---|---|---|
| BPIC2012 | 11.9% | 20.2% | 22.0% | 9.5% | 19.2% |
| BPIC2017 | 58.6% | 55.7% | 74.9% | 59.7% | 70.5% |

Consistently under on BPIC2012 and consistently over on BPIC2017, at every
scale. Finding 10 is untouched by anything in this session.

---

## Fix 19 — `checks/fix5_verdict_control.py` was passing and failing by luck

**Status:** implemented, verified.
**Date:** 2026-08-25

Found while re-running the checks after Fix 16. `fix5` failed on BPIC2015 with
"good rate moved 11.8% between policies — the verdict is still steerable".

It was not caused by Fix 16. The rollouts in that check use no model, and the
environment's dynamics never read the reward, so reward weights cannot affect
it — confirmed by running both `time_penalty_mode` values and getting
byte-identical results (concluded 9/400, good rate 0.8889 in both).

The real cause: the fixed comparison policy (always take the lowest-index
successor) walks into a dead end on BPIC2015 and concludes **9 to 14 episodes
in 400**. Its good rate was a fraction with a single-digit denominator, compared
point-to-point against a rate from ~390 episodes. It swung 10–20% between runs
purely by resampling. The check had been reporting that as evidence about the
verdict.

Three changes:

- **Compare on an interval, not on point estimates.** `diff_ci_low()` returns
  the lower bound of the 95% interval on the difference between two rates; the
  tolerance is applied to that, so a swing has to exceed sampling error.
- **Require 30 concluded episodes per arm**, and **fail** rather than pass when
  no contrast is comparable. An untested claim must not read as a passing one.
- **Add a second fixed policy** (`last`, always the highest-index successor).
  It concludes 47 of 400 on BPIC2015 where `first` concludes 9, so the dataset
  is genuinely tested rather than skipped.

Result — all three datasets now carry a real test, and all three pass:

| Dataset | random | contrast | n | swing | CI lower bound |
|---|---|---|---|---|---|
| BPIC2012 | 17.8% (n=400) | first 16.2% | 334 | 1.6% | 0.0% |
| BPIC2015 | 83.2% (n=388) | last 86.8% | 38 | 3.6% | 0.0% |
| BPIC2017 | 54.5% (n=396) | first 55.6% | 286 | 1.0% | 0.0% |

Every lower bound is 0.0%, so there is no evidence of steerability anywhere.
BPIC2015's contrast arm is thin at n=38, and the interval accounts for that
rather than hiding it.

---

## Fix 20 — One definition of "train a policy"

**Status:** implemented, verified byte-identical.
**Date:** 2026-08-25

Fixes 17 and 18 need dozens of training runs, which means a script. A script
that re-implements notebook 04's training cell is exactly the drift Fix 3 was
about. So the setup moved into `src/experiment.py` and notebook 04 calls it:

- `train_policy(env, seed, ...)` — MaskablePPO, the fixed early-stopping
  settings, the training logger, and the run record for `save_run_config`.
- `measure_policy(env, model, ...)` — reduces a policy to the reported metrics.
  It reuses `insights.collect_trajectories`, the same rollout notebook 06
  analyses, rather than a second loop. `cycle_time_s` was added to that
  function's per-episode record so cycle time comes off the policy's own
  episode.
- `real_self_loop_rate(dataset_dir)` — step-weighted over the event log.
  Reproduces 42.5% / 0.8% / 38.3%, the rates the agent is compared against.
  (Averaging the twin's per-activity loop probabilities does not: it weights a
  rare activity the same as a dominant one.)

`run_experiments.py` drives the studies. The two share runs — the seed study is
the sweep at `effect_scale = 1.0` — so asking for both costs 51 runs rather
than 60, and the shared cell is literally the same run rather than two runs
that ought to agree. Results go to `results/sweep/` only; a sweep cannot
overwrite `output/`.

**Verified, not assumed.** Re-running notebook 04 on BPIC2012 after the
refactor produced identical policy weights (md5 `4a974d00f65e5519…` on the
`policy.pth` tensors) and bit-identical `training_metrics.csv` across all 41
substantive columns and all 32 logged intervals of a 320,000-step run. The only
column that moved was `fps` (417 → 707; the verification run had the machine
to itself). The `.zip` md5 differs because zip archives embed timestamps.

---

## Fix 21 — The managerial recommendations do not survive a change of seed

**Status:** measured. `checks/fix21_insight_stability.py`, results in
`results/fix21_insight_stability/`.
**Date:** 2026-08-25

Fix 17 measured across-seed spread of the *aggregate* metrics. That is a
warning, not an answer — aggregates can move while the underlying routing
preferences stay fixed, and they can stay fixed while the preferences churn.
So this measures the deliverable itself: the contents of
`output/<DATASET>/routing_recommendations.csv`, the file that says things like

> The agent avoids 'W_Afhandelen leads' to 'W_Beoordelen fraude'. This path may
> lead to loops or inefficiency. Consider adding a routing constraint to
> discourage it.

Five seed models per dataset at `effect_scale = 1.0`, 200 episodes each,
routing preference table regenerated from each, compared pairwise.

| | top-10 Jaccard | transitions all 5 seeds exercised | mean shared per seed pair | verdict |
|---|---|---|---|---|
| BPIC2012 | 0.363 | 4 | 6.5 | **seed artefact** |
| BPIC2015 | **0.005** | **0** | 3.5 | **seed artefact** |
| BPIC2017 | 0.297 | 1 | 3.2 | **seed artefact** |

On BPIC2015, five identically-configured runs produce recommendation lists with
essentially **no overlap**, and there is not one transition that all five
policies even visit. The single rank correlation with enough support to be
computed at all is **−0.127** — mildly *anti*-correlated.

This is a stronger statement than "the numbers are noisy". The seeds are not
producing different estimates of the same policy; they are producing policies
that traverse different parts of the process. There is no common object for
them to be noisy estimates *of*.

**Consequence.** Every recommendation currently in `routing_recommendations.csv`
on all three datasets is a property of seed 0. They cannot be reported as
findings, and no amount of re-running fixes that — a median over seeds is not
defined when the seeds share three transitions out of hundreds.

### A defect in the first version of this check

The first run reported BPIC2017 as "partially reproducible" on the strength of
100% sign agreement over **one** shared transition, and a mean Spearman rho of
0.943 computed over seed pairs sharing 1, 1, 2, 3, 3, 3, 4, 4, 4 and 7 points.

That is precisely the defect fixed in `checks/fix5_verdict_control.py` earlier
the same day — a statistic from a single-digit sample compared against a
threshold as though it carried information. Written into a brand-new check
within the hour. `MIN_SHARED_FOR_SIGN` and `MIN_COMMON_FOR_RHO` (both 10) now
guard it, and with the guards in place BPIC2017 is a seed artefact like the
other two. Recorded here because the failure mode is evidently easy to
reintroduce and the guard belongs in any new check by default.

### What this means for the project

The four constraints below are structural, not tuning problems:

1. **The logs cannot ground the management catalogue.** They record what
   happened, never what was *tried*. No causal effect of an intervention is
   estimable from them (Fix 15). Every effect size is an assumption.
2. **The agent cannot affect outcomes**, by design and correctly (Fix 5). The
   only available lever is how a case is handled.
3. **That lever gets gamed.** Fix 16 met the cycle-time target by choosing slow
   activities and repeating them; episode-length fidelity got worse on two of
   three datasets.
4. **The policy is a function of the data *and the seed*,** and this check
   shows the seed term dominates.

So prescriptive managerial insight is not available from BPIC2012/2015/2017 by
this route. What the work has produced instead is a diagnostic apparatus —
eight checks that catch fabricated transitions, self-selected verdicts,
train/eval reward drift, impossibly fast outcomes, unfaithful duration models,
cycle-time terms that reward implausibility, checks that pass by luck, and now
seed-dependent recommendations — plus the seed-variance result itself. That is
a methods contribution about why RL-derived managerial recommendations from
public event logs are not yet trustworthy, and it is stronger for being
negative.

---

## Fix 22 — Validating routing advice against the real log, not the simulator

**Status:** measured. `checks/fix22_route_speed_in_real_log.py`,
results in `results/fix22_route_speed/`.
**Date:** 2026-08-25

### Correcting the framing of Fix 21

Fix 21 concluded that the routing recommendations are seed artefacts and read
that as "the approach cannot deliver managerial insight". Two errors in that
reading, both mine:

1. **"The agent cannot affect outcomes" was listed as a limitation.** It is not.
   The objective is to make the approve/reject decision *faster*, not to change
   what it is. Fix 5 removing outcome control is the design working, and
   time-to-decision is exactly the quantity left.
2. **Instability was treated as invalidity.** Five seeds disagreeing means we
   cannot report *the* list. It does not establish that any list is wrong.
   Different seeds may have found different, equally real shortcuts.

Both are settled by leaving the simulator: the real log already records, for
every case, the route it took and how long it took.

### Method

For each trained policy, take its **preferred transitions** (used more than a
random policy uses them). Score every **real** case by the fraction of its
transitions in that set. Ask whether higher-conformance real cases reached
their decision faster. Time is measured to the *first outcome marker* —
post-decision admin is not time-to-decision. No twin, no reward, no simulated
episode enters the measurement.

Two confounds are controlled because either would manufacture a result:
outcome (rejections are far faster than approvals on BPIC2012, so conformance
could proxy the decision — everything is computed within outcome group), and
length (long cases have more chances to leave the preferred set and are
trivially slower — a partial correlation controls for step count).

### Result

Negative rho = conformant real cases were faster.

| Dataset | Outcome | mean rho | seed range | partial rho | median days low → high |
|---|---|---|---|---|---|
| BPIC2012 | reject | +0.300 | −0.19 … +0.48 | −0.065 | 14.2 → 1.3 |
| BPIC2012 | approve | −0.072 | −0.15 … +0.11 | −0.025 | 16.9 → 13.9 |
| BPIC2015 | reject | +0.008 | −0.11 … +0.11 | +0.015 | 62.6 → 62.7 |
| BPIC2015 | approve | +0.025 | −0.03 … +0.09 | −0.039 | 99.6 → 101.0 |
| **BPIC2017** | **reject** | **−0.222** | **−0.35 … −0.06** | **−0.285** | **30.7 → 17.5** |
| BPIC2017 | approve | +0.086 | −0.01 … +0.11 | −0.103 | 13.7 → 15.4 |

**One cell survives: BPIC2017 rejections.** All four distinct policies agree in
direction (seeds 2 and 3 produced identical preferred sets — Fix 21 pairs.csv
records Jaccard 1.000 — so this is 4 draws, not 5). The partial correlation is
*stronger* than the raw one, so it is not a length artefact. Conformant real
rejections took **43% less time**: median 30.70 → 17.46 days over 15,091 cases.

**This settles the Fix 21 question.** Those four policies barely share a
transition, yet they agree on direction here. Instability and invalidity are
genuinely different, as Fix 21 should have said.

**BPIC2012 reject is a lesson in raw correlations.** Raw +0.300, controlled
−0.065. Its low-conformance cases have a median of **0.05 days** — auto-rejections
finishing in about an hour, matching nothing the agent prefers. Conformance
there tracks "the case was worked on at all", not speed. Reporting the raw
figure would have produced a confident, backwards finding.

### What this changes

The RL cannot certify its own routing advice — Fix 16 showed simulated speed is
a property of the reward. But it can *propose*, with the log adjudicating:

    train N seeds → collect preferred transitions per seed
                  → validate each against the real log (this check)
                  → report only what survives

This makes seed instability a **selection mechanism** rather than a defect, and
makes the speed claim immune to simulator artefacts because validation never
enters the simulator. On this data it yields one finding rather than a
catalogue — thin, but real.

### The limit that must be stated

This is an **association in observational data**. Cases that are easy to reject
may naturally follow those routes, so "route cases this way and they conclude
faster" cannot be separated from "cases that were always fast go this way".
Neither log records interventions, so the distinction is not identifiable here.
The defensible phrasing is a **targeting** claim — where to look — not a
treatment effect.
