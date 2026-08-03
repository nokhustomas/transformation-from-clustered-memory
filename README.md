# Co-activation Assemblies and Clustered Memory in Transformers

Two self-contained empirical papers, all code, and every logged result behind
every number. Everything ran on **2 CPU cores, no GPU**.

Both papers report what happened, including the parts that did not work. Several
numbers in here are corrections of earlier numbers we reported to ourselves; the
corrections are in the papers, not hidden.

---

## Paper 1 — Clustered Memory Augmentation: a three-round null result

**`paper/paper.pdf`** (22 pages) · code in `experiments/`, `cluster_memory.py`

An external memory bank that *clusters* past token representations instead of
retaining them raw, evaluated on MQAR-style associative recall against a
matched raw-top-k baseline.

It does not work, and the paper is about how we established that rather than
about the mechanism. Three candidate explanations for the null were proposed and
each was killed by direct evidence:

| Candidate explanation | How it died |
|---|---|
| Pooled query can't address individual items | The retrieval `top_k` equalled bank capacity, so retrieval selected *everything* regardless of query — the comparison was a mathematical no-op. Found by re-reading our own code before running the fix the reviewer asked for. |
| Memory branch collapsed to the null slot | Measured: null attention weight sits at 0.28–0.52, not 1.0. |
| The task is too hard to separate conditions | Made it easier; full-attention control jumped 34% → 67%, both memory conditions stayed at chance. |

40 logged runs. Clustered memory came out **0.65pp below** the raw baseline —
wrong direction, not merely insignificant. A 6000-step probe found a promising
upward drift in one seed; four held-out seeds did not reproduce it (3 of 4 went
the other way), so we stopped there and said so.

---

## Paper 2 — One Registry, Several Jobs

**`paper2/paper2.pdf`** (18 pages) · code in `assembly/`, `sentinel/`

A different memory idea, and the one that produced a positive result. During
ordinary training, record which hidden units fire together; treat each recorded
set as a group; later, when an input partially activates a known group,
re-activate the rest.

We built it and measured every link of that chain separately.

### What holds

**Groups really do emerge.** On a synthetic language with 8 planted word
classes, training produces exactly 8 groups, 100% class purity, zero cross-class
merges, replicated over 3 seeds — and the registry captures them online, for
free, during training. On real text the same machinery works but needs ~450
groups instead of 8: the inventory has to match the data's intrinsic state count.

**Injected content is causally live.** Injecting the *right* content repairs 78%
of damage when the damage sits at the injection site. Injecting *wrong* content
hurts, in every single experiment — including during training, where it corrupts
the model itself.

### What fails, and exactly why

The completion mechanism does not deliver its promised benefits. The cause is
sharper than "it didn't help":

> **A network's states are context-bound. There is no stored state that is both
> reusable and informative.**

- A **real** state of the **right class**, taken from another sequence: `+0.92` nats. The oracle intervention is `−0.078`. Reusing a real state from elsewhere is an order of magnitude worse than doing nothing.
- Averaging over many contexts: inert. Averaging two: harmful.
- Storing 16 real exemplars per group and using the nearest: recovers nothing — exemplars are themselves other-context states.

Every downstream failure follows from this: the rare tokens that most need
completion are retrieved correctly only 40% of the time (and a confident wrong
substitution costs far more than a right one gains); the stable co-activation
cores encode *how many steps remain in the task*, not composable meaning; repair
dilutes as the group inventory grows.

### The substitution that works

States can't be transported. **The correction can.**

Fit `correct ≈ A·h + b` on (deficient state, correct state) pairs from one set of
contexts, apply it to *disjoint* contexts:

| | synthetic (n=689) | 82M distilgpt2 |
|---|---|---|
| **Global linear map `Ah+b`** | **99.9% of oracle** | **83%** (oracle 87%) |
| Per-group maps | 69% | — |
| Bias only `h+b` *(the original additive form)* | **harmful** | **−40%** |
| Stored mean state | inert | **−246%** |

Same result at two scales three orders of magnitude apart. Note what the
winning row does *not* use: no retrieval, no grouping, no registry at all beyond
the pairs used to fit it.

Honest limits: fitting needs a healthy reference pass as supervision, so we show
such a map *exists* and generalizes across contexts — not that a network can
find one unaided. And the operator is scope-bound: fit on one damage type in one
domain it recovers 83% there, but −10% on a different damage type and −9% on a
far domain. Applying the wrong operator actively harms.

### The by-product that turned out useful

The registry's *readings* — how well the current state matches its best group,
and whether matched units carry typical values — make a zero-calibration,
streaming, layer-localizing **internal-fault sentinel**. It beats a perplexity
baseline precisely where perplexity is weak (internal damage: 0.955 vs 0.732),
and it comes free with training: an archive built *during* 400 steps of real
continued pretraining matches a post-hoc profiling pass.

Then we ran it as an actual monitor instead of scoring it as a classifier, and
it got worse in three specific ways that we report in full:

- **Alarm rate does not track harm.** Freezing 30% of units costs twice the loss of deleting 30% of active ones, and is caught *half* as often. What matters is whether the fault lands in the high-activation set the registry watches.
- **Damage outside the assemblies is invisible** — alarm rate equal to the healthy baseline, to the digit. An alarm is informative; silence is not evidence of health. (The consolation: member units carry 13× the damage, so the blind spot is aimed at the *less* load-bearing part.)
- **A healthy model reading out-of-domain text false-alarms at 95%.** Building the archive across three domains plus judging within-document takes that to 0%, with no loss of detection (100%) or localization (100%).

Once an alarm does fire, 86–100% of alarms name the correct layer, across every
fault type.

---

## Repository layout

```
paper/          Paper 1: tex, pdf, figures, README
experiments/    Paper 1 code: MQAR data, models, 3 experiment rounds, diagnostics
cluster_memory.py, test_cluster_memory.py    Paper 1 mechanism + 8 unit tests

paper2/         Paper 2: tex, pdf, 7 figures, gen_figures.py, README
assembly/       Paper 2 synthetic experiments (expt 1-15, 19, 24-28) + results/
sentinel/       Paper 2 real-model experiments (expt 16-23, 27) + the usable tool
```

Read `paper2/README_paper2.txt` before citing paper 2 — it lists which scripts
are void, which printed labels are wrong, and which results were run inline
rather than saved.

`CLAUDE.md` is the raw working log kept during the project: every hypothesis, run
and result in chronological order, including the explanations that were killed
and the corrections we had to make to our own reported numbers. It is not a
source for any claim — the papers cite the logged JSON — but it makes the process
inspectable. `七小时实验总结.md` and `反复靠拢实验_方法与标准.md` are plain-language
Chinese summaries written during the work (the latter is a pre-registration of
method and success criteria, written before the run it describes).

## Reproducing

CPU only, `torch` + `transformers`. Every `run_expt*.py` is standalone.

```bash
python3 test_cluster_memory.py                # paper 1 mechanism, 8 tests
cd assembly && python3 test_registry.py       # paper 2 mechanism, 8 tests
cd assembly && python3 run_expt1_emergence.py # -> results/
cd sentinel && python3 finetune_with_registry.py   # ~40 min, builds the 82M archive
cd paper2   && python3 gen_figures.py && latexmk -pdf paper2.tex
```

Large checkpoints (a 313 MB finetuned model and four 97 MB registry archives)
are **not** in the repo — they are rebuilt by the scripts above in minutes. All
logged results *are* in the repo, so every number in both papers can be checked
without rerunning anything.

## Status

Both papers carry an `Anonymous` author block and an AI-assistance disclosure
footnote. Before submitting anywhere, fill in real authorship, check the
disclosure against the venue's policy, and verify the 2024–2026 arXiv citation
IDs — they were found by web search during the work and have not been
independently confirmed.
