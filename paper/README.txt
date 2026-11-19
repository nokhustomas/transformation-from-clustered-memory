Clustered Memory Augmentation for Transformers -- paper + code bundle
======================================================================

WHAT THIS IS
------------
An arXiv-style paper (paper.pdf / paper.tex) plus the full code that produced
every number in it. The paper's central finding is a diagnosed NULL RESULT
that survived a self-correction AND two full rounds of follow-up experiments:
at matched memory budget, our clustered-memory method (CMA) does not
outperform a raw-token-retention baseline in these experiments -- both stay
at chance level, under two different retrieval mechanisms and two different
task difficulties (see below). Read the abstract and Section 5 before
anything else; this is not a "CMA wins" paper, and by the end it is not a
"we ran out of time, more experiments might change this" paper either --
the authors deliberately stopped after three independent candidate
explanations for the null result were each ruled out by a direct,
targeted measurement. Section 7 (Conclusion) explains that decision
explicitly.

THIS BUNDLE INCLUDES A SELF-CORRECTION -- READ THIS FIRST
------------------------------------------------------------
An earlier version of this paper attributed the null result to a specific
bug-like cause ("the retrieval-selection query is pooled once per chunk, so
it can't tell which of several queries in that chunk is active"). Before
implementing a fix, we re-read our own retrieve() code and found that
explanation was wrong: in the reported configuration, top_k_sim was set equal
to each memory bank's full capacity, which makes retrieval "selection" a
no-op regardless of the query (torch.topk(sim, k=array_length) always returns
everything; RawTopKMemoryBank.retrieve() doesn't even read its query
argument). Pooled vs. per-token could not have mattered under that
configuration. We built a corrected version with retrieval selection that is
genuinely conditioned on a per-token query (k_sim lowered to half of
capacity so selection actually discriminates), unit-tested it, and reran the
full comparison. Result: the same null pattern. CMA still does not separate
from TopKRaw. This is reported in the paper (Section 5.2) as a ruled-out
hypothesis, not swept under the rug -- the paper shows the wrong explanation
and the correction, not just the final version.

THIS BUNDLE ALSO INCLUDES A THIRD ROUND: RULING OUT TASK DIFFICULTY
------------------------------------------------------------------------
After the corrected (per-token) mechanism produced the same null result, one
confound remained untested: maybe the task itself (4 candidates to
disambiguate from a 16-symbol vocabulary) was simply too hard for a
randomly-initialized cross-attention circuit to get traction on, regardless
of retrieval mechanism. We built a substantially easier version of the same
content-matching task (fewer pairs, smaller vocabulary, wider gap between
writing and querying -- deliberately keeping it pure content-matching, with
no recency/overwrite shortcut available) and reran the comparison a third
time. A full-attention oracle's accuracy nearly doubled on the easier task
(0.336 -> 0.669), confirming the task really did get easier and stayed
learnable in the same step budget -- but CMA and TopKRaw were still pinned
to chance, with CMA nominally trailing TopKRaw rather than leading it
(Section 5.4, Table 8). Task difficulty is therefore also ruled out. While
investigating this, we found and explain (Section 5.4) a striking but fully
diagnosed pattern: three of the bounded-memory runs produced bit-identical
accuracy curves across all 13 training checkpoints, which traces to
verified bit-identical parameter initialization (neither memory bank has
learnable parameters) plus non-saturated null-attention weight (~0.5, not
collapsed to 1.0) -- consistent with, not contradicting, the paper's
leading hypothesis.

BEFORE YOU DO ANYTHING ELSE WITH THIS PAPER
--------------------------------------------
1. Fill in your real name and affiliation on the title page (currently a
   placeholder: "[Author Name -- please insert before submission]").
2. Read the AI-assistance disclosure footnote on page 1 and adjust it to
   match your venue's actual disclosure policy -- the current text is a
   reasonable default, not a guarantee it satisfies every venue's rules.
3. Review every claim yourself before submitting anywhere. You are the
   author of record; I (Claude) am not, and cannot be.

DIRECTORY LAYOUT
-----------------
paper.pdf, paper.tex      The paper itself and its LaTeX source (22 pages).
figures/                  The 5 PDF figures paper.tex includes.
code/cluster_memory.py    Reference module: memory banks, clustered
                           cross-attention (both the original pooled-query
                           forward() and the corrected forward_per_token()),
                           boundary detector. Unit-tested (7 tests), imported
                           by the experiment code below.
code/test_cluster_memory.py   Its unit tests, 7/7 passing (tests 6-7 cover
                           the corrected per-token retrieval mechanism and
                           were added specifically to validate it before it
                           was trusted in a training run).
code/experiments/
  mqar_data.py             Synthetic MQAR long-range recall task generator,
                            parametrized (used, unmodified, by all three
                            rounds -- the third round only changes the
                            parameters it's called with, not its logic).
  models.py                All model conditions (NoMemory / FullAttn /
                            TopKRaw / CMA / TopKRaw-PT / CMA-PT), RoPE
                            self-attention. The "-PT" conditions use the
                            corrected per-token retrieval mechanism and are
                            reused, unmodified, for the third round's
                            easier-task conditions.
  run_experiment.py        Driver for the ORIGINAL 16 runs (pooled-query
                            retrieval; the first, since-corrected diagnosis
                            was based on these).
  run_experiment_pertoken.py   Driver for the 10 SECOND-ROUND runs
                            (corrected, genuinely discriminative per-token
                            retrieval).
  run_experiment_easytask.py  Driver for the 14 THIRD-ROUND runs (same
                            per-token mechanism, substantially easier task).
                            All three run_experiment*.py scripts append to
                            the same results/raw_results.jsonl (condition
                            names don't collide) and are independently
                            resumable.
  make_plots.py             Turns the original 16 runs into figures/tables.
  make_plots_pertoken.py    Turns the second-round 10 runs into
                            fig_pertoken_bar.{pdf,png}.
  make_plots_easytask.py    Turns the third-round 14 runs into
                            fig_easytask_bar.{pdf,png} (also prints the
                            B=8 single-seed reference values).
  diagnose_null_attn.py     Null-fallback-slot attention-weight diagnostic
                            (Section 5.2, Table 4, top two rows: original
                            mechanism; bottom two rows: --pertoken flag).
  diagnose_null_attn_easytask.py   Same diagnostic logic, rerun under the
                            third round's (easier-task) hyperparameters
                            (Section 5.4, Table 10) -- written specifically
                            to explain the bit-identical-curves observation
                            described above, not a copy-paste duplicate.
  test_pertoken.py          Extra unit tests for the parts of the second-round
                            correction that live in models.py (RawTopKMemoryBank's
                            discriminative retrieve_per_token(), and the two
                            "_pertoken" build_model() conditions), 2/2 passing.
  calibrate.py, calibrate2.py, speed_test.py, speed_test2.py
                            Calibration/timing scripts used to pick the
                            hyperparameters before the real runs (kept for
                            transparency, not needed to reproduce the
                            reported numbers).
code/experiments/results/
  raw_results.jsonl         One JSON record per training run (40 total: 16
                             original + 10 second-round + 14 third-round) --
                             the ground truth for every number in the
                             paper's tables.
  run_log.txt                 Console log of run_experiment.py (round 1).
  run_log_pertoken.txt        Console log of run_experiment_pertoken.py
                               (round 2).
  run_log_easytask.txt        Console log of run_experiment_easytask.py
                               (round 3). Note: this run was interrupted
                               twice by background-process/session
                               boundaries during development and resumed
                               each time using the same resumability
                               mechanism described below -- nothing about
                               the reported numbers themselves was affected,
                               since each (condition, seed, budget) triple
                               is only ever run once and appended once.
  diagnostic_null_attn_log.txt            Log of diagnose_null_attn.py
                                           (original mechanism).
  diagnostic_null_attn_pertoken_log.txt   Log of diagnose_null_attn.py
                                           --pertoken (corrected mechanism).
  diagnostic_null_attn_easytask_log.txt   Log of diagnose_null_attn_easytask.py
                                           (round 3 hyperparameters).
  summary.txt, table_data.json   Summaries from make_plots.py (original run).
  fig_*.pdf, fig_*.png       All figures, both formats, from all three
                              make_plots*.py scripts.

HOW TO REPRODUCE
-----------------
Requires: Python 3, PyTorch (CPU is fine -- everything here was run on a
2-core CPU machine, no GPU), matplotlib.

  cd code/experiments
  python3 test_pertoken.py            # sanity-check the corrected mechanism first (~10s)
  python3 run_experiment.py           # round 1, 16 runs, ~50-60 min; writes raw_results.jsonl
  python3 run_experiment_pertoken.py  # round 2, 10 runs, ~45-50 min; appends to the same file
  python3 run_experiment_easytask.py  # round 3, 14 runs, ~35-40 min; appends to the same file
  python3 make_plots.py               # figures/tables from round 1
  python3 make_plots_pertoken.py      # fig_pertoken_bar from round 2
  python3 make_plots_easytask.py      # fig_easytask_bar from round 3
  python3 diagnose_null_attn.py             # Table 4, top two rows
  python3 diagnose_null_attn.py --pertoken  # Table 4, bottom two rows
  python3 diagnose_null_attn_easytask.py    # Table 10 (both rows)

Then, to rebuild the PDF:
  cd ../../paper
  latexmk -pdf paper.tex        # or: pdflatex paper.tex (run twice for refs/citations)

All three run_experiment*.py scripts are resumable: they skip any
(condition, seed, budget) combination already present in
results/raw_results.jsonl, so if interrupted, just run again. This was
exercised for real during development (round 3's run was interrupted twice),
not just designed-but-untested.

HONEST SUMMARY OF WHAT THE EXPERIMENTS SHOW
----------------------------------------------
- Full attention over the whole sequence learns the synthetic recall task
  (~34% accuracy vs. ~6% chance on the primary task) within our step budget;
  a chunk-local model with no memory does not (~7%, indistinguishable from
  chance).
- Adding either a raw top-K token memory (H2O-style) or CMA's clustered
  memory on top of chunk-local attention does NOT lift accuracy above chance,
  at any of the three memory budgets we tried (4, 8, or 16 slots), under the
  ORIGINAL pooled-query retrieval mechanism.
- Round 1's diagnosis of why (a chunk-pooled retrieval query couldn't
  distinguish the task's multiple queries) turned out to be WRONG on
  closer reading of our own code: in this configuration, retrieval
  selection was already a no-op (top_k_sim equaled full capacity), so
  pooled vs. per-token could never have mattered -- there was nothing being
  filtered either way.
- Round 2: we implemented and unit-tested a corrected mechanism with
  retrieval selection genuinely conditioned on a per-token query (k_sim
  lowered below capacity so selection is no longer vacuous), reran the full
  comparison, and got the SAME null result: CMA still does not separate from
  TopKRaw, at any budget, under the corrected mechanism either. We also
  directly measured whether the memory branch had simply learned to ignore
  retrieved content (attention on a learned "null" fallback slot approaching
  1.0) -- it had not, under either mechanism (stabilizes around 0.28-0.44,
  not 1.0).
- Round 3: suspecting task difficulty itself might be the confound, we
  built a substantially easier version of the same content-matching task
  and reran the comparison a third time. A full-attention oracle's accuracy
  nearly doubled (0.336 -> 0.669), confirming the task really did get
  easier and stayed learnable in the same step budget -- but CMA and
  TopKRaw were still pinned to chance (CMA nominally -0.65 percentage
  points BELOW TopKRaw, the wrong direction, well within noise). Task
  difficulty is therefore also ruled out. A striking side observation
  (three runs producing bit-identical accuracy curves across all 13
  checkpoints) is fully investigated and explained in Section 5.4 rather
  than left as a curiosity -- it traces to verified bit-identical parameter
  initialization plus non-saturated null-attention weight, not a bug and
  not evidence of a fourth explanation.
- Net effect: the paper's central question (does clustering organize memory
  better than raw retention, at equal budget?) is NOT answered either way by
  these experiments, under either retrieval mechanism or either task
  difficulty tested. THREE specific candidate explanations for the null
  result are ruled out by direct evidence, each by a measurement targeted
  at that explanation: collapse to the null fallback slot, vacuous
  query-independent retrieval selection, and task difficulty. The leading
  remaining hypothesis -- that the memory cross-attention's own
  content-matching circuit hasn't been learned within the step budget,
  independent of memory organization, retrieval granularity, or task
  difficulty -- has not itself been tested directly.
- We stopped here BY DELIBERATE CHOICE, not because we ran out of compute
  or ideas (Section 7 explains this explicitly). More seeds would sharpen
  the confidence interval around a gap that is already small and
  wrong-signed for CMA -- an effect-size problem, not a statistical-power
  problem, so more seeds were not judged worth running. A substantially
  longer training run (we'd guess 5,000+ steps) is a genuinely different,
  legitimate next experiment -- it would speak to whether the circuit can
  bootstrap at all given more steps, not to whether clustering beats raw
  retention once it can -- and we name it explicitly as unfinished business
  for anyone who wants to pursue it, rather than either running a fourth
  round ourselves or silently dropping it. What we deliver is the method, a
  tested implementation, a documented self-correction, and three
  independently ruled-out explanations for an honestly null result.

This file was written to help you (and anyone you share this with) navigate
the bundle quickly -- it is not a substitute for reading the paper itself.
