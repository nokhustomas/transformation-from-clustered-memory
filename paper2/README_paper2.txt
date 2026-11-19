Paper 2: "One Registry, Several Jobs: An Empirical Map of Online Co-activation
Assemblies in Small Transformers"
================================================================================

WHAT THIS IS
------------
An 18-page paper mapping a single mechanism -- an online co-activation assembly
registry -- across every job it was proposed to do, over 20+ experiments on
synthetic Transformers (34k-112k params) and one 82M pretrained LM (distilgpt2),
all on 2 CPU cores.

The map has four verdicts:
  1. Assemblies DO emerge during ordinary training and are captured online for
     free -- with a capacity law (real text needs ~450 groups, not 8).
  2. The proposed pattern-completion mechanism FAILS, and the paper says exactly
     why: a stored STATE is the wrong carrier. Network states are context-bound.
     A real state of the right class taken from another context is catastrophic
     (+0.92 nats where the oracle gives -0.078); averages are inert; stored
     exemplars are no better.
  3. The substitution that WORKS: store the CORRECTION, not the state. A single
     global linear map A*h+b, fit on one set of contexts and applied to disjoint
     ones, recovers 99.9% of the achievable benefit (synthetic) and 95% of the
     oracle's recovery on distilgpt2. The additive form the original mechanism
     used is harmful at both scales.
  4. The registry's READINGS are a competent internal-fault sentinel: zero
     calibration, streaming, layer-localizing, strongest where perplexity
     monitoring is weakest -- with three honest defects quantified (alarm rate
     does not track harm; damage outside the assemblies is invisible; a healthy
     model reading out-of-domain text false-alarms at 95%, fixable to 0%).

This is a separate work from the earlier paper on external clustered memory
(CMA, in paper/); that paper stands on its own.

FILES
-----
paper2/
  paper2.tex          -- full source
  paper2.pdf          -- compiled, 18 pages (latexmk -pdf paper2.tex)
  gen_figures.py      -- regenerates every figure directly from result files
  figures/*.pdf       -- 7 figures

assembly/   (synthetic-model experiments, CPU minutes to ~1h each)
  registry.py             -- AssemblyRegistry (the mechanism; 8/8 unit tests)
  test_registry.py        -- unit tests (python3 test_registry.py)
  concept_lm_data.py      -- synthetic class language (Expts 1-5, 7)
  build_corpus.py         -- real-text corpus extraction (4MB docstring English)
  run_expt1_emergence.py  -- Expt 1: emergence, 3 seeds
  run_expt2_completion.py -- Expt 2: completion + placebo (no-deficit regime)
  run_expt3_stress.py     -- Expt 3: corruption repair + extreme-rare tail
  run_expt4_multilayer.py -- Expt 4: multi-site completion
  run_expt5_intraining.py -- Expt 5: in-training completion loop
  run_expt6_realtext.py   -- Expt 6: real-text emergence + capacity law
  run_expt6b_capacity_sweep.py -- Expt 6b: the capacity sweep, logged (K x tau x
                             set threshold -> groups created, coverage, alignment)
  run_expt7_twopass.py    -- Expt 7: cross-layer two-pass completion
  run_expt7_oracle_bound.py / run_expt7_snap_variant.py -- oracle bound, variants
  run_expt7b_retrieval_by_rank.py -- Expt 7b: correct-group retrieval by token
                             frequency (the need-reliability crossover), logged
  run_expt8_realrepair.py -- Expt 8: real-text repair (dilution)
  run_expt9b_scaled.py / run_expt9b_complete.py -- Expt 9: composition (v2)
  run_expt9c_group_semantics.py -- Expt 9c: do groups encode modifier meaning or
                             execution progress? (purity against both labels)
  run_expt10_detector.py / run_expt10_baseline.py -- Expt 10: sentinel + perplexity
  run_expt11_harden.py / run_expt11b_valuescore.py -- Expt 11: fault families,
                             seeds, localization, value channel
  run_expt11c_fault_harm.py -- Expt 11c: how much each fault actually costs the
                             model (pairs with the AUC table; detection != harm)
  run_expt12_matched.py   -- Expt 12: magnitude-matched placebo + matched noise
  run_expt13_patching.py  -- Expt 13: PATCHING vs ADDITION (the pivot)
  run_expt14_interp.py    -- Expt 14: interpolation sweep + state-legality boundary
  run_expt15_exemplar.py  -- Expt 15: stored real exemplars (negative)
  run_expt19_transform.py -- Expt 19: STORE THE TRANSFORM (the positive result)
  run_expt24/25/26_*.py   -- settling family (negative; kept for completeness)
  run_expt28_self_correct.py -- VOID BY DESIGN, see NOTES below
  results/                -- every logged result the paper cites (json/jsonl)
                             plus trained model/registry checkpoints (.pt)

sentinel/   (82M distilgpt2 experiments + the usable tool)
  sentinel.py             -- AssemblySentinel: hooks any GPT-2-family model,
                             streaming archive, two readings, length-binned
                             calibration, document-level diagnose(), health_report()
  registry.py             -- same mechanism (duplicated so this dir is standalone)
  build_registry.py / check_health.py / demo_faults.py -- the three user commands
  try_it.py               -- one-shot try-out, with --break-layer to inject a fault
  demo_all.py             -- six scenarios end to end
  finetune_with_registry.py -- continued pretraining WITH the registry registering
                             online (the "free by-product" claim, tested for real)
  validate_real.py        -- fault battery + localization on the real model
  eval_at_scale.py        -- 150 docs/condition, document-level alarm decisions
  eval_blindspot.py       -- members vs non-members vs random neuron kills
  eval_gain_modulation.py -- gain-scaling arm (negative; see NOTES below)
  run_expt16_real_patching.py  -- patch vs add on distilgpt2
  run_expt17_multidomain.py    -- the 95% false-alarm fix
  run_expt20_real_transform.py -- the transform on distilgpt2 (83% vs oracle 87%)
  run_expt21_combined.py  -- both false-alarm fixes together
  run_expt22_transfer.py  -- is A a general repair operator? (no: scope-bound)
  run_expt23_mixed.py     -- mixed fitting: coverage vs peak accuracy
  run_expt27_real_group.py -- group-restricted settling on the real model (negative)
  *.json                  -- every logged result the paper cites
  *_corpus.txt            -- in / near / far domain corpora

REPRODUCING
-----------
CPU-only, ~2 cores, torch + transformers. Each run_expt*.py is standalone.
  Synthetic:  cd assembly  && python3 run_expt1_emergence.py    (-> results/)
  Real model: cd sentinel  && python3 finetune_with_registry.py (builds the 82M
              checkpoint and the during-training archive, ~40 min), then any
              run_expt*.py in that directory.
  Figures:    cd paper2 && python3 gen_figures.py     (reads results directly;
              no hand-copied numbers anywhere in the figure code)
  Paper:      cd paper2 && latexmk -pdf paper2.tex

NOT IN THIS ARCHIVE (rebuildable, too large for git)
----------------------------------------------------
  sentinel/distilgpt2_finetuned.pt      313 MB  <- finetune_with_registry.py
  sentinel/registry_during_training.pt   97 MB  <- same script, same run
  sentinel/registry_ready.pt             97 MB  <- build_registry.py (2-5 min)
  sentinel/*_registry.pt                 97 MB each
Everything else needed to reproduce is here, including every logged result, so
the paper's numbers can be checked without rerunning anything.

NOTES ON WHAT IS AND IS NOT IN THE PAPER (please read before citing)
--------------------------------------------------------------------
- run_expt28_self_correct.py is VOID BY DESIGN and its result is NOT in the
  paper. It applied the correction at the embedding layer, which for this
  vocabulary is a lookup table over 16 distinct rare tokens; reconstruction
  error was 8.5e-14 (exact memorization) and all four arms came out identical.
  The script is kept so the mistake is inspectable, not because it shows
  anything.
- eval_gain_modulation.py's printed significance labels are WRONG: it computed
  a cross-document (unpaired) standard error for what is a paired comparison,
  which labelled a monotone trend as "no change". The trend itself
  (alpha = 0.25/0.5/1.0/2.0 -> +0.0034/+0.0096/+0.0285/+0.0856, monotonically
  harmful) is the reportable result. The paper does not use the labels.
- Four numbers the paper relies on were originally measured inline and were not
  saved to a file. Before this release they were re-run as proper logged scripts
  (6b, 7b, 9c, 11c), and two of them changed: correct-group retrieval at 11
  occurrences is 52.0% at n=230, not the 40% first measured at n=70; and the
  real-text registry creates 452 groups at K=512, not 456. The paper uses the
  re-run values. Experiment 9's answer-region targeted *completion* run (the
  zero-flips result) is still an inline result; 9c re-establishes its diagnostic
  but not the completion arm itself.
- Run-to-run variance across store draws exceeds the quoted standard errors in
  the settling experiments (the same config gave 13.3% and 9.1% with +-0.002
  SEs). Those SEs are within-run only. This is why the settling line is
  reported as negative rather than as a small positive.
- Absolute false-alarm rates are not comparable ACROSS experiments, because the
  archive budget differs (expt17 uses 60 batches x 4, eval_at_scale uses
  150 x 16). Within-table comparisons are valid; that caveat is in the paper too.

WHAT THE AUTHORS (you) STILL NEED TO DO BEFORE SUBMISSION
---------------------------------------------------------
1. Replace the Anonymous author block with real names/affiliations.
2. Review the AI-assistance disclosure footnote against the venue's policy.
3. Verify the four 2024-2026 arXiv citations resolve to the intended works
   (they were found via web search during the project; double-check the IDs).
4. Re-read every claim against results/ files. The paper was written to match
   them and every number in the new sections was re-verified against the JSON
   before this release, but the responsibility transfer requires your own pass.
