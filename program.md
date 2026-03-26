# autoresearch

This is an experiment to have the LLM do its own research.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, data prep, tokenizer, dataloader, evaluation. Do not modify.
   - `train.py` — the file you modify. Model architecture, optimizer, training loop.
4. **Verify data exists**: Check that `~/.cache/autoresearch/` contains data shards and a tokenizer. If not, tell the human to run `uv run prepare.py`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU. The training script runs for a **fixed time budget of 5 minutes** (wall clock training time, excluding startup/compilation). You launch it simply as: `uv run train.py`.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation, data loading, tokenizer, and training constants (time budget, sequence length, etc).
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Modify the evaluation harness. The `evaluate_bpb` function in `prepare.py` is the ground truth metric.

**The goal is simple: get the lowest val_bpb.** Since the time budget is fixed, you don't need to worry about training time — it's always 5 minutes. Everything is fair game: change the architecture, the optimizer, the hyperparameters, the batch size, the model size. The only constraint is that the code runs without crashing and finishes within the time budget.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful val_bpb gains, but it should not blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.001 val_bpb improvement that adds 20 lines of hacky code? Probably not worth it. A 0.001 val_bpb improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

### The speed-capacity tradeoff

Under a fixed 5-minute time budget, **training throughput (steps per minute) is as important as model capacity**. A change that adds 10% more parameters but costs 20% more time per step will almost certainly lose — fewer training steps means less learning. Always check `num_steps` in the output. If it dropped significantly, the experiment likely failed regardless of the architecture's theoretical capacity. When in doubt, prefer changes that make per-step time faster (without destroying capacity) over changes that increase capacity at the cost of speed.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Known constraints (do NOT re-test these)

These have been empirically validated across multiple experiments. Do not waste runs re-testing them.

- **Muon cannot handle 1D parameters.** Do not add bias terms, per-head scalars, learned temperatures, or any parameter that isn't a 2D matrix to the model — unless you also modify the optimizer param grouping to route them to AdamW. Crashes are guaranteed otherwise (torch.stack fails on mismatched shapes).
- **Larger models lose under 5-min budgets on this GPU.** DEPTH>10 has been tested 8+ times (DEPTH=11, 12, 13, 14) and always produces fewer training steps, resulting in worse val_bpb. Do not try increasing depth or width unless you simultaneously find a way to speed up per-step time to compensate.
- **Zero warmup destabilizes training.** Always use WARMUP_RATIO >= 0.05. Tested twice; both times caused large regressions.
- **UNEMBEDDING_LR > 0.01 diverges.** Tested at 0.012 and 0.016 across multiple configurations; catastrophic every time (+0.05-0.06 bpb regression).
- **EMBEDDING_LR > 1.0 is harmful.** 1.5 was too high (+0.016 bpb). The sweet spot is around 1.0.
- **Label smoothing is harmful** at this scale (+0.2 bpb regression with smoothing=0.1).
- **SwiGLU is too slow.** 3 weight matrices per MLP layer instead of 2 means ~40% fewer training steps. Tested twice; never competitive.
- **GQA (n_kv_head=2) is slower than MQA (n_kv_head=1)** at this model scale. Extra KV parameters slow steps without helping. Tested 3 times.
- **torch.compile reduce-overhead crashes on Turing GPUs.** OverflowError in CUDA static launcher. Do not use mode="reduce-overhead".
- **Muon momentum 0.95 is too aggressive for short runs.** The 300-step warmup keeps momentum at ~0.85 for all 47 steps. This is optimal. Do not shorten warmup.
- **Embedding norm hurts** (+0.047 bpb). N(0,1) init with sqrt(dim) scaling already provides correct magnitude.
- **UNEMBEDDING_LR extremely sensitive around 0.008**: 0.006→+0.051, 0.01→+0.037. Do not deviate.
- **resid_lambdas are important** (+0.034 regression when removed). Learned residual scaling interacts with x0 shortcut.
- **Weight tying is catastrophic** (+1.02 bpb). Embedding/unembedding have very different optimal LRs.
- **EMBEDDING_LR sweet spot is 1.6.** Tested 1.2-2.0 range. Clear optimum.
- **Near-misses don't reliably combine.** Tested multiple combinations — interaction effects are unpredictable.
- **Stochastic depth crashes with torch.compile** (inductor crash).
- **Factored embeddings are catastrophic.** Both 256-dim (+0.58) and 512-dim (+0.56). Projection bottleneck kills token discrimination.
- **Multi-token prediction hurts.** Both separate-head (+0.18) and shared-head (+0.035) variants worse. Auxiliary gradients interfere with primary task.
- **Cosine LR is far better than linear** (+0.068 regression with linear). The cosine shape is optimal for 47 steps.
- **Cosine warm restarts hurt** (+0.081). LR spike at midpoint disrupts learning with so few steps.
- **Dropout hurts** (+0.018 with p=0.05). Model is underfitting with 47 steps, not overfitting.
- **Muon ns_steps=4 is worse than 3** (+0.020, fewer steps). Extra Newton-Schulz iteration costs compute.
- **x0_lambda init 0.1 is precisely optimal.** 0.05→+0.010, 0.12→+0.040, 0.15→+0.020, 0.2→+0.021.
- **Depthwise conv before attention hurts** (+0.012, +0.6GB VRAM). Redundant with full attention.
- **EMA weight averaging is a significant win** (-0.019 bpb from baseline without EMA). Adaptive decay 0.88→0.95 is optimal. Use `torch._foreach_lerp_` for speed.
- **SWA (uniform averaging) is much worse than EMA** (+0.028 bpb vs EMA). Exponential weighting is critical.
- **EMA decay sweep**: 0.85→2.221, 0.9→2.211, 0.92→2.208, 0.93→2.211, 0.95→2.220. Adaptive 0.88→0.95 is best (2.205).
- **UNEMBEDDING_LR sensitivity persists even with EMA.** 0.01 still +0.031 bpb worse. Fundamental, not noise.
- **Hyperparameter optima are largely unchanged by EMA.** MATRIX_LR, WARMDOWN_RATIO, FINAL_LR_FRAC all same optima. EMA is orthogonal to training dynamics.
- **Smaller batch + EMA is a massive win.** Batch 8K (DEVICE=32, TOTAL=2^13) with adaptive EMA gave 2.175 vs 2.205 (-0.030 bpb). 80 steps vs 47, only 1.7GB VRAM. EMA smooths gradient noise from smaller batches while more optimizer steps give more learning. This is the single biggest improvement since the initial architecture was set up.
- **EMBEDDING_LR sweet spot was 1.6 at batch 16K.** May need re-tuning at smaller batch sizes — more steps means different effective LR dynamics.
- **Previous batch size tests (pre-EMA) failed.** Run 58 (batch 8K, 2.389), run 119 (batch 8K, 2.275) — both much worse. EMA is what makes small batches viable.

## Strategy: EMA + small batch synergy (current, runs 221+)

**Key insight**: EMA weight averaging and smaller batch sizes have a powerful synergy. EMA smooths the noise from smaller batches, while smaller batches give more optimizer steps in the fixed 5-min budget. This broke through a 50-experiment plateau.

**Executed experiments in this strategy (do NOT repeat):**
- Run 221: EMA decay=0.95 → 2.220 (keep, first EMA win)
- Run 222: EMA decay=0.9 → 2.211 (keep)
- Run 223: EMA decay=0.85 → 2.221 (discard, too much averaging)
- Run 224: EMA decay=0.92 → 2.208 (keep)
- Run 225: EMA decay=0.93 → 2.211 (discard)
- Run 226: EMA decay=0.91 → 2.209 (discard)
- Run 227: EMA with foreach_lerp_ optimization → 2.207 (keep, same quality, faster)
- Run 228: EMA decay=0.9 fast impl → 2.211 (discard)
- Run 229: Late-start EMA (after 30% training) → 2.208 (discard)
- Run 230: WARMDOWN_RATIO=0.2 with EMA → 2.208 (discard)
- Run 231: FINAL_LR_FRAC=0.05 with EMA → 2.211 (discard)
- Run 232: MATRIX_LR=0.025 with EMA → 2.212 (discard)
- Run 233: EMBEDDING_LR=1.7 with EMA → 2.207 (discard, tied)
- Run 234: SWA uniform averaging → 2.232 (discard, much worse than EMA)
- Run 235: UNEMBEDDING_LR=0.01 with EMA → 2.236 (discard)
- Run 236: Adaptive EMA 0.85→0.95 → 2.206 (keep)
- Run 237: Adaptive EMA 0.80→0.95 → 2.212 (discard)
- Run 238: Adaptive EMA 0.88→0.95 → 2.205 (keep, best EMA-only)
- Run 239: Adaptive EMA 0.90→0.95 → 2.205 (discard)
- Run 240: Adaptive EMA 0.88→0.92 → 2.209 (discard)
- Run 241: Adaptive EMA 0.88→0.98 → 2.207 (discard)
- Run 242: WARMDOWN_RATIO=0.4 with EMA → 2.205 (discard)
- Run 243: Batch 8K + EMA → **2.175** (keep, massive -0.030 improvement)
- Run 244: Batch 4K + EMA → pending

**Current exploration direction**: Batch size reduction sweep, then re-tune hyperparameters at new batch size.

## Output format

Once the script finishes it prints a summary like this:

```
---
val_bpb:          0.997900
training_seconds: 300.1
total_seconds:    325.9
peak_vram_mb:     45060.2
mfu_percent:      39.80
total_tokens_M:   499.6
num_steps:        953
num_params_M:     50.3
depth:            8
```

Note that the script is configured to always stop after 5 minutes, so depending on the computing platform of this computer the numbers might look different. You can extract the key metric from the log file:

```
grep "^val_bpb:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit	val_bpb	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. val_bpb achieved (e.g. 1.234567) — use 0.000000 for crashes
3. peak memory in GB, round to .1f (e.g. 12.3 — divide peak_vram_mb by 1024) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	val_bpb	memory_gb	status	description
a1b2c3d	0.997900	44.0	keep	baseline
b2c3d4e	0.993200	44.2	keep	increase LR to 0.04
c3d4e5f	1.005000	44.0	discard	switch to GeLU activation
d4e5f6g	0.000000	0.0	crash	double model width (OOM)
```

## Experiment discipline

- **One variable at a time.** Never change two things simultaneously — you won't know which one helped or hurt.
- **Sweep in one direction first.** When tuning a hyperparameter (e.g., LR), keep going in the direction that's working until it stops, then stop. Don't oscillate.
- **Log step count.** When describing a discard, include the effective step count if it changed — this is the most common explanation for regressions.
- **Expect ~80% failure rate.** A 20% keep rate is normal for hyperparameter search. Do not treat a streak of discards as a signal to make increasingly radical changes — that tends to produce crashes and large regressions. Steady, methodical exploration beats wild swings.

## When progress stalls

After 20+ consecutive discards, the current configuration is likely near a local optimum for incremental changes. Shift strategy:

- **Combine near-misses**: If two independent changes each gave ~0.001 improvement or were tied, try them together — interactions can compound.
- **Revisit discarded ideas in new context**: An idea that lost before may win after the architecture changed. For example, sequential blocks lost at experiment #110 (bpb 2.277 vs 2.270) but won at experiment #131 (bpb 2.259 vs 2.262) after MLP ratio was reduced. Architecture interactions matter.
- **Focus on compute efficiency**: Any change that makes per-step time faster (without hurting capacity too much) is likely to win, because more steps = more learning in the fixed budget.
- **Try fundamentally different approaches**: e.g., different training dynamics (curriculum learning, batch size schedules), alternative optimizer configurations, or novel architectural patterns that stay within the Muon 2D-parameter constraint.

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar5` or `autoresearch/mar5-gpu0`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `uv run train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "^val_bpb:\|^peak_vram_mb:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If val_bpb improved (lower), you "advance" the branch, keeping the git commit
9. If val_bpb is equal or worse, you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should take ~5 minutes total (+ a few seconds for startup and eval overhead). If a run exceeds 10 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~5 minutes then you can run approx 12/hour, for a total of about 100 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!
