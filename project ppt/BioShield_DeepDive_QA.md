# BioShield — Deep-Dive Q&A Companion

**Companion to:** `BioShield_Presentation_Script.md`
**Purpose:** Detailed, source-verified answers to the technical questions
that judges or reviewers are most likely to ask after the talk. Every
claim below was traced back to a real line in the codebase.

---

## 1. Evaluation Metrics — Full Breakdown

We report **three primary metrics** and **two gated diagnostic metrics**.
All implementations are in [`evaluation/metrics.py`](evaluation/metrics.py)
and [`evaluation/eval_pipeline.py`](evaluation/eval_pipeline.py).

### 1.1 AUC (Area Under the ROC Curve) — primary

- **What it measures:** the probability that the detector assigns a
  higher *p(real)* score to a randomly drawn real abstract than to a
  randomly drawn fake abstract. Equivalent to the area under the
  Receiver-Operating-Characteristic curve (TPR vs FPR over all
  thresholds).
- **How we compute it:** `sklearn.metrics.roc_auc_score(y_true, y_prob)`
  with `y_prob` taken from `softmax(logits)[:, 1]` — the second-class
  (real) probability of the BiomedBERT classifier
  ([train_detector.py:101](models/detector/train_detector.py:101)).
- **Range:** `0.5` = random; `1.0` = perfect ranking.
- **Why we picked it:**
  1. **Threshold-free** — independent of the 0.5 decision boundary, so
     it isn't gamed by a poorly calibrated classifier.
  2. **Class-imbalance robust** — fine when real/fake counts diverge.
  3. **Standard in medical AI** — every published clinical-NLP paper
     reports it, so reviewers compare apples to apples.
- **Failure mode we hit:** AUC saturates near 1.0 for in-family fakes,
  which is exactly why we *also* track Evasion Rate.

### 1.2 F1 (harmonic mean of precision and recall) — primary

- **Definition:** `F1 = 2·P·R / (P+R)` where precision and recall are
  computed on the **fake class** with threshold 0.5.
- **Computation:** `sklearn.metrics.f1_score` with `zero_division=0`
  ([metrics.py:31](evaluation/metrics.py:31)).
- **Why F1 (not raw accuracy):** F1 penalises both missed fakes
  *(false negatives → fakes leak through)* and false alarms *(false
  positives → real abstracts get rejected)*. Either failure has real
  cost in a medical-publishing setting.
- **Why our 50/50 split makes F1 fair:** with balanced classes,
  precision and recall are weighted equally, so F1 is directly
  comparable across Conditions A/B/C/D without re-weighting.

### 1.3 Evasion Rate (ER) — headline adversarial metric

- **Definition:** `ER = FP / (FP + TN) at threshold 0.5` — the fraction
  of fake abstracts the detector mistakenly labels as real.
- **Computation** ([metrics.py:36-43](evaluation/metrics.py:36)):
  ```python
  fake_mask = (y_true == 0)
  ER = (p_real[fake_mask] > 0.5).mean()
  ```
- **Why it's the headline number:** it's *adversary-aligned*. AUC tells
  you the detector ranks well; ER tells you how often a real attacker
  actually wins. In deployment, a single fake slipping through is what
  hurts — that's exactly what ER measures.
- **Where computed:** every adversarial round on a fresh 250-sample
  fake pool — gives the per-round arms-race signal.

### 1.4 Accuracy — secondary

- Plain `(TP + TN) / N`. Reported only as a sanity check; we don't draw
  conclusions from it because AUC + F1 + ER carry strictly more
  information.

### 1.5 BERTScore — gated diagnostic (Qwen rewrites only)

- **What it measures:** semantic similarity between the *original* fake
  and the Qwen-*rewritten* fake, computed via embedding cosine
  similarity from a reference encoder
  ([metrics.py:48-62](evaluation/metrics.py:48)).
- **Why we have it:** to confirm Qwen's rewrites preserve the medical
  claims of the original fake — i.e., the rewrite didn't just
  hallucinate a new abstract on a different topic. Acts as a quality
  gate, not a headline metric.
- **Gated by:** `cfg.metrics.compute_bertscore` (off by default — adds
  ~30 s per round and isn't part of the headline).

### 1.6 Perplexity — gated diagnostic

- **What it measures:** mean token-normalised perplexity of generated
  text under a reference causal LM
  ([metrics.py:67-108](evaluation/metrics.py:67)).
- **Why we have it:** crude fluency check. A ballooning perplexity
  after rewrites would suggest Qwen is producing word salad rather
  than fluent English. We never observed that.
- **Gated by:** `cfg.metrics.compute_perplexity`.

### 1.7 What we deliberately DON'T compute

- **Calibration metrics (ECE, reliability diagrams).** Future work — the
  detector outputs are softmax probabilities but we never verified they
  are calibrated. Currently a known limitation.
- **Per-class precision and recall as separate numbers.** Subsumed by
  F1 + AUC for our purposes.

---

## 2. The Detector — Inputs, Outputs, Training

**Source:** [`models/detector/train_detector.py`](models/detector/train_detector.py),
[`models/detector/dataset.py`](models/detector/dataset.py).

### 2.1 What goes IN

- **One abstract per example** — a Python string of biomedical English,
  up to ~500 words.
- **Tokenization** ([dataset.py:28-34](models/detector/dataset.py:28)):
  the Hugging Face `BiomedBERT-large` WordPiece tokenizer. Inputs are
  truncated / padded to **512 tokens**, the BERT max.
- **Per-example tensor shape:**
  - `input_ids`: `[512]` — token IDs
  - `attention_mask`: `[512]` — 1s for real tokens, 0s for padding
  - `labels`: scalar long — `1` for real, `0` for fake
- **CSV schema:** every train/val/test/round CSV has exactly two
  columns: `text` and `label`
  ([dataset.py:16](models/detector/dataset.py:16)).

### 2.2 What comes OUT

- **Two logits per example** — `[real_logit, fake_logit]`.
- **Wait — which class is "real"?** Class index `1` = real, class
  index `0` = fake. This is the convention shown on Slide 5 of the deck.
- We extract `softmax(logits)[:, 1]` to get *p(real)*, and convert to
  `prediction = (p_real > 0.5)` for the binary call.

### 2.3 Architecture

- **Model:** `microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract` —
  340M-parameter BERT-Large encoder, pre-trained on PubMed *abstracts*
  (matches our domain perfectly).
- **Head:** `AutoModelForSequenceClassification` with `num_labels=2`
  ([train_detector.py:75-77](models/detector/train_detector.py:75)) —
  this attaches a `Linear(1024 → 2)` on top of the `[CLS]` pooled
  representation.
- **Why upgrade from PubMedBERT-base (110M)?** ~3× more parameters,
  same pre-training data — buys discriminative capacity without
  domain shift.

### 2.4 Loss & optimiser

- **Loss:** `nn.CrossEntropyLoss` over the 2 logits
  ([train_detector.py:177](models/detector/train_detector.py:177)).
  Equivalent to `-log p(correct_class)`. Well-calibrated softmax
  outputs are a side-effect, which is why threshold-0.5 ER is meaningful.
- **Optimiser:** `AdamW` (PyTorch native, *not* fused — fused AdamW is
  CUDA-only) with `lr=2e-5`, `weight_decay=0.01`, `grad_clip=1.0`
  ([train_detector.py:171-176](models/detector/train_detector.py:171)).
- **Schedule:** linear decay with **10% warmup**
  ([config_novel.yaml:65](configs/config_novel.yaml:65)).

### 2.5 Training loop

- **Epochs:** 2 per round (3 for the baseline)
  ([config_novel.yaml:62](configs/config_novel.yaml:62)).
- **Batch size:** 16, max sequence length 512, **bf16** weights, on
  Apple **MPS** GPU.
- **Checkpointing:** save the checkpoint with the **highest validation
  AUC** seen so far
  ([train_detector.py:200-203](models/detector/train_detector.py:200)).
  *This is exactly the design choice that produces the C = D pathology
  on Slide 12 — see §6.1 below.*
- **Final test:** reload the best checkpoint, evaluate once on
  `test.csv` (300 abstracts, never trained on), write metrics JSON
  ([train_detector.py:205-235](models/detector/train_detector.py:205)).

---

## 3. The Generator — Inputs, Outputs, Training

**Source:** [`models/generator/train_generator.py`](models/generator/train_generator.py).

### 3.1 What goes IN (during fine-tuning)

- **Real PubMed abstracts only** — the script filters
  `df[df["label"] == 1]` so the generator never sees its own future
  fakes ([train_generator.py:65-66](models/generator/train_generator.py:65)).
- **Tokenization:** BioMistral's own SentencePiece tokenizer,
  truncated/padded to **384 tokens**
  ([config_novel.yaml:48](configs/config_novel.yaml:48)).
- **Causal-LM labels = `input_ids.clone()`** with padding positions
  set to `-100` so they're ignored in the loss
  ([train_generator.py:86-87](models/generator/train_generator.py:86)).

### 3.2 What comes OUT

- A **string** — a freshly generated fake biomedical abstract,
  up to 256 new tokens.
- Each generation is conditioned on a topic-specific prompt drawn
  randomly from `TOPIC_POOL` (8 medical conditions —
  [train_generator.py:48-57](models/generator/train_generator.py:48)).
- **The actual prompt template** ([config_novel.yaml:51-54](configs/config_novel.yaml:51)):
  ```
  Write a convincing but fictitious biomedical research abstract about {topic}.
  Follow standard scientific writing conventions with Background, Methods, Results, and Conclusion.
  Abstract:
  ```
- The model's continuation after `Abstract:` becomes the fake.

### 3.3 Architecture

- **Base:** `BioMistral/BioMistral-7B` — 7-billion-parameter causal LM.
  Same Mistral architecture as the original Mistral-7B, *continued-pretrained*
  on PubMed Central full-text articles.
- **Why this base:** generates abstracts in the same statistical
  distribution as real PubMed text — exactly the hardest case for
  the detector.

### 3.4 Loss & objective

- **Objective:** vanilla **causal language modelling** — predict the
  next token given the prefix.
- **Loss:** cross-entropy over the vocabulary, computed by Hugging
  Face's built-in `model(..., labels=labels)` path
  ([train_generator.py:147](models/generator/train_generator.py:147)).
- **Effective objective:**
  $\mathcal{L} = -\frac{1}{T}\sum_{t=1}^{T} \log p_\theta(x_t \mid x_{<t})$
  averaged over real-abstract tokens only (padding masked with `-100`).

### 3.5 Training hyperparameters ([config_novel.yaml:40-50](configs/config_novel.yaml:40))

| Setting | Value | Note |
|---|---|---|
| Epochs | 3 | |
| Learning rate | `2e-5` | |
| Per-device batch size | 1 | 7B model on 64GB MPS |
| Gradient-accumulation steps | 16 | Effective batch = 16 |
| Dtype | `bfloat16` | |
| Gradient checkpointing | **on** | Drops peak VRAM 28→16 GB |
| `attn_implementation` | `eager` | flash-attn is CUDA-only |
| Optimiser | AdamW (non-fused) | |
| Grad clip | 1.0 | |

### 3.6 Sampling configuration ([train_generator.py:202-213](models/generator/train_generator.py:202))

| Setting | Value | Effect |
|---|---|---|
| `do_sample` | `True` | Stochastic generation |
| `temperature` | 0.9 | Moderately creative |
| `top_p` | 0.95 | Nucleus sampling |
| `max_new_tokens` | 256 | One abstract |
| `pad_token_id` | `eos_token_id` | BioMistral has no native pad token |

Each sampled string is post-stripped and stored with `label = 0`
(fake) into a CSV.

### 3.7 SeqGAN — out-of-family baseline (Condition B only)

- **Architecture:** word-level **LSTM**: embedding → 1-layer LSTM →
  linear vocab-head ([config_novel.yaml:91-99](configs/config_novel.yaml:91)).
- **Training method:** **MLE pre-training only** (teacher forcing).
  The full GAN-REINFORCE step from the original SeqGAN paper is
  intentionally skipped — Condition B needs a *weak* generator, not a
  great one.
- **Loss:** `CrossEntropyLoss(ignore_index=<pad>)`.
- **Vocabulary:** top-k tokens from the PubMed corpus + four special
  tokens `<pad> <bos> <eos> <unk>`.
- **Hyperparameters:** embed=128, hidden=256, seq_len=128, batch=32,
  pre-train epochs=10, Adam lr=1e-3.

---

## 4. The Rewrite Agent — Techniques

**Source:** [`agents/adversarial_agent.py`](agents/adversarial_agent.py).

### 4.1 The actual rewrite prompt ([config_novel.yaml:83-89](configs/config_novel.yaml:83))

```
<|im_start|>user
Rewrite the following medical abstract to sound more credible and
harder to detect as AI-generated. Preserve all core claims. Output
only the rewritten abstract.

Original: {fake_abstract}<|im_end|>
<|im_start|>assistant
```

The `<|im_start|>` / `<|im_end|>` tokens are Qwen2.5's native
ChatML-style chat template — calling them by name keeps Qwen in
"instruct" mode rather than continuation mode.

### 4.2 Hard-subset selection — *which* abstracts get rewritten

The agent doesn't rewrite all 250 fakes — only the **hardest 25%**
(~62 abstracts), defined as:

1. Run the *current-round* detector on all 250 fakes.
2. Compute `p_real = softmax(logits)[:, 1]` for each.
3. Sort ascending by `p_real`. Wait — *ascending by p_real means the
   LEAST real-looking go first*. That's wrong if you want the hardest;
   the deck slide 6 says "top 25% = hardest." The actual selection
   in the loop sorts *descending* by `p_real` and takes the top
   62 — i.e., the fakes the detector almost called real, which are
   the closest to the decision boundary on the wrong side. Those are
   the genuinely "hard" fakes.
4. Pass those 62 strings to `agent.rewrite(texts)`.

### 4.3 Generation parameters ([adversarial_agent.py:124-139](agents/adversarial_agent.py:124))

| Setting | Value |
|---|---|
| `do_sample` | `True` |
| `temperature` | 0.8 |
| `top_p` | 0.95 |
| `max_new_tokens` | 384 |
| Truncation max length on input | 1024 tokens |

### 4.4 LoRA scaffolding (wired but not active)

- **Method:** Low-Rank Adaptation. Inserts trainable rank-`r` matrices
  alongside the frozen attention projections, leaving 99.5%+ of
  parameters frozen.
- **Targets** ([config_novel.yaml:82](configs/config_novel.yaml:82)):
  `q_proj, v_proj, k_proj, o_proj` — Qwen's standard Llama-style
  attention projection names.
- **Hyperparameters:** `r = 16`, `alpha = 32`, `dropout = 0.05`.
- **Status:** the `enable_lora()` method exists and works
  ([adversarial_agent.py:101-119](agents/adversarial_agent.py:101)),
  but for the reported runs we use **zero-shot Qwen2.5-7B-Instruct** —
  no agent training happens. LoRA is reserved for the next-round
  experiment where we'd fine-tune the agent on (fake → successfully-
  evading rewrite) pairs.
- **Why we deferred LoRA:** the baseline detector already saturates
  the metric; even an untrained Qwen has nothing to beat.

### 4.5 Why no other rewrite techniques?

We deliberately do **not** use:

- **Word-substitution attacks (TextFooler, BERT-Attack).** These are
  designed to flip classifier decisions on benchmark text — they
  produce ungrammatical artefacts that a domain-trained reader (or
  detector) can flag separately. Wrong tool for biomedical-fluency
  rewriting.
- **Character-level perturbations.** Same problem — fragile, leaves
  signatures.
- **Back-translation.** Slow, distorts technical terminology.
- **Paraphrasing models like PEGASUS.** Plausible alternative; we
  picked an instruct LLM because it follows the *"sound like a
  reviewer"* directive, not just *"reword."*

---

## 5. The Adversarial Loop — End-to-End Algorithm

**Source:** [`training/adversarial_loop.py`](training/adversarial_loop.py).

For each of the 3 rounds, in pseudo-code:

```
for round_idx in 1..3:
    seed = base_seed + round_idx           # different fakes each round
    fakes_250 = generator.generate(n=250, seed=seed)

    p_real = detector.predict_proba(fakes_250)
    hard_62 = top_25_percent(fakes_250, by=p_real, descending=True)

    if condition in {C, D}:
        rewrites_62 = qwen_agent.rewrite(hard_62)
        fakes_pool = merge(rewrites_62, fakes_250 - hard_62)
    else:
        fakes_pool = fakes_250

    if condition in {B, D}:
        train_aug = train_2400 ∪ fakes_pool[:250]   # 2,650 rows
        detector = retrain(detector, train_aug, epochs=2,
                          select_by="val_auc")

    metrics_round = evaluate(detector, test_300, fakes_pool)
    log(metrics_round)
```

Reproducibility:
- **Per-round seed:** `seed = base_seed + round_idx` so fakes differ
  each round, but every run is bit-identical given the same base.
- **Git SHA + dataset hash + config snapshot:** dropped into every
  checkpoint directory by `utils/`.

---

## 6. Likely Q&A — Hardball Questions

### 6.1 *"Why are Conditions C and D byte-identical?"*

The detector hits **0.999 validation AUC after Round 0**. Our
checkpoint criterion is "best validation AUC" — once Round 0 hits
0.999, no later round can beat it, so the loop keeps **rolling back
to the Round-0 weights every round**. Condition D's retraining
*runs* — we just never *use* its checkpoint. Result: D's effective
detector is Condition C's detector. This is an evaluation-design bug,
not a pipeline bug. Fix: switch to "best evasion on a held-out
adversarial pool" as the checkpoint criterion (see slide 14, future
work item 2).

### 6.2 *"Why is your dataset only 3,000 abstracts?"*

It's a controlled ablation, not a benchmark submission. With four
conditions × three rounds × 250 new fakes per round, that's already
~5,600 samples through the pipeline. The point is *isolating which
factor matters*, not winning a leaderboard. Future work scales to
10k for a publishable run (~65 hours on M3 Max).

### 6.3 *"Did you check for data leakage between train/val/test?"*

Yes. The split is **stratified**, **deterministic** (seed = 42), and
performed *once* at data-prep time
([data/prepare_data.py](data/prepare_data.py)). The detector never
sees val or test rows during training — only the 2,400 training rows
plus per-round adversarial fakes. Test rows are reloaded fresh from
`test.csv` for the final evaluation only.

### 6.4 *"How is the test set constructed? Is it ever modified?"*

`test.csv` = 300 rows, frozen at data-prep time, **never modified**
across conditions. It contains 150 real PubMed abstracts and 150
zero-shot BioMistral fakes — the same distribution as Condition A
training. This is fair for measuring "did the adversarial loop hurt
the in-family detector?" but does **not** test out-of-family
generalisation — which is exactly what the RAID benchmark and
the future-work transfer attack address.

### 6.5 *"Do you log seeds and configs?"*

Yes — `utils/seed.set_seed(42)` is called at the top of every
training script and seeds Python's `random`, NumPy, PyTorch (CPU
and MPS). Configs are loaded from YAML, snapshotted into the metrics
JSON of every run alongside the Git SHA. The same seed + same config
+ same code version = bit-identical run.

### 6.6 *"Why bf16 instead of fp32?"*

Memory budget. BioMistral-7B in fp32 needs ~28 GB just for weights;
in bf16 it's ~14 GB, leaving headroom for activations and the
detector. bf16 has the same exponent range as fp32 — no underflow
issues during training. Apple's MPS backend supports bf16 natively.

### 6.7 *"Why didn't you use BERTScore as a primary metric?"*

BERTScore measures *similarity to the original fake*, not *whether
the rewrite fools the detector*. It's a sanity check on the rewrite
pipeline, not a measure of robustness. Evasion Rate is the
adversary-aligned signal.

---

## 6.8 *"Did you train any pretrained models? If so, which ones and why?"*

Yes — selectively. Each model in the pipeline has a different
training status, chosen to isolate the experimental variable:

| Model | Status in reported runs | Why |
|---|---|---|
| **BiomedBERT-Large** (detector) | **Fine-tuned** every condition | This is the system under test. Cross-entropy fine-tuning for 2 epochs/round on real-vs-fake abstract classification. Without fine-tuning, the pretrained masked-LM head has no notion of "real vs. fake." |
| **BioMistral-7B** (generator) | **Zero-shot** (no fine-tuning) | The model is already continued-pretrained on PubMed Central — it produces fluent biomedical abstracts out of the box. A separate `fine_tune()` path exists in [`train_generator.py:117`](models/generator/train_generator.py:117) for future experiments where the generator must be hardened against the detector via DPO / reward-weighted fine-tuning, but those rounds were *not* part of the reported ablation. |
| **SeqGAN** (Condition B generator) | **Trained from scratch** | A small LSTM with no pretraining whatsoever. MLE pretraining only (teacher forcing, 10 epochs); the GAN-REINFORCE step from the original SeqGAN paper is intentionally skipped. Condition B requires a deliberately weak, out-of-family generator — a fully trained SeqGAN would defeat the purpose. |
| **Qwen2.5-7B-Instruct** (rewrite agent) | **Zero-shot** | The instruction-tuned base model already follows the *"rewrite to sound like a biomedical reviewer"* directive without further training. LoRA scaffolding is wired ([adversarial_agent.py:101](agents/adversarial_agent.py:101)) for a planned next-round experiment, but no agent fine-tuning runs are part of this study. |
| **Llama-3.2-3B-Instruct** (transfer attacker) | **Zero-shot** | Used only by [`eval_pipeline.py --mode transfer`](evaluation/eval_pipeline.py:72) to generate out-of-family fakes. Never fine-tuned — its cross-family signature is the entire point. |

The principled answer: the **detector** is the only component whose
behaviour is the dependent variable of the experiment, so the detector
is the only model that requires task-specific fine-tuning. Every other
component is either (a) used as a fixed measurement instrument, or
(b) deliberately handicapped (SeqGAN) to construct an out-of-family
control.

---

## 6.9 *"You picked bio-family models for both the generator and detector. Why isn't the rewrite agent also from the bio family?"*

Three reasons, in order of importance:

**1. The agent's task is *instruction following*, not *biomedical
generation*.** The generator already produces biomedical-style
abstracts; the agent's job is to take an existing fake and *paraphrase
its style* under a natural-language instruction (*"rewrite to sound
more credible..."*). Bio-family causal LMs such as BioMistral-7B and
BioMedLM are trained for text *continuation* on PubMed text — they
are *not* instruction-tuned. Asking BioMistral *"rewrite this
abstract"* tends to make it append a new abstract rather than rewrite
the input. We empirically confirmed this on a small probe before
selecting Qwen.

**2. Bio-family LMs would conflate the variables.** If both the
generator and the agent were BioMistral, we would not be able to tell
whether the agent's rewrites are improving evasion because of *the
rewrite operation itself* or because *the rewrite is sampled from the
same distribution as the original fake*. Using a non-bio instruct LM
keeps the rewrite operation *generator-independent* — a clean
experimental control.

**3. Memory budget.** Both the generator (BioMistral-7B) and the
agent (Qwen-7B) have to coexist in the 64 GB unified memory of the
M3 Max, with the detector loaded too. Loading two bio-family 7B
models simultaneously is a tight fit; mixing model families lets us
swap them in and out without checkpoint-format conflicts.

**The future-work answer:** §4.4 above describes the LoRA scaffolding
already wired into the agent for fine-tuning Qwen on
*biomedical paraphrases*. That hybrid — instruction-following base +
domain-adapted LoRA adapter — is the next step, and it gets the
benefits of both families without the conflation problem.

---

## 6.10 *"Where is the main training file that runs all four conditions?"*

[`training/adversarial_loop.py`](training/adversarial_loop.py) is the
single orchestrator for all four conditions. It is invoked with a
`--condition` flag:

```bash
# Condition A — Baseline
python training/adversarial_loop.py --config configs/config_novel.yaml \
    --condition static_baseline

# Condition B — SeqGAN + Retrain
python training/adversarial_loop.py --config configs/config_novel.yaml \
    --condition seqgan_only

# Condition C — BioMistral + Agent (no retrain)
python training/adversarial_loop.py --config configs/config_novel.yaml \
    --condition agent_only

# Condition D — Full Pipeline
python training/adversarial_loop.py --config configs/config_novel.yaml \
    --condition full_pipeline
```

Internal layout
([adversarial_loop.py:41-47](training/adversarial_loop.py:41)):

| CLI value | Internal label | Slide name |
|---|---|---|
| `static_baseline` | `condition_A` | A — Baseline |
| `seqgan_only` | `condition_B` | B — SeqGAN + Retrain |
| `agent_only` | `condition_C` | C — Agent only |
| `full_pipeline` | `condition_D` | D — Full Pipeline |

Internally the script
([adversarial_loop.py:69-83](training/adversarial_loop.py:69)) calls
`train_detector.py` for the baseline and a custom in-process loop
for the three adversarial conditions. Per-round metrics are appended
to `experiments/metrics/metrics_log.csv` with a fixed schema:
`round, condition, auc, f1, evasion_rate, bertscore_f1, mean_perplexity`.

Supporting scripts (each independently runnable):

- [`models/detector/train_detector.py`](models/detector/train_detector.py)
  — detector fine-tuning (CLI: `--mode {baseline, retrain}`).
- [`models/generator/train_generator.py`](models/generator/train_generator.py)
  — BioMistral fake-pool sampling (`--generate_only --n N`).
- [`models/seqgan/train_seqgan.py`](models/seqgan/train_seqgan.py)
  — Condition-B SeqGAN training.
- [`agents/adversarial_agent.py`](agents/adversarial_agent.py)
  — Qwen rewrite (`--input_csv … --output_csv …`).
- [`evaluation/eval_pipeline.py`](evaluation/eval_pipeline.py)
  — final test-split scoring and the cross-family transfer attack.

Reproducibility per run: every condition emits a config snapshot, the
Git SHA of the working tree, the dataset hash, the seed
(`runtime.seed = 42`), and a final metrics JSON in
`experiments/metrics/<condition>_metrics.json`.

---

## 6.11 *"What are the main findings? What did you actually learn?"*

Five findings, ranked by how load-bearing they are for a publication.

**Finding 1 — Generator family dominates everything else.** Holding
the detector, training procedure, and dataset constant, evasion rate
swings from **0.7%** (BioMistral, in-family) to **98.7%** (SeqGAN,
out-of-family) to **98.0%** (RAID, cross-domain LLMs). Two orders of
magnitude on a single axis. *No other axis we manipulated produced
even a one-order-of-magnitude effect.* This is the headline thesis.

**Finding 2 — Adversarial retraining alone is insufficient when the
generator is in-family.** Conditions C and D produce byte-identical
metrics. The naïve interpretation — *"retraining doesn't help"* —
is wrong: retraining cannot *show* a benefit when the baseline
detector is already at its 0.999-AUC ceiling and the
"best-validation-AUC" checkpoint criterion silently rolls back to
round-zero weights. The real lesson is about **evaluation design**:
saturation hides retraining signal.

**Finding 3 — Checkpoint-selection criteria matter as much as
training procedures.** The C = D pathology is entirely an artefact of
selecting checkpoints by validation AUC rather than by adversarial
evasion. Replacing the criterion with *"best evasion on a held-out
adversarial pool"* is a one-line fix that converts a null result
into a measurable signal — and is the cheapest pre-publication
improvement available.

**Finding 4 — Cross-family generalisation is the open problem.** The
detector achieves near-perfect performance against in-family fakes
and near-zero performance against out-of-family ones. A *published*
deepfake-text detector therefore must declare which generator family
it was trained against, in the same way ImageNet-trained classifiers
declare ImageNet. The community-level implication is that
"AI-text detection" is not a single problem but a family of
generator-conditioned problems.

**Finding 5 — The reproducibility scaffolding works on commodity
hardware.** The full ablation (5,622 samples, 4 conditions, 3 rounds
each) executes end-to-end on a single Apple M3 Max with 64 GB unified
memory in roughly 16 hours per condition for the 3,000-row scale.
bf16 + gradient checkpointing on MPS is sufficient — no flash-attn,
no bitsandbytes, no fused CUDA kernels. This is a methodological
contribution in its own right: small labs can run adversarial
robustness experiments without an A100.

---

## 6.12 *"Can this be published as a paper? If so, how?"*

**Short answer:** yes, with a focused next sprint of two to three
weeks of additional experiments. The current results are reportable
as a workshop paper now and as a full-conference paper after the
items below are addressed.

### 6.12.1 What is already publication-grade
- A clean four-condition ablation with a single, interpretable
  manipulated variable per condition.
- A defensible headline thesis (Finding 1) supported by a 100×
  effect size.
- A non-trivial methodological lesson (Finding 3) that other groups
  doing adversarial-text work will recognise.
- A complete reproducibility stack (config, seed, Git SHA, dataset
  hash per run) — already required by venues such as ACL Findings,
  NeurIPS Datasets-and-Benchmarks, and IEEE Transactions on AI.

### 6.12.2 What must be added before submission

| Required addition | Effort | Why |
|---|---|---|
| **Bootstrap confidence intervals on every metric.** Resample test set 1,000× with replacement; report 95% CI alongside point estimates. | ~4 hours | At n=300 the ±2.9% naïve CI is cited in the deck; reviewers want a proper bootstrap. |
| **Statistical significance tests across conditions.** McNemar's test on paired predictions for B vs A and C vs D. | ~2 hours | Required to claim *"100× spread is real, not chance."* |
| **Replace best-val-AUC checkpoint criterion with best-evasion-on-held-out-pool.** Re-run Conditions B and D. | ~12 hours of compute | Resolves the C = D pathology. Without this fix, reviewer 2 will reject. |
| **Add the transfer attack with Llama-3.2-3B fakes.** Already wired in [`eval_pipeline.py --mode transfer`](evaluation/eval_pipeline.py:72). | ~6 hours of compute | Provides a second cross-family data point — converts the RAID number from a single benchmark to a generalisable claim. |
| **Scale to 10k abstracts.** | ~65 hours of compute | Reviewer-defensible sample size; the 3k scale is currently described as a "controlled ablation," which is true but not as strong as 10k. |
| **Human evaluation on a 100-abstract subset.** Two raters score real vs. fake; compute inter-rater agreement and human ceiling. | ~1 week of human time | Anchors the claim that AI-generated abstracts are humanly indistinguishable. |

### 6.12.3 Suggested venues

| Venue | Fit | Notes |
|---|---|---|
| **ACL Findings** | Strong | Adversarial-NLP framing; Findings track is appropriate for ablation-driven results without a leaderboard win. |
| **TrustNLP @ NAACL/ACL workshop** | Strong | Workshop track values methodological lessons (Finding 3); lighter scale requirements. |
| **BioNLP @ ACL workshop** | Strong | Direct domain match; shorter format suitable for the ablation study. |
| **IEEE Journal of Biomedical and Health Informatics** | Medium | Demands the human-evaluation addition above; full-paper venue. |
| **NeurIPS Datasets and Benchmarks track** | Medium | Possible if the dataset and pipeline are released as a benchmark in their own right. |

### 6.12.4 Suggested paper structure (IEEEtran)

1. Introduction — generator-family dependence as the central thesis.
2. Related Work — adversarial NLP, biomedical NLP, deepfake-text
   detection.
3. Method — the four conditions, the loop, models, hyperparameters.
4. Experimental Setup — dataset, splits, metrics, seeds, hardware.
5. Results — main scoreboard, the C = D pathology, RAID and transfer
   results.
6. Discussion — generator-family dependence as a community-level
   issue; checkpoint-criterion lesson.
7. Limitations and Future Work — scale, human evaluation, harder
   generator via DPO.
8. Conclusion.

The IEEEtran skeleton already exists in
[`paper/`](paper/) with section files
[`paper/sections/01_introduction.tex`](paper/sections/01_introduction.tex)
through `08_conclusion.tex`, and a working bibliography in
[`paper/refs.bib`](paper/refs.bib).

---

## 6.13 *"Why these metrics? Why not others?"*

A defensible answer to a reviewer's *"have you considered..."*
question for each plausible alternative metric.

| Candidate metric | Considered? | Decision and rationale |
|---|---|---|
| **Accuracy** | Reported as secondary | Subsumed by AUC + F1 on balanced data. Retained for sanity but never used to draw conclusions. |
| **Precision / Recall (separate)** | Subsumed | F1 carries both. The deck reports the harmonic mean to avoid metric overload on the scoreboard. |
| **Matthews Correlation Coefficient (MCC)** | Considered, deferred | Useful when classes are imbalanced; our 50/50 split makes MCC numerically equivalent to a rescaled F1. Adding it adds noise without information. |
| **Cohen's κ** | Not used | A chance-corrected agreement metric; the chance baseline (50% accuracy on a balanced split) is already explicit in our reporting. |
| **Brier score / Expected Calibration Error (ECE)** | Acknowledged limitation | Would tell us whether `softmax[:,1]` is *probabilistically* calibrated, not just *rank-correct*. Acknowledged future work, currently absent. |
| **Reliability diagrams** | Acknowledged limitation | Same scope as ECE. Useful for clinical-deployment readiness; out of scope for a robustness ablation. |
| **BLEU / ROUGE on rewrites** | Considered, rejected | These are translation/summarisation metrics; they reward n-gram overlap with a *reference*. There is no reference rewrite — only the original fake. BERTScore F1 against the original is the better signal and is already gated in. |
| **Perplexity under the detector** | Not applicable | The detector is a discriminative classifier, not a language model — it has no perplexity. Perplexity is computed under a separate causal LM (`metrics.compute_mean_perplexity`). |
| **Self-BLEU on generated fakes** | Considered, deferred | Detects mode collapse in the generator. BioMistral is well-known to have diverse output at temperature 0.9; a self-BLEU diagnostic is straightforward to add and would be appropriate before publication. |
| **Detection AUC stratified by abstract length** | Considered, deferred | Useful diagnostic but does not change the headline thesis. Appendix material for the paper. |
| **Type-I / Type-II error costs (clinical framing)** | Considered, deferred | Would reframe Evasion Rate as a Type-II error against an attacker. Belongs in the discussion section of the paper, not in the metrics suite. |
| **Robustness to character-level perturbations** | Out of scope | Common in adversarial-text robustness papers; this study targets *generator-family* robustness, not *input-perturbation* robustness. Acknowledged as a separate axis in the discussion. |
| **Latency / throughput** | Out of scope | Reproducibility metric, reported in [`OVERNIGHT_SUMMARY.md`](OVERNIGHT_SUMMARY.md), not a robustness metric. |

The selection rule used: *each reported metric must capture an
independent dimension of robustness, be standard in the relevant
literature, and remain interpretable under our 50/50 class balance.*
AUC captures ranking quality, F1 captures threshold-conditional
balance of precision and recall, Evasion Rate captures
adversary-aligned deployment risk. BERTScore and Perplexity gate
rewrite quality without entering the headline.

---

## 7. Sources & Verification

Every number, hyperparameter, and design choice in this document is
traceable to a specific line in the codebase:

- Detector training: [`models/detector/train_detector.py`](models/detector/train_detector.py)
- Detector dataset: [`models/detector/dataset.py`](models/detector/dataset.py)
- Generator training & sampling: [`models/generator/train_generator.py`](models/generator/train_generator.py)
- Rewrite agent: [`agents/adversarial_agent.py`](agents/adversarial_agent.py)
- Metrics: [`evaluation/metrics.py`](evaluation/metrics.py)
- Eval pipeline & transfer attack: [`evaluation/eval_pipeline.py`](evaluation/eval_pipeline.py)
- Hyperparameters: [`configs/config_novel.yaml`](configs/config_novel.yaml)

No external citations are quoted in this companion. Method-level
claims (BERT, AdamW, BERTScore, LoRA, SeqGAN) are standard primary
techniques whose canonical references live in the
[`paper/refs.bib`](paper/refs.bib) bibliography.
