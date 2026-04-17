# BioShield — Beginner's Guide

> If you've never touched this project, read this first. No jargon. 10 minutes.

---

## 1. What problem are we solving?

Language models like ChatGPT can write **medical research abstracts that look real but are fake**. That's dangerous — fake abstracts can spread health misinformation, pollute scientific databases, and be used in research fraud.

**BioShield builds a tool that reads a biomedical abstract and tells you: "human-written" or "AI-generated."**

Think of it like a spam filter, but for fake medical writing.

---

## 2. What did we actually build?

Three pieces talking to each other:

```
┌──────────────────┐    writes fake    ┌──────────────────┐   tries to improve   ┌──────────────────┐
│  GENERATOR       │ ────abstracts──>  │  DETECTOR        │ ────its accuracy──>  │  AGENT           │
│  (BioMistral-7B) │                   │  (BiomedBERT)    │                      │  (Qwen2.5-7B)    │
│  a biomedical    │                   │  a binary        │                      │  a rewriter that │
│  LLM that writes │                   │  classifier:     │                      │  edits fakes to  │
│  convincing      │                   │  real vs. fake   │                      │  sound MORE real │
│  fake abstracts  │                   │  abstract        │                      │                  │
└──────────────────┘                   └──────────────────┘                      └──────────────────┘
```

- **Generator** — makes fake abstracts. Used as the "bad guy."
- **Detector** — the thing we actually ship. Tries to catch fakes.
- **Agent** — a "coach" that rewrites fakes to look more real, forcing the detector to get smarter.

The loop: generate fakes → detect them → rewrite the ones that almost fooled the detector → retrain the detector on the new fakes → repeat. This is called **adversarial training**. Each round the detector gets better because it's seeing harder examples.

---

## 3. What data did we use?

| Source | What it is | How many |
|---|---|---|
| **PubMed abstracts** (streamed from HuggingFace) | Real human-written biomedical abstracts | 500 |
| **BioMistral-7B zero-shot output** | We prompted an LLM to write fake abstracts on medical topics | 500 |

Split 80 / 10 / 10:
- **Train**: 800 abstracts (400 real + 400 fake) — what the detector learns from
- **Val**: 100 abstracts — used to pick the best model during training
- **Test**: 100 abstracts — final grade, never seen during training

---

## 4. The 4 experiments (ablation study)

We compared 4 setups to isolate what actually helps:

| Condition | Generator | Agent | Detector retrains? | Plain-English |
|---|---|---|---|---|
| **A** — static_baseline | BioMistral (zero-shot) | — | no | Just train the detector once. Baseline. |
| **B** — seqgan_only | SeqGAN (small neural network) | — | yes, every round | Adversarial retraining, no fancy rewriter. |
| **C** — agent_only | BioMistral | Qwen rewrite | no | Rewriter tries to fool the detector, but detector never retrains. |
| **D** — full_pipeline | SeqGAN + BioMistral | Qwen rewrite | yes, every round | The full adversarial setup — both pieces working together. |

---

## 5. What did we find? (the 2-minute version)

1. **Condition A detector is already really good.** AUC 0.989, F1 0.94 — it catches almost all BioMistral fakes out of the box. That's our paper-worthy baseline.

2. **Condition B proves adversarial training works.** Evasion rate (fakes that fool the detector) drops **0.74 → 0.64 → 0.54** across 3 rounds. The detector genuinely learns from seeing fakes it previously missed.

3. **Conditions C and D show BioMistral fakes are easy.** Even without an agent, the detector catches nearly all of them (evasion ~0.02). The rewriter doesn't help here because the fakes are already being detected — there's no "almost fooled it" pile to sharpen against.

4. **The detector doesn't generalize outside biomedical text.** When we ran it on the RAID benchmark (general web-scraped text from many LLMs), accuracy dropped to 13% and evasion jumped to 96%. **This is important for the paper**: our detector is *biomedical-specialist*, not *universal*.

---

## 6. Why should anyone care?

- **Scientific reviewers**: could use a tool like BioShield to screen suspicious submissions.
- **Research databases** (PubMed, CORD-19): can filter generated content at ingestion time.
- **Health platforms**: block AI-generated "medical advice" before it reaches users.
- **Honest finding for the paper**: adversarial training provides measurable improvement **when the raw fakes are actually hard to detect** (SeqGAN case). When fakes are too easy (zero-shot LLM), the adversarial loop has nothing to optimize against.

---

## 7. How do I run it myself?

Two commands:

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
cd "NOVEL AI PROJECT"

# One autonomous script, 6-7 hours end-to-end on M3 Max
bash scripts/run_presentation.sh
```

That script prepares data, runs all 4 conditions, generates plots, and runs cross-benchmarks. Everything is logged to `/tmp/bioshield_progress.log` while it runs.

For full environment setup see `README.md`. For data flow details see `docs/2_DATA_FLOW.md`. For evaluation details see `docs/3_EVALUATION.md`.
