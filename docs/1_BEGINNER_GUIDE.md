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

The loop: generate fakes → detect them → rewrite the ones that almost fooled the detector → retrain the detector on the new fakes → repeat. This is called **adversarial training**. In principle each round the detector gets better because it's seeing harder examples. We'll see what actually happened.

---

## 3. What data did we use?

| Source | What it is | How many |
|---|---|---|
| **PubMed abstracts** (streamed from HuggingFace) | Real human-written biomedical abstracts | 1,500 |
| **BioMistral-7B zero-shot output** | We prompted an LLM to write fake abstracts on medical topics | 1,500 |

Split 80 / 10 / 10:
- **Train**: 2,400 abstracts (1,200 real + 1,200 fake) — what the detector learns from
- **Val**: 300 abstracts — used to pick the best model during training
- **Test**: 300 abstracts — final grade, never seen during training

---

## 4. The 4 experiments (ablation study)

We compared 4 setups to isolate what actually helps:

| Condition | Generator | Agent | Detector retrains? | Plain-English |
|---|---|---|---|---|
| **A** — static_baseline | BioMistral (zero-shot) | — | no | Just train the detector once. Baseline. |
| **B** — seqgan_only | SeqGAN (small neural network) | — | yes, every round | Adversarial retraining, no fancy rewriter. |
| **C** — agent_only | BioMistral | Qwen rewrite | no | Rewriter tries to fool the detector, but detector never retrains. |
| **D** — full_pipeline | BioMistral | Qwen rewrite | yes, every round | The full adversarial setup — both pieces working together. |

---

## 5. What did we find? (the 2-minute version)

1. **Condition A detector is already really good in-distribution.** AUC 0.9997, F1 0.98 — it catches nearly all BioMistral fakes out of the box. That's our paper-worthy baseline on n=300 test.

2. **Condition B was flat — adversarial retraining did not help against SeqGAN.** Evasion rate (fakes that fool the detector) stayed at **0.993 → 0.980 → 0.987** across 3 rounds. SeqGAN's token distribution is different enough from biomedical English that the detector can't learn it with 2 epochs of retraining. This is a clean negative result at 3k scale.

3. **Conditions C and D collapse onto the baseline.** BioMistral fakes are trivially caught (evasion ~0.007). With nothing slipping through, the retrain loop has no gradient — the checkpoint selector keeps rolling back to Condition A's weights. C and D produce byte-identical test metrics.

4. **The detector doesn't generalize outside biomedical text.** On RAID (cross-domain AI text from many LLMs), evasion jumped to 0.98 and accuracy dropped to 11%. **This is important for the paper**: BioShield is *biomedical-specialist*, not *universal*.

---

## 6. The headline finding

**"Our detector is generator-family specific."**

| Fake source | How often it evaded detection |
|---|---|
| SeqGAN (Condition B) | **99%** |
| BioMistral zero-shot (Conditions C, D) | **0.7%** |
| Mixed LLMs on RAID (cross-domain) | **98%** |

That's a **100× spread** in evasion across generator families. The family matters more than rounds or rewrites. Any published detector has to disclose *which* generator family it was trained to catch.

---

## 7. Why should anyone care?

- **Scientific reviewers**: could use a tool like BioShield to screen suspicious submissions from known generator families.
- **Research databases** (PubMed, CORD-19): can filter generated content at ingestion — but only against the generators they've trained against.
- **Health platforms**: block AI-generated "medical advice" before it reaches users — again, family-specific.
- **Honest finding for the paper**: adversarial retraining's ceiling depends almost entirely on how close the generator family is to the detector's domain. Out-of-family generators (SeqGAN, RAID's mixed LLMs) evade at 98–99% regardless of retraining.

---

## 8. How do I run it myself?

Two commands:

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
cd "NOVEL AI PROJECT"

# One autonomous script, ~10 hours end-to-end on M3 Max
bash scripts/run_presentation.sh
```

That script prepares data, runs all 4 conditions, generates plots, and runs cross-benchmarks. Everything is logged to `/tmp/bioshield_progress.log` while it runs.

For full environment setup see `README.md`. For data flow details see `docs/2_DATA_FLOW.md`. For evaluation details see `docs/3_EVALUATION.md`.
