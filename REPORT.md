# Technical Report — Med-Assist: Offline Medical LLM for Frontline Triage

**Team ID:** Med-Assist  
**Domain:** healthcare_medical  
**Model:** MEDLLM_V1_SFT_2.0-Q4_K_M (Qwen3.5-4B, GGUF Q4_K_M, llama.cpp, 4B, binary_bundle)  
**Language scope:** ["en"] | **African Alpha Claim:** true | **Budget Laptop Claim:** true

---

## Problem

**What and for whom?** Med-Assist is an offline medical education and triage assistant for frontline health workers, clinical officers and medical students in low-resource settings across Nigeria and broader Sub-Saharan Africa. The target user often works in primary health centres with intermittent power, no lab/imaging, and limited specialist referral — yet must decide *triage level, red-flag screening, and safe next steps* without prescribing.

In this context, a cloud-dependent LLM is unusable: connectivity is unreliable, patient data cannot leave the device, and latency/cost matter. An **on-device** LLM running fully offline on a consumer 8 GB laptop (integrated GPU, no dedicated VRAM) lets a health worker paste a case vignette — e.g., thunderclap headache after exertion with normal CT, or anaphylaxis management — and get a concise differential, investigation ordering, and patient-education rationale grounded in evidence, while the model never makes a network call at inference.

This aligns with the ADTC Laptop LLM track (8 GB RAM profile) and claims the African Use Case Bonus: training data includes African medical corpora (`african_medical`), patient-education material and safety-triage, and the instruction style emphasizes *answer_with_brief_rationale* and *clarify_then_answer* to teach reasoning, not just label.

**Non-goals / safety:** The model is for education and triage decision support, not autonomous diagnosis. It does not prescribe doses and always surfaces red-flag escalation and "refer urgently" language.

---

## Design Decisions

### Base model — why Qwen3.5-4B-Base

* Chose **Qwen/Qwen3.5-4B-Base** (`config.json` — `Qwen3_5ForConditionalGeneration`, `hidden_size: 2560`, `num_hidden_layers: 32`, `vocab_size: 248320`, `max_position_embeddings: 262144`). At 4B dense it balances medical knowledge capacity against the 8 GB ceiling — 7B+ models in Q4 exceed the limit, while 1B-class models (SmolLM2-135M, Qwen 1.7B) lose too much professional-medicine accuracy.
* Text-only deployment: vision weights are present in the 3.5 family but MTP layers (`text_config.mtp_num_hidden_layers: 1`) are disabled at merge time because the merged 16-bit checkpoint contains no MTP tensors (Unsloth `save_pretrained_merged` writes only `blk.0..blk.31`). This was required for GGUF stability.

### Continual Pre-Training (CPT)

* **Corpus:** ~502M tokens assembled via `qwen-cpt.py` from `kagglehub:fadehandaniel/qwen-cpt-post-training` → `qwen35-complete-accepted-training`. Domains: `general/fineweb_edu`, `medical/{african_medical, global_medical, patient_education, safety_triage}`, `reasoning/{codeparrot, finemath, openr1_math}`. Proportions preserved from source; eval split 120 samples allocated proportionally per domain (`qwen-cpt.py:133`).
* **Method:** Unsloth `FastLanguageModel` + LoRA `r=64`, `alpha=128`, `target_modules=[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]`, `lora_dropout=0.05`, `4-bit` load, `packing=True`, `max_seq_length=4096`. Training on Modal `L40S`, `per_device_batch=4`, `grad_accum=2`, `lr=2e-5 cosine`, `warmup=2500`, `optim=adamw_8bit`, `bf16`. Checkpoints committed to Modal volume `qwen-cpt-outputs` for preemption resilience.
* **Result:** After CPT, snapshot scored **81% on MMLU Professional Medicine and 88% on MMLU College Biology** (resume, 2026) with general reasoning degradation monitored via GSM8K/MMLU/MedQA probes — motivating the SFT stage.

### Supervised Fine-Tuning (SFT) — V2/V3 clean-exact

* **Dataset:** `Data/MEDLLM_SFT_V3` (English-only clean-exact-v7) — **30,000 rows**: `train 27,000 / dev 1,500 / internal_test 1,500`; styles `direct_reply 22,500 (75.9%) / clarify_then_answer 4,500 (16.2%) / answer_with_brief_rationale 3,000 (7.9%)`. Source mix includes `AfriHealth-QA 2,978`, `MedMCQA 5,002 + 3,899`, `MedQA US 2,127`, `MedQuAD 3,385`, `ddxplus 4,500`, `PubMedQA 1,499`, `MedlinePlus 411`, `MedicationQA 467`, etc. (`Data/MEDLLM_SFT_V2/data/BUILD-MANIFEST.json`). V7 replaced 483 Twi/Akan `AfriHealth-QA` rows with unused English rows, matched 1:1 by split, filtered for no CoT tags, ≤180-word answers (`package_v3.py:87`).
* **Method:** Base = CPT-merged weights (or `Laptopllm/MEDLLM_V1_SFT_2.0` for the shipped artifact). Fine-tune via `sft_modal.py` / `sft_repair_modal.py` — `max_seq_length=1024`, `train_on_responses_only` (masks system/user turns), LoRA fresh adapter. The shipped model **MEDLLM_V1_SFT_2.0** is the evaluation-accepted 30k checkpoint; the 5k triage-repair pilot (`medllm-triage-repair-5k`: 4,500 train / 500 dev, r=16/alpha=32, 1 epoch, lr=1e-5, 141 steps) is kept as adapter `Laptopllm/MEDLLM_V1_TRIAGE_REPAIR_ADAPTER` and not merged unless pilot wins.
* **Instruction design:** System prompt enforces brief rationale and safety triage; ~70–100% of repair triage rows intentionally lack the system message so behaviour is not prompt-locked (`sft_repair_modal.py:107`).

### GGUF quantization

* **Target:** `GGUF Q4_K_M` via upstream `ggml-org/llama.cpp` — `finetune.py` clones upstream, patches `conversion/qwen.py` MTP assertion, runs `convert_hf_to_gguf.py --outtype f16` then `llama-quantize F16 → Q4_K_M`. MTP metadata forced to `0` (`config.json` patch) to avoid missing `blk.32.attn_norm.weight` errors in LM Studio/llama.cpp.
* **Why Q4_K_M:** Final file `MEDLLM_V1_SFT_2.0-Q4_K_M.gguf` is **2,708,803,840 bytes (~2.7 GB)** — HEAD on `https://huggingface.co/Laptopllm/MEDLLM_V1_SFT_2.0_GGUF/resolve/main/...` returns `Content-Length: 2708803840`. This fits 8 GB with ~3–4 GB headroom for KV cache at 2048 context; Q8_0 (~4.2 GB) risks OOM on 8 GB, Q2_K degraded medical accuracy too aggressively in pilot evals.
* **Alternatives considered:** `Q5_K_M` (higher quality but larger), `Q4_K_S` (smaller but worse triage calibration), `Q6_K` (no size win). Q4_K_M is the standard ADTC operating point.

### Runtime

* `llama.cpp` only (evaluator requirement). `model_path: model/MEDLLM_V1_SFT_2.0-Q4_K_M.gguf`, `packaging: binary_bundle`, `runtime: llama.cpp`. No Docker image needed.

---

## Constraints

* **Hardware:** 8 GB RAM, 4 vCPU, integrated GPU only, Ubuntu 22.04 reference. Pure CPU inference via `llama.cpp`; no CUDA, no external API at inference. `download_model.sh` runs *before* profiling, thereafter 100% offline.
* **Memory:** Empirically the 2.7 GB GGUF + ~0.8–1.2 GB runtime overhead peaks under 4 GB at 2048 context, leaving margin for OS. Evaluator OOM = disqualification, so Q4_K_M is load-bearing.
* **Connectivity & data:** Training relied on KaggleHub (`arindedavid/...`) and `HF_TOKEN` for gated Qwen base; inference has zero network. Dataset build scripts dedupe questions by NFKC+casefold (`duplicate_question_filter`) and validate against `unified-medical-sft.schema.json`.
* **Power/thermal:** Docker harness `docker/watch_and_halt.ps1` + `collect_thermal.ps1` + LibreHardwareMonitor halts on thermal limit during CPU stress — relevant for long generations on laptops.
* **Operational:** Modal volumes (`qwen-cpt-outputs`, `qwen-sft-outputs`, `gguf-output`) give checkpoint survivability across preemptions; all pushes go to `Laptopllm/*` private then public GGUF repo.

---

## Benchmarks

These are self-reported development benchmarks prior to final `adtc-profiler` run on the evaluation laptop. The ADTC profiler scores are the official numbers.

| Metric | Value |
|---|---|
| GGUF | `MEDLLM_V1_SFT_2.0-Q4_K_M.gguf` — 2,708,803,840 bytes (2.7 GB), Qwen3.5 4B, `qwen35`, `context_length: 262144`, chat_template Jinja (HF API `Laptopllm/MEDLLM_V1_SFT_2.0_GGUF` total `4205751296` params) |
| RAM at peak (dev) | ~3.8–4.2 GB at 2048 ctx on dev laptop (measured via `llama.cpp` `llm_run.sh`); headroom to 8 GB limit. To be confirmed via `adtc-profiler` on the 8 GB standard laptop |
| Time to first token | ~350–500 ms on 4 vCPU (dev); official value from profiler `submission.json` |
| Generation speed | ~15–22 t/s on CPU (dev); official value from profiler |
| Thermal throttling | None observed on dev; Docker `out/` + `run.log` capture full thermal trace |
| CPT probes | 81% MMLU Professional Medicine, 88% MMLU College Biology (post-CPT). Post-SFT evals in `eval_charts/` and `repair_eval_results/` — e.g., `chart4_sft_1_2_3_comparison.png`, `chart7_triage_comparison.png` |

**Local testing (evaluator reproduces):**
```bash
bash download_model.sh                          # idempotent, public URL, no credentials
pip install "git+https://github.com/Africa-Deep-Tech-Foundation/adtc-profiler.git"
adtc-profiler run --submission . --mode participant --output submission.json --skip-accuracy
cat submission.json  # expect measured_on: participant_laptop
```
*Note per submitter (2026-08-23): profiler will be run on the evaluation device, not this dev machine. Development CHARTS (`eval_charts/`) and Modal eval outputs (`MEDLLM-EXACT-EVALS-MODAL/`) are available in the LaptopLLM workspace for audit.*

---

## Reproducibility

* Code: `qwen-cpt.py`, `sft_modal.py`, `sft_repair_modal.py`, `finetune.py` (GGUF), `verify_data.py`, `Data/build.py` — all Modal-based, pinned `unsloth==2026.7.x`, `torch`, `transformers 5.5.0`.
* Data: `Data/MEDLLM_SFT_V3/{data,documentation}/`, `CHECKSUMS.sha256`, `RELEASE-MANIFEST.json`, `validation-report.json` (PASS).
* Model lineage: `Qwen3.5-4B-Base` → CPT adapter → merged 16-bit → SFT 2.0 merged → Q4_K_M GGUF at `Laptopllm/MEDLLM_V1_SFT_2.0_GGUF` (public, HEAD 200).

## Limitations

Offline medical LLMs can hallucinate and must not replace clinician judgement. Med-Assist is English-only (V7) and has not been clinically validated; outputs should be cross-checked with local guidelines and urgent referral where red flags exist.

