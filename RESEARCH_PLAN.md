# Research Plan: The Efficiency of Reason

**Subtitle:** Quantifying the Computational Sparsity of Chain-of-Thought in Large Language Models



## 1. The Core Research Question (The "Hook")

**Current Assumption:** Common intuition suggests that "Reasoning" (System 2 thinking) requires *more* energy and *more* neural activation than simple text generation.

**Our Hypothesis:** Reasoning is actually **computationally sparse**.

* **Hypothesis:** When a model enters a "Reasoning Trace" (inside `<think>` tags), it activates a highly specialized, narrow subset of attention heads (low Circuit Density), whereas "General Knowledge Retrieval" (e.g., reciting Wikipedia) activates a diffuse, messy web of neurons.

* **Why this matters:** If reasoning is distinct and sparse, we can isolate it. If we can isolate it, we can protect it from **Catastrophic Forgetting** during fine-tuning. This is a critical problem in Continual Learning (the focus of labs like Christopher Kanan's).



## 2. Methodology & Approach



### A. The Model

* **Primary:** `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`

* **Why:** It is State-of-the-Art (SOTA) for open-weights reasoning, small enough to run on one GPU, and crucially, **it outputs explicit `<think>` ... `</think>` tags**.

* **Advantage:** We no longer need heuristic windowing. We have Ground Truth labels for "Reasoning" vs. "Generation."



### B. The Metrics (The "REV" Toolkit)

We will repurpose the project's existing metrics to measure the structural differences between thinking and speaking:

1.  **CUD (Circuit Utilization Density):** What % of attention heads are active?

    * *Prediction:* Lower inside `<think>` tags (High Focus/Sparsity).

2.  **APE (Attention Process Entropy):** How "distributed" is the attention?

    * *Prediction:* Lower inside `<think>` tags (Structured flow).

3.  **AE (Activation Energy):** L2 Norm of hidden states.

    * *Prediction:* May be lower or more peaked during specific reasoning steps.



### C. The Dataset

1.  **Reasoning Task:** GSM8K (Grade School Math) – Requires step-by-step logic.

2.  **Knowledge Task:** Wikipedia / MMLU (General Facts) – Requires broad retrieval.

3.  **Control:** Random text / Nonsense.



---



## 3. The Experiments (Week-by-Week)



### Experiment 1: The Sparsity Gap (Weeks 1-3)

* **Goal:** Prove that `<think>` tokens look structurally different from `Response` tokens inside the GPU.

* **Procedure:**

    1.  Feed 100 GSM8K questions to DeepSeek-R1.

    2.  Parse the outputs into `Think_Segment` and `Answer_Segment`.

    3.  Compute CUD, APE, and REV scores for both segments separately.

* **Expected Result:** A visualization showing `Think_Segment` has significantly **lower** CUD (is sparser) than `Answer_Segment`.



### Experiment 2: Task Contrast (Weeks 4-6)

* **Goal:** Prove this isn't just a token artifact, but a task difference.

* **Procedure:**

    1.  Run the model on GSM8K (Math).

    2.  Run the model on a "Recite History" task (Knowledge).

    3.  Compare the internal metrics of the processing traces.

* **Expected Result:** "Math" traces utilize a smaller, more consistent sub-graph of the network than "History" traces.



### Experiment 3: The "Kanan" Validation (Weeks 7-9)

* **Goal:** Prove modularity for Continual Learning applications.

* **Procedure (Ablation):**

    1.  Identify the Top-1% of Attention Heads that are active during `<think>`.

    2.  **Zero-Ablate** these heads.

    3.  Test: Does the model fail at Math? (Yes).

    4.  Control: Does the model still speak fluent English? (Yes).

* **Significance:** Proves that reasoning circuits are distinct from language modeling circuits, suggesting they can be frozen or protected during fine-tuning.



---



## 4. Literature Review Priorities

* **Chain-of-Thought (Wei et al., 2022):** Foundation of reasoning traces.

* **Mechanistic Interpretability (Anthropic/Olah):** Theory of Induction Heads and Circuits.

* **Inference-Time Compute (OpenAI o1, DeepSeek R1):** The shift from training-time to test-time compute.

* **Sparse Autoencoders (Anthropic 2024):** The theoretical backing for "Sparsity" as a feature of intelligence.



## 5. Deliverables

* **Code:** A reproducible pipeline for `DeepSeek-R1` analysis (The `Weight_of_Reasoning` repo).

* **Paper:** "Internal Traces of System 2: Sparse Circuit Utilization in Reasoning Models."

* **Target:** NeurIPS 2025 Workshop or ArXiv Preprint.

