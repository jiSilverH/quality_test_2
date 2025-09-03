# Section 11. Fine Tuning LLM

# **Fine-Tuning LLMs on Custom Datasets – Study Notes**

### **1. Context**

- Previous fine-tuning: small/medium Transformer models (e.g., BERT – 110M, DistilBERT – 67M).
- Now: **Large Language Models (LLMs)** with **billions of parameters** (e.g., Microsoft **Phi-3**).
- Traditional fine-tuning = **computationally expensive** → needs new approaches.

---

### **2. Why Fine-Tune LLMs?**

- Pre-trained LLMs are generic.
    - Custom fine-tuning improves relevance, reduces hallucination, and boosts performance.
- Fine-tuning aligns them to **specific tasks/domains** (chat, text generation, classification).

---

### **3. Challenges with Large Models**

- Phi-3 Mini = **3.8B parameters**; Phi-3 Vision = **4.2B**.
- Loading weights in memory (FP32, 4 bytes/param) = ~16 GB RAM needed just for weights.
- With FP16 → ~8 GB RAM, but training needs **much more** (gradients, activations, optimizer states).
- Fine-tuning traditionally **impractical** on local machines.

---

### **4. Parameter-Efficient Fine-Tuning (PEFT)**

- Techniques: **Adapters, LoRA, QLoRA**.
- Reduce compute & memory by updating only a **small subset** of parameters.
- Quantization (8-bit/4-bit) → fits LLMs on consumer GPUs / laptops.

---

### **5. About Phi-3 Model (Microsoft)**

- Successor of Phi-2, comparable to GPT-3.5 & Mistral 7B.
- **Trained on 3.3 trillion tokens** (very large data).
- Rule of thumb: need ≥20× tokens than parameters.
    - For 4B params → ≥80B tokens required; Phi-3 used 3.3T → excellent scaling.
- **Context length**:
    - 128K (long-context version).
    - 4K (smaller version).
- **Vocabulary size**: ~32K (same as standard BPE tokenizers in BERT/T5).
- **Architecture**:
    - 3072 hidden size.
    - 32 heads.
    - 32 layers (encoder blocks).
    - FP16 training → more memory efficient.

---

### **6. Phi-3 Vision**

- Extension of Phi-3 for multimodal tasks.
- ~4.2B parameters.
- Trained for both **text + vision** inputs.

---

### **7. Key Takeaways**

- **LLMs are massive** → direct fine-tuning infeasible without PEFT.
- **Phi-3** demonstrates high performance with efficient scaling and dataset design.
- Fine-tuning requires **quantization + PEFT (LoRA, QLoRA, adapters)**.
- Pre-training dataset quality & size (trillions of tokens) = main reason for Phi-3’s success.
- Next practical step: **train Phi-3 on custom dataset** in either *text generation* or *chat format*.

# 📘 Paper Review: Parameter-Efficient Transfer Learning (2019)

### **1. Context**

- Published **2019** (pre-GPT-3/phi family era).
- Focused on **medium models** (BERT, GPT-1, T5, BART).
- Problem: **Full fine-tuning** = train **all parameters** (e.g., BERT-large = 110M).
    - Expensive in time + GPU memory.
- Proposed solution: **Parameter-Efficient Transfer Learning (PETL)** → fine-tune **small subsets only**.

---

### **2. Core Idea**

- Instead of updating all weights, **freeze most parameters**.
- Add **lightweight components (adapters)** inside transformer layers.
- Only fine-tune those adapter parameters (few million vs. hundreds of millions).

---

### **3. Key Results**

- **BERT-large full fine-tuning**: 110M trainable parameters.
- **Adapter fine-tuning**: ~3.6M trainable parameters (≈3.6%).
- Performance: almost identical to full fine-tuning (≤0.4% accuracy gap on GLUE benchmark).
- Major savings: speed, memory, and feasibility.

---

### **4. Categories of PETL**

1. **Additive Fine-Tuning (Adapters)**
    - Insert small trainable MLP layers inside each Transformer block.
    - Only these are updated.
    - Trade-off: Slight **latency increase at inference** (due to added layers).
2. **Reparameterized Fine-Tuning (LoRA)**
    - Replace some weight matrices with **low-rank decompositions**.
    - Train only the low-rank factors.
    - More efficient → **no inference latency increase**.
3. **Partial Fine-Tuning**
    - Freeze most layers, train only selected ones (e.g., top layer, or a few encoder blocks).
4. **Hybrid Fine-Tuning**
    - Combine different approaches (e.g., adapters + partial fine-tuning).

---

### **5. Adapters – How They Work**

- Transformer encoder block structure:
    
    ```
    Multi-head Attention → Add & Norm → Feedforward → Add & Norm
    
    ```
    
- Adapters are **inserted between feedforward and Add & Norm**.
- They are small bottleneck networks:
    - Down-projection (reduce dimension).
    - Non-linearity.
    - Up-projection (expand back).
- Only adapter weights are updated during training.

---

### **6. Strengths & Weaknesses**

✅ **Strengths**:

- Massive reduction in trainable parameters.
- Comparable accuracy to full fine-tuning.
- Lower training cost → feasible on limited hardware.

⚠️ **Weaknesses**:

- **Added inference latency** (due to adapter layers).
- Memory footprint still grows slightly (~3–5%).
- Later methods (LoRA, QLoRA) improve efficiency further.

---

- 2019: GPUs were weaker, fine-tuning BERT was already expensive.

### **7. Historical Significance**

- Adapters made fine-tuning **practical for real-world use**.
- Paved the way for **LoRA (2021)** and **QLoRA (2023)**, which are now the **state-of-the-art** in PEFT.

---

✅ **Exam Tip:**

- Remember:
    - **Full fine-tuning** = all parameters.
    - **Adapters (2019)** = add small modules, fine-tune only them (3–4% params).
    - **LoRA (2021)** = reparameterization, no added latency.
    - **QLoRA (2023)** = quantized LoRA, even more efficient.

# 📘 Low-Rank Adaptation (LoRA, 2021)

### **1. Context**

- Published in **2021**, when **GPT-3 (175B parameters)** was already out.
- Challenge:
    - Full fine-tuning impossible for startups/smaller labs (hundreds of GBs of memory).
    - Adapter-based methods (2019) reduced training cost but:
        - **Added 3–4% more parameters** → model grows larger.
        - **Extra inference latency** (slower serving).

---

### **2. Core Idea**

- **LoRA = Reparameterization technique**.
- Instead of **adding new modules** (adapters), it **decomposes existing weight matrices** into **low-rank matrices**.
- Only the small low-rank matrices are trained.
- At inference, they are **merged back into original weights**, so **no extra latency**.

---

### **3. How LoRA Works**

- A large weight matrix **W₀ ∈ ℝᵈˣᵏ** is frozen.
- Train two small matrices:
    - **A ∈ ℝᵈˣʳ**
    - **B ∈ ℝʳˣᵏ**
    - where **r ≪ min(d,k)** (low rank).
- During training:
    
    ```
    W = W₀ + BA
    
    ```
    
    - `BA` approximates the update to `W₀`.
- At inference: `BA` is merged into `W₀` → no runtime overhead.

---

### **4. Example of Parameter Savings**

- Suppose `W₀` is **10×10** (100 params).
- LoRA with **rank = 1**:
    - `A` = 10×1 (10 params)
    - `B` = 1×10 (10 params)
    - Total trainable = 20 params (vs 100).
- General result: **10,000× fewer trainable parameters possible**.

---

### **5. Advantages over Adapters**

| Feature | Adapters (2019) | LoRA (2021) |
| --- | --- | --- |
| Trainable params | ~3–4% extra | ~0.01–0.1% (depending on rank) |
| Model size | Increases (extra modules) | No increase |
| Inference latency | **Slower** (extra layers) | **No change** |
| Memory efficiency | Better than full FT, but still grows | Very efficient |
| Accuracy | Near full fine-tuning | Near/full fine-tuning |

---

### **6. Significance**

- Made **fine-tuning GPT-3-scale models feasible** on commodity hardware.
- Became the **default standard** for PEFT (Parameter-Efficient Fine-Tuning).
- Widely adopted in industry (HuggingFace, OpenAI, etc.).

---

✅ **Exam Tip:**

- *Adapters = additive → more params + latency.*
- *LoRA = reparameterization → same params, no latency.*
- Think: **“Adapters add, LoRA reuses.”**

# 📘 QLoRA (Dettmers et al., 2023)

### **1. Context & Motivation**

- After **LoRA (2021)**, fine-tuning became efficient but still required **FP16 weights** → high memory footprint.
- Example: GPT-3 (175B params) in FP16 → **350 GB just to load**.
- Even LoRA fine-tuning required **large GPUs** → not feasible for most labs.
- **QLoRA = LoRA + Quantization** → reduces memory drastically while maintaining accuracy.

---

### **2. Core Idea**

- Keep LoRA’s low-rank adaptation.
- Add **quantization of base model weights** (before fine-tuning).
- Fine-tune in **4-bit (NF4)** instead of 16-bit.
- This allows fitting models like **65B LLaMA** on a **single 48–64 GB GPU**.

---

### **3. Technical Innovations**

1. **NF4 (NormalFloat-4) Quantization**
    - Custom 4-bit data type, optimized for normally distributed weights.
    - Allocates **more resolution where weights are dense**, fewer bits where sparse → less reconstruction error.
2. **Double Quantization**
    - Quantizes not just weights but also **quantization constants**.
    - Cuts another **10% memory** without accuracy loss.
3. **Paged Optimizers**
    - GPU memory spikes during backpropagation → often crash.
    - QLoRA offloads optimizer states to **CPU memory in pages**, preventing OOM.

---

### **4. Memory Efficiency**

- Example: **LLaMA-65B**
    - Standard fine-tuning: **780 GB GPU RAM**.
    - LoRA fine-tuning: still hundreds of GB.
    - **QLoRA: ~64 GB GPU RAM** → fits on one A100 GPU.
- Practical impact: **consumer-level fine-tuning** possible.

---

### **5. Performance**

- Accuracy:
    - **QLoRA 4-bit ≈ FP16 full fine-tuning** (within 0.3–0.7%).
    - On benchmarks, **99.3% of full FT performance**.
- Speed:
    - Slightly slower than LoRA due to quantization overhead.
    - But huge tradeoff: **accessibility > speed**.

---

### **6. Comparison (Adapters → LoRA → QLoRA)**

| Aspect | Adapters (2019) | LoRA (2021) | QLoRA (2023) |
| --- | --- | --- | --- |
| Parameters trained | ~3–4% extra | ~0.01–0.1% | ~0.01–0.1% |
| Memory (65B model) | TB-scale | ~100s of GB | ~64 GB |
| Inference size | Larger | Same as base | Same as base |
| Accuracy | Near full FT | Near full FT | Near full FT (≥99%) |
| Extra latency | Yes | No | No |
| Key trick | Add modules | Low-rank updates | Quantized base + LoRA |

---

### **7. Significance**

- **Shift from “lab-only” to “personal workstation” fine-tuning**.
- Popularized in HuggingFace ecosystem → widely adopted for **LLaMA-2, Falcon, Mistral**.
- Today: QLoRA is the **default PEFT method** for LLMs.

---

✅ **Exam Tip:**

- *LoRA = low-rank update, saves training compute.*
- *QLoRA = LoRA + quantization, saves memory → fits huge models on a single GPU.*

# 📘 Fine-Tuning Pipeline of LLMs

### **1. Three Main Stages**

1. **Pre-Training (Language Modeling)**
    - Also called **LM objective**.
    - Self-supervised → no explicit labels needed.
    - Train on huge dataset (e.g., *Phi-3 used ~4.8T tokens*).
    - Techniques:
        - **Causal LM (autoregressive)** → predict next token using right-shifted input. (Used by GPT models).
        - **Masked LM (MLM)** → randomly mask tokens and predict them (used in BERT).
        - **Seq2Seq LM** → encoder–decoder setup (used in T5).

---

1. **Supervised Fine-Tuning (SFT)**
    - Requires **labeled pairs** (prompt → response).
    - Adapts pre-trained LLM to specific downstream tasks (chat, QA, summarization, domain tasks).
    - Techniques:
        - **PEFT methods** like **Adapters, LoRA, QLoRA**.
        - Train only a *tiny fraction* of parameters → efficient & feasible on smaller GPUs.
    - This is where we’ll apply **adaptive LoRA / QLoRA**.

---

1. **Preference Fine-Tuning (Alignment)**
    - Also known as **RLHF** (Reinforcement Learning with Human Feedback).
    - Shapes model behavior according to **human preferences** (safety, helpfulness, alignment).
    - Process:
        - Collect human rankings on model outputs.
        - Train a **reward model**.
        - Optimize LLM with **RL (e.g., PPO)** against that reward.
    - Alternative: **DPO (Direct Preference Optimization)** → more stable, simpler.

---

### **2. Other Lightweight Tuning Methods**

- **Zero-Shot / Few-Shot Prompting**
    - No parameter updates → rely on clever prompt design.
- **Prompt Tuning / Prefix Tuning / P-Tuning**
    - Only optimize **prompt embeddings** while freezing the model.
    - Cheaper but less powerful than SFT.

---

### **3. Key Takeaway Workflow**

```
Pre-Training (self-supervised, trillions of tokens)
        ↓
Supervised Fine-Tuning (labeled data, PEFT like LoRA/QLoRA)
        ↓
Preference Fine-Tuning (RLHF / DPO for alignment & safety)

```

---

✅ **Exam tip**:

- *Pre-training teaches the language.*
- *SFT teaches the task.*
- *Preference fine-tuning teaches values & alignment.*

# 📘 Supervised Fine-Tuning (SFT)

### **1. Core Idea**

- Pretrained models (e.g., GPT, Phi-3) are *generalists* trained on **huge unlabeled corpora** (Wikipedia, CommonCrawl, books, etc.).
- They learn to predict the *next token* but don’t know how to follow instructions or perform domain-specific tasks.
- **SFT adapts the pretrained model to specific tasks using smaller, labeled datasets.**

---

### **2. How It Works**

- **Data**:
    - Input = text (e.g., review, question, document).
    - Label = target (e.g., sentiment, answer, summary).
- **Process**:
    1. Tokenize input and labels.
    2. Pass through model → predict label tokens.
    3. Optimize loss between predicted vs. actual label tokens.

👉 Unlike pretraining (self-supervised, next-token prediction), SFT is **supervised** (explicit labels provided).

---

### **3. Two Modes of SFT**

1. **Instruction Fine-Tuning (Instruction → Response)**
    - Data is formatted as:
        
        ```
        Instruction: Summarize the text
        Input: "The cat climbed the tree..."
        Output: "A cat went up a tree."
        
        ```
        
    - Makes the model behave like a helpful assistant.
    - Enables **multi-task learning**: QA, summarization, translation, sentiment, etc.
2. **Open-ended Text Generation Fine-Tuning**
    - Focused on domain-specific text continuation.
    - Example: Legal contracts, medical notes, research papers.
    - Adapts LLMs to **specialized vocabularies** that aren’t well-covered in internet-scale pretraining.

---

### **4. Why SFT is Necessary**

- **Pretrained Model Limitation**:
    
    Example → if you ask base GPT “What is 1+1?”, it may continue with nonsense (“What is 1+1+1...”), because it’s not *instructed*.
    
- **SFT adds structure**: teaches the model *how to follow instructions* and *produce answers* instead of endless text.

---

### **5. Efficiency via PEFT**

- Full fine-tuning of billions of parameters is expensive.
- **Parameter-Efficient Fine-Tuning (PEFT)** methods train only small subsets:
    - **Adapters**
    - **LoRA (Low-Rank Adaptation)**
    - **QLoRA (Quantized LoRA, even cheaper)**
    - **Adaptive LoRA (dynamic rank allocation)**
- Greatly reduces cost while retaining performance.

---

### **6. Key Contrast: Pretraining vs. SFT**

| **Aspect** | **Pre-Training** | **Supervised Fine-Tuning** |
| --- | --- | --- |
| Data | Unlabeled (web-scale) | Labeled (task-specific) |
| Objective | Next-token prediction (self-supervised) | Input → Output mapping (supervised) |
| Scale | Trillions of tokens | Millions or fewer examples |
| Cost | Extremely high (thousands of GPUs) | Moderate (consumer-level GPUs with PEFT) |
| Purpose | Learn general language | Specialize to tasks, domains, or instructions |

---

✅ **Exam Tip**:

- Think of **Pretraining = general language learning**,
- **SFT = teaching the model tasks**,
- **Preference Fine-Tuning (RLHF/DPO) = teaching values & alignment.**

# 📘 Q-LoRA Quantization: Key Concepts

### **1. What is Quantization?**

- Pretrained model weights are usually in **32-bit (FP32)** or **16-bit (FP16)** precision.
- **Quantization** reduces this to **8-bit** or even **4-bit**, drastically lowering memory and compute requirements.
- Example: FP32 → Int8 or NF4 representation.

---

### **2. Standard Quantization**

- **8-bit quantization**: relatively straightforward (sign + exponent + mantissa).
- **Naive 4-bit quantization**: introduces large errors because resolution is too coarse.

---

### **3. Normalized Float 4-bit (NF4)**

- **Problem**: Neural network weights follow a *normal distribution* (most values near 0).
- If quantized naively, many close-together weights collapse into the same bucket → **precision loss**.
- **NF4 (Normalized Float 4-bit)** is a **dynamic quantization technique**:
    - Allocates more buckets (higher precision) near regions with high weight density (around 0).
    - Ensures small differences (e.g., 0.9512 vs. 0.9514) are preserved instead of merged.

✅ **Result**: NF4 preserves fine-grained weight differences, reducing reconstruction error and improving accuracy compared to naive 4-bit.

---

### **4. Workflow in Q-LoRA**

1. **Storage**: Model weights stored in NF4 (4-bit).
2. **Loading**: When model loads, weights are decompressed back to 16-bit (or 8-bit) for computation.
3. **Training**: LoRA (Low-Rank Adapters) are trained, while the main model remains quantized and frozen.
4. **Inference**: Uses quantized weights efficiently with minimal accuracy drop.

---

### **5. Use Cases**

- **Open-ended text generation model**: trained to produce free-flowing text (domain adaptation).
- **Chat model**: trained with structured prompts like
    
    ```
    User: question ...
    <eos>
    Model: answer ...
    
    ```
    
- Q-LoRA supports both types by making fine-tuning **memory efficient**, even on consumer GPUs.

---

✅ **Key takeaway**:

Q-LoRA with **NF4 quantization** allows training/fine-tuning **very large models** on **consumer hardware**, preserving precision where it matters most (around the mean of weight distributions).

---

Would you like me to also **make a visual diagram** that shows the difference between:

- Standard FP32 → Int4 quantization (precision loss),
- NF4 dynamic quantization (precision preserved)?

That way you’ll have a single infographic summarizing this lecture.