# Section6 : Knowledge Distillation for BERT model

**MobileBERT – Key Summary**

- **Overview**:
    - A *thin* and *deep* version of the BERT-Large model, developed by Google.
    - **Task-agnostic**: pre-trained without being tied to any specific downstream task.
    - Designed for on-device NLP (e.g., Pixel 4), with low latency (62 ms).
- **Efficiency Gains**:
    - **4.3× smaller** and **5.5× faster** than BERT-Base.
    - Comparable or better accuracy than BERT-Base on some datasets (e.g., +1.5–2.1% on SQuAD dev set).
- **Why Deep but Thin**:
    - Reducing encoder layers makes a model *shallow* → faster but poor representation power.
    - MobileBERT keeps **24 encoder layers** (same as BERT-Large) for strong representation but narrows layers to reduce parameters.
    - Uses an *inverted bottleneck* architecture for better efficiency.
- **Training Approach**:
    - Knowledge transfer from an **inverted bottleneck BERT** to MobileBERT.
    - Distillation is done during pre-training, not fine-tuning → preserves task-agnostic nature.
- **Comparison with Other Models**:
    - **DistilBERT**: 60% faster, 40% fewer parameters, but less compact than MobileBERT.
    - **TinyBERT**: Good compression, but distillation happens in both pre-training & fine-tuning stages → task-specific, not task-agnostic.
    - MobileBERT outperforms or matches these while staying task-agnostic.
- **Practical Impact**:
    - Better accuracy in some tasks despite being much smaller & faster.
    - Strong candidate to replace BERT-Base in many applications.

| Aspect | **Knowledge Transfer (KT)** | **Knowledge Distillation (KD)** |
| --- | --- | --- |
| **Definition** | Broad process of passing knowledge from a teacher model to a student model. | A specific type of KT that uses *output probability distributions* (soft labels) from the teacher as targets. |
| **Scope** | Can include **intermediate features**, **attention maps**, and **outputs**. | Focuses only on **final outputs** (logits → probabilities). |
| **Typical Loss Functions** | - **MSE** (Mean Squared Error) for feature maps.- **KL Divergence** for attention distributions. | - **Cross-Entropy** between teacher & student probability distributions.- Often uses temperature scaling. |
| **MobileBERT Examples** | - **Feature Map Transfer**: MSE between hidden states of teacher & student.- **Attention Transfer**: KL divergence between attention head distributions. | - KD loss for **Masked Language Modeling (MLM)** outputs.- KD loss for **Next Sentence Prediction (NSP)** outputs. |
| **Goal** | Align **internal representations** and behaviors between teacher and student. | Mimic the **final decision patterns** of the teacher. |
| **Stage of Use** | Can be applied at any layer during training. | Applied at the **output layer** during training. |
| **Granularity** | Fine-grained (layer-by-layer or module-by-module). | Coarse-grained (overall task prediction). |

## **MobileBERT – Training Strategies & Final Findings**

### **Training Strategies**

1. **Auxiliary Knowledge Transfer (Aux-KT)**
    - Uses a **single loss function**:
        
        **Loss = Knowledge Transfer Loss + Pre-training Distillation Loss** (linear combination).
        
    - Updates all student weights jointly from this combined loss.
2. **Joint Knowledge Transfer (Joint-KT)**
    - Separates the two loss terms.
    - First trains with **all layer-wise knowledge transfer losses** at once (feature maps + attention maps).
    - All layers are trained simultaneously.
3. **Progressive Knowledge Transfer (PKT)**
    - Trains **one layer at a time** in **L stages** (L = number of layers, e.g., 24).
    - At each stage:
        - Train one layer’s transfer.
        - **Freeze** all other layer parameters.
    - Found to perform best among the three.

---

### **Key Experimental Findings**

- **Parameter & Accuracy Trade-offs**:
    - Reducing **internal embedding size** greatly lowers parameters **but** hurts accuracy significantly.
    - Reducing **number of attention heads** has little effect on accuracy or parameter count.
    - Conclusion: **Internal embedding size** is far more critical for accuracy than number of heads.
- **Teacher Choice**:
    - Used **Inverted Bottleneck BERT-Large** as teacher to ensure effective feature transfer.
- **Optimization Impact**:
    - Removing LayerNorm + replacing ReLU with **GELU** drastically improved speed without major accuracy loss.

---

### **Benchmark Results**

- **MobileBERT (no optimization)**:
    - 25M parameters, **192 ms** inference, GLUE score: 78.5.
- **MobileBERT (optimized)**:
    - Same 25M parameters, **62 ms** inference (~5× faster than BERT-Base), GLUE score: 77.7.
- **BERT-Base**:
    - 109M parameters, **342 ms** inference, GLUE score: 78.3.
- **MobileBERT-Tiny**:
    - ~40 ms inference, smaller size, GLUE score: 75.4.
- **Overall**:
    - MobileBERT is **smaller, faster, and competitive in accuracy** vs. BERT-Base.
    - Outperforms TinyBERT in being **task-agnostic** (TinyBERT’s distillation is task-specific).

## **TinyBERT – Key Summary**

### **Overview**

- Smallest BERT variant: **~15M parameters**.
- Accuracy: ~97% of BERT-Base on GLUE benchmark.
- **7.5× smaller** and **~9.4× faster** in inference than BERT-Base.
- Faster & smaller than MobileBERT (which is ~5× smaller & faster).

---

### **Similarity to Other Distilled Models**

- **Same teacher–student framework** as DistilBERT & MobileBERT:
    - Teacher model guides student through **knowledge distillation**.
    - Loss functions include:
        - Knowledge distillation loss (soft logits).
        - Cosine similarity loss (attention alignment).
        - Feature map transfer (hidden states).
        - KL divergence.
        - Cross-entropy loss.
- **Supports Next Sentence Prediction (NSP)** and token type IDs (unlike DistilBERT).

---

### **Key Difference**

- **Task-specific distillation**:
    - DistilBERT & MobileBERT → **task-agnostic** (only pre-training distillation).
    - TinyBERT → **two-stage distillation**:
        1. **General pre-training distillation** (task-agnostic).
        2. **Task-specific distillation** on *augmented dataset* for downstream tasks.

---

### **Distillation Process in TinyBERT**

1. **Pre-training stage**:
    - Teacher → Student transfers:
        - **Hidden states** (feature maps) via MSE.
        - **Attention matrices** via MSE.
    - Embedding layer outputs also matched.
2. **Task-specific stage**:
    - Fine-tune TinyBERT on augmented data.
    - Teacher–student distillation repeated for the target task.

---

### **Performance**

- Nearly matches BERT-Base accuracy while being much smaller and faster.
- Outperforms MobileBERT in speed and size, but MobileBERT is more flexible due to task-agnostic design.

| Model | Params (M) | Speed vs BERT-Base | Accuracy (GLUE) | Distillation Type | NSP Support | Task-Agnostic? | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **BERT-Base** | 109M | 1× (baseline) | 78.3 | None (original pre-training) | ✅ | ✅ | Full-size baseline, high accuracy but slow & large. |
| **DistilBERT** | ~66M | ~1.6× faster | ~97% of BERT-Base | Pre-training KD only | ❌ | ✅ | Drops NSP & token type IDs; smaller & faster. |
| **MobileBERT** | 25M | ~5× faster | 77.7 | Pre-training KD + knowledge transfer (feature & attention) | ✅ | ✅ | Thin & deep (24 layers), inverted bottleneck; optimized version removes LayerNorm + uses GELU. |
| **TinyBERT** | 15M | ~9.4× faster | ~97% of BERT-Base | **Two-stage KD**: pre-training + **task-specific** distillation on augmented data | ✅ | ❌ | Smallest & fastest; matches BERT-Base accuracy closely; optimized for specific downstream tasks. |

## **TinyBERT – Conclusion Summary**

### **Background**

- Published between **DistilBERT** and **MobileBERT** (2019–2020).
- Goal: **Accelerate inference** and **reduce model size** for real-world deployment (especially mobile/edge devices).
- Achieves:
    - **7.5× smaller** model size (15M parameters).
    - **~9.4–10× faster** inference than BERT-Base.
    - Accuracy ~97% of BERT-Base (GLUE ~77 vs. ~79 for BERT-Base).

---

### **Two-Stage Transformer Distillation**

1. **General-domain distillation** (task-agnostic pre-training).
2. **Task-specific distillation** on augmented datasets for downstream tasks.

---

### **Distillation Targets (Loss Functions)**

1. **Embedding Layer Output** – compare student vs. teacher embeddings, update via backpropagation.
2. **Hidden States & Attention Matrices** –
    - Attention transfer is critical (captures most semantic information).
    - Uses MSE for hidden states and attention maps.
3. **Logits (pre-softmax output)** – compare distributions before softmax.

---

### **Multi-Head Attention Transfer**

- Standard MHA: Query–Key–Value mechanism with scaling factor 1dk\frac{1}{\sqrt{d_k}}dk1.
- Attention weights from teacher are transferred to student.

---

### **Training Method**

- Teacher model behavior FTF_TFT and student model behavior FSF_SFS compared.
- Multiple teacher layers can be **compressed into fewer student layers**.
- Combined loss integrates embedding, hidden state, attention, and logit comparisons.

---

### **Performance**

- **Parameters**: ~15M vs. BERT-Base’s 109M.
- **Speed**: ~10× faster than BERT-Base.
- **Accuracy**: ~2 points lower on GLUE but still highly competitive.
- **Trade-off**: Requires both pre-training and task-specific distillation (more complex training).

---

### **Practical Takeaway**

- Excellent choice for industry deployment when small size & high speed are critical.
- Outperforms DistilBERT & MobileBERT in many cases, especially with available pre-trained/fine-tuned versions (e.g., on Hugging Face).
- Task-specific distillation makes it **less task-agnostic** but very strong for targeted use cases.

## **Named Entity Recognition (NER) – Key Points**

- **Definition**:
    
    NER is the task of identifying and classifying key information (entities) in text into predefined categories (e.g., person, organization, location).
    
- **Example**:
    
    *Sentence*: “Jeff Dean is a computer scientist at Google in California.”
    
    - **Jeff Dean** → PERSON
    - **Google** → ORGANIZATION
    - **California** → LOCATION
- **Process**:
    1. **Sentence tokenization** → break text into words/tokens.
    2. **Token classification** → assign a label (entity type) to each token.
    3. **Output** → list of tokens with entity tags.
- **NER = Token Classification**:
    - Similar to multi-class classification, but done **per token**.
    - Labels depend on dataset (e.g., `PER`, `ORG`, `LOC`, `DATE`, `GPE`).
    - Special tagging schemes like BIO/IOB are used (e.g., `B-PER`, `I-PER`, `O`).
- **Example Output**:
    
    *Input*: “My name is Omar. I live in Zurich.”
    
    - Omar → PERSON
    - Zurich → GPE (Geopolitical Entity)
- **How it works in practice**:
    - **Input text** → tokenized (subword or word-level).
    - Each token → classified into a label.
    - Model trained on annotated dataset (e.g., MIT Restaurant dataset).

## **IOB / BIO Tagging for NER – Key Points**

- **Definition**:
    - **B** = Beginning of an entity
    - **I** = Inside an entity
    - **O** = Outside any entity
    - Also called **Inside–Outside–Beginning (IOB)** or **BIO** format.
- **Purpose**:
    
    Ensures that multi-token entities are recognized as a single unit.
    
    Example: *"John Smith"* →
    
    - `John` → **B-PER**
    - `Smith` → **I-PER**
- **Why Needed**:
    
    Without B/I distinction, the model can’t tell if adjacent entity tokens belong to the same entity or separate entities.
    
- **Example**:
    
    Sentence: *"Sarah lives in New York City"*
    
    - Sarah → **B-PER**
    - lives → **O**
    - in → **O**
    - New → **B-LOC**
    - York → **I-LOC**
    - City → **I-LOC**
        
        → Combines into **Location = "New York City"**
        
- **Model Workflow (NER with Transformers)**:
    1. **Tokenization** → Split text into tokens.
    2. **Embeddings** → Token embeddings + positional embeddings.
    3. **Transformer Encoder** → Produces contextualized hidden states.
    4. **Token Classification Head** → Linear layer maps hidden states to tag logits.
    5. **BIO Tagging** → Assigns `B-`, `I-`, `O` labels.
    6. **Post-processing** → Combine tokens with B/I labels into complete entities.
- **Architecture Notes**:
    - Only **encoder** part of transformer is used (decoder is not needed for NER).
    - Token classification head is similar to sequence classification head but works at **per-token level**.