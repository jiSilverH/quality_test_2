# Section4 : BERT Architecture Theory

**BERT – Key Study Points**

1. **Definition**
    - **BERT** = *Bidirectional Encoder Representations from Transformers* (Google, 2018).
    - Built by stacking **transformer encoder** layers.
    - Uses **bidirectional context** (looks at words before and after a token).
2. **Why BERT was Different**
    - Previous models:
        - **GPT** → left-to-right (unidirectional).
        - **ELMo** → semi-bidirectional (two separate directions combined).
    - **BERT** → fully bidirectional → more accurate word embeddings.
3. **Core Training Tasks**
    - **MLM (Masked Language Modeling)**:
        - Mask a word in a sentence, predict the missing word using surrounding context.
    - **NSP (Next Sentence Prediction)**:
        - Given two sentences, predict if the second follows the first in the original text.
4. **Model Size**
    - **BERT Base**: ~110 million parameters.
    - Modern large language models (e.g., GPT-4, LLaMA) have **billions** of parameters.
5. **Language Modeling Purpose**
    - A *probabilistic model* that predicts the next word (token) based on context.
    - Produces a **probability distribution (PDF)** over the vocabulary.
    - Chooses the token with the highest probability during generation.
    - Example: Prompt “Shanghai is a city in …” → highest probability token = “China.”
6. **Applications**
    - Improves NLP tasks: text classification, Q&A, sentiment analysis, etc.
    - Foundation for many later LLMs.

## **BERT – Context & Training Tasks**

### **1. Context in Language Understanding**

- **Context** = meaning of a word determined by surrounding words (before & after).
- Example: the word **“right”** can mean:
    - **Direction** (“They were on the right side of the street”)
    - **Correctness** (“They were on the right side of history”)
- **Bidirectional context** is crucial — looking only in one direction (as in early GPT or ELMo) can miss the intended meaning.
- BERT captures **different embeddings** for the same word in different contexts.

### **2. Transformer Components**

- **Transformer** = Encoder + Decoder architecture.
- **BERT** uses **only encoders** → produces *context vectors* for words (used for classification, QA, etc.).
- **GPT** uses **only decoders** → autoregressive (text generation).

### **3. BERT’s Main Pretraining Tasks**

**A. MLM – Masked Language Modeling**

- Mask tokens in the sentence.
- Model predicts the missing tokens using **both** left and right context.
- In “multi-mask” setups, multiple tokens may be masked and predicted.

**B. NSP – Next Sentence Prediction**

- Input: Sentence A + Sentence B.
- Task: Predict whether B logically follows A (True/False or 1/0).
- Example:
    - A: “The legislators believed they were on the right side of history.”
    - B: “So they changed the law.” → **True** (connected)
    - B: “The bunny ate the carrot.” → **False** (unrelated)

### **4. Why This Matters**

- Earlier models lacked full bidirectional understanding → context was incomplete.
- BERT’s bidirectional attention allows richer, more accurate embeddings.
- Pretrained on MLM + NSP, then fine-tuned for **downstream tasks** (classification, sentiment analysis, QA, NER, etc.).

## **BERT – Pretraining, Architecture & Key Concepts**

### **1. Related Work Context**

- **ELMo**: Uses LSTM networks, semi-bidirectional.
- **GPT (v1)**: Uses decoder-only transformers, unidirectional.
- **BERT**: Encoder-only transformer, **fully bidirectional**.

---

### **2. Two-Step Process**

1. **Pretraining**
    - On **two main tasks**:
        - **MLM (Masked Language Modeling)**: Randomly mask 15% of tokens.
            - 80% replaced with `[MASK]`
            - 10% replaced with a random token
            - 10% left unchanged
        - **NSP (Next Sentence Prediction)**: Predict if Sentence B follows Sentence A in original text.
    - Uses large unlabeled text corpus.
2. **Fine-tuning**
    - On **downstream tasks** (e.g., SQuAD for QA, MNLI for NLI, NER).
    - Adds task-specific layers on top of pretrained BERT.

---

### **3. Special Tokens & Input Representation**

- **[CLS]**: Placed at the start of sequence; its final vector represents the whole input (used for classification tasks).
- **[SEP]**: Separates sentences (used for single or paired inputs).
- **Segment embeddings**: Learned embeddings to indicate whether a token belongs to Sentence A or Sentence B.
- **Final input embedding** for each token =
    
    **Token embedding** + **Segment embedding** + **Position embedding**.
    

---

### **4. Tokenization**

- **WordPiece** tokenization.
- Vocabulary size: **30,000 tokens**.

---

### **5. BERT Architectures**

| Model | Layers (Encoders) | Hidden Size | Attention Heads | Params |
| --- | --- | --- | --- | --- |
| BERT Base | 12 | 768 | 12 | 110M |
| BERT Large | 24 | 1024 | 16 | 340M |

---

### **6. Downstream Task Examples**

- **QA (Question Answering)** – SQuAD
- **NLI (Natural Language Inference)** – MNLI
- **NER (Named Entity Recognition)**
- **Classification tasks** (sentiment, topic)

---

### **7. Key Takeaways for Exam**

- BERT is **encoder-only** transformer → produces **contextual embeddings**.
- Fully **bidirectional attention** → uses both left and right context.
- Pretrained with MLM & NSP, then fine-tuned for many NLP tasks.
- Special tokens ([CLS], [SEP]) and segment embeddings are crucial for handling single vs. paired sentences.
- WordPiece tokenization allows handling out-of-vocabulary words efficiently.

## **BERT – Additional Key Terminology & Insights**

### **1. Special Tokens & Input Embedding Components**

- **[CLS]** → First token in every sequence; final hidden state used for classification tasks.
- **[SEP]** → Separates Sentence A and Sentence B.
- **Token Embedding** → Word meaning representation (WordPiece tokens).
- **Segment Embedding** → Learned embedding indicating sentence membership (A or B).
- **Position Embedding** → Indicates word order in sequence.
- **Final input vector** for each token = Token + Segment + Position embeddings (same dimension as model size).

---

### **2. Pretraining Data**

- **BooksCorpus** → 800M words.
- **English Wikipedia** → 2.5B words.
- Pretraining done on Cloud TPUs (costly & time-consuming).
- **Fine-tuning**: Much faster (≈1 hour on TPU, few hours on GPU).

---

### **3. Evaluation Benchmarks**

- **GLUE** benchmark: BERT-Large scored **86.7% on MNLI** vs GPT’s 82.1%.
- Outperformed prior models (GPT, ELMo) on multiple datasets.

---

### **4. Model Architecture Experiments**

- **L** = # encoder layers, **H** = hidden size, **A** = attention heads.
- Found that increasing encoders and attention heads generally improves accuracy — but limited by computational cost.
- Final chosen configs:
    - **BERT Base**: L=12, H=768, A=12 → 110M params.
    - **BERT Large**: L=24, H=1024, A=16 → 340M params.

---

### **5. Comparison to ELMo & GPT**

- **ELMo**: Semi-bidirectional (separate LSTMs for left-to-right and right-to-left, then concatenate).
- **GPT**: Unidirectional (left-to-right only).
- **BERT**: True bidirectional — attends to both left & right context for every token at once.

---

### **6. Why Bidirectionality Matters**

- Increases ability to capture nuanced meaning of words.
- Higher probability of accurate predictions across diverse NLP tasks.

![image.png](Section4%20BERT%20Architecture%20Theory%2024a4d6b253d38010be9beb1d1a3cf37a/image.png)

## **BERT – Architecture & Key Details Refresher**

### **1. Transformer Parts**

- **BERT** = *Encoder-only* transformer.
- **GPT** = *Decoder-only* transformer.
- BERT stacks **N encoders** (BERT Base: 12; BERT Large: 24).
- After encoders, adds a **task-specific head**:
    - **Classification head** (e.g., sentiment analysis, NSP).
    - **Token classification head** (e.g., NER).

---

### **2. Positional Encoding vs. Embedding**

- **Original Transformer**: *Positional encoding* → fixed, formula-based, not learned.
- **BERT**: *Positional embedding* → learnable during training.

---

### **3. Input Representation**

Each token’s input vector =

**Token embedding** (WordPiece vocab, 30k tokens)

- **Segment embedding** (Sentence A or B)
- **Position embedding** (max 512 positions)

---

### **4. Multi-Head Attention**

- Inside **each encoder layer**, BERT Base has **12 attention heads** (parallel attention mechanisms).
- Outputs from all heads are concatenated → linear projection → final embedding.
- **Embedding size**:
    - Base: 768
    - Large: 1024

---

### **5. Sequence Length & Tokenization**

- Max sequence length = **512 tokens** (due to position embedding limit).
- Uses **WordPiece tokenizer** to split words into smaller sub-tokens.

---

### **6. Classification Head Output**

- For **classification tasks**:
    - Output neurons = number of classes.
    - **Softmax** → probability distribution over classes.
- For **token classification (NER)**:
    - Classifies each token individually.

---

### **7. Workflow**

1. Tokenize input → WordPiece tokens.
2. Create token, segment, and position embeddings.
3. Pass through stacked encoders (12 or 24).
4. Output final hidden states.
5. Apply task-specific head → **softmax** → probability.

---

✅ **Key Exam Pointers**

- BERT uses **positional embeddings** (learned) vs. Transformer’s positional encodings (fixed).
- Attention heads are **per layer**, not equal to number of encoders.
- Input length limit = 512 tokens.
- Token embeddings + Segment embeddings + Position embeddings are summed before entering encoders.

## **BERT – Input Embeddings Breakdown**

### **1. Tokenization (WordPiece)**

- **Purpose**: Reduce vocabulary size to ~30,000 tokens.
- **Method**:
    - Break unknown or rare words into **subword units**.
    - Root word kept, suffix prefixed with `##` (e.g., `playing` → `play`, `##ing`).
    - Helps reuse common roots across many words (reduces parameters).

---

### **2. Special Tokens**

- **[CLS]** → First token of every sequence; final embedding used for classification.
- **[SEP]** → Separates two sentences; also placed at the end of each sentence.

---

### **3. Embedding Types**

For each token, BERT sums **three embeddings** of the same dimension (Base: 768, Large: 1024):

1. **Token Embedding**
    - Represents the meaning of the token (learned, like Word2Vec).
    - Different for each token.
2. **Segment Embedding**
    - Indicates whether token belongs to Sentence A or Sentence B.
    - All tokens in the same sentence share the same segment embedding.
3. **Position Embedding**
    - Represents token position (0, 1, 2, … up to 511).
    - All positions have **different** embeddings.
    - Learned (unlike original Transformer’s fixed positional encoding).

---

### **4. Final Input Embedding**

- **Final embedding = Token + Segment + Position embeddings (element-wise sum)**.
- Passed to the encoder stack for processing.

---

### **5. Why These Matter**

- **Token embedding** → semantic meaning.
- **Segment embedding** → sentence origin (A or B).
- **Position embedding** → word order and relative position.

## **BERT – Output Processing & Training Objectives**

### **1. Input Recap**

- Tokenized via **WordPiece** → adds special tokens **[CLS]** (start), **[SEP]** (end/separator).
- **Three embeddings summed**: Token + Segment + Position → **Final input embedding**.
- Dimension: **768** (Base) / **1024** (Large).

---

### **2. Output Representations**

- After passing through all encoder layers:
    - **Token vectors** = context-aware embeddings for each token.
    - **[CLS] vector** = context vector for the entire input (used for classification/NSP).
- These are **context vectors**, not static word embeddings — same word changes meaning by context.

---

### **3. Pretraining Objectives**

### **A. MLM – Masked Language Modeling**

- Randomly mask **15% of tokens**:
    - 80% → replace with `[MASK]`
    - 10% → replace with random token
    - 10% → keep unchanged
- For each masked position:
    - Linear layer output size = **30,000 neurons** (vocab size)
    - **Softmax** → probability distribution over all tokens
    - Loss = **Categorical Cross-Entropy**
- Trains the model to predict missing words using **bidirectional context**.

### **B. NSP – Next Sentence Prediction**

- Uses the **[CLS] vector** to predict if Sentence B follows Sentence A.
- Output = 2 neurons (Yes / No)
- Loss = **Binary Cross-Entropy** (or Categorical Cross-Entropy if >2 classes)

---

### **4. Loss Functions Used**

- **Categorical Cross-Entropy** → MLM (multi-class, vocab size 30k).
- **Binary Cross-Entropy** → NSP (binary classification).
- Both losses are combined during pretraining.

---

✅ **Key Exam Pointers**:

- MLM predicts masked words **over entire vocab** (softmax over 30k tokens).
- NSP is based **only on [CLS] vector**.
- Same architecture can be fine-tuned for other tasks by swapping the head.
- Context vectors vary with position and surrounding tokens.

## **BERT – Fine-Tuning & Downstream Tasks**

### **1. Two-Step Process**

1. **Pretraining**
    - Tasks: **MLM** (Masked Language Modeling) + **NSP** (Next Sentence Prediction).
    - Produces general-purpose **contextual embeddings**.
2. **Fine-tuning**
    - Adapt pretrained BERT to specific **downstream tasks** by adding a **task-specific head** on top of embeddings.
    - Same BERT architecture can be reused; only the head changes.

---

### **2. Examples of Downstream Tasks**

- **Sentence Pair Classification (e.g., NSP, entailment)**
    - Input: Sentence A + Sentence B.
    - Use **[CLS] vector** → classification head → binary/multi-class output.
    - Loss: Binary Cross-Entropy or Categorical Cross-Entropy.
- **Single Sentence Classification** (e.g., sentiment, emotion classification)
    - Input: Sentence A (Sentence B empty).
    - Use [CLS] vector → classification head.
- **Question Answering (QA)** (e.g., SQuAD dataset)
    - Input: Question (Sentence 1) + Passage (Sentence 2).
    - Predict **start** and **end** token positions of the answer.
    - Loss: Categorical Cross-Entropy for start and end positions.
- **Named Entity Recognition (NER) / Token Classification**
    - Output classification for each token.
    - Multi-class classification over NER tag set.
    - Loss: Categorical Cross-Entropy.
- **Paraphrase detection / sentence similarity**
    - Sentence pair → classification head for similarity score.

---

### **3. Why BERT is Flexible**

- **Same pretrained model** can be fine-tuned for many tasks just by changing the final head and loss function.
- **[CLS] vector** captures whole-sequence context.
- Token embeddings capture per-token context for tagging/NER.

---

### **4. Evaluation Benchmarks**

- **GLUE (General Language Understanding Evaluation)**
    - Multiple datasets for tasks: MNLI, QQP, QNLI, SST-2, RTE, etc.
    - Covers classification, similarity, entailment.
    - **BERT-Large** outperformed previous models (including GPT) on most GLUE tasks.
- **SQuAD** (QA benchmark)
    - BERT achieved **85–87%** F1 score (human avg ~80–86%).
- **NER (e.g., CoNLL-2003 dataset)**
    - Entities: Person, Organization, Location, Miscellaneous.
    - BERT achieved ~97% accuracy.

---

✅ **Key Exam Pointers**

- Fine-tuning is fast compared to pretraining; only a small head is trained from scratch.
- [CLS] vector is used for **sequence-level** tasks; per-token outputs are used for **token-level** tasks.
- BERT’s flexibility comes from **bidirectional embeddings** + transformer encoder stack.
- Evaluation shows **BERT-Large** consistently beats GPT and ELMo across multiple NLP benchmarks.