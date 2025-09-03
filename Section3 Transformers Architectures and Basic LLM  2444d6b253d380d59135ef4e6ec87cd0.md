# Section3: Transformers Architectures and Basic LLM Concepts

### 🔑 Key Points: Introduction to Transformers via Sequence-to-Sequence Models

### 1. **Course Structure (High-Level)**:

- Start with **sequence-to-sequence (seq2seq)** models.
- Understand **why Transformers are needed** (limitations of RNNs/LSTMs).
- Learn about **attention mechanisms**.
- Finally, move into the **Transformer architecture**.

---

### 2. **What is Sequence-to-Sequence (Seq2Seq)?**

- A model architecture introduced by Google in **2014** for tasks like **machine translation**.
- Consists of:
    - **Encoder**: Takes input sequence (e.g., sentence in English).
    - **Decoder**: Produces output sequence (e.g., sentence in French).

---

### 3. **Key Features of Seq2Seq**:

- **Input and output can have different lengths**.
    - E.g., English sentence with 5 words → French sentence with 4 words.
- Originally used **RNNs**, later replaced by **LSTMs/GRUs** to deal with:
    - **Vanishing/exploding gradient problems**.
- **Processes data sequentially**:
    - Word-by-word, which slows down training/inference.
- Has a **fixed-length context vector** (a bottleneck):
    - Entire input is compressed into a single hidden state before decoding.
    - This limits performance on **long sequences**.

---

### 4. **Why Seq2Seq Matters for Transformers**:

- Transformers improve on seq2seq by:
    - Allowing **parallel processing**.
    - Replacing the fixed context with **attention** (dynamic access to all input tokens).
- Understanding seq2seq helps you appreciate **how Transformers solve key problems**.

---

### 🔑 Key Points: Sequence-to-Sequence Encoder and Decoder (LSTM-based)

---

### 🧩 **1. Encoder (LSTM-based)**

- **Input**: A sequence of words → tokenized → passed through **embedding layers**.
- Each word embedding is processed **sequentially** through the **LSTM**.
- LSTM maintains and updates a **hidden state** `h₁ → h₂ → h₃ → h₄` (for 4 tokens, as example).
- **Final hidden state (`h₄`)** is used to summarize all input — it's the **context vector** passed to the decoder.

⚠️ **Issue**:

- All the information from the input sequence is **compressed into one fixed-length hidden state (`h₄`)**.
- As input length increases, this **compression causes information loss**.
- Longer inputs = more loss → **performance degradation**.

---

### 🧩 **2. Decoder (LSTM-based)**

- Starts with a **Start-of-Sequence (SOS)** token → embedded and passed to LSTM.
- Uses the **final encoder hidden state (`h₄`)** as input for the initial state.
- At each step:
    - Takes the **previous output word** and **hidden state** to predict the **next word**.
- Stops once **End-of-Sequence (EOS)** token is generated.

⚠️ **Issues**:

- Like the encoder, decoder works **sequentially**, which:
    - Prevents **parallelization**.
    - Slows down training and inference.
- Still relies on a **single hidden state** as context → same **bottleneck problem**.

---

### 🚫 Limitations of LSTM-based Seq2Seq:

| Limitation | Explanation |
| --- | --- |
| **Compression Bottleneck** | Entire input is encoded into a single vector (`h₄`). |
| **Information Loss** | Long sequences = more loss during compression. |
| **No Parallelism** | Processing is strictly sequential (both in encoder and decoder). |
| **Fixed Memory Size** | Context vector is fixed in size regardless of input length. |

---

### ✅ Motivation for Attention / Transformers:

- Instead of using **only the last hidden state**, **use all encoder hidden states** (`h₁, h₂, h₃, h₄`).
- Decoder can **dynamically "attend" to different parts** of the input as needed.
- This reduces:
    - **Compression loss**
    - **Sequential bottlenecks**
    - **Training time**

This sets up the motivation for **attention mechanisms**, which we’ll cover next.

### 🔑 Key Concepts: Evolving Seq2Seq with Attention

---

### 💡 Problem in Vanilla Seq2Seq (LSTM):

- Previously, only the **final hidden state** (e.g., `h₄`) from the encoder was passed to the decoder.
- This caused:
    - **Information bottleneck**
    - **Loss of detail** in longer sequences
    - Limited decoder performance

---

### ✅ Solution – Use **All Encoder Hidden States**:

- Instead of passing only `h₄`, pass **all hidden states**: `h₁, h₂, h₃, h₄` to the decoder.
- This allows the decoder to **selectively focus** on the most relevant parts of the input sequence at each output step.

> Think of it as giving the decoder a full map instead of a single summary.
> 

---

### 🧠 How Does the Decoder Use These States?

1. **Each decoder step** (e.g., generating one word) can **attend** to different encoder hidden states.
    - For example, when generating the second output word, it may focus more on `h₂`.
2. These encoder hidden states can be combined in various ways:
    - **Concatenation**
    - **Summation**
    - **Averaging**
    - Or via a learned **attention mechanism** (preferred)

---

### 🧭 Two Types of Attention to Understand:

| Type | What it Does |
| --- | --- |
| **Cross-Attention** | Decoder attends to the **encoder hidden states** |
| **Self-Attention** | A module (encoder or decoder) attends to **its own input tokens** |

---

### 🚀 Why This Matters:

- Enables the model to **dynamically use the full context**.
- **Reduces compression loss**.
- Supports **parallel computation** when used in Transformer architecture.
- Leads to **much better performance**, especially on longer sequences and complex tasks.

### 🔑 Key Concepts: Attention Mechanism in Sequence-to-Sequence Models

---

### 🧩 1. **Problem Recap – Traditional Seq2Seq with LSTM**

- Encoder produces **final hidden state (`h₄`)** as a compressed representation of all inputs.
- Issues:
    - **Information loss** for long sequences (compression bottleneck)
    - **Vanishing gradients** in long input chains
    - Cannot dynamically focus on specific parts of input
    - Decoding depends on a **fixed-length context vector**

---

### ✅ 2. **Solution – Use Attention Mechanism**

### ➤ Instead of using just `h₄`, pass **all encoder hidden states** (`h₁, h₂, h₃, h₄`) to the decoder.

- Decoder **decides which part to focus on** dynamically, per output token.

---

### 🎯 3. **Types of Attention**

| Type | What it Focuses On |
| --- | --- |
| **Self-Attention** | Within the same sequence (e.g., decoder → decoder) |
| **Cross-Attention** | Decoder focuses on encoder hidden states |

---

### 📊 4. **Performance Gains with Attention**

- Traditional Seq2Seq performance **drops sharply** as sentence length increases.
- Attention-based models **maintain performance** even with longer sequences.
- Shown in the **original attention paper (2014/2015)**, before Transformers (2017).

---

### 📏 5. **Evaluation Metrics for Text Generation**

| Metric | Focus | Use Case |
| --- | --- | --- |
| **BLEU** | Precision | Translation, summarization |
| **ROUGE** | Recall | Summarization, generation |
| **F1 Score** | Balance of precision and recall | Combined metric |
- `F1 = 2 × (BLEU × ROUGE) / (BLEU + ROUGE)`

---

### 🧠 6. **How Attention Weighs Inputs**

### ❌ *Naive Approach*:

- Just **add or average** all encoder hidden states:
    
    `C = h₁ + h₂ + h₃ + h₄`
    
- Problem: No focus control — all hidden states are equally weighted.

### ✅ *Weighted Attention*:

- Use **weights (`w₁, w₂, w₃, w₄`)** to assign **importance** to each hidden state:
    
    ```
    ini
    복사편집
    C = w₁·h₁ + w₂·h₂ + w₃·h₃ + w₄·h₄
    
    ```
    
- Weights (`w`) are calculated using the **decoder’s previous hidden state** (so attention is dynamic).
- This lets the decoder **focus more on relevant tokens** in the input (e.g., `h₃`) during generation.

---

### 🚀 Next Step: Transformers

- Transformers **replace RNNs** entirely.
- Use **only attention mechanisms** (no recurrence).
- Introduce concepts like:
    - **Multi-head attention**
    - **Positional encoding**
    - **Layer normalization**

### 🔑 Key Concepts: Queries, Keys, Values, and Attention

---

### 📘 Background

- **“Attention is All You Need”** (2017): Introduced the **Transformer** architecture.
- Builds on earlier **attention mechanisms** from 2014–2015 that enhanced Seq2Seq models.

---

### 🧩 Core Components: Query, Key, Value (Q, K, V)

- Each token in a sentence is **mapped to three vectors**:
    - **Query (Q)**: What are we looking for?
    - **Key (K)**: What do we have?
    - **Value (V)**: What information do we retrieve?

> 🔍 Analogy:
> 
> - **Query** = your search input
> - **Keys** = titles & metadata of documents
> - **Values** = the actual content returned

---

### ⚙️ How Attention Works (Scaled Dot-Product Attention)

Given a query vector **Q**, attention is computed over all **K-V pairs**:

1. **Score**: Compute similarity between the query and each key:score=Q⋅KT
    
    score=Q⋅KT\text{score} = Q \cdot K^T
    
    (This is a dot product.)
    
2. **Scale**: To prevent large dot product values from destabilizing softmax:scaled score=dkQ⋅KT
    
    scaled score=Q⋅KTdk\text{scaled score} = \frac{Q \cdot K^T}{\sqrt{d_k}}
    
3. **Softmax**: Convert the scores into **attention weights**:α=softmax(dkQ⋅KT)
    
    α=softmax(Q⋅KTdk)\alpha = \text{softmax}\left(\frac{Q \cdot K^T}{\sqrt{d_k}}\right)
    
4. **Weighted Sum**: Multiply each value **V** by its corresponding weight and sum:output=∑αi⋅Vi
    
    output=∑αi⋅Vi\text{output} = \sum \alpha_i \cdot V_i
    

---

### 🔁 Self-Attention vs Cross-Attention

| Type | Meaning |
| --- | --- |
| **Self-Attention** | Q, K, V all come from the **same sequence** (e.g., encoder attends to itself) |
| **Cross-Attention** | Q comes from decoder; K and V come from encoder (e.g., decoder attends to encoder output) |

---

### 🧠 Why Attention Helps

- **No bottleneck** like in LSTM-based Seq2Seq (no single context vector).
- **Long sequences** handled better.
- Enables **parallelization** of computation.
- Learns **which parts of input are relevant** for generating each output token.

---

### 📊 Evaluation Metrics Mentioned

| Metric | Focus | Use Case |
| --- | --- | --- |
| **BLEU** | Precision | Common in translation tasks |
| **ROUGE** | Recall | Common in summarization |
| **F1** | Combined | 2⋅BLEU⋅ROUGEBLEU+ROUGE2 \cdot \frac{\text{BLEU} \cdot \text{ROUGE}}{\text{BLEU} + \text{ROUGE}}2⋅BLEU+ROUGEBLEU⋅ROUGE |

### 🔑 Key Concept: **Scaled Dot-Product Attention** (from *Attention Is All You Need*)

---

### ⚙️ **1. The Core Attention Formula**

Attention(Q,K,V)=softmax(QKTdk)V\text{Attention}(Q, K, V) = \text{softmax}\left( \frac{QK^T}{\sqrt{d_k}} \right) V

Attention(Q,K,V)=softmax(dk

QKT)V

- **Q** = Queries
- **K** = Keys
- **V** = Values
- **dkd_kdk** = Dimension of key vectors (used to scale)

---

### 🧠 2. **Why Scaling?**

- Dot products of large vectors can produce **very large values**.
- Applying softmax on large values can lead to **very small gradients** → training becomes unstable.
- So we **scale** by dk\sqrt{d_k}dk to keep the softmax values in a stable range.

---

### 🧩 3. **Where Do Q, K, V Come From?**

- All are derived from **input embeddings** using **learned linear layers**:Q=XWQ,K=XWK,V=XWV
    
    Q=XWQ,K=XWK,V=XWVQ = XW^Q,\quad K = XW^K,\quad V = XW^V
    
- WQW^QWQ, WKW^KWK, WVW^VWV are **learned weight matrices** (i.e., neural layers).
- Multiple such transformations allow the model to capture **different linguistic aspects** (gender, syntax, POS, etc.)

---

### 🧠 4. **Masked Attention (Optional)**

- Used in **decoder during training** to prevent attention on **future tokens**.
- Ensures autoregressive generation.
- Implemented by setting future scores to **∞\infty∞** before applying softmax.

---

### 🎯 5. **Flexible Focus – Not Just Diagonal!**

- Attention is **not restricted to word-aligned positions**.
- A word in position 4 can attend strongly to position 0, 6, etc.
- This is powerful for:
    - **Long-range dependencies**
    - **Reordering in translation**
    - **Coreference resolution**

---

### 🧪 6. **Visualization**

- Attention is often visualized as a heatmap:
    - Rows: output tokens (queries)
    - Columns: input tokens (keys)
    - Brightness: attention weight (how much that input is focused on)

Example:

```
vbnet
복사편집
Query: "t"
Focuses on Input: "tea"
→ High attention weight

```

In **practice**, attention maps **do not follow diagonal lines**, proving that the model learns **flexible, non-local relationships**.

---

### 🧠 7. **Multi-Head Attention (Coming Next)**

- Multiple Q-K-V projections allow the model to attend to different features **in parallel**.
- Enables capturing **multiple types of dependencies** (e.g., grammar, position, meaning).

---

### ✅ Summary

| Component | Role |
| --- | --- |
| **Q (Query)** | What am I looking for? |
| **K (Key)** | What do I have? |
| **V (Value)** | What info do I return if there's a match? |
| **Softmax(Q·Kᵀ / √dₖ)** | How similar is the query to each key (attention weights)? |
| **Multiply by V** | Get the final representation for the query's context |

## 🔍 Overview: What is a Transformer?

- Introduced in the paper **“Attention is All You Need”** (2017).
- Replaces recurrence (RNN/LSTM) with **self-attention** and **parallel processing**.
- Major parts:
    - **Encoder stack** (understands input)
    - **Decoder stack** (generates output)

---

## 🧩 1. Input Processing (Encoder Side)

### Example sentence: `"I love coffee"`

### Step-by-step:

1. **Tokenization**
    
    → Breaks into tokens: `["I", "love", "coffee"]`
    
2. **Token Encoding**
    
    → Assigns IDs: `[101, 2071, 4513]` (example IDs)
    
3. **Embedding**
    
    → Maps token IDs to vectors (using learned embeddings)
    
4. **Positional Encoding**
    
    → Adds position info (since Transformers are not sequential by default)
    
5. **Input to Encoder Stack**
    
    → Final input = word embedding + positional encoding
    

---

- E.g., index 0 gets a unique vector, index 1 gets a different one, etc.

## 🧠 2. Encoder Stack

Each encoder layer contains:

1. **Multi-head self-attention**
    
    → Each word can focus on other words (even itself)
    
2. **Feed-forward neural network**
    
    → Applied to each token vector independently
    
3. **Add & Normalize**
    
    → Stabilizes training
    

### Result:

- Produces a **contextualized vector** for each input token.
    - E.g., “coffee” now knows it’s in the context of "I love"

---

## 📤 3. Output Processing (Decoder Side)

Let’s say you’re doing **machine translation**:

Input: `"I love coffee"`

Expected Output: `"J'aime le café"`

### Decoder Stack Steps:

1. **Start with [START] token**
    
    → Decoder begins with `[START]` and empty context
    
2. **Embedding + Positional Encoding**
    
    → Same as in encoder
    
3. **Masked Multi-head self-attention**
    
    → Prevents the model from seeing “future” words during training (e.g., no peeking at “café” while generating “J'aime”)
    
4. **Cross-attention**
    
    → Decoder looks at encoder output (context of input sentence)
    
5. **Final feed-forward + softmax**
    
    → Outputs probabilities over the vocabulary for the next word
    
6. **Prediction**
    
    → Choose the word with highest probability (or use sampling strategies)
    

---

## 📦 4. How Attention Connects Encoder and Decoder

### Key Idea:

- Decoder needs to “pay attention” to relevant words in the input sentence.

So:

- **Encoder output** = keys and values
- **Decoder hidden state** = query

This allows the decoder to **dynamically attend to the right parts of the input** for each output token.

---

## 💡 Why This Is Powerful

| Benefit | Explanation |
| --- | --- |
| **Parallelization** | All tokens can be processed at once (great for GPUs) |
| **No recurrence** | No more slow, step-by-step sequence modeling |
| **Flexible attention** | Any word can attend to any other word (not just nearby) |
| **Better long-sequence modeling** | Handles long context far better than LSTM or GRU |

---

## 🔁 Summary Flow

**Encoder side**:

```
Input → Tokenize → Embed → Add Position → Encoder stack → Contextual vectors
```

**Decoder side**:

```
Start token → Embed → Add Position
 → Masked self-attention
 → Cross-attention (with encoder output)
 → FFN → Softmax → Predict next token
```

## 🔑 **What Is the Encoder Doing?**

**Goal**: Take a sentence (like "I am not going") and turn it into **contextualized vectors** — where each word is represented not just by itself, but by **what it means in context**.

---

## 🧩 **Step-by-Step Encoder Process**

Let’s use this example sentence:

> "I am not going"
> 

---

### 🔹 Step 1: Input Tokenization

- Split into tokens: `["I", "am", "not", "going"]`

---

### 🔹 Step 2: Embedding

- Each token gets converted to a vector (via learned embeddings):
    
    ```
    css
    복사편집
    I      → [0.1, 0.3, ...]
    am     → [0.5, 0.2, ...]
    not    → [0.9, 0.6, ...]
    going  → [0.3, 0.8, ...]
    
    ```
    
- These vectors go into the encoder.

---

### 🔹 Step 3: Add Positional Encoding

- Since Transformers are not sequential by default, we **add position info** to each token vector.
    - E.g., token at position 0 is treated differently from token at position 3.

---

### 🔹 Step 4: Calculate Q, K, V for Each Token

- From each word vector (embedding), we compute:
    - **Q** = query vector
    - **K** = key vector
    - **V** = value vector
- These are calculated using **learned linear layers**:
    
    ```
    ini
    복사편집
    Q = X × W_Q
    K = X × W_K
    V = X × W_V
    
    ```
    
- Example for “going”:
    
    ```
    ini
    복사편집
    Q_going = embedding("going") × W_Q
    
    ```
    

---

### 🔹 Step 5: **Self-Attention** in Encoder

Each word **attends** to all other words — including itself.

Let’s say we want to re-represent the word **“going”**:

> How important are “I”, “am”, and “not” when processing “going”?
> 
1. Compute similarity between **Q_going** and each **K_i** (from all tokens).
2. Apply **softmax** to get attention weights.
3. Multiply each **V_i** (value vector) by the corresponding weight.
4. Sum them up to get the **new vector for “going”**, now aware of its context.

---

### 🧠 Example Output:

> “going” becomes:
> 

```
ini
복사편집
new_vector = 0.1×V_I + 0.3×V_am + 0.6×V_not + 0.0×V_going

```

This tells the model:

- "not" is **highly relevant** to "going" (e.g., "not going").
- So the model builds a representation of "going" that reflects **negation**.

---

### 🔸 Why Multi-Head Attention?

Instead of doing this attention just **once**, we do it **multiple times in parallel**, each with its own Q/K/V projections.

Each head can focus on:

- Head 1: Syntax
- Head 2: Subject-object relation
- Head 3: Negation
- Head 4: Time indicators

At the end, the outputs of all heads are **concatenated** and passed through a **feed-forward neural network**.

---

## ✅ Final Output of Encoder

- For each token, you get a **contextualized vector**.
- Now "going" doesn’t just mean “to move” — it reflects its **context**, like “not going”.

These **context-rich vectors** go to the **decoder**, or are used for classification (in BERT-like models).

---

## 📌 Summary of Encoder Flow:

| Step | Description |
| --- | --- |
| 1. Tokenization | Break text into tokens |
| 2. Embedding + Positional Encoding | Convert tokens into vectors and add position info |
| 3. Q/K/V Calculation | Linear projections from input vectors |
| 4. Multi-head Self-Attention | Each token attends to all others |
| 5. Feed-Forward Network | Applies non-linearity to each token vector |
| 6. Output | Contextual vectors for each token |

## 🧠 Why Do We Need Positional Encoding?

Transformers **do not have recurrence** (like RNNs) or convolution (like CNNs).

That means:

➡️ They treat each word **independently** unless we explicitly give them **position information**.

Without position info:

> "I love you" and "you love I" would look the same to the model!
> 

---

## 📘 The Idea of Positional Encoding

Instead of learning positions (which adds more parameters), the authors used a **static mathematical function**:

- Uses **sine and cosine** functions
- Gives a **unique vector** for each position in the sequence
- Adds it to the word embedding so the model knows **where the word is**

---

## 🧮 Formula (From Paper)

For a position `pos` and dimension index `i`, the positional encoding vector is:

PE(pos,2i)=sin⁡(pos100002i/dmodel)\text{PE}_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i / d_{\text{model}}}}\right)

PE(pos,2i)=sin(100002i/dmodelpos)

PE(pos,2i+1)=cos⁡(pos100002i/dmodel)\text{PE}_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i / d_{\text{model}}}}\right)

PE(pos,2i+1)=cos(100002i/dmodelpos)

- `d_model` = the total dimension of the embedding (e.g., 512)
- Even indices → use **sine**
- Odd indices → use **cosine**

---

## 🎯 What This Does

- **Low-frequency waves** capture **coarse** position (early layers)
- **High-frequency waves** capture **fine-grained** position (later layers)
- Together, they provide a **rich, unique pattern** per position that helps attention know “where” tokens are.

---

## 📊 Example: Simple 4-Dimensional Positional Encoding

Let’s say:

- `d_model = 4` (small for simplicity)
- Sentence: `"I am not going"`
- Positions = 0, 1, 2, 3

### 🔹 For position = 0:

PE(0)=[sin⁡(0),cos⁡(0),sin⁡(0),cos⁡(0)]=[0,1,0,1]\text{PE}(0) = [\sin(0), \cos(0), \sin(0), \cos(0)] = [0, 1, 0, 1]

PE(0)=[sin(0),cos(0),sin(0),cos(0)]=[0,1,0,1]

### 🔹 For position = 1:

You apply the formula with different divisors (based on `10000^(2i/d_model)`)

Let’s compute approximate values (rounded):

PE(1)=[sin⁡(1/1),cos⁡(1/1),sin⁡(1/100),cos⁡(1/100)]≈[sin⁡(1),cos⁡(1),sin⁡(0.01),cos⁡(0.01)]≈[0.84,0.54,0.01,0.9999]\text{PE}(1) = [\sin(1/1), \cos(1/1), \sin(1/100), \cos(1/100)]
\approx [\sin(1), \cos(1), \sin(0.01), \cos(0.01)]
\approx [0.84, 0.54, 0.01, 0.9999]

PE(1)=[sin(1/1),cos(1/1),sin(1/100),cos(1/100)]≈[sin(1),cos(1),sin(0.01),cos(0.01)]≈[0.84,0.54,0.01,0.9999]

You’ll notice that:

- First two dimensions change more (low-frequency)
- Later ones change less (high-frequency)

---

## 📌 Key Properties

| Feature | Explanation |
| --- | --- |
| **No training** | It's fixed (no learnable parameters) |
| **Fast** | No extra computation during training |
| **Unique** | Each position has a distinct pattern |
| **Similar positions are similar** | Helps capture distance/direction |
| **Works well in practice** | Learned embeddings didn't show significant improvement |

---

## 🧠 Analogy

Imagine you're writing music:

- Each **note** (word) has its **pitch** (embedding)
- But to make a melody, you also need to know **where it lands on the beat** (position)

Positional encoding adds **rhythm** to the **sound** — without it, words are just floating in space.

---

## ✅ Summary

| Without Positional Encoding | With Positional Encoding |
| --- | --- |
| Words have no sense of order | Position info is encoded into the vector |
| All word embeddings are same regardless of position | Same word gets slightly different vector at each position |
| Can’t handle sequence tasks | Can model syntax and structure well |

## 🔍 Three Types of Attention in the Transformer

| Type | Where it occurs | What it does |
| --- | --- | --- |
| **Self-Attention** | Encoder & Decoder | Helps a word focus on other words in the same sequence |
| **Masked Self-Attention** | Decoder only | Prevents peeking into future words while generating output |
| **Cross-Attention (Encoder–Decoder Attention)** | Decoder | Helps output tokens attend to the input tokens (from encoder) |

---

## 🧠 1. **Self-Attention** (Used in **Encoder** and **Decoder**)

### 💬 Example: `"It's time for tea"`

Each word builds its **contextualized representation** by attending to all words **in the same sentence**:

| Predicting "tea" → attention weights might be: |

|--------------|---------------------|

| "It's"       | 0.2                 |

| "time"       | 0.1                 |

| "for"        | 0.1                 |

| **"tea"**    | **0.6**             |

So the model learns that "tea" heavily relates to itself but might also consider "for".

### 🔁 Happens in **parallel**:

Every word computes self-attention **simultaneously**, not sequentially.

---

## 🔐 2. **Masked Self-Attention** (Used **only in Decoder** during training)

### 🤔 Why "Masked"?

When generating output like:

> "Je t'aime beaucoup"
> 

At each step, the model should **not look ahead**.

So, when predicting `"t'aime"` (step 2), the model **can only use previous tokens**:

```
cpp
복사편집
Input so far: ["Je", "t'aime"]
Mask: ✔️ "Je", ✔️ "t'aime", ❌ future words

```

### 📉 How masking works:

In attention score matrix (Q·Kᵀ), we **fill upper triangle** with **−∞**, so that softmax = 0 (i.e., no attention to future tokens).

|  | Je | t'aime | beaucoup |
| --- | --- | --- | --- |
| Je | ✔️ | ❌ | ❌ |
| t'aime | ✔️ | ✔️ | ❌ |
| beaucoup | ✔️ | ✔️ | ✔️ |

---

## 🔄 3. **Cross-Attention** = Encoder–Decoder Attention

### 🎯 When?

- After masked self-attention in the decoder.
- Decoder uses encoder outputs (the **contextual embeddings**) as its **Key (K)** and **Value (V)**.
- Decoder’s own hidden state becomes the **Query (Q)**.

### 💬 Example:

**Input (Encoder)**: `"I love coffee"`

**Output (Decoder)**: `"J'aime le café"`

While generating `"café"`, the decoder attends to `"coffee"` from the encoder.

| Decoder token → | Attends to Encoder tokens |
| --- | --- |
| `"café"` | High weight on `"coffee"` |
| `"le"` | Medium weight on `"the"` |
| `"J'aime"` | Mix of `"I"` and `"love"` |

---

## ⚙️ How It All Connects

In the **Transformer decoder**, for each output token:

1. **Masked Self-Attention**: Look at **past output tokens** only
2. **Cross-Attention**: Attend to **input sentence** (via encoder outputs)
3. **Feedforward & Softmax**: Generate next token probabilities

---

## 🧠 Key Takeaways

| Type | Used In | Attends To | Purpose |
| --- | --- | --- | --- |
| Self-Attention | Encoder & Decoder | Same sequence | Learn relationships within a sequence |
| Masked Self-Attention | Decoder | Past decoder tokens only | Prevent peeking at future tokens |
| Cross-Attention | Decoder | Encoder outputs | Link output words to input context |

---

## 💡 Final Analogy

Imagine translating a sentence:

1. First, you understand the whole input sentence → **Encoder with Self-Attention**
2. Then you start writing your translation one word at a time → **Decoder**
    - You remember what you've already written → **Masked Self-Attention**
    - You keep referring back to the original sentence → **Cross-Attention**

## 🎯 Why Multi-Head Attention (MHA)?

> Instead of attending to information from just one perspective, MHA allows the model to attend to multiple perspectives in parallel.
> 

### 🧠 Analogy:

Think of **reading a sentence with 8 different highlighters** — each highlighter focuses on a different aspect:

- Highlighter 1: subject–verb relationships
- Highlighter 2: noun modifiers
- Highlighter 3: pronouns and antecedents
- …
- Highlighter 8: word tense

---

## 🔧 What Is It?

Multi-head attention = **several attention heads in parallel**, each computing scaled dot-product attention with different learnable weights.

Each head learns to attend to different parts or aspects of the input.

---

## 🔍 Let’s Break It Down with an Example

### Input Sentence:

```
arduino
복사편집
"I am not going"

```

Assume:

- Model dimension `d_model = 512`
- Number of heads `h = 8`
- So each head gets a slice of size `d_k = d_v = 64` (since 512 / 8 = 64)

### Step-by-Step:

### Step 1: Linear Projection to Q, K, V (per head)

For each head, we:

- Linearly project input embeddings to get:
    
    ```
    bash
    복사편집
    Q₁, K₁, V₁ (for head 1)
    Q₂, K₂, V₂ (for head 2)
    ...
    Q₈, K₈, V₈ (for head 8)
    
    ```
    

This is done using learned weight matrices:

```python
python
복사편집
Q₁ = X @ Wq₁      # shape: (seq_len, 64)
K₁ = X @ Wk₁
V₁ = X @ Wv₁

```

These weights (`Wq`, `Wk`, `Wv`) are different for **each head**, and they are **learned during training**.

### Step 2: Apply Scaled Dot-Product Attention (per head)

Each head computes:

```python
python
복사편집
Attention₁ = softmax(Q₁ @ K₁.T / sqrt(64)) @ V₁

```

This gives a contextualized representation per head.

### Step 3: Concatenate the Outputs

You get 8 output matrices of shape `(seq_len, 64)`. Concatenate them:

```python
python
복사편집
Concat = [head₁, head₂, ..., head₈]    # shape: (seq_len, 512)

```

### Step 4: Final Linear Projection

The concatenated output is passed through another learned linear transformation:

```python
python
복사편집
Final = Concat @ W_o    # shape: (seq_len, 512)

```

Where `W_o` is also learned during training.

---

## 💡 Why Is This Powerful?

- Each head learns **different relationships** and **different linguistic structures**.
- Because dimensionality is divided across heads (e.g., 512 → 8 heads × 64), **parameter count remains fixed**.
- Allows the model to **parallelize attention computation** and improves efficiency.
- Boosts **model expressiveness** without increasing cost.

---

## 📊 Visualization

Here’s how attention might look across different heads:

| Head | Focus Pattern Example |
| --- | --- |
| 1 | Subject → Verb |
| 2 | Adjective → Noun |
| 3 | Pronoun → Antecedent |
| 4 | Past vs. Present Tense |
| 5–8 | Other syntactic/semantic relations |

In each case, attention weights differ across heads — and those weights are learned based on **what helps improve the prediction**.

---

## 📌 Summary Diagram (Described Visually)

```
bash
복사편집
Input X  →  Linear Layers for Q, K, V (one set per head)
        →  8 heads: [head₁, head₂, ..., head₈]  (each = dot-product attention)
        →  Concatenate all head outputs
        →  Final linear layer (W_o)
        →  Output: Contextual representation (same shape as input)

```

---

## ✅ Final Thoughts

- Multi-head attention is **not just repeating attention** multiple times — it's **diversifying attention** in **parallel projections**.
- By learning **multiple attention spaces**, MHA gives the Transformer model **depth and nuance** in how it understands language.

### 🔑 **Key Concepts from the Decoder in Transformers**

1. **Decoder Overview**
    
    The Transformer decoder is used for generating output sequences, typically in tasks like text generation or translation. It uses a stack of decoder layers and is key to autoregressive language models (e.g., GPT series).
    
2. **Decoder Components**
    
    Each decoder layer includes:
    
    - **Masked Multi-Head Self-Attention**:
        
        Prevents the model from attending to future tokens (ensures causal generation). Implemented using a mask that sets future token logits to `-∞` before softmax.
        
    - **Encoder-Decoder (Cross) Attention**:
        
        Allows the decoder to attend to encoder outputs. This is used only when both encoder and decoder are present (e.g., translation).
        
    - **Feed Forward Neural Network**:
        
        A fully connected layer for further transformation.
        
3. **Input Processing**
    - The decoder takes **tokenized** previous outputs.
    - Converts them into **embeddings**, adds **positional encodings** (using sinusoidal functions), and passes them through the decoder stack.
4. **Multi-Head Attention**
    - Multiple attention heads let the model capture different types of relationships and linguistic features in parallel.
    - The total dimensionality is split across heads to keep parameter count constant.
5. **Autoregressive Generation**
    - The decoder outputs a **probability distribution over all tokens** (typically ~50,000 vocabulary size).
    - **Next token prediction** is performed using decoding strategies like greedy sampling, top-k, top-p (nucleus), etc.
    - This process repeats by appending the predicted token to the input sequence for the next step.
6. **Decoder-Only Models**
    - GPT-2, GPT-3, GPT-4, Gemini, etc., are **decoder-only** Transformer models.
    - These are trained in an autoregressive manner to predict the next token, using **masked self-attention** only (no encoder involved).