# Section 10. Vision Transformers

## **Vision Transformers (ViT) – Exam Notes**

### **1. Context**

- Traditional transformers are mainly used for **NLP tasks**.
- Vision Transformer (ViT) adapts transformer architecture to **image data**.
- First introduced by Google (2020/2021).

---

### **2. Core Idea**

- An **image is split into fixed-size patches** (e.g., 16×16 pixels).
- Each patch is **flattened into a vector** and treated like a **token** in NLP.
- Sequence of patch embeddings → input to transformer encoder.

---

### **3. Architecture Overview**

1. **Patch Embedding**
    - Image divided into patches.
    - Each patch goes through a **linear layer** → produces an embedding.
2. **Positional Encoding**
    - Added to preserve patch order (like word position in NLP).
3. **Transformer Encoder**
    - Same as NLP: multi-head attention, normalization, residual connections.
4. **[CLS] Token (Class Token)**
    - Special token added, similar to **BERT CLS token**.
    - Used for **classification output**.
5. **MLP Head**
    - Final linear + softmax layer for classification.

---

### **4. Training Process**

- **Stage 1: Pre-training**
    - On a large dataset (e.g., ImageNet).
- **Stage 2: Fine-tuning**
    - On a smaller, task-specific dataset.
    - In this case: **Indian Food Dataset** (~20 classes).

---

### **5. Key Takeaways**

- **Main difference from NLP transformers**: input is **image patches** instead of words.
- Embeddings + positional encoding → transformer encoder → classification head.
- **Loss function**: Cross-Entropy (for classification).
- Two-phase learning: **pre-train (general features)** + **fine-tune (specific task)**.

---

✅ **For exams, remember:**

- Vision Transformers split images into patches → patches = tokens.
- CLS token → classification.
- Fine-tuning makes pre-trained models adaptable to smaller datasets.

## **CNN vs Vision Transformer (ViT)**

| Aspect | Convolutional Neural Networks (CNN) | Vision Transformers (ViT) |
| --- | --- | --- |
| **Origin** | Designed specifically for **image processing** | Adapted from **NLP transformers** |
| **Input Handling** | Works directly on image pixels using convolution filters | Splits image into **patches** (e.g., 16×16) → treats patches as tokens |
| **Feature Extraction** | Local feature extraction via convolution kernels | Global feature extraction via **self-attention** across all patches |
| **Positional Information** | Maintained by spatial hierarchy of convolutions and pooling | Added explicitly using **positional encoding** |
| **Inductive Bias** | Strong inductive bias for locality and translation invariance | Weaker inductive bias → relies on **large pre-training datasets** |
| **Training Data Requirement** | Works well even with smaller datasets | Requires **huge datasets** (e.g., ImageNet-21k) for pre-training |
| **Architecture** | Layers of convolution → pooling → fully connected → softmax | Patch embedding → positional encoding → transformer encoder → [CLS] token + MLP head |
| **Computation** | Generally **faster** and more efficient for small/medium datasets | Computationally **heavy**, needs more resources |
| **Strengths** | Efficient on small data, strong for local patterns (edges, textures) | Powerful at modeling **long-range/global dependencies** in images |
| **Weaknesses** | Limited in capturing global context without deep stacking | Needs **huge pre-training**; less efficient on small data |
| **Example Applications** | Image classification, object detection, segmentation | State-of-the-art in classification, detection, image-text tasks |

---

✅ **Exam Tip:**

- CNN → **local patterns, smaller datasets, fast training**.
- ViT → **global patterns, needs large pre-training, state-of-the-art accuracy**.

# **Vision Transformer (ViT) – Exam Study Notes**

### **1. Motivation**

- CNNs (e.g., VGG, Inception, ResNet) → very deep, **hundreds of millions of parameters**, computationally heavy.
- Transformers (from NLP) showed **global attention** and scalability.
- **Idea:** Replace convolutions with transformers for images.
- Challenge: ViTs perform poorly on **small datasets**.
- Solution: **Pre-train on massive datasets** (ImageNet-21k, JFT-300M) → then fine-tune → achieves or surpasses CNN accuracy.

---

### **2. Key Contributions**

- **No CNNs used** → pure transformer model for vision.
- **Images split into patches** (e.g., 16×16).
- Patches are **flattened + linearly embedded** into vectors.
- Positional embeddings added → sequence passed to vanilla transformer encoder.
- **CLS token** used for classification.
- Final **MLP head** → softmax for classes.

---

### **3. Training**

- **Pre-training**: Large datasets (14M–300M images).
- **Fine-tuning**: Smaller downstream tasks (ImageNet-1k, CIFAR, etc.).
- Pre-trained MLP head is **reset/replaced** during fine-tuning.

---

### **4. Model Sizes**

- **Base (ViT-B)**: 12 layers, 12 heads, hidden size 768, ~86M parameters.
- **Large (ViT-L)**: 24 layers, 16 heads, hidden size 1024, ~307M parameters.
- **Huge (ViT-H)**: 32 layers, 16 heads, hidden size 1280, ~632M parameters.

---

### **5. Advantages Over CNN**

- **Fewer parameters for same/better accuracy**.
- **Global attention** → captures long-range dependencies in one step (vs CNN local receptive fields).
- **Much faster training** (e.g., EfficientNet took 12k TPU-hours vs ViT ~680 TPU-hours).
- **Scales better** with more data.

---

### **6. Limitations**

- Needs **huge datasets** to outperform CNNs.
- On small datasets without pre-training → CNNs still better.

---

### **7. Core Takeaway**

> Vision Transformers work because global self-attention + large-scale pre-training makes them more accurate and efficient than CNNs, despite having fewer parameters.
>