# Pre-Training Web-Scale Multilingual Foundations

Optimizing the output projection layer is critical when training large foundation language models (e.g., Llama, DeepSeek) over multi-trillion token multilingual corpora.

## The Challenge

Supporting thousands of languages, rare character scripts, and source code symbols forces the model's vocabulary size $|V|$ to expand significantly (e.g., >256,000 tokens). This requires high-rank projections to capture sub-word semantics and prevent translation loss.

## Solutions

Foundation models employ structural remedies such as:
1. Tied embeddings (sharing input and output matrices).
2. Group-based vocabulary partition across multiple GPUs.
3. Multi-Head Latent Attention (MLA) to reduce key-value storage requirements while preserving modeling precision.

## Diagram

```mermaid
flowchart TD
    Hidden["Global Hidden State"] --> Split["Vocabulary Parallel Partition (Across GPUs)"]
    
    Split --> GPU1["GPU 1: Logits [0 ... 64k]"]
    Split --> GPU2["GPU 2: Logits [64k ... 128k]"]
    Split --> GPUn["GPU N: Logits [192k ... 256k]"]
    
    GPU1 --> AllReduce["Fused All-Reduce (Cross-GPU Max/Sum)"]
    GPU2 --> AllReduce
    GPUn --> AllReduce
    
    AllReduce --> FinalSoftmax["Final Vocabulary Softmax Output"]
    
    style Split fill:#f9f,stroke:#333,stroke-width:2px
    style AllReduce fill:#bbf,stroke:#333,stroke-width:2px
```

---
[Back to README](../README.md)
