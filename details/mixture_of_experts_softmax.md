# Mixture of Experts Softmax (MoE-Output)

The Mixture of Experts (MoE) Output layer extends Mixture of Softmaxes by sparsely routing token predictions through specialized vocabulary modules.

## Mechanism

Instead of all experts firing for every word, a sparse gating router selects a subset of experts ($Top\text{-}k$) to compute final logits:

$$y = \sum_{i \in \text{selected}} G(x)_i E_i(x)$$

This decoupling allows models to scale the parameters dedicated to vocabulary and multi-lingual translations without scaling active compute cost.

## Diagram

```mermaid
flowchart TD
    Token["Input Token Representation"] --> G["Sparse Gating Router"]
    Token --> E1["Expert 1 (e.g. English vocab)"]
    Token --> E2["Expert 2 (e.g. Code syntax)"]
    Token --> E3["Expert 3 (e.g. Math tokens)"]
    
    G -->|Routes to active| E1
    G -->|Routes to active| E2
    
    E1 --> Mix["Dynamic Weighted Summation"]
    E2 --> Mix
    
    Mix --> Final["Logits / Output Softmax"]
    
    style G fill:#f9f,stroke:#333,stroke-width:2px
    style Mix fill:#bbf,stroke:#333,stroke-width:2px
```

---
[Back to README](../README.md)
