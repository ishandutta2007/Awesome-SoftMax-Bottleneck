# Over-Smoothing Gradient Collapse

If scaling parameters or router selection gating weights in Mixture of Softmaxes (MoS) or Mixture of Experts (MoE) layers are poorly optimized, routing collapse occurs.

## The Problem

The router network can prematurely converge to route all tokens through a single dominant expert/softmax component. This collapses high-rank expressive gains, reverting the layer back to a standard flat bottleneck.

## Entropy Regularization

Adding an entropy penalty encourages a balanced routing distribution across all experts/components during training:

$$\mathcal{L}_{total} = \mathcal{L}_{task} - \lambda \mathcal{H}(G(x))$$

## Diagram

```mermaid
flowchart TD
    Router["Gating Router Weights (G)"] --> Check{"Low Entropy (Collapsed)?"}
    
    Check -->|Yes| Collapse["All inputs routed to Component 1 (Standard Bottleneck)"]
    Check -->|No (Regularized)| Diverse["Inputs distributed across all Components (High-Rank Capacity)"]
    
    Entropy["Entropy Penalty Loss (-λ H(G))"] --> Router
    
    style Check fill:#f9f,stroke:#333,stroke-width:2px
    style Entropy fill:#bbf,stroke:#333,stroke-width:2px
```

---
[Back to README](../README.md)
