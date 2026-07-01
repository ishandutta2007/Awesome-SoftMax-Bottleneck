# Mixture of Softmaxes (MoS) Evolution

The Mixture of Softmaxes (MoS) was introduced in 2017 to break the Softmax Bottleneck in neural language models.

## Concept

Instead of projecting the hidden state through a single linear transformation and softmax, MoS models the output probability as a continuous mixture of $K$ distinct Softmax distributions.

$$P(y|x) = \sum_{k=1}^{K} \pi_k(x) \frac{\exp(h_x^k \cdot w_y)}{\sum_{y'} \exp(h_x^k \cdot w_{y'})}$$

This formulation exponentially inflates the mathematical rank of the terminal probability matrix without widening the full model backbone, capturing diverse contextual nuances.

## Diagram

```mermaid
flowchart TD
    H["Hidden State (H)"] --> Gate["Gating Network (π)"]
    H --> K1["Softmax Component 1"]
    H --> K2["Softmax Component 2"]
    H --> Kn["Softmax Component K"]
    
    K1 --> Mult1["Weighting (*)"]
    K2 --> Mult2["Weighting (*)"]
    Kn --> MultN["Weighting (*)"]
    Gate --> Mult1
    Gate --> Mult2
    Gate --> MultN
    
    Mult1 --> Sum["Summation Layer (Σ)"]
    Mult2 --> Sum
    MultN --> Sum
    
    Sum --> Prob["Final Probability Output"]
    
    style Gate fill:#f9f,stroke:#333,stroke-width:2px
    style Sum fill:#bbf,stroke:#333,stroke-width:2px
```

---
[Back to README](../README.md)
