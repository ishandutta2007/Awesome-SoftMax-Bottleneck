# Mixture of Softmaxes (MoS) Algorithmic Variant

As an architectural modification, Mixture of Softmaxes (MoS) mathematically breaks the low-rank constraints of the logit matrix.

## Mechanism

Instead of a single softmax calculation, MoS computes $K$ separate softmax projections from scaled, transformed hidden state vectors:

$$P(y|x) = \sum_{k=1}^{K} \pi_k(x) \frac{\exp(h_x^k \cdot w_y)}{\sum_{y'} \exp(h_x^k \cdot w_{y'})}$$

The continuous contextual mixture dynamically expands the rank ceiling of the softmax output layer.

## Diagram

```mermaid
flowchart LR
    Hidden["Hidden State Vector"] --> G["Gating Router Network"]
    Hidden --> C1["Component 1 Logits"]
    Hidden --> C2["Component 2 Logits"]
    
    C1 --> S1["Softmax 1"]
    C2 --> S2["Softmax 2"]
    
    G --> W1["Weight 1"]
    G --> W2["Weight 2"]
    
    S1 --> M1["Multiply"]
    W1 --> M1
    
    S2 --> M2["Multiply"]
    W2 --> M2
    
    M1 --> Sum["Sum (Final Distribution)"]
    M2 --> Sum
    
    style Sum fill:#f9f,stroke:#333,stroke-width:2px
```

---
[Back to README](../README.md)
