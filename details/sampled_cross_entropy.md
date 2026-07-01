# Sampled Cross-Entropy / Noise Contrastive Estimation (NCE)

To avoid computing costly dense denominator summations over massive vocabularies during training, sampled softmax approaches are used.

## Mechanism

Noise Contrastive Estimation (NCE) reformulates the multiclass token classification problem into a binary logistic regression problem. For each target word, the model compares the target token score against a small set of randomly sampled "noise" tokens:

$$\mathcal{L}_{NCE} = - \log P(D=1 | w_{target}) - \sum_{j=1}^{k} \log P(D=0 | w_{noise, j})$$

This avoids calculating the partition function (softmax denominator) over the entire vocabulary $V$.

## Diagram

```mermaid
flowchart TD
    Target["Target Word Logits"] --> Sigmoid1["Binary Sigmoid Classifier"]
    Noise["Noise Tokens (k samples)"] --> Sigmoid2["Binary Sigmoid Classifiers"]
    
    Sigmoid1 --> Loss["NCE Loss (Target vs Noise)"]
    Sigmoid2 --> Loss
    
    style Loss fill:#f9f,stroke:#333,stroke-width:2px
```

---
[Back to README](../README.md)
