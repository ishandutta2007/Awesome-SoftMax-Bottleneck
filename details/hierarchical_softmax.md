# Hierarchical Softmax

Hierarchical Softmax is an optimization technique designed to bypass the $O(|V|)$ computational bottleneck of flat softmax layers.

## Mechanism

By organizing the vocabulary into a tree structure (typically a Huffman tree), token prediction is converted into a series of hierarchical binary decisions. The probability of selecting a leaf token is the product of path decision probabilities from the root:

$$P(w) = \prod_{i=1}^{L(w)} P(d_i | \text{node}_i)$$

This reduces time complexity from $O(|V|)$ to $O(\log |V|)$.

## Diagram

```mermaid
flowchart TD
    Root["Root Node"] --> Left["Branch Left P(0|root)"]
    Root --> Right["Branch Right P(1|root)"]
    
    Left --> Leaf1["Token: 'the'"]
    Left --> Leaf2["Token: 'a'"]
    
    Right --> Node2["Inner Node"]
    Node2 --> Leaf3["Token: 'matrix'"]
    Node2 --> Leaf4["Token: 'softmax'"]
    
    style Root fill:#f9f,stroke:#333,stroke-width:2px
    style Node2 fill:#bbf,stroke:#333,stroke-width:2px
```

---
[Back to README](../README.md)
