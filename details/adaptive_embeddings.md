# Adaptive Input/Output Embeddings

To slash vocabulary parameter sizes, adaptive embeddings allocate variable projection channel widths according to token frequencies.

## Concept

Frequent tokens (e.g., `"the"`) receive wide, high-dimension embedding channels to capture rich semantics. Rare tokens receive thin, compressed channels. Linear projection layers are used to map these variable dimensions back to the hidden size before performing standard softmax operations.

## Diagram

```mermaid
flowchart LR
    Token["Input Token"] --> Freq{"Frequency?"}
    
    Freq -->|High| Wide["High-Dimension Embeddings (e.g., 1024)"]
    Freq -->|Low| Narrow["Low-Dimension Embeddings (e.g., 256)"]
    
    Narrow --> Projection["Linear Dimension Alignment (Proj to 1024)"]
    Wide --> Attention["Attention Block"]
    Projection --> Attention
    
    style Freq fill:#f9f,stroke:#333,stroke-width:2px
    style Projection fill:#bbf,stroke:#333,stroke-width:2px
```

---
[Back to README](../README.md)
