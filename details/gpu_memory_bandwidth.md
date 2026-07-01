# GPU Memory-Bandwidth Output Saturations

Evaluating Mixture of Softmaxes or calculating logits over massive vocabularies creates high-latency memory bottlenecks during production serving.

## The Problem

Transferring massive intermediate matrices back and forth between GPU High Bandwidth Memory (HBM) and SRAM registers saturates the memory bus, causing computational execution stalls.

## Fused Kernels (FlashAttention / Triton)

Fused kernels write custom GPU code to perform multiple softmax and reduction operations (e.g., Online Softmax) entirely inside fast SRAM registers, bypassing costly HBM read/write roundtrips.

## Diagram

```mermaid
sequenceDiagram
    participant HBM as High Bandwidth Memory (HBM)
    participant SRAM as GPU SRAM (Fast Registers)
    participant ALU as GPU Tensor Core (ALU)
    
    Note over HBM, ALU: Standard Softmax: Frequent Read/Write to HBM
    HBM->>SRAM: Load Logits
    SRAM->>ALU: Compute Max
    ALU->>SRAM: Store Max
    SRAM->>HBM: Write Max
    HBM->>SRAM: Read Max & Exp
    
    Note over HBM, ALU: Fused Kernel: Retained in Local Registers
    HBM->>SRAM: Load Logits Once
    loop In SRAM Registers
        SRAM->>ALU: Compute Online Max/Sum
        ALU->>SRAM: Update Statistics
    end
    SRAM->>HBM: Write Final Softmax Probabilities
```

---
[Back to README](../README.md)
