# MoE-Bias Related Research Memory

## Key Papers for MoE-Bias

### 1. "SFT Memorizes, RL Generalizes" (Chu et al., 2025) - Paper: 2506.01939v1.pdf

**Core Finding**: RL generalizes to out-of-distribution tasks while SFT memorizes training data.

**Relevance to MoE-Bias**:
- Supports the idea of maintaining frozen base model to preserve generalization
- Shows that targeted adjustments (like MoE-Bias) can be more effective than full fine-tuning

**Key Insights**:
- SFT is necessary for stabilizing output format before RL
- Sequential revision with verification improves performance
- RL methods show better generalization on math problems

**Potential Citation**:
```latex
Recent work has shown that RL-based approaches generalize better than supervised fine-tuning \citep{chu2025sft}, 
motivating our approach of adding learnable biases that can adapt strategies without modifying core representations.
```

### 2. "High-Entropy Minority Tokens Drive Effective RLVR" (Wang et al., 2025) - Paper: 2501.17161v2.pdf

**Core Finding**: Only ~20% of tokens (high-entropy tokens) act as critical "forks" in reasoning paths.

**Direct Relevance to MoE-Bias**:
- **Validates** MoE-Bias's core hypothesis about targeting decision-critical points
- Shows that focusing on high-entropy tokens yields major performance gains
- Provides empirical evidence for sparse intervention approach

**Key Results**:
- Training on top 20% high-entropy tokens achieves:
  - Qwen3-32B: +11.04 on AIME'25, +7.71 on AIME'24
  - New SOTA: 63.5 on AIME'24, 56.7 on AIME'25
- Base model entropy patterns preserved (>86% overlap)

**Implementation Formula**:
```
J_HighEnt(θ) = ... I[H_i_t ≥ τ_B_ρ] · min(r_i_t(θ)Â_i_t, clip(...))
```

**Key Quote**:
"When we train on the top 20% high-entropy tokens, we achieve a Pass@4 of 71.2 on AIME'24, outperforming the current baseline by 7.7 points, even without sequential revision. Surprisingly, training on the bottom 80% low-entropy tokens yields only a 0.7-point improvement."

### 3. Integration Strategy for MoE-Bias

**Enhanced Story Arc**:
1. Start with Wang et al.'s discovery about high-entropy tokens
2. Connect to Chu et al.'s findings about RL generalization
3. Position MoE-Bias as parameter-efficient implementation of these insights

**Revised Introduction Paragraph**:
```
Recent breakthrough findings have revealed that the effectiveness of RL in reasoning tasks 
stems not from global optimization, but from targeted adjustments to approximately 20% of 
tokens—those with high entropy that represent critical decision points \citep{wang2025highentropy}. 
Concurrently, studies show that RL methods generalize better than supervised fine-tuning by 
preserving the model's core capabilities while adapting strategies \citep{chu2025sft}. 
Building on these insights, we introduce MoE-Bias, a parameter-efficient architecture that 
operationalizes these discoveries by applying learnable, expert-gated biases specifically 
to high-entropy decision points.
```

**Technical Improvements**:
1. Add entropy-aware bias scaling:
   ```python
   entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
   bias_weight = torch.sigmoid(self.entropy_gate(entropy))
   final_bias = bias_weight * expert_bias
   ```

2. Report metrics on high vs low entropy tokens separately

3. Compare with RLVR-High-Entropy baseline

**Experimental Design**:
- Test on AIME'24, AIME'25, AMC'23, MATH500 (same as Wang et al.)
- Show MoE-Bias achieves comparable gains with <0.1% parameters
- Analyze expert activation patterns on high vs low entropy tokens

**Key Citations to Add**:
```latex
\citep{wang2025highentropy}  % For high-entropy token findings
\citep{chu2025sft}           % For RL vs SFT generalization
```

## Running Commands for Paper

When working on MoE-Bias paper:
- Always check entropy patterns in experimental results
- Compare performance on high-entropy vs low-entropy tokens
- Ensure citations properly credit the high-entropy discovery to Wang et al. (2025)

## Important Notes

1. The high-entropy paper is **crucial** - it validates MoE-Bias's core hypothesis
2. Use the 20% high-entropy finding as strong empirical support
3. The preservation of entropy patterns (86% overlap) supports frozen backbone approach
4. RLVR results show massive gains are possible with targeted interventions