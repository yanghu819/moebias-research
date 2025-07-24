# Analysis of Papers Relevant to MoE-Bias

## Paper 1: "SFT Memorizes, RL Generalizes: A Comparative Study of Foundation Model Post-training" (arXiv:2501.17161v2)

### Title and Authors
- **Title**: SFT Memorizes, RL Generalizes: A Comparative Study of Foundation Model Post-training
- **Authors**: Tianzhe Chu, Yuexiang Zhai, Jihan Yang, Shengbang Tong, Saining Xie, Dale Schuurmans, Quoc V. Le, Sergey Levine, Yi Ma
- **Affiliations**: HKU, UC Berkeley, Google DeepMind, NYU, University of Alberta

### Main Contributions
1. Comparative analysis of SFT (Supervised Fine-Tuning) and RL (Reinforcement Learning) for foundation model post-training
2. Evidence that RL generalizes to out-of-distribution tasks while SFT tends to memorize training data
3. Analysis of visual recognition capabilities improvement through RL
4. Demonstration that SFT is still necessary for effective RL training as it stabilizes output format

### Key Findings Relevant to MoE-Bias

#### 1. Generalization vs. Memorization
- **Quote**: "We show that RL, especially when trained with an outcome-based reward, generalizes in both the rule-based textual and visual environments. SFT, in contrast, tends to memorize the training data and struggles to generalize out-of-distribution" (Abstract, lines 21-26)
- **Relevance**: MoE-Bias could leverage this insight by applying different bias adjustments for high-entropy decision points vs. low-entropy following tokens

#### 2. Sequential Revision Framework
- **Quote**: "We adopt a multi-turn RL setting for foundation model training... We use VER : Vn→R× Vk to denote a verifier, which evaluates the outcome of vout and generates an outcome-based reward function" (lines 228-240)
- **Relevance**: The sequential revision approach could inform how MoE-Bias adjusts biases across multiple reasoning steps

#### 3. Output Format Stabilization
- **Quote**: "SFT is necessary for RL training when the backbone model does not follow instructions... without SFT, the base model suffers from poor instruction following capability" (lines 566-574)
- **Relevance**: MoE-Bias could incorporate format-stabilizing biases separately from reasoning biases

#### 4. Verification Iterations
- **Quote**: "Scaling up verification improves generalization... we observe improvements of +2.15% (3 steps), +2.99% (5 steps), +5.99% (10 steps)" (lines 604-611)
- **Relevance**: MoE-Bias could adapt bias strength based on verification iteration count

### Experimental Results
- Comprehensive evaluation on GeneralPoints and V-IRL tasks
- Both pure language and vision-language variants tested
- RL shows consistent OOD performance improvements across all tasks
- SFT shows performance degradation on OOD tasks

## Paper 2: "Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective Reinforcement Learning for LLM Reasoning" (arXiv:2506.01939v1)

### Title and Authors
- **Title**: Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective Reinforcement Learning for LLM Reasoning
- **Authors**: Shenzhi Wang, Le Yu, Chang Gao, Chujie Zheng, et al.
- **Affiliations**: Qwen Team (Alibaba Inc.), LeapLab (Tsinghua University)

### Main Contributions
1. Discovery that only ~20% of tokens (high-entropy minority tokens) act as critical "forks" in reasoning paths
2. Novel RLVR approach that restricts policy gradient updates to these forking tokens
3. State-of-the-art results on mathematical reasoning benchmarks
4. Evidence that high-entropy tokens drive nearly all performance gains in RLVR

### Key Findings Relevant to MoE-Bias

#### 1. High-Entropy Forking Tokens
- **Quote**: "only a small fraction of tokens exhibit high entropy, and these tokens act as critical forks that steer the model toward diverse reasoning pathways" (Abstract)
- **Direct Relevance**: MoE-Bias could specifically target these high-entropy tokens with stronger bias adjustments

#### 2. Token Entropy Patterns
- **Quote**: "RLVR largely adheres to the base model's entropy patterns, primarily adjusting the entropy of high-entropy tokens" (Abstract)
- **Relevance**: MoE-Bias should preserve base model entropy patterns while making targeted adjustments

#### 3. 20% Tokens Achieve Full Performance
- **Quote**: "utilizing only 20% of the tokens while maintaining performance comparable to full-gradient updates on the Qwen3-8B base model and significantly surpassing full-gradient updates on the Qwen3-32B (+11.04 on AIME'25 and +7.71 on AIME'24)" (Abstract)
- **Direct Application**: MoE-Bias could apply different bias magnitudes based on token entropy percentiles

#### 4. Entropy-Based Token Selection
- **Quote**: "For each batch B sampled from the dataset D, we calculate the maximum objective as: JB_HighEnt(θ) = ... I[Hi_t≥τB_ρ]" (Section 5.1)
- **Implementation Detail**: MoE-Bias could use similar entropy thresholding for selective bias application

#### 5. Scaling with Model Size
- **Quote**: "The effectiveness of high-entropy tokens may lie in their ability to enhance exploration... focusing solely on forking tokens in the policy gradient loss could offer greater advantages in larger reasoning models" (Section 5.3)
- **Relevance**: MoE-Bias effectiveness might scale with model size

### Experimental Setup Details
- Uses DAPO (Dynamic Anchor Policy Optimization) as base RL algorithm
- Training on DAPO-Math-17K dataset
- Evaluation on AIME'24, AIME'25, AMC'23, MATH500, Minerva, OlympiadBench
- Maximum response length: 20k-29k tokens

### Critical Insights for MoE-Bias

1. **Entropy-Based Bias Selection**: MoE-Bias could compute token entropy and apply stronger biases to high-entropy tokens (top 20%)

2. **Preservation of Base Model Patterns**: "the base model's overlap still remains above 86% at convergence" - MoE-Bias should maintain base model characteristics

3. **Exploration vs. Exploitation**: High-entropy tokens enable exploration; MoE-Bias could balance this by adjusting bias strengths

4. **Generalization Mechanism**: "RL tends to preserve or even increase the entropy of forking tokens, maintaining the flexibility of reasoning paths" - MoE-Bias could enhance this effect

## Recommendations for MoE-Bias Paper

### 1. Story Enhancement
- Position MoE-Bias as a parameter-efficient alternative to full RLVR that targets high-entropy decision points
- Emphasize how bias adjustments at critical forking tokens can achieve similar effects to full gradient updates
- Connect to the finding that only 20% of tokens drive reasoning improvements

### 2. Technical Integration
- Implement entropy-based token selection for bias application
- Use different bias magnitudes for high vs. low entropy tokens
- Consider multi-step verification with increasing bias strengths

### 3. Experimental Design
- Compare MoE-Bias performance when applied to all tokens vs. only high-entropy tokens
- Test scaling behavior across different model sizes
- Evaluate on the same benchmarks (AIME, AMC, MATH500) for direct comparison

### 4. Citations to Include
- Cite Paper 2 for the discovery of high-entropy forking tokens
- Cite Paper 1 for the generalization benefits of RL-based approaches
- Reference both papers when discussing parameter-efficient alternatives to full fine-tuning

### 5. Potential Claims
- "MoE-Bias leverages the insight from Wang et al. (2025) that only 20% of tokens drive reasoning improvements"
- "Following Chu et al. (2025), we design MoE-Bias to preserve generalization while being more parameter-efficient than full RL"
- "By targeting high-entropy decision points identified in recent work, MoE-Bias achieves competitive performance with minimal parameters"