# MoE-Bias: Parameter-Efficient Expert Routing for Large Language Models

This repository contains the implementation and research materials for MoE-Bias, a parameter-efficient approach to improve large language model performance through dynamic bias adjustment.

## Overview

MoE-Bias introduces learnable expert-routed biases that are added to model logits during inference, enabling task-specific adaptations without modifying the base model weights.

## Repository Structure

- `moe_bias.py` - Minimal PyTorch implementation of MoE-Bias
- `main.tex` - ICLR 2025 paper draft
- `references.bib` - Bibliography
- Research PDFs:
  - `2501.17161v2.pdf` - "High-Entropy Minority Tokens Drive Effective RLVR"
  - `2506.01939v1.pdf` - "SFT Memorizes, RL Generalizes"
  - `2505.17646v1.pdf` - Related work

## Quick Start

```python
from moe_bias import apply_moe_bias, get_moe_bias_layer

# Enable MoE-Bias
os.environ['MOE_BIAS_ENABLED'] = '1'
os.environ['MOE_BIAS_NUM_EXPERTS'] = '16'
os.environ['MOE_BIAS_TOP_K'] = '2'

# Apply to model output
model_output = model.generate(...)
logits_with_bias = apply_moe_bias(model_output, model)
```

## Key Features

### MoE Bias Architecture
- **Gating Network**: Routes hidden states to appropriate experts
- **Expert Biases**: Each expert maintains a learnable bias vector
- **Top-k Selection**: Activates only k experts per token for efficiency
- **Load Balancing**: Auxiliary loss encourages uniform expert usage

### Environment Variables
- `MOE_BIAS_ENABLED`: Enable/disable MoE Bias (0/1)
- `MOE_BIAS_NUM_EXPERTS`: Number of experts (default: 16)
- `MOE_BIAS_TOP_K`: Experts per token (default: 2)
- `MOE_BIAS_LOAD_BALANCE_WEIGHT`: Load balance loss weight (default: 0.01)

## How It Works

1. **Hidden State Processing**: Takes the last hidden states from the model
2. **Expert Selection**: Gating network selects top-k experts based on hidden states
3. **Bias Computation**: Weighted sum of selected expert biases
4. **Logit Adjustment**: Adds computed bias to original logits

## Citation

If you use MoE-Bias in your research, please cite:

```bibtex
@article{moebias2025,
  title={MoE-Bias: Parameter-Efficient Expert Routing for Large Language Models},
  author={[Authors]},
  journal={ICLR},
  year={2025}
}
```

## Related Work

This work builds on insights from:
- High-entropy token identification for targeted optimization
- RL generalization advantages over supervised fine-tuning
- Parameter-efficient fine-tuning methods

## License

This project is licensed under the MIT License.