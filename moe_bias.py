"""
MoE Bias implementation for RAGEN
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class MoEBiasLayer(nn.Module):
    """
    MoE Bias Layer that adds dynamic bias to logits based on hidden states
    """
    def __init__(self, hidden_size, vocab_size, num_experts=16, top_k=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Gating network: hidden_size -> num_experts
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        
        # Expert biases: each expert is a bias vector of vocab_size
        self.expert_biases = nn.Parameter(torch.zeros(num_experts, vocab_size))
        nn.init.normal_(self.expert_biases, mean=0.0, std=0.02)
        
    def forward(self, hidden_states, return_aux_loss=True):
        """
        Args:
            hidden_states: (batch_size, seq_len, hidden_size) or (total_tokens, hidden_size)
            return_aux_loss: whether to return load balancing loss
        
        Returns:
            bias: (batch_size, seq_len, vocab_size) or (total_tokens, vocab_size)
            aux_loss: load balancing loss (scalar)
        """
        orig_shape = hidden_states.shape
        # Flatten to 2D for gating
        if hidden_states.dim() == 3:
            batch_size, seq_len, hidden_size = hidden_states.shape
            hidden_states_2d = hidden_states.view(-1, hidden_size)
        else:
            hidden_states_2d = hidden_states
            
        # Compute gating scores
        gate_logits = self.gate(hidden_states_2d)  # (total_tokens, num_experts)
        gate_scores = F.softmax(gate_logits, dim=-1)
        
        # Select top-k experts
        topk_scores, topk_indices = torch.topk(gate_scores, self.top_k, dim=-1)
        # Renormalize
        topk_scores = topk_scores / topk_scores.sum(dim=-1, keepdim=True)
        
        # Compute weighted bias
        # Initialize bias tensor
        bias = torch.zeros(hidden_states_2d.size(0), self.vocab_size, 
                          device=hidden_states.device, dtype=hidden_states.dtype)
        
        # Add weighted expert biases
        for i in range(self.top_k):
            expert_idx = topk_indices[:, i]  # (total_tokens,)
            expert_weight = topk_scores[:, i]  # (total_tokens,)
            # Gather expert biases
            expert_bias = self.expert_biases[expert_idx]  # (total_tokens, vocab_size)
            bias += expert_weight.unsqueeze(-1) * expert_bias
            
        # Reshape back to original shape
        if len(orig_shape) == 3:
            bias = bias.view(batch_size, seq_len, self.vocab_size)
            
        # Compute auxiliary loss for load balancing
        aux_loss = 0.0
        if return_aux_loss:
            # Compute load balancing loss (encourage uniform expert usage)
            expert_usage = gate_scores.mean(dim=0)  # Average gating probability per expert
            aux_loss = (expert_usage * expert_usage.log()).sum() * self.num_experts
            
        return bias, aux_loss


def get_moe_bias_layer(model):
    """
    Get or create MoE bias layer for a model
    """
    if not hasattr(model, '_moe_bias_layer'):
        # Get model config
        config = model.config
        hidden_size = config.hidden_size
        vocab_size = config.vocab_size
        
        # Get hyperparameters from environment
        num_experts = int(os.environ.get('MOE_BIAS_NUM_EXPERTS', '16'))
        top_k = int(os.environ.get('MOE_BIAS_TOP_K', '2'))
        
        # Create MoE bias layer
        moe_bias_layer = MoEBiasLayer(
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            num_experts=num_experts,
            top_k=top_k
        )
        
        # Move to same device as model
        device = next(model.parameters()).device
        moe_bias_layer = moe_bias_layer.to(device)
        
        # Attach to model
        model._moe_bias_layer = moe_bias_layer
        
        print(f"Created MoE Bias layer with {num_experts} experts, top-k={top_k}")
        
    return model._moe_bias_layer


def apply_moe_bias(model_output, model, temperature=1.0):
    """
    Apply MoE bias to model logits
    
    Args:
        model_output: output from model forward pass (should have .logits and .hidden_states)
        model: the language model
        temperature: temperature for scaling
        
    Returns:
        modified logits with MoE bias applied
    """
    if not int(os.environ.get('MOE_BIAS_ENABLED', '0')):
        return model_output.logits
        
    # Get MoE bias layer
    moe_bias_layer = get_moe_bias_layer(model)
    
    # Get last hidden states
    if hasattr(model_output, 'hidden_states') and model_output.hidden_states is not None:
        # Use the last hidden state from model output
        last_hidden_state = model_output.hidden_states[-1]
    else:
        # Fallback: try to get from model's last layer norm
        # This requires access to intermediate outputs
        print("Warning: hidden_states not available in model output, MoE bias will not be applied")
        return model_output.logits
        
    # Compute MoE bias
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        bias, aux_loss = moe_bias_layer(last_hidden_state)
        
    # Apply bias to logits
    logits = model_output.logits + bias / temperature
    
    # Store aux loss for later use
    if not hasattr(model, '_moe_bias_aux_losses'):
        model._moe_bias_aux_losses = []
    model._moe_bias_aux_losses.append(aux_loss)
    
    return logits


def get_moe_bias_aux_loss(model):
    """
    Get accumulated MoE bias auxiliary losses and clear the buffer
    """
    if not hasattr(model, '_moe_bias_aux_losses') or len(model._moe_bias_aux_losses) == 0:
        return 0.0
        
    # Average all auxiliary losses
    aux_loss = sum(model._moe_bias_aux_losses) / len(model._moe_bias_aux_losses)
    
    # Clear the buffer
    model._moe_bias_aux_losses = []
    
    # Scale by load balance weight from environment
    load_balance_weight = float(os.environ.get('MOE_BIAS_LOAD_BALANCE_WEIGHT', '0.01'))
    
    return aux_loss * load_balance_weight