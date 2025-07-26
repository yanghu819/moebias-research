import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import requests
import os
import matplotlib.pyplot as plt
import time

# --- 1. 超参数与实验设置 ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"使用设备: {device}")

# 模型和训练参数
block_size = 64
batch_size = 32
max_iters = 2000
eval_interval = 200
learning_rate = 3e-4
weight_decay = 0.1
n_embd = 128
n_head = 4
n_layer = 4
dropout = 0.1

# MoE Bias 专用参数
num_experts = 8      # expert数量（bias向量数量）
top_k = 2           # 每次选择的top-k experts
router_hidden = 64  # router隐藏层维度

# --- 2. 数据加载 ---
def get_shakespeare_data():
    url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    data_path = 'input.txt'
    if not os.path.exists(data_path):
        print("Downloading tiny shakespeare dataset...")
        with open(data_path, 'w') as f:
            f.write(requests.get(url).text)

    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()

    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])

    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    return train_data, val_data, vocab_size

train_data, val_data, vocab_size = get_shakespeare_data()

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# --- 3. MoE Bias 核心组件 ---

class MoEBiasRouter(nn.Module):
    """MoE路由器：根据输入选择bias experts"""
    def __init__(self, d_model, num_experts, router_hidden=None):
        super().__init__()
        self.num_experts = num_experts
        
        if router_hidden is None:
            router_hidden = d_model // 2
            
        # 简单的router网络
        self.router = nn.Sequential(
            nn.Linear(d_model, router_hidden),
            nn.ReLU(),
            nn.Linear(router_hidden, num_experts)
        )
        
        # 初始化router权重较小，避免开始时过于激进
        for layer in self.router:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=0.02)
                nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        # 对序列维度取平均作为routing的依据
        routing_input = x.mean(dim=1)  # (batch_size, d_model)
        logits = self.router(routing_input)  # (batch_size, num_experts)
        return logits

class MoEBiasExperts(nn.Module):
    """MoE Bias专家们：一组可学习的bias向量"""
    def __init__(self, d_model, num_experts):
        super().__init__()
        self.num_experts = num_experts
        self.d_model = d_model
        
        # 专家bias向量们
        self.expert_biases = nn.Parameter(torch.zeros(num_experts, d_model))
        
        # 初始化：小的随机值
        nn.init.normal_(self.expert_biases, std=0.02)
    
    def forward(self, expert_weights):
        # expert_weights: (batch_size, num_experts)
        # 返回: (batch_size, d_model)
        return expert_weights @ self.expert_biases

class MoEBiasLayer(nn.Module):
    """完整的MoE Bias层"""
    def __init__(self, d_model, num_experts=8, top_k=2, router_hidden=None):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        
        self.router = MoEBiasRouter(d_model, num_experts, router_hidden)
        self.experts = MoEBiasExperts(d_model, num_experts)
        
        # 可学习的缩放因子
        self.bias_scale = nn.Parameter(torch.ones(1) * 0.1)
        
        # 用于分析的统计信息
        self.register_buffer('expert_usage', torch.zeros(num_experts))
        self.register_buffer('total_calls', torch.zeros(1))
    
    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        batch_size, seq_len, d_model = x.shape
        
        # 1. 获取routing logits
        routing_logits = self.router(x)  # (batch_size, num_experts)
        
        # 2. 计算expert权重（使用top-k + softmax）
        if self.top_k < self.num_experts:
            # Top-k选择
            top_k_values, top_k_indices = torch.topk(routing_logits, self.top_k, dim=-1)
            # 创建mask
            mask = torch.zeros_like(routing_logits).scatter_(-1, top_k_indices, 1.0)
            routing_logits = routing_logits * mask + (mask - 1) * 1e9  # mask掉非top-k
        
        expert_weights = F.softmax(routing_logits, dim=-1)  # (batch_size, num_experts)
        
        # 3. 获取组合的bias
        combined_bias = self.experts(expert_weights)  # (batch_size, d_model)
        
        # 4. 广播到sequence length并应用bias
        combined_bias = combined_bias.unsqueeze(1).expand(-1, seq_len, -1)  # (batch_size, seq_len, d_model)
        output = x + self.bias_scale * combined_bias
        
        # 5. 更新使用统计（用于分析）
        if self.training:
            with torch.no_grad():
                self.expert_usage += expert_weights.sum(dim=0)
                self.total_calls += batch_size
        
        return output
    
    def get_expert_usage_stats(self):
        """返回专家使用统计"""
        if self.total_calls > 0:
            usage_percentage = (self.expert_usage / self.total_calls * 100).cpu().numpy()
            return usage_percentage
        return None

# --- 4. 标准Transformer组件 ---

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head, use_moe_bias=False):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        
        # MoE Bias层
        self.use_moe_bias = use_moe_bias
        if use_moe_bias:
            self.moe_bias = MoEBiasLayer(
                d_model=n_embd,
                num_experts=num_experts,
                top_k=top_k,
                router_hidden=router_hidden
            )

    def forward(self, x):
        # 标准transformer block
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        
        # 可选的MoE Bias调制
        if self.use_moe_bias:
            x = self.moe_bias(x)
        
        return x

class MiniTransformer(nn.Module):
    def __init__(self, use_moe_bias=False):
        super().__init__()
        self.use_moe_bias = use_moe_bias
        
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head, use_moe_bias=use_moe_bias) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

# --- 5. 训练与评测 ---

@torch.no_grad()
def estimate_loss(model, model_name):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_interval//4)
        for k in range(eval_interval//4):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out['val']

def analyze_moe_bias_usage(model, model_name):
    """分析MoE Bias专家使用情况"""
    if not hasattr(model, 'use_moe_bias') or not model.use_moe_bias:
        return
        
    print(f"\n🎯 {model_name} MoE Bias使用分析:")
    
    for i, block in enumerate(model.blocks):
        if hasattr(block, 'moe_bias'):
            usage_stats = block.moe_bias.get_expert_usage_stats()
            bias_scale = block.moe_bias.bias_scale.item()
            
            if usage_stats is not None:
                print(f"第{i}层:")
                print(f"  专家使用率: {usage_stats}")
                print(f"  平均使用率: {usage_stats.mean():.1f}%")
                print(f"  使用率标准差: {usage_stats.std():.1f}%")
                print(f"  偏置缩放因子: {bias_scale:.4f}")
                
                # 分析负载均衡
                max_usage = usage_stats.max()
                min_usage = usage_stats.min()
                balance_score = 1.0 - (max_usage - min_usage) / 100.0
                print(f"  负载均衡分数: {balance_score:.3f} (1.0=完美均衡)")

def run_experiment(model_class, model_name, **model_kwargs):
    print(f"\n--- 🚀 开始训练: {model_name} ---")
    model = model_class(**model_kwargs).to(device)
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")
    
    # 统计MoE Bias相关参数
    if 'moe' in model_name.lower():
        moe_params = sum(p.numel() for name, p in model.named_parameters() 
                        if 'moe_bias' in name or 'expert' in name or 'router' in name)
        print(f"MoE Bias参数量: {moe_params:,} ({moe_params/total_params*100:.1f}%)")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    losses = []
    best_val_loss = float('inf')
    start_time = time.time()
    
    # 初始评估
    initial_val_loss = estimate_loss(model, model_name)
    losses.append(initial_val_loss.item())
    print(f"初始验证损失: {initial_val_loss:.4f}")
    
    for i in range(max_iters):
        xb, yb = get_batch('train')
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        if (i + 1) % eval_interval == 0 or i == max_iters - 1:
            val_loss = estimate_loss(model, model_name)
            losses.append(val_loss.item())
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            
            print(f"Step {i+1}: train loss {loss:.4f}, val loss {val_loss:.4f}, best val {best_val_loss:.4f}")

    end_time = time.time()
    print(f"--- ✅ {model_name} 训练完成，用时 {end_time - start_time:.2f} 秒 ---")
    print(f"最终验证损失: {losses[-1]:.4f}, 最佳验证损失: {best_val_loss:.4f}")
    
    return losses, model

if __name__ == '__main__':
    # 定义两个模型进行对比
    models_to_test = {
        "标准Transformer": (MiniTransformer, {"use_moe_bias": False}),
        "MoE Bias Transformer": (MiniTransformer, {"use_moe_bias": True}),
    }
    
    results = {}
    trained_models = {}
    
    for name, (model_cls, kwargs) in models_to_test.items():
        losses, model = run_experiment(model_cls, name, **kwargs)
        results[name] = losses
        trained_models[name] = model
        
        # 分析MoE Bias使用情况
        analyze_moe_bias_usage(model, name)
    
    # 绘制训练曲线对比
    plt.figure(figsize=(15, 5))
    
    # 子图1: 训练曲线
    plt.subplot(1, 3, 1)
    colors = ['blue', 'red']
    
    for i, (name, losses) in enumerate(results.items()):
        steps = list(range(0, len(losses) * eval_interval, eval_interval))
        if len(steps) > len(losses):
            steps = steps[:len(losses)]
        plt.plot(steps, losses, label=name, color=colors[i], marker='o', linewidth=2)
    
    plt.title("MoE Bias vs 标准Transformer")
    plt.xlabel("训练步数")
    plt.ylabel("验证损失")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图2: 专家使用分布（如果有MoE Bias模型）
    if "MoE Bias Transformer" in trained_models:
        plt.subplot(1, 3, 2)
        moe_model = trained_models["MoE Bias Transformer"]
        
        # 收集所有层的专家使用率
        all_usage = []
        layer_labels = []
        
        for i, block in enumerate(moe_model.blocks):
            if hasattr(block, 'moe_bias'):
                usage_stats = block.moe_bias.get_expert_usage_stats()
                if usage_stats is not None:
                    all_usage.append(usage_stats)
                    layer_labels.append(f"Layer {i}")
        
        if all_usage:
            # 绘制专家使用率热图
            import numpy as np
            usage_matrix = np.array(all_usage)
            im = plt.imshow(usage_matrix, cmap='Blues', aspect='auto')
            plt.colorbar(im, label='使用率 (%)')
            plt.xlabel('专家ID')
            plt.ylabel('层')
            plt.title('专家使用率分布')
            plt.yticks(range(len(layer_labels)), layer_labels)
    
    # 子图3: 参数效率对比
    plt.subplot(1, 3, 3)
    model_names = list(results.keys())
    final_losses = [results[name][-1] for name in model_names]
    param_counts = [sum(p.numel() for p in trained_models[name].parameters()) for name in model_names]
    
    # 散点图：参数量 vs 性能
    colors_scatter = ['blue', 'red']
    for i, name in enumerate(model_names):
        plt.scatter(param_counts[i]/1e6, final_losses[i], 
                   color=colors_scatter[i], s=100, label=name, alpha=0.7)
        plt.annotate(name, (param_counts[i]/1e6, final_losses[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.xlabel('参数量 (M)')
    plt.ylabel('最终验证损失')
    plt.title('参数效率对比')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("moe_bias_comparison.png", dpi=150, bbox_inches='tight')
    print("\n📊 对比图表已保存到 'moe_bias_comparison.png'")
    plt.show()
    
    # 结果总结
    print(f"\n📈 实验结果总结:")
    print(f"{'='*60}")
    
    baseline_loss = results["标准Transformer"][-1]
    moe_bias_loss = results["MoE Bias Transformer"][-1]
    
    improvement = (baseline_loss - moe_bias_loss) / baseline_loss * 100
    
    standard_params = sum(p.numel() for p in trained_models["标准Transformer"].parameters())
    moe_bias_params = sum(p.numel() for p in trained_models["MoE Bias Transformer"].parameters())
    param_overhead = (moe_bias_params - standard_params) / standard_params * 100
    
    print(f"标准Transformer     : 损失={baseline_loss:.4f}, 参数={standard_params:,}")
    print(f"MoE Bias Transformer: 损失={moe_bias_loss:.4f}, 参数={moe_bias_params:,}")
    print(f"性能改进: {improvement:+.2f}%")
    print(f"参数开销: +{param_overhead:.1f}%")
    
    if improvement > 0:
        efficiency_ratio = improvement / param_overhead
        print(f"效率比 (改进%/参数开销%): {efficiency_ratio:.2f}")
        print(f"🎉 MoE Bias有效！以{param_overhead:.1f}%的参数开销获得了{improvement:.2f}%的性能提升")
    else:
        print(f"🤔 MoE Bias在当前设置下效果不明显，可能需要调整超参数")
    
    print(f"\n💡 MoE Bias技术总结:")
    print(f"✅ 极低的参数开销（主要是bias向量）")
    print(f"✅ 极低的计算开销（只是加法操作）") 
    print(f"✅ 动态的内容相关调制能力")
    print(f"✅ 可解释的专家使用模式")
