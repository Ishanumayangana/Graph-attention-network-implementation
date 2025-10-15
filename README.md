# Graph Attention Network (GAT) Implementation



A complete implementation of **Graph Attention Networks (GAT)** from scratch using PyTorch. This project demonstrates how attention mechanisms can be applied to graph-structured data for node classification tasks.

## 🎯 Overview

Graph Attention Networks (GAT) are neural network architectures that operate on graph-structured data, leveraging masked self-attention layers to address the shortcomings of prior methods based on graph convolutions. This implementation includes:

- **Core GAT Layer**: Single attention head implementation
- **Multi-Head Attention**: Multiple attention heads for learning diverse representations
- **Complete Training Pipeline**: Training, validation, and testing with visualization
- **Optimized Model**: Improved version with 100% accuracy

## ✨ Features

- ✅ **From-scratch implementation** of Graph Attention Layer
- ✅ **Multi-head attention mechanism** for capturing different aspects of graph structure
- ✅ **Adaptive learning** of neighbor importance weights
- ✅ **Visualization tools** for attention weights and training curves
- ✅ **Two model versions**: Original (baseline) and Optimized (100% accuracy)
- ✅ **Early stopping** and learning rate scheduling
- ✅ **Comprehensive documentation** and comments

## 🚀 Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Install Dependencies

```bash
pip install torch torchvision numpy matplotlib networkx
```

Or install from requirements file:

```bash
pip install -r requirements.txt
```

### Requirements

```
torch>=2.8.0
torchvision>=0.23.0
numpy>=2.2.0
matplotlib>=3.10.0
networkx>=3.5
```
## 🧠 How It Works

### Graph Attention Mechanism

The GAT layer computes attention coefficients between connected nodes:

**1. Attention Coefficients:**
```
α_ij = softmax_j(e_ij)
where e_ij = LeakyReLU(a^T [W·h_i || W·h_j])
```

**2. Node Feature Aggregation:**
```
h'_i = σ(Σ_{j∈N_i} α_ij · W·h_j)
```

Where:
- `h_i`: Input features for node i
- `W`: Learnable weight matrix
- `a`: Attention mechanism parameters
- `α_ij`: Attention weight from node j to node i
- `N_i`: Neighbors of node i

### Multi-Head Attention

Multiple attention heads learn different aspects of the graph:
```
h'_i = ||_{k=1}^K σ(Σ_{j∈N_i} α^k_ij · W^k·h_j)
```

Where `||` denotes concatenation and K is the number of heads.

## 📊 Results

### Original Model (Baseline)

| Metric | Accuracy |
|--------|----------|
| **Training** | 50.0% |
| **Validation** | 0.0% |
| **Test** | 0.0% |

**Configuration:**
- 7 nodes, 5 features
- 4 attention heads, 8 hidden units per head
- Dropout: 0.6
- 200 epochs

### Improved Model (Optimized)

| Metric | Accuracy |
|--------|----------|
| **Training** | 100.0% ✅ |
| **Validation** | 100.0% ✅ |
| **Test** | 100.0% ✅ |

**Configuration:**
- 20 nodes, 10 features
- 6 attention heads, 16 hidden units per head
- Dropout: 0.3
- 500 epochs with early stopping

### Performance Comparison

| Aspect | Original | Improved | Change |
|--------|----------|----------|--------|
| Nodes | 7 | 20 | +186% |
| Features | 5 | 10 | +100% |
| Hidden Units/Head | 8 | 16 | +100% |
| Attention Heads | 4 | 6 | +50% |
| Dropout | 0.6 | 0.3 | -50% |
| Parameters | 326 | 1,446 | +344% |
| **Test Accuracy** | **0%** | **100%** | **+100pp** |

## 💻 Usage

### Running the Original Model

```python
# Model parameters
nfeat = 5
nhid = 8
nclass = 3
dropout = 0.6
nheads = 4

# Initialize and train
model = GAT(nfeat, nhid, nclass, dropout, alpha=0.2, nheads=nheads)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
# ... training loop
```

### Running the Improved Model

```python
# Optimized parameters
nfeat = 10
nhid = 16
nclass = 3
dropout = 0.3
nheads = 6

# Initialize with scheduler
model = GAT(nfeat, nhid, nclass, dropout, alpha=0.2, nheads=nheads)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
# ... training loop with early stopping
```

## 🏗️ Model Architecture

### GraphAttentionLayer Class

```python
class GraphAttentionLayer(nn.Module):
    """
    Single Graph Attention Head
    
    Input: (N, in_features), adjacency_matrix (N, N)
    Output: (N, out_features)
    """
```

**Components:**
- Linear transformation: `W` (in_features × out_features)
- Attention mechanism: `a` (2 × out_features × 1)
- LeakyReLU activation (α = 0.2)
- Softmax normalization
- Dropout regularization

### GAT Class

```python
class GAT(nn.Module):
    """
    Multi-Head Graph Attention Network
    
    Layer 1: K parallel attention heads (concatenated)
    Layer 2: Single attention head for classification
    Output: log_softmax for node classification
    """
```

## ⚙️ Optimization Techniques

### What Made the Difference?

1. **🔢 More Training Data**
   - Increased dataset size from 7 to 20 nodes
   - Better train/val/test split (60%/15%/25%)

2. **📊 Meaningful Features**
   - Created community-specific feature patterns
   - 10-dimensional features instead of 5

3. **🧠 Larger Model Capacity**
   - More attention heads (4 → 6)
   - Larger hidden dimensions (8 → 16 per head)
   - Total parameters: 326 → 1,446

4. **⚙️ Better Regularization**
   - Reduced dropout from 0.6 to 0.3
   - Avoided over-regularization on small dataset

5. **📚 Training Improvements**
   - Early stopping (patience = 100)
   - Learning rate scheduler (step decay)
   - Longer training (up to 500 epochs)

6. **🏗️ Structured Data**
   - Clear community structure in graph
   - Balanced class distribution
   - Meaningful node connections

## 🔍 Key Insights

### Advantages of GAT

- ✅ **Adaptive Attention**: Learns which neighbors are most important
- ✅ **Inductive Learning**: Can generalize to unseen graphs
- ✅ **Parallel Computation**: All attention heads compute in parallel
- ✅ **Interpretability**: Attention weights show model decisions
- ✅ **Flexible**: Works with variable-sized neighborhoods

### When to Use GAT

- Node classification tasks
- Graph-structured data with variable connectivity
- When neighbor importance varies
- Need for interpretable model decisions


## 🛠️ Customization

### Create Your Own Graph

```python
# Define custom edges
edges = [(0, 1), (1, 2), (2, 3), ...]

# Create adjacency matrix
adj = torch.zeros((num_nodes, num_nodes))
for i, j in edges:
    adj[i][j] = 1
adj = adj + torch.eye(num_nodes)  # Add self-loops

# Define features and labels
features = torch.randn((num_nodes, num_features))
labels = torch.tensor([...])  # Your labels
```

### Tune Hyperparameters

```python
# Experiment with:
- Number of attention heads (nheads)
- Hidden units per head (nhid)
- Dropout rate (dropout)
- Learning rate (lr)
- Number of epochs
- Weight decay
```

## 📈 Visualization

The notebook includes:

1. **Graph Structure**: NetworkX visualization of node connections
2. **Training Curves**: Loss and accuracy over epochs
3. **Attention Heatmap**: Learned attention weights between nodes
4. **Confusion Matrix**: Per-class performance analysis
