---
title: Machine Learning Notes
category: ai
priority: high
---

# Machine Learning

## What is ML?

Machine learning is a subset of artificial intelligence that enables systems to learn from data.

## Types of Learning

### Supervised Learning

Training with labeled data. Examples:
- Classification
- Regression

### Unsupervised Learning

Finding patterns in unlabeled data. Examples:
- Clustering
- Dimensionality reduction

## Popular Algorithms

| Algorithm | Use Case |
|-----------|----------|
| Linear Regression | Prediction |
| Random Forest | Classification |
| K-Means | Clustering |
| Neural Networks | Complex patterns |

## Deep Learning

Neural networks with multiple layers.

```python
import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)
```