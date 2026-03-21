"""
FinBERT hierarchical news classifier.

Multi-head architecture:
  - Shared backbone: Chinese FinBERT
  - Level-1 head: 8 major categories
  - Level-2 head: 28 subcategories
  - Joint training: Loss = α × L1_loss + β × L2_loss
"""
