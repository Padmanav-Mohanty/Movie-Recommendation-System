"""
Serving layer.

Imports are intentionally lazy to avoid pulling in heavy dependencies
(torch, faiss) unless actually needed.
"""

# Public API is accessed via direct imports:
#   from src.serving.recommender import load_recommender, BaseRecommender
