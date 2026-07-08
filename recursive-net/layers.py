# Node Class Definition
class Node:
  def __init__(self, feature_vector=None, children=None):
    self.feature_vector = feature_vector
    self.children = children if children is not None else []
    self.hidden_state = None

  def is_leaf(self):
    return len(self.children) == 0

  def __repr__(self):
        if self.is_leaf():
            return f"LeafNode(features_shape={self.feature_vector.shape if self.feature_vector is not None else 'None'})"
        else:
            return f"InternalNode(num_children={len(self.children)})"