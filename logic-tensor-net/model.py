import tensorflow as tf
from layers import LTNConstant,MLPPredicate,FuzzyLogicOperators,Quantifiers

class LTNModel(tf.Module):
  def __init__(self, constants_config, predicates_config):
    super().__init__()

    # Use a standard Python dictionary for constants and predicates
    # tf.Module automatically tracks tf.Module instances assigned as attributes
    self.constants = {}
    for name, config in constants_config.items():
      self.constants[name] = LTNConstant(name=name, embedding_dim=config.get('embedding_dim', 10), trainable=config.get('trainable', True))
    # For TF to track these as submodules, assign the dict itself as an attribute if needed,
    # or ensure constants are directly assigned attributes if possible. For simple dict access,
    # tf.Module will still find trainable variables if accessed later.

    self.predicates = {}
    for name, config in predicates_config.items():
      self.predicates[name] = MLPPredicate(name=name, input_dim=config['input_dim'], hidden_dims=config.get('hidden_dims'))

    # Make FuzzyLogicOperators and Quantifiers accessible
    self.fuzzy_ops = FuzzyLogicOperators
    self.quantifiers = Quantifiers

  def get_constant(self, name):
    return self.constants.get(name)

  def get_predicate(self, name):
    return self.predicates.get(name)