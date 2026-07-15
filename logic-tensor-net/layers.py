import tensorflow as tf

# 1. Implement LTNConstant
class LTNConstant(tf.Module):
  def __init__(self,name,initial_value=None,embedding_dim=10,trainable=True):
    super().__init__(name=name)
    if initial_value is None:
      self.embedding=tf.Variable(tf.random.uniform(shape=(embedding_dim,),minval=-1.0,maxval=1.0),
                               name=f"embedding_{name}",
                               trainable=trainable)
    else:
      # If initial_value is provided, it should be a tensor of the correct shape.
      # For simplicity in this context, if initial_value is given, we assume it's
      # already a tensor and use it directly, but still make it trainable if specified.
      self.embedding=tf.Variable(initial_value, name=f"embedding_{name}", trainable=trainable)
  def __call__(self):
    return self.embedding
  def __str__(self):
        return f"Constant('{self.name}', embedding_dim={self.embedding.shape[-1]})"

# 2. Implement fuzzy logic operators as Python functions using TensorFlow

class FuzzyLogicOperators:
    @staticmethod
    @tf.function
    def neg(a):
        """Fuzzy Negation: 1 - a"""
        return 1.0 - a

    # T-norms (Fuzzy AND)
    @staticmethod
    @tf.function
    def t_norm_prod(a, b):
        """Product T-norm: a * b"""
        return a * b

    @staticmethod
    @tf.function
    def t_norm_min(a, b):
        """Gödel T-norm (Minimum T-norm): min(a, b)"""
        return tf.minimum(a, b)

    @staticmethod
    @tf.function
    def t_norm_lukasiewicz(a, b):
        """Lukasiewicz T-norm: max(0, a + b - 1)"""
        return tf.maximum(0.0, a + b - 1.0)

    # T-conorms (Fuzzy OR)
    @staticmethod
    @tf.function
    def t_conorm_prob_sum(a, b):
        """Probabilistic Sum T-conorm: a + b - (a * b)"""
        return a + b - (a * b)

    @staticmethod
    @tf.function
    def t_conorm_max(a, b):
        """Gödel T-conorm (Maximum T-conorm): max(a, b)"""
        return tf.maximum(a, b)

    @staticmethod
    @tf.function
    def t_conorm_lukasiewicz(a, b):
        """Lukasiewicz T-conorm: min(1, a + b)"""
        return tf.minimum(1.0, a + b)

    # Fuzzy Implication
    @staticmethod
    @tf.function
    def imp_lukasiewicz(a, b):
        """Lukasiewicz Implication: min(1, 1 - a + b)"""
        return tf.minimum(1.0, 1.0 - a + b)

    # Optional: Gödel Implication (non-differentiable points)
    @staticmethod
    @tf.function
    def imp_godel(a, b):
        """Gödel Implication: 1 if a <= b else b"""
        return tf.where(a <= b, 1.0, b)

    # Optional: Residuum of Product (Goguen) Implication (careful with a=0)
    @staticmethod
    @tf.function
    def imp_goguen(a, b):
        """Goguen Implication: 1 if a <= b else b/a. Handles division by zero.
        For a=0, this is typically defined as 1 if b is also 0, or 1 otherwise.
        Here, we approximate to 1.0 if a is very close to 0.
        """
        safe_a = tf.where(tf.abs(a) < tf.keras.backend.epsilon(), tf.keras.backend.epsilon(), a)
        return tf.where(a <= b, 1.0, b / safe_a)

# 3. Implement mechanisms for universal and existential quantifiers

class Quantifiers:
    @staticmethod
    @tf.function
    def forall_soft_min(truth_values, p=2.0):
        """Universal Quantifier (FORALL) using a soft-minimum approximation (p-norm based).
        A higher 'p' approximates the true minimum more closely.
        """
        # This approximates 1 - max(1 - tv) which is equivalent to min(tv)
        # (1 - truth_values) gives falsity values. We want to find the max falsity.
        # The formula used is based on a generalized mean approximation for max
        # applied to falsity values, then converting back to truth.
        # More common soft-min for truth values:
        # 1 - (tf.reduce_mean(tf.pow(1.0 - truth_values, p))**(1.0/p))

        # Simpler differentiable soft-min (LogSumExp trick or similar can be used for numerical stability)
        # For simplicity, using a generalized mean approximation of the falsity and then (1 - result)
        # Or directly, a soft-min: exp(log_sum_exp(values * -p) / p)
        # For now, let's use a straightforward p-norm based approximation
        # (1 - (sum((1-tv)^p) / N)^(1/p))
        N = tf.cast(tf.shape(truth_values)[0], tf.float32)
        if N == 0: return tf.constant(1.0, dtype=tf.float32) # If no elements, FORALL is true
        # truth_values are in [0, 1]. Falsity is 1 - truth_values, also in [0, 1].
        # We want to approximate min(truth_values) or 1 - max(1 - truth_values)
        # max(x) approx= (sum(x^p) / N)^(1/p)
        # So max(1-tv) approx= (sum((1-tv)^p) / N)^(1/p)
        # then forall_approx = 1 - (sum((1-tv)^p) / N)^(1/p)
        falsity_values = 1.0 - truth_values
        max_falsity_approx = tf.pow(tf.reduce_mean(tf.pow(falsity_values, p)), 1.0 / p)
        return 1.0 - max_falsity_approx

    @staticmethod
    @tf.function
    def exists_soft_max(truth_values, p=2.0):
        """Existential Quantifier (EXISTS) using a soft-maximum approximation (p-norm based).
        A higher 'p' approximates the true maximum more closely.
        """
        N = tf.cast(tf.shape(truth_values)[0], tf.float32)
        if N == 0: return tf.constant(0.0, dtype=tf.float32) # If no elements, EXISTS is false
        # max(x) approx= (sum(x^p) / N)^(1/p)
        return tf.pow(tf.reduce_mean(tf.pow(truth_values, p)), 1.0 / p)

class LTN_Predicate(tf.Module):
  def __init__(self,name,input_shape=None):
    super().__init__(name=name)
    self.input_shape=input_shape
  @tf.function
  def __cal__(self,*args):
    raise NotImplementedError("Subclasses must implement the __call__ method to compute truth values.")
  def __str__(self):
        return f"Predicate('{self.name}')"
        
class MLPPredicate(LTN_Predicate):
  def __init__(self, name, input_dim, hidden_dims=None):
    super().__init__(name=name)
    self.input_dim = input_dim

    if hidden_dims is None:
      hidden_dims = [64, 32] # Default hidden layers

    layers = []
    for dim in hidden_dims:
      layers.append(tf.keras.layers.Dense(dim, activation='relu'))
    layers.append(tf.keras.layers.Dense(1, activation='sigmoid')) # Output a single truth value [0, 1]

    self.mlp = tf.keras.Sequential(layers, name=f"mlp_{name}")

  @tf.function
  def __call__(self, *args):
    processed_args_raw = []
    for arg in args:
        if isinstance(arg, LTNConstant):
            processed_args_raw.append(arg()) # Get the tf.Variable embedding
        else:
            # For anything else, assume it's already a tensor-like object
            # (tf.Tensor, tf.Variable, numpy array etc.) and let convert_to_tensor handle it.
            processed_args_raw.append(tf.convert_to_tensor(arg, dtype=tf.float32)) # Convert early with explicit dtype

    if len(processed_args_raw) == 0:
      raise ValueError("Predicate requires at least one argument embedding.")

    # All items in processed_args_raw should now be tf.Tensor
    processed_args_as_tensors = processed_args_raw

    # Determine if inputs are batched based on the first argument's rank
    first_arg_tensor = processed_args_as_tensors[0]

    # If inputs are unbatched (e.g., single constant embeddings), add a batch dimension for the MLP.
    original_was_unbatched = first_arg_tensor.shape.rank == 1
    if original_was_unbatched:
        processed_args_batched = [tf.expand_dims(arg, axis=0) for arg in processed_args_as_tensors]
    else:
        processed_args_batched = processed_args_as_tensors

    # Concatenate along the last axis (feature dimension)
    concatenated_embeddings = tf.concat(processed_args_batched, axis=-1)

    # Pass through MLP to get truth value
    truth_value = self.mlp(concatenated_embeddings)

    # If input was originally unbatched, squeeze the batch dimension back out.
    if original_was_unbatched and truth_value.shape.rank > 1:
        truth_value = tf.squeeze(truth_value, axis=0)

    return truth_value