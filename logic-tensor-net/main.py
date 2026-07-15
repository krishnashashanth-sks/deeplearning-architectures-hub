import tensorflow as tf
from model import LTNModel
from evaluate import evaluate_forall_formula
from layers import LTNConstant

# 1. Define a domain of discourse by creating LTNConstant instances
embedding_dim = 10

# Example constants
constant_a = LTNConstant(name='a', embedding_dim=embedding_dim, trainable=True)
constant_b = LTNConstant(name='b', embedding_dim=embedding_dim, trainable=True)
constant_c = LTNConstant(name='c', embedding_dim=embedding_dim, trainable=True)
constant_d = LTNConstant(name='d', embedding_dim=embedding_dim, trainable=True)

domain_constants = [constant_a, constant_b, constant_c, constant_d]

print("Defined domain constants:")
for const in domain_constants:
    print(f"- {const.name}: embedding shape {const.embedding.shape}")

# 2. Prepare constants_config and predicates_config dictionaries

# constants_config should reflect the constants we just defined
constants_config = {
    'a': {'embedding_dim': embedding_dim, 'trainable': True},
    'b': {'embedding_dim': embedding_dim, 'trainable': True},
    'c': {'embedding_dim': embedding_dim, 'trainable': True},
    'd': {'embedding_dim': embedding_dim, 'trainable': True},
}

# predicates_config for 'PredicateA' and 'PredicateB'
# Each predicate takes one constant embedding as input, so input_dim = embedding_dim
predicates_config = {
    'PredicateA': {'input_dim': embedding_dim, 'hidden_dims': [32, 16]},
    'PredicateB': {'input_dim': embedding_dim, 'hidden_dims': [32, 16]},
}

print("Constants Configuration:", constants_config)
print("Predicates Configuration:", predicates_config)

# 3. Instantiate the LTNModel using the configurations
model = LTNModel(constants_config=constants_config, predicates_config=predicates_config)

print("LTNModel instantiated.")
print("Managed constants:", list(model.constants.keys()))
print("Managed predicates:", list(model.predicates.keys()))

# 5. Call the model.evaluate_formula() method, passing your evaluate_forall_formula callable function
result_truth_value = evaluate_forall_formula(model,domain_constants)

# 6. Print the calculated truth value of the logical formula
print(f"Truth value of 'FORALL x (PredicateA(x) IMPLIES PredicateB(x))': {result_truth_value.numpy():.4f}")