import tensorflow as tf

#  Define a callable Python function that encapsulates the logical formula

def evaluate_forall_formula(model,domain_constants):
    # Retrieve PredicateA and PredicateB instances
    predicate_A = model.get_predicate('PredicateA')
    predicate_B = model.get_predicate('PredicateB')

    # Create a list to store the truth values of the implication for each constant
    implication_truth_values = []

    # Iterate through each constant in the defined domain of discourse
    for constant_obj in domain_constants:
        # Get the constant's embedding
        constant_embedding = constant_obj()

        # Evaluate PredicateA with the constant's embedding
        truth_A = predicate_A(constant_embedding)

        # Evaluate PredicateB with the constant's embedding
        truth_B = predicate_B(constant_embedding)

        # Compute the implication: A IMPLIES B using Lukasiewicz implication
        implication_truth = model.fuzzy_ops.imp_lukasiewicz(truth_A, truth_B);

        # Add the implication_truth to the list
        implication_truth_values.append(implication_truth)

    # Convert the list of truth values into a TensorFlow tensor
    implication_tensor = tf.convert_to_tensor(implication_truth_values, dtype=tf.float32)

    # Apply the universal quantifier (forall_soft_min) on the tensor of implication truth values
    final_truth_value = model.quantifiers.forall_soft_min(implication_tensor)

    return final_truth_value
