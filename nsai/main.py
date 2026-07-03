from layers import *

# Example symbolic facts (visual and common-sense) and rules
visual_facts = [
    Fact('is_a', 'obj1', 'refrigerator'),
    Fact('has_color', 'obj1', 'white'),
    Fact('has_state', 'obj1', 'closed'),
    Fact('is_a', 'obj2', 'apple'),
    Fact('is_on', 'obj2', 'obj1'), # Apple is on the refrigerator
    Fact('is_a', 'obj3', 'table'),
    Fact('has_color', 'obj3', 'brown'),
    Fact('is_next_to', 'obj1', 'obj3'),
    Fact('scene_type', 'kitchen')
]

common_sense_facts = [
    Fact('is_appliance', 'refrigerator'),
    Fact('stores', 'refrigerator', 'food'),
    Fact('stores', 'refrigerator', 'milk'),
    Fact('is_edible', 'apple'), # This refers to the type 'apple'
    Fact('supports', 'table', 'objects') # General rule for what a table supports
]

# Rules are represented as dictionaries with 'antecedent' and 'consequent'
# Variables are represented as strings, e.g., 'X', 'Y'
rules = [
    # Rule: If something is a refrigerator and stores food, then it's a food storage appliance
    {
        'antecedent': [
            Fact('is_a', 'X', 'refrigerator'),
            Fact('stores', 'refrigerator', 'food')
        ],
        'consequent': Fact('is_food_storage_appliance', 'X')
    },
    # Rule: If an object is on another object, and that other object is a table, then the first object is supported by the table
    {
        'antecedent': [
            Fact('is_on', 'X', 'Y'),
            Fact('is_a', 'Y', 'table')
        ],
        'consequent': Fact('is_supported_by', 'X', 'Y')
    },
    # Rule: If an object is edible and stored in a refrigerator, it is likely fresh
    {
        'antecedent': [
            Fact('is_a', 'X', 'TYPE_X'),       # X is an instance of TYPE_X
            Fact('is_edible', 'TYPE_X'),      # TYPE_X is edible (e.g., apple is edible)
            Fact('is_on', 'X', 'FRIDGE_INSTANCE'), # X is on FRIDGE_INSTANCE
            Fact('is_a', 'FRIDGE_INSTANCE', 'refrigerator') # FRIDGE_INSTANCE is a refrigerator
        ],
        'consequent': Fact('is_fresh', 'X')
    }
]


# Mock neural network outputs (from previous steps)
mock_object_detections = [
    {'class': 'refrigerator', 'confidence': 0.95, 'object_id': 'obj1'},
    {'class': 'apple', 'confidence': 0.88, 'object_id': 'obj2'},
    {'class': 'table', 'confidence': 0.92, 'object_id': 'obj3'},
    {'class': 'person', 'confidence': 0.65, 'object_id': 'obj4'}, # Will be filtered by threshold
    {'class': 'chair', 'confidence': 0.82, 'object_id': 'obj5'}
]

mock_attribute_predictions = [
    {
        'object_id': 'obj1',
        'attributes': {
            'color': {'white': 0.90, 'silver': 0.08},
            'state': {'closed': 0.93, 'open': 0.05}
        }
    },
    {
        'object_id': 'obj2',
        'attributes': {
            'color': {'red': 0.85, 'green': 0.10},
            'state': {'fresh': 0.78, 'rotten': 0.15}
        }
    },
    {
        'object_id': 'obj3',
        'attributes': {
            'color': {'brown': 0.91, 'black': 0.07}
        }
    }
]

mock_relationship_predictions = [
    {'object_id1': 'obj2', 'object_id2': 'obj1', 'relation_type': 'is_on', 'confidence': 0.80},
    {'object_id1': 'obj1', 'object_id2': 'obj3', 'relation_type': 'is_next_to', 'confidence': 0.75},
    {'object_id1': 'obj5', 'object_id2': 'obj3', 'relation_type': 'is_on', 'confidence': 0.60} # Will be filtered
]

mock_scene_understanding = {
    'kitchen': 0.98,
    'living_room': 0.01,
    'bedroom': 0.005
}

# Instantiate Integrator and generate symbolic facts from mock neural outputs
integrator = NeuroSymbolicIntegrator(confidence_threshold=0.7)
symbolic_facts = integrator.integrate_neural_outputs(
    mock_object_detections,
    mock_attribute_predictions,
    mock_relationship_predictions,
    mock_scene_understanding
)

# Instantiate Reasoning Engine with common-sense facts and rules
reasoning_engine = SymbolicReasoningEngine(common_sense_facts, rules)

# Add neural-derived symbolic facts to the reasoning engine
for fact in symbolic_facts:
    reasoning_engine.add_fact(fact)

# Perform inference to derive new facts
derived_facts = reasoning_engine.infer(max_iterations=5)

print("--- Neuro-Symbolic AI System Initialization Complete ---")
print(f"Initial Facts: {len(common_sense_facts)} common-sense, {len(symbolic_facts)} neural-derived")
print(f"Total Facts in Engine after adding neural outputs: {len(reasoning_engine._facts)}")
print(f"Newly Derived Facts after inference: {len(derived_facts)}")

print("\n--- All Facts in Reasoning Engine after Inference ---")
for fact in sorted(list(reasoning_engine._facts), key=lambda x: str(x)):
    print(fact)

print("\n--- Query 1: What color is obj1? ---")
query_color = Fact('has_color', 'obj1', 'COLOR')
results_color = reasoning_engine.query(query_color)
if results_color:
    for fact, bindings in results_color:
        print(f"Found: {fact}, Bindings: {bindings}")
else:
    print("No facts found for the query.")

print('\n--- Query 2: What objects are appliances? ---')
query_appliance = Fact('is_appliance', 'OBJECT')
results_appliance = reasoning_engine.query(query_appliance)
if results_appliance:
    unique_appliances = set()
    for fact, bindings in results_appliance:
        unique_appliances.add(bindings['OBJECT'])
    print(f"Found appliances: {', '.join(sorted(list(unique_appliances)))}")
else:
    print("No appliances found.")

print('\n--- Query 3: What is fresh? ---')
query_fresh = Fact('is_fresh', 'ITEM')
results_fresh = reasoning_engine.query(query_fresh)
if results_fresh:
    unique_fresh_items = set()
    for fact, bindings in results_fresh:
        unique_fresh_items.add(bindings['ITEM'])
    print(f"Found fresh items: {', '.join(sorted(list(unique_fresh_items)))}")
else:
    print("No fresh items found.")