class Fact:
    def __init__(self, predicate, *args):
        self.predicate = predicate
        self.args = args

    def __repr__(self):
        return f"Fact('{self.predicate}', {', '.join(map(str, self.args))})"

    def __eq__(self, other):
        if not isinstance(other, Fact):
            return NotImplemented
        return self.predicate == other.predicate and self.args == other.args

    def __hash__(self):
        return hash((self.predicate, self.args))


class NeuroSymbolicIntegrator:
    def __init__(self, confidence_threshold=0.7):
        self.confidence_threshold = confidence_threshold
        self.object_id_counter = 0

    def _generate_object_id(self):
        self.object_id_counter += 1
        return f"obj{self.object_id_counter}"

    def _process_object_detections(self, detections):
        facts = []
        for det in detections:
            obj_id = det.get('object_id', self._generate_object_id())
            predicted_class = det['class']
            confidence = det['confidence']
            if confidence >= self.confidence_threshold:
                facts.append(Fact('is_a', obj_id, predicted_class))
        return facts

    def _process_attribute_predictions(self, attributes_list):
        facts = []
        for attr in attributes_list:
            obj_id = attr["object_id"]
            for attr_type, predictions in attr["attributes"].items():
                for pred_attr, confidence in predictions.items():
                    if confidence >= self.confidence_threshold:
                        if attr_type == 'color':
                            facts.append(Fact('has_color', obj_id, pred_attr))
                        elif attr_type == 'state':
                            facts.append(Fact('has_state', obj_id, pred_attr))
        return facts

    def _process_relationship_predictions(self, relationships):
        facts = []
        for rel in relationships:
            obj_id1 = rel["object_id1"]
            obj_id2 = rel["object_id2"]
            relation_type = rel["relation_type"]
            confidence = rel["confidence"]
            if confidence >= self.confidence_threshold:
                facts.append(Fact(relation_type, obj_id1, obj_id2))
        return facts

    def _process_scene_understanding(self, scene_predictions):
        facts = []
        for scene_type, confidence in scene_predictions.items():
            if confidence >= self.confidence_threshold:
                facts.append(Fact('scene_type', scene_type))
                break
        return facts

    def integrate_neural_outputs(self, object_detections, attribute_predictions, relationship_predictions, scene_understanding):
        all_facts = []
        all_facts.extend(self._process_object_detections(object_detections))
        all_facts.extend(self._process_attribute_predictions(attribute_predictions))
        all_facts.extend(self._process_relationship_predictions(relationship_predictions))
        all_facts.extend(self._process_scene_understanding(scene_understanding))
        return all_facts


class SymbolicReasoningEngine:
    def __init__(self, initial_facts, rules):
        self._facts = set()
        for fact in initial_facts:
            self.add_fact(fact)
        self._rules = rules

    def add_fact(self, fact):
        if not isinstance(fact, Fact):
            raise ValueError("Only Fact objects can be added to the knowledge base.")
        self._facts.add(fact)

    def __repr__(self):
        return f"SymbolicReasoningEngine with {len(self._facts)} facts and {len(self._rules)} rules."

    def _unify(self, pattern_fact, concrete_fact):
        if pattern_fact.predicate != concrete_fact.predicate:
            return None
        if len(pattern_fact.args) != len(concrete_fact.args):
            return None

        bindings = {}
        for i in range(len(pattern_fact.args)):
            p_arg = pattern_fact.args[i]
            c_arg = concrete_fact.args[i]

            if p_arg == c_arg:
                continue
            # Relaxed variable check: any all-uppercase string is a variable
            if isinstance(p_arg, str) and p_arg.isupper():
                if p_arg in bindings:
                    if bindings[p_arg] != c_arg:
                        return None
                else:
                    bindings[p_arg] = c_arg
            elif p_arg != c_arg:
                return None
        return bindings

    def _apply_bindings(self, fact, bindings):
        new_args = []
        for arg in fact.args:
            # Relaxed variable check: any all-uppercase string is a variable
            if isinstance(arg, str) and arg.isupper() and arg in bindings:
                new_args.append(bindings[arg])
            else:
                new_args.append(arg)
        return Fact(fact.predicate, *new_args)

    def query(self, query_fact):
        results = []
        for concrete_fact in self._facts:
            bindings = self._unify(query_fact, concrete_fact)
            if bindings is not None:
                results.append((concrete_fact, bindings))
        return results

    def _combine_bindings(self, old_bindings, new_bindings):
        if old_bindings is None:
            return new_bindings
        if new_bindings is None:
            return old_bindings

        combined = dict(old_bindings)
        for var, val in new_bindings.items():
            if var in combined:
                if combined[var] != val:
                    return None
            else:
                combined[var] = val
        return combined

    def infer(self, max_iterations=10):
        derived_facts = set()

        for _ in range(max_iterations):
            new_facts_derived_in_iteration = False
            for rule in self._rules:
                antecedent_patterns = rule['antecedent']
                consequent_pattern = rule['consequent']

                # Find all possible matches for each antecedent pattern
                all_antecedent_matches = [self.query(ap) for ap in antecedent_patterns]

                # Recursive helper to find consistent bindings across all antecedents
                def find_consistent_bindings(antecedent_idx, current_combined_bindings):
                    if antecedent_idx == len(antecedent_patterns):
                        yield current_combined_bindings  # All antecedents matched with consistent bindings
                        return

                    # Iterate through matches for the current antecedent pattern
                    for _, match_bindings in all_antecedent_matches[antecedent_idx]:
                        # Combine current match's bindings with previously combined bindings
                        combined_bindings_for_step = self._combine_bindings(current_combined_bindings, match_bindings)

                        if combined_bindings_for_step is not None:
                            # If bindings are consistent, recurse for the next antecedent
                            yield from find_consistent_bindings(antecedent_idx + 1, combined_bindings_for_step)

                # Start finding consistent bindings from the first antecedent
                for final_bindings in find_consistent_bindings(0, {}):
                    if final_bindings is not None:  # Ensure bindings were found
                        derived_consequent = self._apply_bindings(consequent_pattern, final_bindings)

                        # Check if this derived fact is truly new
                        if derived_consequent not in self._facts and derived_consequent not in derived_facts:
                            derived_facts.add(derived_consequent)
                            self.add_fact(derived_consequent)  # Add to the engine's facts
                            new_facts_derived_in_iteration = True

            if not new_facts_derived_in_iteration:
                break  # No new facts derived in this iteration, stop.

        return list(derived_facts)
