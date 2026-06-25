import torch

def predict_masked_token(model, plm_head, config, input_ids, index_to_predict):
    """
    Predicts a masked token at a specified index within an input sequence.
    """
    inference_seq_len = input_ids.size(1)
    batch_size = input_ids.size(0)

    # Create attention permutation mask for the model's forward pass
    attn_perm_mask_for_inference = torch.zeros(batch_size, inference_seq_len, inference_seq_len).bool()
    # Mask the current position from attending to itself for the query stream ('g' stream).
    attn_perm_mask_for_inference[:, index_to_predict, index_to_predict] = True

    # Create prediction mask for the PLM head (plm_prediction_mask)
    plm_prediction_mask_for_inference = torch.zeros(batch_size, inference_seq_len).bool()
    plm_prediction_mask_for_inference[:, index_to_predict] = True

    # Dummy target_ids, as we are only interested in logits for inference, not loss calculation here.
    dummy_target_ids = torch.zeros_like(input_ids)

    # Initialize memories (all None for the first inference step of a new sequence)
    inference_mems = [None] * config['n_layer']

    attention_mask_for_inference = torch.ones_like(input_ids).bool()

    with torch.no_grad():
        # Forward pass through the main XLNet model
        _, g_out_infer, _ = model(
            input_ids,
            attention_mask=attention_mask_for_inference,
            mems=inference_mems,
            perm_mask=attn_perm_mask_for_inference
        )

        # Get logits from the PLM head for the predicted position
        _, logits = plm_head(g_out_infer, dummy_target_ids, plm_prediction_mask_for_inference)

        # Get the predicted token ID (highest probability)
        predicted_token_id = torch.argmax(logits, dim=-1).item()

    return predicted_token_id

def generate_text(model, plm_head, config, start_token_id, num_generate_tokens):
    """
    Generates text autoregressively using the model's memory mechanism.
    """
    current_input_ids = torch.tensor([[start_token_id]]) # (batch_size=1, qlen=1) for the first token
    current_mems = [None] * config['n_layer'] # Initialize memory as None for the very first step
    generated_token_ids = [start_token_id]

    # Dummy target_ids for plm_head, as we only need logits
    dummy_target_ids = torch.zeros_like(current_input_ids)

    for i in range(num_generate_tokens):
        qlen_step = current_input_ids.size(1) # This will be 1 for each new token generated

        attention_mask_step = torch.ones_like(current_input_ids).bool()

        # Create an attention permutation mask for the current token for the model's forward pass.
        # Mask self-attention for the query stream
        attn_perm_mask_step = torch.ones(1, qlen_step, qlen_step).bool() 

        # The plm_prediction_mask for the head should be True for this single token
        plm_prediction_mask_step = torch.ones_like(current_input_ids).bool()

        with torch.no_grad():
            _, g_out_step, new_mems_step = model(
                current_input_ids,
                attention_mask=attention_mask_step,
                mems=current_mems,
                perm_mask=attn_perm_mask_step
            )

            # Get logits for the predicted token
            _, logits_step = plm_head(g_out_step, dummy_target_ids, plm_prediction_mask_step)

            predicted_next_token_id = torch.argmax(logits_step, dim=-1).item()
            generated_token_ids.append(predicted_next_token_id)

            # Update memory for the next step: new_mems_step becomes current_mems
            current_mems = new_mems_step
            # The input for the next step is the token just predicted
            current_input_ids = torch.tensor([[predicted_next_token_id]])

    return generated_token_ids