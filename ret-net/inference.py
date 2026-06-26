import torch
import torch.nn.functional as F

def generate_text(model, prompt, max_new_tokens, encode, decode, device='cpu'):
    # 1. Set the model to evaluation mode and move to CPU
    model.eval()
    model.to(device)

    # 2. Encode the initial prompt
    encoded_prompt = encode(prompt)
    # Convert to tensor and add batch dimension
    input_ids = torch.tensor(encoded_prompt, dtype=torch.long, device=device).unsqueeze(0)

    # 3. Initialize past_key_values to None for the first token generation
    past_key_values = None

    # List to store generated tokens
    generated_tokens = input_ids.tolist()[0] # Start with the encoded prompt

    # Loop for max_new_tokens iterations
    for _ in range(max_new_tokens):
        # Take the last token from the current sequence
        # For recurrent mode, we only pass the *last* token generated/prompt token
        current_token_tensor = torch.tensor([[generated_tokens[-1]]], dtype=torch.long, device=device)

        # Pass this single token and the past_key_values to the model
        with torch.no_grad():
            logits, new_past_key_values = model(current_token_tensor, mode='recurrent', past_key_values=past_key_values)

        # Update past_key_values with the new_past_key_values returned by the model
        past_key_values = new_past_key_values

        # Apply softmax to logits to get probabilities and sample the next token
        # The logits for recurrent mode will be (batch_size, vocab_size)
        probabilities = F.softmax(logits[0], dim=-1) # Get probabilities for the single token in the batch

        # Sample the next token (using multinomial for probabilistic sampling)
        next_token_id = torch.multinomial(probabilities, num_samples=1).item()

        # Append the generated token to the current sequence of tokens
        generated_tokens.append(next_token_id)

    # Decode the entire sequence of tokens back into a string
    full_generated_text = decode(generated_tokens)

    # Return the generated text string
    model.train() # Set model back to training mode
    return full_generated_text
