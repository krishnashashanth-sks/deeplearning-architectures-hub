import torch
from main import SOS_IDX,EOS_IDX

# ---  Inference Function ---

def translate_sentence(sentence, model, sp_model, device, max_len):
    model.eval() # Set model to evaluation mode
    with torch.no_grad():
        # Tokenize source sentence (NL input has no SOS/EOS for encoder)
        indexed = sp_model.encode_as_ids(sentence)
        src_tensor = torch.LongTensor(indexed).unsqueeze(0).to(device) # Add batch dimension

        # Encode source sentence
        hidden, cell = model.encoder(src_tensor)

        # Prepare first input to decoder (<s> token)
        trg_indexes = [SOS_IDX]
        input_token = torch.LongTensor([SOS_IDX]).to(device) # Initial input token for decoder

        for _ in range(max_len):
            # Decoder takes input_token as [1] so unsqueeze for batch dimension if needed for batch_first=True
            output, hidden, cell = model.decoder(input_token.unsqueeze(0), hidden, cell)
            pred_token = output.argmax(1).item()
            trg_indexes.append(pred_token)

            if pred_token == EOS_IDX:
                break

            input_token = torch.LongTensor([pred_token]).to(device)

        # Convert target indexes to text
        # Remove SOS and EOS tokens for decoding if they are there
        if trg_indexes and trg_indexes[0] == SOS_IDX:
            trg_indexes = trg_indexes[1:]
        if trg_indexes and trg_indexes[-1] == EOS_IDX:
            trg_indexes = trg_indexes[:-1]

        if not trg_indexes:
            return "<empty prediction>"

        return sp_model.decode_ids(trg_indexes)
