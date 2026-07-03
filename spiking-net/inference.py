import torch

def inference(model, input_image, num_steps, device):
    model.eval() # Set model to evaluation mode
    with torch.no_grad():
        input_image = input_image.unsqueeze(0).to(device) # Add batch dimension and move to device
        spk_rec = model(input_image, num_steps)
        # For inference, sum spikes over time and find the class with the most spikes
        # spk_rec shape: (num_steps, 1, num_outputs)
        spike_count = spk_rec.sum(dim=0).squeeze(0) # sum over time, remove batch dim
        predicted_class = torch.argmax(spike_count, dim=0)
    return predicted_class.item()
