import torch

# ---  Inference Function ---

def run_inference(model, dataloader):
    """Runs inference on the model with the given dataloader."""
    print("\nPerforming inference...")
    model.eval() # Set model to evaluation mode
    with torch.no_grad(): # Disable gradient calculation
        for batch in dataloader:
            output = model(batch)
            print(f"Input shape: {batch.x.shape}")
            print(f"Output shape: {output.shape}")
            print("Inference successful.")
            return output, batch # Return output and the input batch for visualization
    return None, None

