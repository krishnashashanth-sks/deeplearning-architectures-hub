from sklearn.metrics import mean_squared_error, r2_score
import torch

# --- Evaluation Loop Definition ---
def evaluate_model(model, dataloader, loss_fn):
    model.eval() # Set the model to evaluation mode
    total_loss = 0
    all_predictions = []
    all_true_affinities = []

    with torch.no_grad(): # Disable gradient calculation for evaluation
        for batch_idx, (rdk_feats, maccs_feats, ecfp_feats, protein_feats, affinities) in enumerate(dataloader):
            # Forward pass
            predictions = model(rdk_feats, maccs_feats, ecfp_feats, protein_feats)

            # Calculate loss
            loss = loss_fn(predictions, affinities)
            total_loss += loss.item()

            all_predictions.extend(predictions.cpu().numpy())
            all_true_affinities.extend(affinities.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    mse = mean_squared_error(all_true_affinities, all_predictions)
    r2 = r2_score(all_true_affinities, all_predictions)

    print(f"\nEvaluation Results:")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"R-squared (R2): {r2:.4f}")

    return avg_loss, mse, r2