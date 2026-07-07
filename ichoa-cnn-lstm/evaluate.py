# ---  Evaluate and Perform Inference Function ---
def evaluate_and_infer_model(final_model, X_test, y_test):
    """
    Evaluates the final model on test data and performs inference.
    """
    print("\nEvaluating final model on test data...")
    test_loss, test_accuracy = final_model.evaluate(X_test, y_test, verbose=1)
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print("Final model evaluation complete.")

    print("\nPerforming model inference on test data...")
    y_pred_probabilities = final_model.predict(X_test)
    y_pred_classes = (y_pred_probabilities > 0.5).astype(int)

    print("\nComparing Actual vs. Predicted Labels (first 10 samples):")
    for i in range(min(10, len(y_test))):
        print(f"  Sample {i+1}: Actual = {y_test[i]}, Predicted = {y_pred_classes[i][0]}")
    print("Model inference complete.")
