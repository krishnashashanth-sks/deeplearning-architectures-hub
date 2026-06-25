def predict_with_dannet(model, new_data):
  """Performs inference with the trained Advanced DanNet model."""
  predictions = model.predict(new_data)
  return predictions