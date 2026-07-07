from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, LSTM, Dense, Dropout

def build_model(hyperparameters, n_timesteps, n_features):
    filters1 = int(hyperparameters['filters1'])
    kernel_size1 = int(hyperparameters['kernel_size1'])
    pool_size1 = int(hyperparameters['pool_size1'])
    dropout1 = hyperparameters['dropout1']

    filters2 = int(hyperparameters['filters2'])
    kernel_size2 = int(hyperparameters['kernel_size2'])
    pool_size2 = int(hyperparameters['pool_size2'])
    dropout2 = hyperparameters['dropout2']

    lstm_units = int(hyperparameters['lstm_units'])
    dropout_lstm = hyperparameters['dropout_lstm']

    model = Sequential()
    
    # Explicitly define the input shape using an Input layer
    model.add(Input(shape=(n_timesteps, n_features)))
    
    # First Conv block (input_shape removed from Conv1D)
    model.add(Conv1D(filters=filters1, kernel_size=kernel_size1, activation='relu'))
    model.add(MaxPooling1D(pool_size=pool_size1))
    model.add(Dropout(dropout1))

    # Second Conv block
    model.add(Conv1D(filters=filters2, kernel_size=kernel_size2, activation='relu'))
    model.add(MaxPooling1D(pool_size=pool_size2))
    model.add(Dropout(dropout2))

    # Recurrent block and Output
    model.add(LSTM(lstm_units, activation='relu', return_sequences=False))
    model.add(Dropout(dropout_lstm))
    model.add(Dense(1, activation='sigmoid')) # Assuming binary classification
    
    return model