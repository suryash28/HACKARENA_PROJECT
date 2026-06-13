import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import pickle

DATASET_PATH = "dataset"

X = []
y = []

print("Loading Dataset...")

for gesture in os.listdir(DATASET_PATH):

    gesture_path = os.path.join(DATASET_PATH, gesture)

    if not os.path.isdir(gesture_path):
        continue

    for file in os.listdir(gesture_path):

        if file.endswith(".npy"):

            sequence = np.load(
                os.path.join(gesture_path, file)
            )

            X.append(sequence)
            y.append(gesture)

X = np.array(X)
y = np.array(y)

print("X Shape:", X.shape)
print("Y Shape:", y.shape)

# Encode Labels

le = LabelEncoder()
y_encoded = le.fit_transform(y)

num_classes = len(np.unique(y_encoded))

y_categorical = to_categorical(
    y_encoded,
    num_classes=num_classes
)

# Save Label Encoder

os.makedirs("models", exist_ok=True)

with open("models/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

# Split Data

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_categorical,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

print("Training Samples:", len(X_train))
print("Testing Samples:", len(X_test))

# LSTM Model

model = Sequential()

model.add(
    LSTM(
        64,
        return_sequences=True,
        input_shape=(30, 63)
    )
)

model.add(Dropout(0.2))

model.add(
    LSTM(128)
)

model.add(Dropout(0.2))

model.add(
    Dense(
        64,
        activation="relu"
    )
)

model.add(
    Dense(
        num_classes,
        activation="softmax"
    )
)

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=16,
    callbacks=[early_stop]
)

loss, accuracy = model.evaluate(
    X_test,
    y_test
)

print("\nAccuracy:", accuracy * 100)

model.save(
    "models/gesture_lstm.keras"
)

print("\nModel Saved!")