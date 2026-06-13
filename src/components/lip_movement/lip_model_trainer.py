import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


class LipModelTrainer:

    def __init__(
        self,
        feature_file_path="data/lip_features.csv",
        model_path="models/lip_model.pkl"
    ):

        self.feature_file_path = feature_file_path
        self.model_path = model_path

    def train_model(self):

        print("Loading dataset...")

        df = pd.read_csv(
            self.feature_file_path,
            header=None
        )

        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        print(f"Total Samples: {len(df)}")
        print("\nClass Distribution:")
        print(y.value_counts())

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        model = RandomForestClassifier(
            n_estimators=1000,
            max_depth=30,
            min_samples_split=3,
            random_state=42,
            n_jobs=-1
        )

        print("\nTraining Model...")

        model.fit(X_train, y_train)

        accuracy = model.score(X_test, y_test)

        print(f"\nAccuracy: {accuracy:.4f}")

        y_pred = model.predict(X_test)

        print("\nClassification Report:\n")
        print(
            classification_report(
                y_test,
                y_pred
            )
        )

        os.makedirs(
            os.path.dirname(self.model_path),
            exist_ok=True
        )

        joblib.dump(
            model,
            self.model_path
        )

        print(f"\nModel Saved -> {self.model_path}")

        return model