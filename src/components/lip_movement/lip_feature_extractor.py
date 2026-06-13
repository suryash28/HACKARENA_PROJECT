import pandas as pd


class LipFeatureExtractor:

    def __init__(
        self,
        input_path="data/lip_data.csv",
        output_path="data/lip_features.csv"
    ):

        self.input_path = input_path
        self.output_path = output_path

    def transform_data(self):

        print("Loading dataset...")

        df = pd.read_csv(
            self.input_path,
            header=None
        )

        # Remove empty rows
        df = df.dropna()

        X = df.iloc[:, :-1]
        y = df.iloc[:, -1].astype(str).str.strip().str.lower()

        # Normalize coordinates row-wise
        X = X.sub(
            X.mean(axis=1),
            axis=0
        )

        final_df = pd.concat(
            [X, y],
            axis=1
        )

        final_df.to_csv(
            self.output_path,
            index=False,
            header=False
        )

        print(
            f"Features saved at {self.output_path}"
        )

        print("\nLabels Found:")
        print(y.value_counts())

        return self.output_path


if __name__ == "__main__":

    feature_extractor = LipFeatureExtractor()

    feature_extractor.transform_data()