from src.components.lip_movement.lip_feature_extractor import (
    LipFeatureExtractor
)

from src.components.lip_movement.lip_model_trainer import (
    LipModelTrainer
)


class LipTrainPipeline:

    def __init__(self):
        pass

    def run_pipeline(self):

        print("Starting Feature Extraction...")

        extractor = LipFeatureExtractor()

        feature_file = extractor.transform_data()

        print("Feature Extraction Completed")

        print("Starting Model Training...")

        trainer = LipModelTrainer(
            feature_file_path=feature_file
        )

        trainer.train_model()

        print("Training Completed")


if __name__ == "__main__":

    pipeline = LipTrainPipeline()

    pipeline.run_pipeline()