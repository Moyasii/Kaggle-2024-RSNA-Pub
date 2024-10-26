import json
from pathlib import Path
from dataclasses import dataclass


@dataclass
class Settings:
    raw_data_dir: Path
    train_data_clean_dir: Path
    model_checkpoint_dir: Path
    pretrained_checkpoint_dir: Path
    submission_dir: Path


def load_settings(json_path: str = 'SETTINGS.json') -> Settings:
    with open(json_path, 'r') as f:
        data = json.load(f)
        settings = Settings(
            raw_data_dir=Path(data['RAW_DATA_DIR']),
            train_data_clean_dir=Path(data['TRAIN_DATA_CLEAN_DIR']),
            model_checkpoint_dir=Path(data['MODEL_CHECKPOINT_DIR']),
            pretrained_checkpoint_dir=Path(data['PRETRAINED_CHECKPOINT_DIR']),
            submission_dir=Path(data['SUBMISSION_DIR'])
        )
    return settings


if __name__ == '__main__':
    settings = load_settings()
    print(settings)
