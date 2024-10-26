import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

from src.utils import load_settings


def main():
    settings = load_settings()
    raw_data_dir = settings.raw_data_dir
    pretrained_checkpoint_dir = settings.pretrained_checkpoint_dir
    raw_data_dir.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()

    datasets = [
        {
            'type': 'dataset',
            'name': 'brendanartley/lumbar-coordinate-pretraining-dataset',
            'extract_dir': raw_data_dir / 'lumbar_coordinate_dataset'
        },
        {
            'type': 'competition',
            'name': 'rsna-2024-lumbar-spine-degenerative-classification',
            'extract_dir': raw_data_dir / 'rsna-2024-lumbar-spine-degenerative-classification'
        },
        {
            'type': 'dataset',
            'name': 'akinosora/rsna2024-lsdc-3rd-models-pub',
            'extract_dir': pretrained_checkpoint_dir,
        },
    ]

    for item in datasets:
        if item['type'] == 'dataset':
            print(f"Downloading dataset {item['name']}")
            api.dataset_download_files(item['name'], path=str(raw_data_dir), quiet=False)
            zip_filename = item['name'].split('/')[-1] + '.zip'
        elif item['type'] == 'competition':
            print(f"Downloading competition data {item['name']}")
            api.competition_download_files(item['name'], path=str(raw_data_dir), quiet=False)
            zip_filename = item['name'] + '.zip'

        zip_file = raw_data_dir / zip_filename
        extract_dir = item['extract_dir']

        print(f'Extracting {zip_file} to {extract_dir}')
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        print(f'Removing {zip_file}')
        os.remove(zip_file)


if __name__ == '__main__':
    main()
