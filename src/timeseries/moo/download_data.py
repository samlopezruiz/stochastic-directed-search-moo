from src.timeseries.utils.config import read_config
from src.timeseries.utils.dataset import download_datasets

if __name__ == '__main__':
    project = 'snp'
    dataset_cfg = read_config('model_data_s3', project)
    download_datasets(dataset_cfg, project, '', 'saved_models', base='..')

    dataset_cfg = read_config('model_results_s3', project)
    download_datasets(dataset_cfg, project, '', 'results', base='..')

    dataset_cfg = read_config('train_test_ds', project)
    download_datasets(dataset_cfg, project, '', 'data', base='..')