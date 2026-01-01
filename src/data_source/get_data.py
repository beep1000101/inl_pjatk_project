from datasets import load_dataset


from config.config_parser import get_data_source_dataset_id
from config.paths import CACHE_DATA_DIR, dataset_cache_dir


def get_dataset():
    dataset_id = get_data_source_dataset_id()
    dataset = load_dataset(dataset_id)
    return dataset


def save_dataset_in_cache(dataset):
    CACHE_DATA_DIR.mkdir(parents=True, exist_ok=True)

    dataset_id = get_data_source_dataset_id()
    target_dir = dataset_cache_dir(dataset_id)

    if target_dir.exists() and any(target_dir.iterdir()):
        return

    dataset.save_to_disk(str(target_dir))


def main():
    dataset = get_dataset()
    save_dataset_in_cache(dataset=dataset)


if __name__ == "__main__":
    main()
