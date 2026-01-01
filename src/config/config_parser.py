import tomllib

from src.config.paths import CONFIG_TOML_PATH


def load_config() -> dict:

    if not CONFIG_TOML_PATH.is_file():
        raise FileNotFoundError(f"Missing config file: {CONFIG_TOML_PATH}")

    with CONFIG_TOML_PATH.open("rb") as file:
        return tomllib.load(file)


def get_data_source_dataset_id() -> str:
    config = load_config()
    dataset_id = config.get("data_source", {}).get("dataset")
    if not isinstance(dataset_id, str) or not dataset_id.strip():
        raise ValueError(
            "Expected a non-empty string at [data_source].dataset in src/config/config.toml"
        )
    return dataset_id
