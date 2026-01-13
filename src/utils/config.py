import os

import yaml


class Config:
    """
    Singleton-like configuration loader.
    """

    _instance = None
    _config_data = None

    def __new__(cls, config_path="config.yaml"):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._load_config(config_path)
        return cls._instance

    def _load_config(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Configuration file not found at: {path}")

        with open(path, "r") as f:
            self._config_data = yaml.safe_load(f)
            print(f"--> [Config] Loaded configuration from {path}")

    def get(self, key, default=None):
        """
        Retrieve a value using dot notation (e.g., 'topic_modeling.hdbscan.min_cluster_size')
        """
        keys = key.split(".")
        value = self._config_data
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    @property
    def data(self):
        return self._config_data


# Instantiate globally to be imported elsewhere
cfg = Config()
