from dataclasses import dataclass

from omegaconf import MISSING, OmegaConf

from mridc.core.conf.hydra_runner import hydra_runner


@dataclass
class DefaultConfig:
    """
    This is structured config for this application.
    It provides the schema used for validation of user-written spec file
    as well as default values of the selected parameters.
    """

    # Dataset. Available options: [imdb, sst2]
    dataset_name: str = MISSING


@hydra_runner(config_name="DefaultConfig", schema=DefaultConfig)
def tmp_launch(cfg):
    """Launch the application."""
    print(OmegaConf.to_yaml(cfg))
    _ = cfg.dataset_name


if __name__ == "__main__":
    tmp_launch()
