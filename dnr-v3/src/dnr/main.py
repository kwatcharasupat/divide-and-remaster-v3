import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from .generate import generate_split


@hydra.main(version_base=None, config_path="/root/aa-cass/aa-cass-rdnr/config")
def generate(cfg: DictConfig) -> None:
    """Main entry point for the data generation process.

    This function is decorated with @hydra.main, making it the primary entry point
    for the command-line interface. It initializes the random state and iterates
    through the dataset splits defined in the configuration, calling generate_split
    for each one.

    Args:
        cfg: The Hydra DictConfig object containing the full configuration.
    """
    print(OmegaConf.to_yaml(cfg))

    random_state = np.random.default_rng(cfg.seed)

    splits = cfg.splits

    for split in splits:
        generate_split(split=split, cfg=cfg, random_state=random_state)


if __name__ == "__main__":
    generate()
