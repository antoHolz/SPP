# Add the parent directory to the Python path (because we are executing hydra from within runs)
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import os
import json
import gc
import logging

import hydra
from omegaconf import DictConfig
import torch

from modules.training.training import lightning_train_model
from modules.loader.loaders import trainScaler
from modules.utils.seeds import seed_everything
from modules.utils.hydraqol import run_decorator


@hydra.main(config_path="../data/config", config_name="train_model", version_base="1.3")
@run_decorator
def main(cfg: DictConfig) -> None:
    """Train an SPP time-series model with PyTorch Lightning.

    The full configuration tree is composed by Hydra. The SPP defaults are:
    ``dataset=SPP5s_DR model=InceptionTime10 training=adamw_r_patient
    transforms=tsHFbasic loader=M`` (see ``config/train_model.yaml``).

    Examples
    --------
    Train the default SPP model::

        python runs/train.py

    Override the dataset / model / seed::

        python runs/train.py dataset=SPP5s_TC model=ResNet1D18 seed=1

    Run the SPP5 sweep with the joblib launcher::

        python runs/train.py --multirun +experiment=SPP5 +launcher=joblib
    """
    ##############################
    # Step 1: Preliminaries
    ##############################
    logger = logging.getLogger("training")
    save_dir = Path(cfg.save_dir)
    seed_everything(cfg.seed)
    device = "cuda" if (cfg.training.device == "cuda" and torch.cuda.is_available()) else "cpu"

    # Fit the input/output scaler if any transform requires standardization
    if cfg.transforms.standardize:
        cfg.dataset.mean, cfg.dataset.std = trainScaler(cfg.dataset, cfg.loader)

    ##############################
    # Step 2: Instantiate objects from config
    ##############################
    # 1) Transforms (pretransform lives on the dataset config; aug transforms on the transforms config)
    pretransform = hydra.utils.instantiate(cfg.dataset.pretransform)
    train_transform = hydra.utils.instantiate(cfg.transforms.train)
    valid_transform = hydra.utils.instantiate(cfg.transforms.val)
    test_transform = hydra.utils.instantiate(cfg.transforms.test)

    # 2) Datasets (HFDataset wrapping a datasets.load_from_disk source)
    train_dataset = hydra.utils.instantiate(cfg.dataset.train, transform=train_transform, pretransform=pretransform)
    valid_dataset = hydra.utils.instantiate(cfg.dataset.val, transform=valid_transform, pretransform=pretransform)
    test_dataset = hydra.utils.instantiate(cfg.dataset.test, transform=test_transform, pretransform=pretransform)

    # 3) Data loaders (train/val/test split is handled inside create_dataloaders)
    train_loader, valid_loader, test_loader = hydra.utils.instantiate(
        cfg.loader.object,
        dataset_train=train_dataset,
        dataset_val=valid_dataset,
        dataset_test=test_dataset,
        task=cfg.dataset.task,
    )
    logger.info(f"Dataset splits -- train: {len(train_dataset)}, valid: {len(valid_dataset)}, test: {len(test_dataset)}")

    # 4) Model
    model = hydra.utils.instantiate(cfg.model.object)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Training: {cfg.name} (params: {n_params:,})".replace(",", " "))

    ##############################
    # Step 3: Execute task (Lightning training loop)
    ##############################
    results = lightning_train_model(model, cfg, train_loader, valid_loader, test_loader)

    ##############################
    # Step 4: Persist artifacts
    ##############################
    if int(os.environ.get("SLURM_PROCID", 0)) == 0:
        with open(save_dir / "training_results.json", "w") as f:
            json.dump(results, f, indent=4)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    return results["last"]["test_loss"] if results is not None else None


if __name__ == '__main__':
    main()
