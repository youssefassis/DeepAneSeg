import logging
import os
import platform
import random
import shutil
import subprocess
from datetime import datetime, timezone

import numpy as np
import torch

from hydra.utils import instantiate


def save_checkpoint(state, is_best, checkpoint_dir, logger=None):
    """Saves model and training parameters at '{checkpoint_dir}/last_checkpoint.pytorch'.
    If is_best==True saves '{checkpoint_dir}/best_checkpoint.pytorch' as well.
    Args:
        state (dict): contains model's state_dict, optimizer's state_dict, epoch
            and best evaluation metric value so far
        is_best (bool): if True state contains the best model seen so far
        checkpoint_dir (string): directory where the checkpoint are to be saved
    """

    def log_info(message):
        if logger is not None:
            logger.info(message)

    if not os.path.exists(checkpoint_dir):
        log_info(f"Checkpoint directory does not exists. Creating {checkpoint_dir}")
        os.mkdir(checkpoint_dir)

    last_file_path = os.path.join(checkpoint_dir, "last_checkpoint.pytorch")
    torch.save(state, last_file_path)
    if is_best:
        best_file_path = os.path.join(checkpoint_dir, "best_checkpoint.pytorch")
        log_info(f"Saving best checkpoint to '{best_file_path}'")
        shutil.copyfile(last_file_path, best_file_path)
        return True
    return False


def load_checkpoint(
    checkpoint_path, model, optimizer=None, model_key="model_state_dict", optimizer_key="optimizer_state_dict"
):
    """Loads model and training parameters from a given checkpoint_path
    If optimizer is provided, loads optimizer's state_dict of as well.
    Args:
        checkpoint_path (string): path to the checkpoint to be loaded
        model (torch.nn.Module): model into which the parameters are to be copied
        optimizer (torch.optim.Optimizer) optional: optimizer instance into
            which the parameters are to be copied
    Returns:
        state
    """
    if not os.path.exists(checkpoint_path):
        raise IOError(f"Checkpoint '{checkpoint_path}' does not exist")

    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state[model_key])

    if optimizer is not None:
        optimizer.load_state_dict(state[optimizer_key])

    return state


def load_model(model_cfg, checkpoint_path):
    """Builds the model described by model_cfg (see configs/model) and loads its weights from checkpoint_path."""
    if not os.path.exists(checkpoint_path):
        raise IOError(f"Checkpoint '{checkpoint_path}' does not exist")
    model = get_model(model_cfg)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu")["model_state_dict"])
    return model


loggers = {}


def get_logger(name, level=logging.INFO):
    """Returns a named logger; handlers are left to the application (Hydra configures console and file logs)."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger


def seed_everything(seed):
    """Seeds Python, NumPy and PyTorch (CPU and CUDA). DataLoader workers derive their seeds from PyTorch's."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _git_commit():
    """Commit of the checkout the code runs from, or DEEPANESEG_GIT_COMMIT (set in the Docker image)."""
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True)
        status = subprocess.run(["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True, check=True)
        return commit.stdout.strip(), bool(status.stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        return os.environ.get("DEEPANESEG_GIT_COMMIT", "unknown"), None


def run_info(**extra):
    """Code and environment versions of a run, to be stored with its outputs."""
    commit, dirty = _git_commit()
    return {
        "git_commit": commit,
        "git_uncommitted_changes": dirty,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "numpy": np.__version__,
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **extra,
    }


def get_number_of_learnable_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])


class RunningAverage:
    """Computes and stores the average"""

    def __init__(self):
        self.count = 0
        self.sum = 0
        self.avg = 0

    def update(self, value, n=1):
        self.count += n
        self.sum += value * n
        self.avg = self.sum / self.count


def get_model(model_cfg, weight_init="pytorch", device="cpu"):
    """
    Instantiates the model described by model_cfg (a Hydra config with a _target_ class, see configs/model).
    weight_init: "keras" (Xavier uniform, as in the paper) or "pytorch" (PyTorch defaults)
    """
    logger = get_logger("Model creation")
    if weight_init not in ("keras", "pytorch"):
        raise ValueError(f"Unsupported weight_init '{weight_init}', expected 'keras' or 'pytorch'")
    model = instantiate(model_cfg, _convert_="all")

    logger.info(f"The model '{type(model).__name__}' was chosen to be trained")
    nparams = get_number_of_learnable_parameters(model)
    mem_params = sum([param.nelement() * param.element_size() for param in model.parameters()])
    size = (mem_params + sum([buf.nelement() * buf.element_size() for buf in model.buffers()])) * 1e-6

    logger.info(f"Number of learnable params {nparams}")
    logger.info(f"Size allocated by the model is {size:.2f} mb")

    # initialization
    if weight_init == "keras":
        logger.info("Apply Keras default initialization")
        model.apply(init_model)
    else:
        logger.info("Apply PyTroch default initialization")

    # Move to gpu if available
    logger.info(f"Sending the model to '{device}'")
    model = model.to(device)
    if torch.cuda.device_count() > 1 and not device == "cpu":
        model = torch.nn.DataParallel(model)
        logger.info(f"Using {torch.cuda.device_count()} GPUs for training (Data Parallel)")
    return model


def init_model(m):
    if isinstance(m, (torch.nn.Conv3d, torch.nn.ConvTranspose3d)):
        torch.nn.init.xavier_uniform_(m.weight, gain=torch.nn.init.calculate_gain("relu"))
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    elif isinstance(m, (torch.nn.BatchNorm3d)):
        if m.weight is not None:  # gamma
            torch.nn.init.ones_(m.weight)
        if m.bias is not None:  # beta
            torch.nn.init.zeros_(m.bias)


########### WarmUp scheduler ########
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau


class GradualWarmupScheduler(_LRScheduler):
    """Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.0:
            raise ValueError("multiplier should be greater thant or equal to 1.")
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        optimizer.zero_grad()
        optimizer.step()
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [
                base_lr * ((self.multiplier - 1.0) * self.last_epoch / self.total_epoch + 1.0)
                for base_lr in self.base_lrs
            ]

    def step_reduce_lr_on_plateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = (
            epoch if epoch != 0 else 1
        )  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [
                base_lr * ((self.multiplier - 1.0) * self.last_epoch / self.total_epoch + 1.0)
                for base_lr in self.base_lrs
            ]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group["lr"] = lr
        else:
            self.after_scheduler.step(metrics)  # , None)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_reduce_lr_on_plateau(metrics, epoch)
