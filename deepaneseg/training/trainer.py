import os

import torch
import torch.nn as nn
from tqdm import tqdm
from deepaneseg.utils import GradualWarmupScheduler

from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau

from deepaneseg.utils import get_logger, RunningAverage, save_checkpoint, load_checkpoint

from prefetch_generator import BackgroundGenerator


logger = get_logger("Model Trainer")


def create_trainer(
    config, device, model, optimizer, lr_scheduler, loss_criterion, eval_criterion, loaders, max_iterations
):
    assert config is not None, "Could not find trainer configuration"
    model_path = config["model_file"]
    if os.path.isfile(model_path):
        logger.info(f"Continue training from a checkpoint: '{model_path}/'")
        return Trainer.from_checkpoint(
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            loss_criterion=loss_criterion,
            eval_criterion=eval_criterion,
            device=device,
            loaders=loaders,
            checkpoint_dir=config["test_dir"],
            model_path=model_path,
            max_num_epochs=config["n_epochs"],
            max_iterations=max_iterations,
            earlystop=config.get("early_stop", 100),
        )
    else:
        logger.info("Training the model from scratch")
        return Trainer(
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            loss_criterion=loss_criterion,
            eval_criterion=eval_criterion,
            device=device,
            loaders=loaders,
            checkpoint_dir=config["test_dir"],
            model_path=model_path,
            max_num_epochs=config["n_epochs"],
            max_iterations=max_iterations,
            earlystop=config.get("early_stop", 100),
            config=config,
        )


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        lr_scheduler,
        loss_criterion,
        eval_criterion,
        device,
        loaders,
        checkpoint_dir,
        model_path,
        max_num_epochs=100,
        num_epoch=0,
        num_iterations=0,
        eval_score_higher_is_better=True,
        best_eval_score=None,
        max_iterations=0,
        monitor="valid",
        earlystop=100,
        nonimproved_epoch=0,
        config=None,
        **kwargs,
    ):

        self.model = model
        self.optimizer = optimizer
        self.scheduler = lr_scheduler
        self.loss_criterion = loss_criterion
        self.eval_criterion = eval_criterion
        self.device = device
        self.loaders = loaders
        self.checkpoint_dir = checkpoint_dir
        self.model_path = model_path
        self.max_num_epochs = max_num_epochs
        self.max_iterations = (
            max_iterations  # maximum number of iterations per epoch dict(train, valid) just to speedup tqdm
        )
        self.eval_score_higher_is_better = eval_score_higher_is_better
        self.num_epoch = num_epoch
        self.monitor = monitor
        self.nonimproved_epoch = 0  # nonimproved_epoch
        self.earlystop = earlystop

        if best_eval_score is not None:
            self.best_eval_score = best_eval_score
        else:
            if eval_score_higher_is_better:
                self.best_eval_score = float("-inf")
            else:
                self.best_eval_score = float("+inf")
        self.writer = SummaryWriter(log_dir=os.path.join(checkpoint_dir, "logs"))

    @classmethod
    def from_checkpoint(
        cls,
        model,
        optimizer,
        lr_scheduler,
        loss_criterion,
        eval_criterion,
        device,
        loaders,
        checkpoint_dir,
        model_path,
        max_num_epochs=0,
        max_iterations=0,
        earlystop=100,
        **kwargs,
    ):

        state = load_checkpoint(model_path, model, optimizer)
        logger.info(f"Checkpoint loaded. Epoch: {state['epoch']}; Best metric score: {state['best_eval_score']}")

        return cls(
            model,
            optimizer,
            lr_scheduler,
            loss_criterion,
            eval_criterion,
            device,
            loaders,
            checkpoint_dir,
            model_path,
            eval_score_higher_is_better=state["eval_score_higher_is_better"],
            best_eval_score=state["best_eval_score"],
            num_epoch=state["epoch"],
            max_num_epochs=max_num_epochs,
            max_iterations=max_iterations,
            earlystop=earlystop,
        )

    def fit(self, sanity_check=False, deep_supervision=False):
        torch.backends.cudnn.benchmark = True
        self.adjust_lr = True
        logger.info(f"Start training the model for the next {self.max_num_epochs - self.num_epoch} epochs")
        for epoch in range(self.num_epoch, self.max_num_epochs):
            should_terminate = self.train(sanity_check, deep_supervision)
            if should_terminate:
                return
            should_terminate = self.validate(sanity_check, deep_supervision)
            if should_terminate:
                return
            self.num_epoch += 1
        logger.info(f"Reached maximum number of epochs: {self.max_num_epochs}. Finishing training.")

    ######################## BASIC ########################
    def train(self, sanity_check=False, deep_supervision=False):
        logger.info(f"Epoch [{self.num_epoch}/{self.max_num_epochs}]")
        self.model.train()
        loop = tqdm(
            BackgroundGenerator(self.loaders["train"]), leave=True, unit="batch", total=self.max_iterations["train"]
        )
        self.lr = self.optimizer.param_groups[0]["lr"]

        train_losses = RunningAverage()
        metric_scores = RunningAverage()
        for batch in loop:
            input, target = self._split_training_batch(batch)

            output = self.model(input)
            if deep_supervision:
                target = get_multiscale_gt(target)

            loss, metric = self.loss_batch(self.loss_criterion, self.eval_criterion, output, target, opt=self.optimizer)

            train_losses.update(loss, self._batch_size(input))
            metric_scores.update(metric, self._batch_size(input))
            loss, metric = train_losses.avg, metric_scores.avg

            # Progress bar
            loop.set_description("Training")
            loop.set_postfix(loss=loss, dice=metric, lr=self.lr)
            if sanity_check is True:
                break
        self._log_stats("train", loss, metric, self.num_epoch)
        self._save_best("train", metric)

        if self.should_stop():
            logger.info("Stopping criterion is satisfied. Finishing training.")
            return True
        return False

    @torch.no_grad()  # turn off gradients
    def validate(self, sanity_check=False, deep_supervision=False):
        if self.loaders["valid"] is not None:
            self.model.eval()  # Switch model to evaluation mode
            eval_losses = RunningAverage()
            eval_metric = RunningAverage()

            loop = tqdm(
                BackgroundGenerator(self.loaders["valid"]), leave=True, unit="batch", total=self.max_iterations["valid"]
            )
            for batch in loop:
                input, target = self._split_training_batch(batch)

                output = self.model(input)
                if deep_supervision:
                    target = get_multiscale_gt(target)

                loss, metric = self.loss_batch(self.loss_criterion, self.eval_criterion, output, target, opt=None)

                eval_losses.update(loss, self._batch_size(input))
                eval_metric.update(metric, self._batch_size(input))
                loss, metric = eval_losses.avg, eval_metric.avg

                loop.set_description("Validat.")
                loop.set_postfix(eval_loss=loss, eval_dice=metric)
                if sanity_check is True:
                    break
            self._log_stats("val", loss, metric, self.num_epoch)
            saved = self._save_best("valid", metric)
            self._update_lr(loss, saved)

            if self.should_stop(train=False):
                logger.info("Stopping criterion is satisfied. Finishing training.")
                return True
            return False

    def _update_lr(self, loss, saved):
        if not self.adjust_lr:
            return
        if isinstance(self.scheduler, ReduceLROnPlateau):
            self.scheduler.step(loss)
        elif isinstance(self.scheduler, GradualWarmupScheduler):
            self.scheduler.step(self.num_epoch, loss)
        else:
            self.scheduler.step()

    def _save_best(self, monitor, score):
        assert monitor in ["train", "valid"], (
            "Please specify a valid monitor for to save the checkpoint from ('train' or 'valid')"
        )
        if self.monitor == monitor:
            # remember best validation metric
            is_best = self._is_best_eval_score(score)
            if not is_best:
                self.nonimproved_epoch += 1
            else:
                self.nonimproved_epoch = 0
            # save checkpoint
            return self._save_checkpoint(is_best)
        return False

    def should_stop(self, train=True):
        """
        Training will terminate if maximum number of iterations is exceeded or the learning rate drops below
        some predefined threshold (1e-8 in this case)
        Specify more criterions
        """
        if train:
            min_lr = 1e-8
            lr = self.optimizer.param_groups[0]["lr"]
            if lr < min_lr:
                logger.info(f"Learning rate below the minimum {min_lr}.")
                return True

        if not train:
            # Early stopping
            if self.nonimproved_epoch > 0:
                logger.info(f"Early Stopping: [{self.nonimproved_epoch}/{self.earlystop}]")
                if self.nonimproved_epoch >= self.earlystop:
                    logger.info("Early stopping criterion is satisfied.")
                    return True
        return False

    def _is_best_eval_score(self, eval_score):
        if self.eval_score_higher_is_better:
            is_best = eval_score > self.best_eval_score
        else:
            is_best = eval_score < self.best_eval_score

        if is_best:
            logger.info(f"Saving new best evaluation metric: {eval_score}")
            self.best_eval_score = eval_score
        return is_best

    def _save_checkpoint(self, is_best):
        if isinstance(self.model, nn.DataParallel):
            state_dict = self.model.module.state_dict()
        else:
            state_dict = self.model.state_dict()

        return save_checkpoint(
            {
                "epoch": self.num_epoch + 1,
                "model_state_dict": state_dict,
                "best_eval_score": self.best_eval_score,
                "eval_score_higher_is_better": self.eval_score_higher_is_better,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "num_epoch": self.num_epoch,
                "max_num_epochs": self.max_num_epochs,
            },
            is_best,
            checkpoint_dir=self.checkpoint_dir,
            logger=logger,
        )

    def _log_stats(self, phase, loss_avg, dice_avg, step):
        def _log_lr():
            lr = self.optimizer.param_groups[0]["lr"]
            self.writer.add_scalar("learningRate", lr, self.num_epoch)
            self.writer.close()

        # Log learning rate
        if phase == "train":
            _log_lr()

        # Log loss and metric
        self.writer.add_scalars("Loss", {f"{phase}": loss_avg}, step)
        self.writer.add_scalars("Metrics/Dice", {f"{phase}": dice_avg}, step)
        self.writer.close()

    def _split_training_batch(self, t):
        def _move_to_device(input):
            if isinstance(input, tuple) or isinstance(input, list):
                return tuple([_move_to_device(x) for x in input])
            else:
                return input.to(self.device)

        return _move_to_device(t)

    @staticmethod
    def _batch_size(input):
        if isinstance(input, list) or isinstance(input, tuple):
            return input[0].size(0)
        else:
            return input.size(0)

    @staticmethod
    def loss_batch(loss_func, metrics_func, output, target, opt=None):
        loss = compute_loss(loss_func, output, target)
        with torch.no_grad():
            if isinstance(output, list) and isinstance(target, list):
                metric_b = metrics_func(output[-1], target[-1])
            else:
                metric_b = metrics_func(output, target)
        if opt is not None:
            opt.zero_grad()  # reset gradient to zero
            loss.backward()  # backward pass
            opt.step()  # weight updates
        return loss.item(), metric_b.item()


def compute_loss(loss_func, output, target):
    if isinstance(output, list) and isinstance(target, list):
        assert len(output) == len(target), "Output and Target tensors should have the save length"
        loss = []
        for out, tar in zip(output, target):
            loss.append(loss_func(out, tar))
        return sum(loss) / len(loss)
    else:
        return loss_func(output, target)


def get_multiscale_gt(target):
    """return a list of features"""
    gt_features_levels = []
    gt_features_levels.append(torch.nn.Upsample(size=6, mode="trilinear", align_corners=True)(target))
    gt_features_levels.append(torch.nn.Upsample(size=12, mode="trilinear", align_corners=True)(target))
    gt_features_levels.append(torch.nn.Upsample(size=24, mode="trilinear", align_corners=True)(target))
    gt_features_levels.append(target)
    return gt_features_levels
