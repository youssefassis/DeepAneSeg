"""The installed commands and the configs they read."""

import shutil
import subprocess
from importlib.metadata import entry_points
from importlib.resources import files

import pytest

COMMANDS = {
    "deepaneseg-prepare": "prepare",
    "deepaneseg-remove-skull": "remove_skull",
    "deepaneseg-extract-points": "extract_points",
    "deepaneseg-preprocess": "preprocess",
    "deepaneseg-train": "train",
    "deepaneseg-predict": "predict",
    "deepaneseg-evaluate": "evaluate",
}


def test_every_command_is_installed():
    installed = {ep.name: ep for ep in entry_points(group="console_scripts") if ep.name.startswith("deepaneseg-")}

    assert set(installed) == set(COMMANDS)
    assert all(callable(ep.load()) for ep in installed.values())


@pytest.mark.parametrize("config", sorted(set(COMMANDS.values())))
def test_every_command_has_its_config_in_the_package(config):
    assert (files("deepaneseg.configs") / f"{config}.yaml").is_file()


@pytest.mark.parametrize("command", ["deepaneseg-train", "deepaneseg-predict"])
def test_commands_run_from_any_folder(command, tmp_path):
    result = subprocess.run([shutil.which(command), "--help"], cwd=tmp_path, capture_output=True, text=True)

    assert result.returncode == 0 and "Configuration groups" in result.stdout
