# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import argparse
import sys
from types import ModuleType, SimpleNamespace

import pytest

from relax.utils.env import validate_env
from tests.utils.test_arguments_opd_teacher_colocate import arguments_module as arguments_module_fixture


base_arguments_module = arguments_module_fixture


@pytest.fixture
def arguments_module(monkeypatch, request):
    # OPD is unrelated to this parser branch and imports PyTorch eagerly.
    # Keep the actual Agentic argument registration/validation under test.
    opd = ModuleType("relax.utils.opd.opd_utils")
    for name in (
        "add_opd_arguments",
        "is_managed_opd_teacher_enabled",
        "teacher_sglang_parse_args",
        "validate_managed_opd_teacher_colocate_args",
        "validate_opd_args",
    ):
        setattr(opd, name, lambda value, **kwargs: value)
    monkeypatch.setitem(sys.modules, opd.__name__, opd)
    return request.getfixturevalue("base_arguments_module")


def parse(module, tmp_path, extra=()):
    module.RouterArgs = SimpleNamespace(add_cli_args=lambda parser, **kwargs: parser)
    parser = argparse.ArgumentParser()
    module.get_slime_extra_args_provider()(parser)
    return parser.parse_args(
        [
            "--use-agentic-rollout",
            "--agent-command",
            "python agent.py",
            "--agent-cwd",
            str(tmp_path),
            *extra,
        ]
    )


def test_publication_argument_uses_existing_agentic_parser(arguments_module, tmp_path):
    args = parse(arguments_module, tmp_path, ["--lora-publication-url", "http://gateway"])
    arguments_module._validate_agentic_rollout_args(args)
    assert args.lora_publication_url == "http://gateway"
    assert args.rollout_function_path == "relax.agentic.rollout.generate_rollout"


@pytest.mark.parametrize("url", ["file:///tmp/gateway", "http://", "http://gateway?q=1", "http://gateway#fragment"])
def test_publication_argument_rejects_invalid_gateway_url(arguments_module, tmp_path, url):
    args = parse(arguments_module, tmp_path, ["--lora-publication-url", url])
    with pytest.raises(ValueError, match="HTTP"):
        arguments_module._validate_agentic_rollout_args(args)


@pytest.mark.parametrize("flag", ["--agentic-session-lifecycle", "--agentic-program-admission"])
def test_publication_argument_rejects_incompatible_lifecycles(arguments_module, tmp_path, flag):
    args = parse(arguments_module, tmp_path, ["--lora-publication-url", "http://gateway", flag])
    with pytest.raises(ValueError, match="immutable LoRA"):
        arguments_module._validate_agentic_rollout_args(args)


def test_publication_export_environment_is_registered():
    validate_env({"RELAX_LORA_PUBLICATION_URL": "http://gateway", "RELAX_LORA_VERSION_PREFIX": "run-1"})
