"""`docker/trace-proxy/config.yaml`: the litellm proxy config the trace
capture pipeline actually loads.
"""

from __future__ import annotations

from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).parents[1] / "docker" / "trace-proxy" / "config.yaml"


def load():
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def test_it_parses_as_yaml():
    config = load()
    assert isinstance(config, dict)


def test_it_wires_the_free_custom_callback_not_the_enterprise_gcs_bucket_integration():
    # litellm's own `gcs_bucket` success callback is Enterprise-licensed (see
    # docker/trace-proxy/gcs_logger.py's module docstring for the citation
    # against the pinned litellm version's source); this proxy must load the
    # free substitute instead.
    callbacks = load()["litellm_settings"]["callbacks"]
    assert callbacks == "gcs_logger.proxy_handler_instance"
    assert "gcs_bucket" not in callbacks


def test_it_sets_no_message_redaction_option():
    settings = load()["litellm_settings"]
    # Full prompts and completions in the logged payload are the entire
    # point; the only way that happens is by never setting this.
    assert "turn_off_message_logging" not in settings
    assert "redact" not in {key.lower() for key in settings}


def test_every_deployment_specific_field_is_an_os_environ_reference():
    litellm_params = load()["model_list"][0]["litellm_params"]
    assert litellm_params["api_base"] == "os.environ/TRACE_PROXY_UPSTREAM_BASE_URL"


def test_the_custom_callback_module_it_names_actually_exists():
    callback_module = load()["litellm_settings"]["callbacks"].split(".")[0]
    assert (CONFIG_PATH.parent / f"{callback_module}.py").is_file()
