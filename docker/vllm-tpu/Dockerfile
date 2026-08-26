# Debian 12 / Python 3.12, matching the base the upstream TPU image uses.
# glibc 2.36 satisfies libtpu's manylinux_2_31 wheels.
FROM python:3.12-slim-bookworm@sha256:a116514e19457bcb7af7efe9c3dd0b9b71e85b317694e7882a1c52aa15a78134

# The lock is the whole contract: 241 pinned distributions with hashes,
# resolved from vllm-tpu==0.27.0, which hard-pins tpu-inference==0.27.0.
COPY vllm-tpu.lock /tmp/vllm-tpu.lock

RUN pip install --no-cache-dir uv \
 && uv pip install --system --require-hashes --no-cache \
      --requirement /tmp/vllm-tpu.lock \
 && rm /tmp/vllm-tpu.lock

# Fail the build rather than ship an image whose contents do not match the pin.
RUN python -c "\
import importlib.metadata as m; \
assert m.version('vllm-tpu') == '0.27.0', m.version('vllm-tpu'); \
assert m.version('tpu-inference') == '0.27.0', m.version('tpu-inference'); \
print('vllm-tpu', m.version('vllm-tpu'), '/ tpu-inference', m.version('tpu-inference'))"

# Work around an upstream tpu-inference defect: its shape lookups dereference
# hf_config.text_config eagerly as a getattr default, which only multimodal
# configs have, so a flat text config crashes the engine core at startup. Each
# patch documents its own defect and fails the build if the code it targets has
# changed. Patch contents are part of the image tag, so a change forces a
# rebuild rather than silently reusing an unpatched image.
COPY patches /tmp/patches
RUN for patch in /tmp/patches/*.py; do python "$patch"; done \
 && rm -rf /tmp/patches

# The rewritten expression must resolve on a flat text config under the pinned
# transformers, not just parse.
RUN python -c "\
from transformers import Qwen2Config; \
config = Qwen2Config(hidden_size=1536); \
assert not hasattr(config, 'text_config'), 'flat config unexpectedly nested'; \
assert config.get_text_config().hidden_size == 1536; \
print('get_text_config resolves on a flat Qwen2Config')"

ENTRYPOINT ["vllm"]
