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

ENTRYPOINT ["vllm"]
