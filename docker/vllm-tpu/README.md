# vLLM TPU image

This directory defines the vLLM TPU service image that evaluations serve
models with. It is built on the TPU VM that runs it and is not pushed to a
registry.

The image is identified by a tag derived from the SHA-256 of `Dockerfile`,
`vllm-tpu.lock` and every patch in `patches/`;
`scripts/run_vllm_tpu_container.sh --print-image` prints it. An image ID would
not do, since it differs between builds on different machines.

## Build and inspect

From the repository root on the TPU VM:

```bash
scripts/run_vllm_tpu_container.sh --build
scripts/run_vllm_tpu_container.sh --check
scripts/run_vllm_tpu_container.sh --provenance
```

`--build` uses the digest-pinned `python:3.12-slim-bookworm` base declared in
the Dockerfile. The build and `--provenance` both verify that the installed
service reports `vllm-tpu 0.27.0` and `tpu-inference 0.27.0`, and the build
applies and checks the patches below. None of these uses the TPU. To free disk
space, remove superseded images with `docker image rm`.

## Patches

`patches/` holds build-time fixes for defects in the pinned upstream service.
Each is a stdlib-only Python script that rewrites the installed package and
exits non-zero when the code it targets is absent, so an upstream change fails
the build instead of silently shipping unpatched. The Dockerfile runs every
`patches/*.py` in name order after the install.

- `lazy_text_config_fallback.py` — `tpu_inference` reads shape parameters as
  `getattr(cfg, "hidden_size", cfg.text_config.hidden_size)`. Python evaluates
  the default eagerly, so `.text_config`, which only multimodal Hugging Face
  configs have, is always dereferenced, and a flat config such as
  `Qwen2Config` kills the engine core during startup with `AttributeError:
  'Qwen2Config' object has no attribute 'text_config'`. The patch rewrites the
  idiom to `cfg.get_text_config().hidden_size`, which returns the nested text
  config when there is one and the config itself otherwise.

Patches feed the image tag, so editing one changes the tag, and the wrapper
refuses the image built before the change until it is rebuilt.

## Regenerate the lock

Resolve inside the same Linux base image. `libtpu` publishes only
`manylinux_2_31_x86_64` wheels, so resolving from a laptop or with a host-side
platform override is unsupported.

```bash
sudo docker run --rm \
  --volume "$PWD/docker/vllm-tpu:/work" \
  --workdir /work \
  python:3.12-slim-bookworm \
  bash -c 'pip install --no-cache-dir uv &&
           uv pip compile --generate-hashes --output-file vllm-tpu.lock vllm-tpu.in'
```

The result must contain 241 distributions, including `vllm-tpu==0.27.0`,
`tpu-inference==0.27.0`, `libtpu==0.0.44`, `jax==0.11.0`, `jaxlib==0.11.0`,
`flax==0.12.8`, `torch==2.10.0`, `transformers==5.14.1` and `numpy==2.3.5`.
Review the complete diff before accepting a lock change. `vllm-tpu` hard-pins
`tpu-inference`, so the two service versions cannot drift apart.
