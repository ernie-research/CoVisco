# Contributing to CoVisco

Thanks for your interest in contributing. This guide covers the basics for
reporting issues and submitting changes.

## Repository layout

CoVisco is a two-stage pipeline, with one directory per stage:

- `CoVisco_ViT/` — Stage 1: vision encoder pretraining (built on OpenCLIP).
- `CoVisco_sft/` — Stage 2: multimodal alignment and instruction tuning
  (LLaVA-OneVision-style training on Qwen3).

Each stage has its own README and dependency files. Read the relevant stage
README before making changes.

## Reporting issues

When opening an issue, please include:

- What you were trying to do and which stage (`CoVisco_ViT` or `CoVisco_sft`).
- Environment: OS, Python version, and `torch` / `transformers` versions.
- Steps to reproduce, the full command, and the complete error output.

Note that training and evaluation deliberately pin different `transformers`
versions (training stays on an older pin, evaluation requires a newer one for
Qwen3-VL). Please state which environment you are in — many reported errors are
version-mismatch issues.

## Development setup

Install the stage you are working on in editable mode:

```bash
# Stage 1
cd CoVisco_ViT && pip install -e '.[training]'

# Stage 2
cd CoVisco_sft && pip install -r requirements.txt
```

## Code style

The repository uses [Ruff](https://docs.astral.sh/ruff/) for both linting and
formatting, configured in the root `ruff.toml` (shared by both stages).

Before submitting, run from the repository root:

```bash
pip install ruff
ruff check .          # lint
ruff format --check . # formatting check (run `ruff format .` to apply)
```

Guidelines:

- First-party code and comments are in English.
- Do not hand-edit anything under `third_party/`; it is a vendored upstream
  snapshot. Local changes to vendored code must be documented in the
  corresponding `UPSTREAM.txt`.

## Pull requests

- Keep each PR focused on a single change; avoid bundling unrelated edits.
- Ensure `ruff check .` and `ruff format --check .` pass.
- Describe what changed, why, and how you verified it (commands run, results).
- Update the relevant README/docs when you change user-facing behavior.

## License

By contributing, you agree that your contributions are licensed under the
project's [MIT License](LICENSE).
