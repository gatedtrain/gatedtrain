"""Building blocks of registry rows (MULTIMODAL-SPEC section 6) that several adapters' rows share."""

from __future__ import annotations

# Deployable components of an HF LLM directory (08 section 3.6, R12-6; EXT-14, design choice of list):
# weights, model config, tokenizer files, generation config. Top-level files only.
HF_LLM_COMPONENTS = ("config.json", "generation_config.json", "model.safetensors", "model-*-of-*.safetensors",
                     "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json",
                     "special_tokens_map.json", "vocab.json", "merges.txt", "added_tokens.json",
                     "chat_template.jinja", "tokenizer.model")

# Read-only mounts can preserve their host path so absolute symlinks keep working.
# ext-launch normalizes paths and applies system directory restrictions to this layout (EXT-112).
CONTAINER_PATHS = {"ro_identity": True, "ro_under": ("/inputs",), "rw": ("/out",), "tmpfs": ("/tmp",)}


# --- a recipe's runtime directory and its venv (rows that name no host path) ---

_WARNED: set[str] = set()      # deprecated variables already named on stderr by this process

def runtime_dir(env_var: str, anchor: str, parents: int = 2, *, pack: str | None = None) -> str:
    """The recipe's runtime directory: `env_var` when set (a deprecated workaround, named on stderr, removed in
    `ENV_REMOVED_IN`), the host file's `[packs.<pack>] runtime_dir` when `pack` is given, else `<repository>/runtime`,
    the repository being the `parents`-th parent of `anchor` (the rows module's `__file__`; an editable or source
    checkout). The variable and the file key both set to different values is Refused (EXT-HOST): neither outranks the
    other."""
    import os
    import sys
    from pathlib import Path

    from .launch import DEFAULT_HOST_CONFIG, ENV_HOST_CONFIG, ENV_REMOVED_IN, env_conflict, pack_setting

    from ..errors import Refused

    from_env = os.environ.get(env_var)
    if from_env:
        try:
            from_file = pack_setting(pack, "runtime_dir") if pack else None
        except Refused:
            from_file = None                              # a malformed file: named below, as for no variable
        if from_file and from_file != from_env:           # the owner's file key is never outranked silently
            raise env_conflict(env_var, f"packs.{pack}.runtime_dir", from_env, from_file,
                               os.path.expanduser(os.environ.get(ENV_HOST_CONFIG) or DEFAULT_HOST_CONFIG))
        if env_var not in _WARNED:                        # once per variable and process
            _WARNED.add(env_var)
            print(f"warning: {env_var} is deprecated: set `runtime_dir` in the host file's "
                  f"[packs.{pack or '<pack name>'}] table instead; the variable is removed in {ENV_REMOVED_IN}",
                  file=sys.stderr)
        return from_env
    default = str(Path(anchor).resolve().parents[parents] / "runtime")
    try:
        return (pack_setting(pack, "runtime_dir") if pack else None) or default
    except Refused as exc:
        # A pack is imported by every gatedtrain command, `--help` included: a malformed host file must not take them
        # all down. The package's own runtime directory is used and the problem is named; a stage that needs the
        # host file (the owner's bounds) refuses on its own.
        print(f"warning: the host file is malformed, so {pack}'s runtime_dir is the package default: "
              f"{exc.problems[0].line()}", file=sys.stderr)
        return default


def python_tree(runtime: str) -> str | None:
    """The interpreter tree the runtime's venv was made from (`.venv/pyvenv.cfg` `home` = `<tree>/bin`); None
    without a venv (ext-launch then refuses the interpreter)."""
    from pathlib import Path

    try:
        cfg = (Path(runtime) / ".venv" / "pyvenv.cfg").read_text(encoding="utf-8")
    except OSError:
        return None
    for line in cfg.splitlines():
        key, _, value = line.partition("=")
        if key.strip() == "home" and value.strip():
            return str(Path(value.strip()).parent)
    return None


def venv_runtime_rows(runtime: str, out_name: str) -> dict:
    """The host-path keys of a `venv_in_container` training row (EXT-79, EXT-93, EXT-113): the input roots (the
    interpreter install directory, the runtime directory, the recipe's directory under the host's output root), the
    scan skips (the runtime venv, the interpreter's site-packages), the interpreter trees and the interpreter."""
    from pathlib import Path

    tree = python_tree(runtime)
    version = Path(tree).name.split("-")[1].rsplit(".", 1)[0] if tree else None    # cpython-3.12.16-... -> 3.12
    return {"input_roots": tuple(([str(Path(tree).parent)] if tree else []) + [runtime, "{output_root}/" + out_name]),
            "scan_skip": tuple([f"{runtime}/.venv"] + ([f"{tree}/lib/python{version}/site-packages"] if tree else [])),
            "interpreter_trees": (tree,) if tree else (),
            "interpreter": f"{runtime}/.venv/bin/python"}
