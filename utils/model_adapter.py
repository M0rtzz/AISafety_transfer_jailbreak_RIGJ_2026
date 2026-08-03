from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaConfig,
    LlamaForCausalLM,
    PretrainedConfig,
)

MODEL_FAMILIES = ("qwen", "llama", "mistral", "vicuna", "internlm")

VICUNA_CHAT_TEMPLATE = r"""{%- if messages[0]['role'] == 'system' -%}
    {%- set loop_messages = messages[1:] -%}
    {%- set system_message = messages[0]['content'] -%}
{%- else -%}
    {%- set loop_messages = messages -%}
    {%- set system_message = 'A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user\'s questions.' -%}
{%- endif -%}
{%- for message in loop_messages -%}
    {%- if (message['role'] == 'user') != (loop.index0 % 2 == 0) -%}
        {{- raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') -}}
    {%- endif -%}
    {%- if loop.index0 == 0 -%}
        {{- system_message -}}
    {%- endif -%}
    {%- if message['role'] == 'user' -%}
        {{- ' USER: ' + message['content'].strip() -}}
    {%- elif message['role'] == 'assistant' -%}
        {{- ' ASSISTANT: ' + message['content'].strip() + eos_token -}}
    {%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
    {{- ' ASSISTANT:' -}}
{%- endif -%}"""


def resolve_model_reference(
    value: str, aliases: dict[str, str] | None = None
) -> tuple[str, bool]:
    aliases = aliases or {}
    resolved = aliases.get(value, value)
    candidate = Path(resolved).expanduser()
    looks_local = (
        candidate.is_absolute()
        or resolved.startswith((".", "~"))
        or resolved.startswith("Models/")
    )
    if candidate.exists():
        return str(candidate.resolve()), True
    if looks_local:
        raise FileNotFoundError(
            f"local model path does not exist: {candidate} (cwd={Path.cwd()})"
        )
    return resolved, False


def _read_local_config(reference: str) -> dict[str, Any]:
    path = Path(reference)
    config_path = path / "config.json"
    if not config_path.is_file():
        return {}
    try:
        value = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def infer_model_family(
    reference: str,
    *,
    explicit_family: str | None = None,
    config: Any | None = None,
) -> str:
    if explicit_family:
        family = explicit_family.strip().lower()
        if family not in MODEL_FAMILIES:
            raise ValueError(
                f"unsupported model family {explicit_family!r}; expected one of {MODEL_FAMILIES}"
            )
        return family

    text = str(reference).rstrip("/").lower()
    config_dict = _read_local_config(reference)
    model_type = str(
        getattr(config, "model_type", "") or config_dict.get("model_type", "")
    ).lower()
    architectures = " ".join(
        str(item).lower()
        for item in (
            getattr(config, "architectures", None)
            or config_dict.get("architectures", [])
            or []
        )
    )
    combined = f"{text} {model_type} {architectures}"
    if "vicuna" in combined:
        return "vicuna"
    if "internlm" in combined:
        return "internlm"
    if "qwen" in combined:
        return "qwen"
    if "mistral" in combined or "mixtral" in combined:
        return "mistral"
    if "llama" in combined or "guanaco" in combined:
        return "llama"
    raise ValueError(
        f"cannot infer model family from {reference!r}; pass --source-family/--target-family"
    )


def ensure_known_chat_template(
    tokenizer: Any,
    family: str,
) -> bool:
    if getattr(tokenizer, "chat_template", None):
        return False
    if family == "vicuna":
        tokenizer.chat_template = VICUNA_CHAT_TEMPLATE
        return True
    raise ValueError(
        f"{family} tokenizer has no chat_template; provide a tokenizer with one"
    )


def chat_template_generation_kwargs(tokenizer: Any) -> dict[str, Any]:
    template = getattr(tokenizer, "chat_template", None)
    if template is not None and "enable_thinking" in str(template):
        return {"enable_thinking": False}
    return {}


def render_user_prompt(
    tokenizer: Any,
    text: str,
    *,
    family: str,
    add_generation_prompt: bool = True,
    system_prompt: str = "",
) -> str:
    ensure_known_chat_template(tokenizer, family)
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": str(text)})
    return str(
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            **chat_template_generation_kwargs(tokenizer),
        )
    )


def _resolve_dtype(dtype: str | torch.dtype | None) -> str | torch.dtype | None:
    if isinstance(dtype, torch.dtype):
        return dtype
    if dtype in (None, "", "none"):
        return None
    if dtype == "auto":
        return "auto"
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if dtype not in mapping:
        raise ValueError(f"unsupported torch dtype: {dtype}")
    return mapping[dtype]


def _config_value(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _llama_config_from_internlm3_config(config: Any) -> LlamaConfig:
    rope_parameters = _config_value(config, "rope_parameters")
    rope_scaling = _config_value(config, "rope_scaling")
    rope_theta = float(_config_value(config, "rope_theta", 10000.0))
    normalized_rope_scaling = dict(rope_scaling or {})
    rope_type = normalized_rope_scaling.get("rope_type") or normalized_rope_scaling.get(
        "type"
    )
    if rope_type is not None:
        normalized_rope_scaling["rope_type"] = rope_type
        normalized_rope_scaling["type"] = rope_type
    normalized_rope_parameters = dict(rope_parameters or normalized_rope_scaling)
    if rope_type is not None:
        normalized_rope_parameters["rope_type"] = rope_type
    normalized_rope_parameters["rope_theta"] = rope_theta

    config_kwargs: dict[str, Any] = {
        "vocab_size": int(_config_value(config, "vocab_size")),
        "hidden_size": int(_config_value(config, "hidden_size")),
        "intermediate_size": int(_config_value(config, "intermediate_size")),
        "num_hidden_layers": int(_config_value(config, "num_hidden_layers")),
        "num_attention_heads": int(_config_value(config, "num_attention_heads")),
        "num_key_value_heads": int(_config_value(config, "num_key_value_heads")),
        "hidden_act": _config_value(config, "hidden_act"),
        "max_position_embeddings": int(
            _config_value(config, "max_position_embeddings")
        ),
        "initializer_range": float(_config_value(config, "initializer_range", 0.02)),
        "rms_norm_eps": float(_config_value(config, "rms_norm_eps", 1e-6)),
        "use_cache": bool(_config_value(config, "use_cache", True)),
        "pad_token_id": _config_value(config, "pad_token_id"),
        "bos_token_id": _config_value(config, "bos_token_id"),
        "eos_token_id": _config_value(config, "eos_token_id"),
        "tie_word_embeddings": False,
        "attention_bias": False,
        "mlp_bias": False,
        "attention_dropout": float(_config_value(config, "attention_dropout", 0.0)),
        "head_dim": int(
            _config_value(
                config,
                "head_dim",
                int(_config_value(config, "hidden_size"))
                // int(_config_value(config, "num_attention_heads")),
            )
        ),
        "architectures": ["InternLM3ForCausalLM"],
    }
    llama_params = inspect.signature(LlamaConfig.__init__).parameters
    if "rope_parameters" in llama_params:
        if normalized_rope_parameters:
            config_kwargs["rope_parameters"] = normalized_rope_parameters
    else:
        config_kwargs["rope_theta"] = rope_theta
    if normalized_rope_scaling and "rope_scaling" in llama_params:
        config_kwargs["rope_scaling"] = normalized_rope_scaling
    return LlamaConfig(**config_kwargs)


def _load_internlm3_as_llama(
    model_reference: str,
    *,
    local_files_only: bool,
    torch_dtype: str | torch.dtype | None,
    device_map: str | None,
) -> tuple[Any, dict[str, Any]]:
    source_config, _ = PretrainedConfig.get_config_dict(
        model_reference,
        local_files_only=local_files_only,
    )
    model_type = str(source_config.get("model_type", "")).lower()
    if model_type != "internlm3":
        raise ValueError(f"expected InternLM3 config, got model_type={model_type!r}")
    llama_config = _llama_config_from_internlm3_config(source_config)
    kwargs: dict[str, Any] = {
        "config": llama_config,
        "local_files_only": local_files_only,
        "torch_dtype": torch_dtype,
        "output_loading_info": True,
        "low_cpu_mem_usage": True,
    }
    if device_map not in (None, "", "none"):
        kwargs["device_map"] = device_map
    model, loading_info = LlamaForCausalLM.from_pretrained(model_reference, **kwargs)
    missing = sorted(loading_info.get("missing_keys") or [])
    unexpected = sorted(loading_info.get("unexpected_keys") or [])
    mismatched = loading_info.get("mismatched_keys") or []
    if missing or unexpected or mismatched:
        raise RuntimeError(
            "InternLM3 native Llama loader did not match weights exactly: "
            f"missing={missing[:20]} unexpected={unexpected[:20]} "
            f"mismatched={mismatched[:20]}"
        )
    return model, {
        "internlm3_native_llama_loader": True,
        "internlm3_source_model_type": model_type,
    }


def load_model_and_tokenizer(
    model_reference: str,
    *,
    aliases: dict[str, str] | None = None,
    family: str | None = None,
    tokenizer_reference: str | None = None,
    device_map: str | None = "auto",
    torch_dtype: str | torch.dtype | None = "bfloat16",
    trust_remote_code: bool = False,
    local_files_only: bool = False,
    use_fast: bool = False,
) -> tuple[Any, Any, dict[str, Any]]:
    resolved_model, is_local_model = resolve_model_reference(model_reference, aliases)
    resolved_tokenizer, is_local_tokenizer = resolve_model_reference(
        tokenizer_reference or model_reference,
        aliases,
    )
    resolved_family = infer_model_family(
        resolved_model,
        explicit_family=family,
    )
    use_internlm_loader = (
        resolved_family == "internlm"
        and str(_read_local_config(resolved_model).get("model_type", "")).lower()
        == "internlm3"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        resolved_tokenizer,
        use_fast=use_fast,
        trust_remote_code=trust_remote_code or use_internlm_loader,
        local_files_only=local_files_only or is_local_tokenizer,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        if (
            resolved_family == "llama"
            and "llama-2" in resolved_model.lower()
            and tokenizer.unk_token
        ):
            tokenizer.pad_token = tokenizer.unk_token
        elif tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            raise ValueError(
                f"tokenizer for {resolved_model} has neither pad_token nor eos_token"
            )
    injected_template = ensure_known_chat_template(tokenizer, resolved_family)

    resolved_dtype = _resolve_dtype(torch_dtype)
    loader_meta: dict[str, Any] = {}
    if use_internlm_loader:
        model, loader_meta = _load_internlm3_as_llama(
            resolved_model,
            local_files_only=local_files_only or is_local_model,
            torch_dtype=resolved_dtype,
            device_map=device_map,
        )
    else:
        kwargs: dict[str, Any] = {
            "torch_dtype": resolved_dtype,
            "low_cpu_mem_usage": True,
            "trust_remote_code": trust_remote_code,
            "local_files_only": local_files_only or is_local_model,
        }
        if device_map not in (None, "", "none"):
            kwargs["device_map"] = device_map
        model = AutoModelForCausalLM.from_pretrained(resolved_model, **kwargs)
    model.requires_grad_(False)
    model.eval()
    return (
        model,
        tokenizer,
        {
            "requested_model_reference": model_reference,
            "resolved_model_reference": resolved_model,
            "resolved_tokenizer_reference": resolved_tokenizer,
            "family": resolved_family,
            "is_local_model": is_local_model,
            "is_local_tokenizer": is_local_tokenizer,
            "trust_remote_code": bool(trust_remote_code),
            "injected_chat_template": injected_template,
            **loader_meta,
        },
    )


def model_input_device(model: Any) -> torch.device:
    embeddings = model.get_input_embeddings()
    if embeddings is None:
        raise ValueError(f"model {type(model)} has no input embeddings")
    try:
        return next(embeddings.parameters()).device
    except StopIteration:
        return getattr(model, "device", torch.device("cpu"))


def model_context_limit(model: Any, tokenizer: Any) -> int | None:
    candidates: list[int] = []
    for value in (
        getattr(getattr(model, "config", None), "max_position_embeddings", None),
        getattr(tokenizer, "model_max_length", None),
    ):
        try:
            parsed = int(value)
        except (TypeError, ValueError, OverflowError):
            continue
        if 0 < parsed < 10_000_000:
            candidates.append(parsed)
    return min(candidates) if candidates else None
