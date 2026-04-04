from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from app.config import settings


@dataclass(slots=True)
class GenerationResult:
    answer: str
    model_name: str
    device: str
    thinking: str | None = None


def generate_answer(
    question: str,
    model_obj: AutoModelForCausalLM,
    tok: AutoTokenizer,
    system_prompt: str,
    thinking: bool = False,
    max_new_tokens: int = 512,
) -> dict[str, str]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    if hasattr(tok, "apply_chat_template"):
        text = tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=thinking,
        )
    else:
        text = f"[System]\n{system_prompt}\n\n[User]\n{question}"

    inputs = tok([text], return_tensors="pt")
    inputs = {key: value.to(model_obj.device) for key, value in inputs.items()}

    gen_kwargs: dict[str, object] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "temperature": 0.6 if thinking else 0.7,
        "top_p": 0.95 if thinking else 0.8,
        "pad_token_id": tok.eos_token_id,
    }
    # Keep top_k only when sampling is enabled.
    if gen_kwargs["do_sample"]:
        gen_kwargs["top_k"] = 20

    with torch.no_grad():
        generated = model_obj.generate(**inputs, **gen_kwargs)

    output_ids = generated[0][len(inputs["input_ids"][0]):].tolist()

    if thinking:
        try:
            idx = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            idx = 0
        return {
            "thinking": tok.decode(output_ids[:idx], skip_special_tokens=True).strip(),
            "answer": tok.decode(output_ids[idx:], skip_special_tokens=True).strip(),
        }

    return {"answer": tok.decode(output_ids, skip_special_tokens=True).strip()}


class RagAnswerGenerator:
    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or settings.llm_model
        self.device = self._detect_device()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model_kwargs = self._build_model_kwargs()
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
        if self.device == "mps":
            self.model.to("mps")
        elif self.device == "cpu":
            self.model.to("cpu")
        self.model.eval()

    def _detect_device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _build_model_kwargs(self) -> dict[str, object]:
        if self.device == "cuda":
            return {"dtype": torch.float16, "device_map": "auto"}
        if self.device == "mps":
            return {"dtype": torch.float16}
        return {"dtype": torch.float32}

    def generate(self, system_prompt: str, user_prompt: str, thinking: bool = False) -> GenerationResult:
        output = generate_answer(
            question=user_prompt,
            model_obj=self.model,
            tok=self.tokenizer,
            system_prompt=system_prompt,
            thinking=thinking,
            max_new_tokens=settings.llm_max_new_tokens,
        )
        return GenerationResult(
            answer=output.get("answer", "").strip(),
            thinking=output.get("thinking"),
            model_name=self.model_name,
            device=self.device,
        )

    def _render_prompt(
        self,
        messages: list[dict[str, Any]],
        thinking: bool = False,
        tools: list[dict[str, Any]] | None = None,
        add_generation_prompt: bool = True,
    ) -> str:
        if hasattr(self.tokenizer, "apply_chat_template"):
            kwargs: dict[str, Any] = {
                "tokenize": False,
                "add_generation_prompt": add_generation_prompt,
            }
            if tools is not None:
                kwargs["tools"] = tools
            try:
                kwargs["enable_thinking"] = thinking
                return self.tokenizer.apply_chat_template(messages, **kwargs)
            except TypeError:
                kwargs.pop("enable_thinking", None)
                return self.tokenizer.apply_chat_template(messages, **kwargs)

        parts = [
            f"[{m.get('role', 'user')}]\n{m.get('content', '')}" for m in messages if m.get("content")
        ]
        return "\n\n".join(parts)

    def _build_inputs(self, text: str) -> dict[str, torch.Tensor]:
        inputs = self.tokenizer([text], return_tensors="pt")
        return {key: value.to(self.model.device) for key, value in inputs.items()}

    def generate_from_messages(
        self,
        messages: list[dict[str, Any]],
        thinking: bool = False,
        max_new_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        skip_special_tokens: bool = False,
    ) -> str:
        text = self._render_prompt(messages, thinking=thinking, tools=tools, add_generation_prompt=True)
        inputs = self._build_inputs(text)
        token_count = len(inputs["input_ids"][0])

        with torch.no_grad():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens or settings.llm_max_new_tokens,
                do_sample=True,
                temperature=0.6 if thinking else settings.llm_temperature,
                top_p=0.95 if thinking else settings.llm_top_p,
                top_k=20,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        output_ids = generated[0][token_count:].tolist()
        return self.tokenizer.decode(output_ids, skip_special_tokens=skip_special_tokens).strip()

    def stream_from_messages(
        self,
        messages: list[dict[str, Any]],
        thinking: bool = False,
        max_new_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        skip_special_tokens: bool = True,
    ) -> tuple[TextIteratorStreamer, threading.Thread]:
        text = self._render_prompt(messages, thinking=thinking, tools=tools, add_generation_prompt=True)
        inputs = self._build_inputs(text)

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=skip_special_tokens,
            timeout=120.0,
        )

        gen_kwargs: dict[str, object] = {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": max_new_tokens or settings.llm_max_new_tokens,
            "do_sample": True,
            "temperature": 0.6 if thinking else settings.llm_temperature,
            "top_p": 0.95 if thinking else settings.llm_top_p,
            "pad_token_id": self.tokenizer.eos_token_id,
            "top_k": 20,
        }

        thread = threading.Thread(target=self.model.generate, kwargs=gen_kwargs, daemon=True)
        thread.start()
        return streamer, thread

    def stream_generate(
        self,
        system_prompt: str,
        user_prompt: str,
        thinking: bool = False,
    ) -> tuple[TextIteratorStreamer, threading.Thread]:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return self.stream_from_messages(
            messages,
            thinking=thinking,
            max_new_tokens=settings.llm_max_new_tokens,
            tools=None,
            skip_special_tokens=True,
        )