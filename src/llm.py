from __future__ import annotations

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def parse_json_response(text: str) -> dict:
    text = (text or '').strip()
    if '<think>' in text:
        end = text.rfind('</think>')
        if end != -1:
            text = text[end + len('</think>'):].strip()
    if text.startswith('```'):
        lines = text.splitlines()
        if lines and lines[0].startswith('```'):
            lines = lines[1:]
        if lines and lines[-1].strip() == '```':
            lines = lines[:-1]
        text = '\n'.join(lines).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass
    return {'parse_error': True, 'raw_response': text}


class APIBackend:
    def __init__(self, model: str, api_key_env: str = 'OPENAI_API_KEY', base_url: str = '', max_new_tokens: int = 1500):
        from openai import OpenAI
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(f'{api_key_env} is not set')
        kwargs = {'api_key': api_key}
        if base_url:
            kwargs['base_url'] = base_url
        self.client = OpenAI(**kwargs)
        self.model = model
        self.max_new_tokens = max_new_tokens

    def run_one(self, messages: list[dict]) -> dict:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0,
            max_tokens=self.max_new_tokens,
            response_format={'type': 'json_object'},
        )
        return parse_json_response(response.choices[0].message.content or '')


class LocalQwenBackend:
    def __init__(self, model_path: str, enable_thinking: bool = False, max_new_tokens: int = 1500):
        self.model_path = Path(model_path)
        self.enable_thinking = enable_thinking
        self.max_new_tokens = max_new_tokens
        self._client = None
        self._mode = 'transformers'

        if self._vllm_available():
            from openai import OpenAI
            logger.info('Using local vLLM server at http://localhost:8000/v1')
            self._mode = 'vllm'
            self._client = OpenAI(api_key='EMPTY', base_url='http://localhost:8000/v1')
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        if not self.model_path.exists():
            raise FileNotFoundError(f'Local Qwen model not found: {self.model_path}')
        logger.info('Loading local Qwen model from %s', self.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path), trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            str(self.model_path),
            torch_dtype=torch.bfloat16,
            device_map='auto',
            trust_remote_code=True,
        )
        self.model.eval()

    def _vllm_available(self) -> bool:
        try:
            import urllib.request
            with urllib.request.urlopen('http://localhost:8000/v1/models', timeout=2) as response:
                return response.status == 200
        except Exception:
            return False

    def run_one(self, messages: list[dict]) -> dict:
        if self._mode == 'vllm':
            response = self._client.chat.completions.create(
                model='Qwen3-8B',
                messages=messages,
                temperature=0.01 if not self.enable_thinking else 0.6,
                max_tokens=self.max_new_tokens,
                extra_body={'chat_template_kwargs': {'enable_thinking': self.enable_thinking}},
            )
            return parse_json_response(response.choices[0].message.content or '')

        import torch
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )
        inputs = self.tokenizer([text], return_tensors='pt').to(self.model.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=0.01 if not self.enable_thinking else 0.6,
                do_sample=bool(self.enable_thinking),
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_ids = output_ids[0][inputs['input_ids'].shape[1]:]
        return parse_json_response(self.tokenizer.decode(new_ids, skip_special_tokens=True))


class NoLLMBackend:
    def run_one(self, messages: list[dict]) -> dict:
        return {'skipped': True}


def make_backend(cfg: dict):
    backend = cfg.get('backend', 'local')
    if backend == 'none':
        return NoLLMBackend()
    if backend == 'local':
        return LocalQwenBackend(
            model_path=cfg.get('local_model_path', ''),
            enable_thinking=bool(cfg.get('enable_thinking', False)),
            max_new_tokens=int(cfg.get('max_new_tokens', 1500)),
        )
    if backend == 'api':
        return APIBackend(
            model=cfg.get('api_model', 'gpt-4o-mini'),
            api_key_env=cfg.get('api_key_env', 'OPENAI_API_KEY'),
            base_url=cfg.get('api_base_url', ''),
            max_new_tokens=int(cfg.get('max_new_tokens', 1500)),
        )
    raise ValueError(f'Unknown LLM backend: {backend}')
