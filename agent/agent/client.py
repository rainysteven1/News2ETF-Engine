"""GLM-4-Flash LLM client wrapper — delegates to src.utils.llm_client factory."""

from __future__ import annotations

import json

from openai import OpenAI

from agent.utils.llm_client import get_llm_client


class LLMClient:
    """Thin wrapper around the LLM client factory for structured chat completions."""

    def __init__(self, model: str = "glm-4-flash", temperature: float = 0.0):
        self.model = model
        self.temperature = temperature
        self._client: OpenAI | None = None

    def _get_client(self) -> OpenAI:
        if self._client is None:
            self._client = get_llm_client(self.model)
        return self._client

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        """Send a chat completion request and return the text content."""
        client = self._get_client()
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature,
        )
        return response.choices[0].message.content

    def chat_json(self, system_prompt: str, user_prompt: str) -> dict:
        """Send a chat request and parse JSON from the response."""
        text = self.chat(system_prompt, user_prompt).strip()
        # Handle markdown code blocks
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1])
        return json.loads(text)
