import requests
from typing import List, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatResult, ChatGeneration

class FastGroqLLM(BaseChatModel):
    """Fast Groq LLM using direct API calls - minimal dependencies."""

    api_key: str
    model: str = "llama-3.3-70b-versatile"
    temperature: float = 0
    base_url: str = "https://api.groq.com/openai/v1/chat/completions"

    @property
    def _llm_type(self) -> str:
        return "groq"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> ChatResult:
        # Convert LangChain messages to OpenAI format
        openai_messages = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                openai_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                openai_messages.append({"role": "assistant", "content": msg.content})
            elif isinstance(msg, SystemMessage):
                openai_messages.append({"role": "system", "content": msg.content})
            else:
                openai_messages.append({"role": "user", "content": str(msg.content)})

        response = requests.post(
            self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": self.model,
                "messages": openai_messages,
                "temperature": self.temperature,
                "stop": stop
            }
        )
        response.raise_for_status()

        content = response.json()["choices"][0]["message"]["content"]

        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=content))]
        )

    @property
    def _identifying_params(self):
        return {"model": self.model, "temperature": self.temperature}
