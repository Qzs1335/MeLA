from .openai import OpenAIClient
import logging

try:
    from zhipuai import ZhipuAI
except ImportError:
    ZhipuAI = "zhipuai"

logger = logging.getLogger(__name__)

class ZhipuAIClient(OpenAIClient):

    ClientClass = ZhipuAI

    def _chat_completion_api(self, messages: list[dict], temperature: float, n: int = 1):
        assert n == 1
        if isinstance(self.ClientClass, str):
            logger.fatal(f"Package `{self.ClientClass}` is required")
            exit(-1)
        response = self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=min(temperature, 1.0),
        )
        return response.choices