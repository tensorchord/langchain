"""Wrapper around modelz.ai API."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Mapping

from pydantic import BaseModel, root_validator

from langchain.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain.chat_models.base import BaseChatModel
from langchain.schema import (
    AIMessage,
    BaseMessage,
    ChatGeneration,
    ChatMessage,
    ChatResult,
    HumanMessage,
    SystemMessage,
)
from langchain.utils import get_from_dict_or_env

if TYPE_CHECKING:
    import modelz


class ChatModelzError(Exception):
    pass


def _response_to_result(
    response: modelz.client.ModelzResponse,
    stop: Optional[List[str]],
) -> ChatResult:
    """Converts a response into a LangChain ChatResult."""

    generations: List[ChatGeneration] = []

    generations.append(
        ChatGeneration(
            text=response.data[0],
            message=ChatMessage(role="ai", content=response.data[0]),
        )
    )

    return ChatResult(generations=generations)


def _messages_to_prompt_dict(
    input_messages: List[BaseMessage],
) -> str:
    """Converts a list of LangChain messages into a general format."""

    context: str = ""
    examples: List[str] = []
    messages: List[str] = []

    remaining = list(enumerate(input_messages))

    while remaining:
        index, input_message = remaining.pop(0)

        if isinstance(input_message, SystemMessage):
            if index != 0:
                raise ChatModelzError("System message must be first input message.")
            context = input_message.content
        elif isinstance(input_message, HumanMessage) and input_message.example:
            if messages:
                raise ChatModelzError(
                    "Message examples must come before other messages."
                )
            _, next_input_message = remaining.pop(0)
            if isinstance(next_input_message, AIMessage) and next_input_message.example:
                examples.extend(
                    [
                        "me: " + input_message.content,
                        "ai: " + next_input_message.content,
                    ]
                )
            else:
                raise ChatModelzError(
                    "Human example message must be immediately followed by an "
                    " AI example response."
                )
        elif isinstance(input_message, AIMessage) and input_message.example:
            raise ChatModelzError(
                "AI example message must be immediately preceded by a Human "
                "example message."
            )
        elif isinstance(input_message, AIMessage):
            messages.append(
                "ai: " + input_message.content.replace("\n", " ").strip()
            )
        elif isinstance(input_message, HumanMessage):
            messages.append(
                "me: " + input_message.content.replace("\n", " ").strip()
            )
        elif isinstance(input_message, ChatMessage):
            messages.append(
                "other: " + input_message.content.replace("\n", " ").strip()
            )
        else:
            raise ChatModelzError(
                "Messages without an explicit role not supported by PaLM API."
            )

    return ''.join(messages)


class ChatModelz(BaseChatModel, BaseModel):
    """Wrapper around modelz.ai API.

    To use you must have the modelz-py Python package installed and
    either:

        1. The `MODELZ_API_KEY` environment variable set with your API key, or
        2. Pass your API key using the modelz_api_key kwarg to the ChatModelz
           constructor.

    Example:
        .. code-block:: python

            from langchain.chat_models import ChatModelz
            chat = ChatModelz()

    """

    client: Any  #: :meta private:
    deployment: str
    """Model name to use."""
    modelz_api_key: Optional[str] = None
    timeout: Optional[float] = None
    n: int = 1
    """Number of chat completions to generate for each prompt. Note that the API may
       not return the full n completions if duplicates are generated."""

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate api key exists."""
        modelz_api_key = get_from_dict_or_env(
            values, "modelz_api_key", "MODELZ_API_KEY"
        )

        return values

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            **{"deployment": self.deployment},
            **{"timeout": self.timeout},
        }

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> ChatResult:
        try:
            import modelz
        except ImportError:
            raise ValueError(
                "Could not import modelz python package. "
                "Please install it with `pip install modelz-py`."
            )

        prompt = _messages_to_prompt_dict(messages)

        if self.timeout is None:
            self.timeout = 120

        client = modelz.ModelzClient(key=self.modelz_api_key,
                                     deployment=self.deployment, timeout=self.timeout)
        response = client.inference(params=prompt)

        return _response_to_result(response, stop)

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
    ) -> ChatResult:
        pass
