"""Wrapper around modelz.ai API."""
import logging
from typing import Any, Dict, List, Mapping, Optional

from pydantic import Extra, Field, root_validator

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens
from langchain.utils import get_from_dict_or_env

logger = logging.getLogger(__name__)


class Modelz(LLM):
    """Wrapper around modelz.ai models.

    To use, you should have the ``modelz-py`` python package installed,
    and the environment variable ``MODELZ_API_KEY`` set with your API key.
    You can find your key here: https://cloud.modelz.ai/settings

    Any parameters that are valid to be passed to the call can be passed
    in, even if not explicitly saved on this class.

    Example:
        .. code-block:: python
            from langchain.llms import Modelz
            modelz = Modelz(deployment="stable-diffusion-5mncvma3xs9zpe2a",
                            input={"image_dimensions": "512x512"})
    """

    deployment: str
    input: Dict[str, Any] = Field(default_factory=dict)
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    modelz_api_key: Optional[str] = None

    class Config:
        """Configuration for this pydantic config."""

        extra = Extra.forbid

    @root_validator(pre=True)
    def build_extra(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Build extra kwargs from additional params that were passed in."""
        all_required_field_names = {field.alias for field in cls.__fields__.values()}

        extra = values.get("model_kwargs", {})
        for field_name in list(values):
            if field_name not in all_required_field_names:
                if field_name in extra:
                    raise ValueError(f"Found {field_name} supplied twice.")
                logger.warning(
                    f"""{field_name} was transfered to model_kwargs.
                    Please confirm that {field_name} is what you intended."""
                )
                extra[field_name] = values.pop(field_name)
        values["model_kwargs"] = extra
        return values

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key exists in environment."""
        modelz_api_key = get_from_dict_or_env(
            values, "modelz_api_key", "MODELZ_API_KEY"
        )
        values["modelz_api_key"] = modelz_api_key
        return values

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            **{"deployment": self.deployment},
            **{"model_kwargs": self.model_kwargs},
        }

    @property
    def _llm_type(self) -> str:
        """Return type of model."""
        return "modelz"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        """Call to replicate endpoint."""
        try:
            import modelz
        except ImportError:
            raise ValueError(
                "Could not import modelz python package. "
                "Please install it with `pip install modelz-py`."
            )

        params = self.model_kwargs or {}
        model_inputs = prompt
        # if params is not {} and params is not None:
        #     model_inputs = {"prompt": prompt, **params}

        client = modelz.ModelzClient(key=self.modelz_api_key,
                                     deployment=self.deployment)
        try:
            response = client.inference(params=model_inputs)
            text = response.data[0]
        except ValueError:
            returned = response
            raise ValueError(
                f"\nUnexpected Response: {returned}"
            )
        except (KeyError, TypeError):
            returned = response
            raise ValueError(
                f"\nUnexpected Response: {returned}"
            )
        if stop is not None:
            # I believe this is required since the stop tokens
            # are not enforced by the model parameters
            text = enforce_stop_tokens(text, stop)
        return text
