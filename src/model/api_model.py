"""Client wrapper for OpenAI-compatible inference APIs."""

from openai import OpenAI

from src.shared.load_config import config


class APIModel:
    """Client for interacting with an OpenAI-compatible inference API.

    This class provides a simple interface for sending chat completion
    requests to an API endpoint compatible with the OpenAI API.

    Attributes:
        client (OpenAI): OpenAI client instance configured with the API endpoint.

    """

    def __init__(
        self,
        base_url: str = config["api_generator"]["base_url"],
        api_key: str = config["api_generator"]["api_key"],
    ) -> None:
        """Initialize the API model client.

        Args:
            base_url (str): Base URL of the OpenAI-compatible API endpoint.
            api_key (str): API key used for authentication.

        """
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def generate(
        self,
        messages: list[dict[str, str]],
        model: str = config["api_generator"]["name"],
        response_format: dict | None = None,
    ) -> str:
        """Generate a text response from the model.

        Sends a chat completion request to the configured inference API
        and returns the generated text.

        Args:
            messages (list[dict[str, str]]): List of message dictionaries containing conversation
                history. Each dict should have 'role' and 'content' keys,
                basically OpenAI Chat Completions messages schema.
            model (str): Name of the model to use for generation. Defaults to
                value from config.
            response_format (dict | None): Optional format specification for the response
                (e.g., JSON schema). Defaults to None.

        Returns:
            str: Generated text content from the model.

        Raises:
            ConnectionError: If the API server cannot be reached.
            ValueError: If the request format is invalid or the model is unavailable.

        Example:
            >>> messages = [
            ...     {"role": "user", "content": "What is machine learning?"}
            ... ]
            >>> response = model.generate(messages)
            >>> print(response)

        """
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            response_format=response_format,
            max_tokens=config["api_generator"]["max_tokens"],
        )
        answer = response.choices[0].message.content
        return answer
