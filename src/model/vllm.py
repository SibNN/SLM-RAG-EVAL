"""VLLM class for text generation."""

from openai import OpenAI

from src.shared.load_config import config


class VllmModel:
    """Client for interacting with vLLM inference server.

    This class provides a convenient interface to communicate with a vLLM
    server using the OpenAI client format. It supports both standard text
    generation and thinking-enabled models also json guided generation.

    Attributes:
        client: OpenAI client instance configured to connect to vLLM server.

    """

    def __init__(
        self,
        host: str = config["generator"]["host"],
        port: str = config["generator"]["port"],
        api_key: str | None = None,
    ) -> None:
        """Initialize the vLLM model client.

        Constructs the server URL from host and port configuration and
        initializes the OpenAI client to connect to the vLLM server.

        Args:
            host (str): Hostname or IP address of the vLLM server. Defaults to value from config .
            port (str): Port number of the vLLM server. Defaults to value from config.
            api_key (str | None): optional api key

        Example:
            >>> model = VllmModel()
            >>> model = VllmModel(host="localhost", port="8000")

        """
        api_key = api_key if api_key else "EMPTY"
        url = f"http://{host}:{port}/v1"
        self.client = OpenAI(base_url=url, api_key=api_key)

    def generate(
        self,
        messages: list[dict[str, str]],
        model: str = config["generator"]["name"],
        response_format: dict | None = None,
    ) -> str:
        """Generate text completion using the vLLM server.

        Sends a chat completion request to the vLLM server and returns
        the generated text response.

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
            ConnectionError: If unable to connect to the vLLM server.
            ValueError: If messages format is invalid or model not found.

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
            temperature=config["generator"]["temperature"],
            extra_body={
                "chat_template_kwargs": {
                    "enable_thinking": config["generator"]["thinking"],
                },
            },
            response_format=response_format,
            max_tokens=config["generator"]["max_tokens"],
        )
        answer = response.choices[0].message.content
        return answer
