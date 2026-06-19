"""Module for managing vllm server."""

import subprocess
import time
from http import HTTPStatus
from pathlib import Path

import requests

from src.model.vllm import VllmModel
from src.shared.load_config import config


class VLLMRunner:
    """Class for running vLLM server and correctly stopping it, even by Ctrl-C."""

    def __init__(self) -> None:
        """Init process."""
        self.process: subprocess.Popen | None = None

    def _wait_until_ready(self, host: str, port: int, timeout: float = 120.0) -> bool:
        """Wait till service is ready, Ask server its status, then can continue.

        Args:
            host (str): service ip adress
            port (int): service port
            timeout (float): maximum time to wait

        Returns:
            bool: True if successfully loaded, False othewise

        """
        start_time = time.time()
        url = f"http://{host}:{port}/health"

        while time.time() - start_time < timeout:
            try:
                r = requests.get(url, timeout=3)
                if r.status_code == HTTPStatus.OK:
                    print("vLLM is ready")
                    return True
            except requests.exceptions.RequestException:
                pass
            time.sleep(5)
        return False

    def start(self, config: dict) -> None:
        """Start the service, wait till it's ready to take user's request.

        Args:
            config (dict): Configuration, that contain information about service properties.
                Contains values:
                    - name (str): model name
                    - host (str): ip adress to host
                    - port (int): port number
                    - max_model_len (int): number of tokens, that model can take
                    - gpu_util (float): percent of gpu utilization - scales betweeen 0.0 and 1.0

        """
        name = config["name"]
        host = config.get("host", "localhost")
        gpu_util = config.get("gpu_util", 0.9)
        port = config.get("port", 8000)
        max_model_len = config.get("max_model_len", 8192)
        venv_path = Path(".venv").resolve()
        python_bin = venv_path / "bin" / "python"

        if not python_bin.exists():
            raise RuntimeError(f"Python binary not found at {python_bin}")

        cmd = [
            str(python_bin),
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            name,
            "--host",
            host,
            "--port",
            str(port),
            "--gpu-memory-utilization",
            str(gpu_util),
            "--max-model-len",
            str(max_model_len),
        ]

        print(f"Starting vLLM via venv: {' '.join(cmd)}")

        self.process = subprocess.Popen(  # noqa: S603
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        if not self._wait_until_ready(host, port):
            raise RuntimeError("vLLM did not start in time.")

    def stop(self) -> None:
        """Manually stop server."""
        if self.process is None:
            print("vLLM process not running.")
            return

        print("Stopping vLLM...")
        self.process.terminate()
        self.process.wait()
        self.process = None
        print("vLLM stopped.")

    def __del__(self) -> None:
        """Stop server when interrupted, or end of programm running."""
        self.stop()


if __name__ == "__main__":
    runner = VLLMRunner()
    runner.start(config["generator"])

    model = VllmModel()
    answer = model.generate(
        messages=[
            {"role": "system", "content": "You're helpfull assistant"},
            {
                "role": "assistant",
                "content": "Найденные документы: 1, 2, 3",
            },
            {"role": "user", "content": "Как дойти до собора?"},
        ],
    )
    print(answer)

    runner.stop()
