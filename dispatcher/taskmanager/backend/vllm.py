import logging
import subprocess
import time
import requests
import sys
import os
from typing import Dict, Any, Optional, List

from openai import OpenAI
from openai.types.chat import ChatCompletion
from openai.types.completion import Completion

from .base import BackendManager
from .request import Request, Response

logger = logging.getLogger(__name__)

class VLLMServerManager:
    """Manages the lifecycle of a locally launched vLLM OpenAI API server."""

    def __init__(self, process: subprocess.Popen):
        self.process = process
        if process:
            logger.info(f"VLLM Server Manager initialized for process PID: {self.process.pid}")

    def terminate(self):
        """Attempts to terminate the managed vLLM server process."""
        if self.process and self.process.poll() is None:  # Check if process exists and is running
            logger.info(f"Terminating launched vLLM server (PID: {self.process.pid})...")
            try:
                # Send SIGTERM first for graceful shutdown
                self.process.terminate()
                try:
                    # Wait a bit for graceful shutdown
                    self.process.wait(timeout=10)
                    logger.info("vLLM server terminated gracefully.")
                except subprocess.TimeoutExpired:
                    logger.warning("vLLM server did not terminate gracefully after 10s, sending SIGKILL.")
                    self.process.kill()  # Force kill if terminate didn't work
                    logger.info("vLLM server killed.")
            except Exception as e:
                logger.error(f"Error terminating vLLM server process (PID: {self.process.pid}): {e}")
        self.process = None  # Indicate process is dealt with

    @staticmethod
    def launch_and_wait(
        model_name: str,
        host: str,
        port: int,
        tensor_parallel_size: int,
        gpu_memory_utilization: float,
        api_key: Optional[str],
        chat_template: Optional[str],
        max_model_len: Optional[int],
        startup_timeout: int,
        disable_log_requests: bool = True,
        disable_output: bool = False
    ) -> 'VLLMServerManager':
        """
        Launches the vLLM OpenAI API server and waits for it to become healthy.

        Args:
            model_name: Path or ID of the model.
            host: Host to bind the server to.
            port: Port to bind the server to.
            tensor_parallel_size: Tensor parallel degree.
            gpu_memory_utilization: GPU memory utilization fraction.
            api_key: Optional API key for the server.
            chat_template: Optional path to a chat template file.
            max_model_len: Optional max model length override.
            startup_timeout: Max seconds to wait for the server to pass health check.
            disable_log_requests: Whether to add --disable-log-requests flag.

        Returns:
            An instance of VLLMServerManager managing the launched process.

        Raises:
            RuntimeError: If the server fails to launch or become healthy within the timeout.
        """
        cmd = [
            sys.executable,
            "-m", "vllm.entrypoints.openai.api_server",
            "--model", model_name,
            "--host", host,
            "--port", str(port),
            "--tensor-parallel-size", str(tensor_parallel_size),
            "--gpu-memory-utilization", str(gpu_memory_utilization),
        ]
        
        if disable_log_requests:
            cmd.append("--disable-log-requests")
        if api_key:
            cmd.extend(["--api-key", api_key])
        if chat_template:
            if not os.path.exists(chat_template):
                logger.warning(f"Provided vLLM chat template file not found: {chat_template}")
            cmd.extend(["--chat-template", chat_template])
        if max_model_len:
            cmd.extend(["--max-model-len", str(max_model_len)])

        stdout, stderr = subprocess.DEVNULL, subprocess.DEVNULL
        if not disable_output:
            stdout, stderr = sys.stdout, sys.stderr


        logger.info(f"Launching vLLM server with command: {' '.join(cmd)}")
        process = None
        try:
            process = subprocess.Popen(cmd, stdout=stdout, stderr=stderr, text=True)

            logger.info(f"vLLM server process started (PID: {process.pid}). Waiting up to {startup_timeout} seconds for health check...")

            # Health Check Loop
            start_time = time.monotonic()
            health_url = f"http://{host}:{port}/health"
            server_ready = False
            while time.monotonic() - start_time < startup_timeout:
                poll_result = process.poll()
                if poll_result is not None:
                    stdout, stderr = process.communicate()
                    error_msg = (f"vLLM server process terminated unexpectedly during startup check. "
                                f"Exit code: {poll_result}\nStderr:\n{stderr}\nStdout:\n{stdout}")
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)

                try:
                    response = requests.get(health_url, timeout=5)
                    if response.status_code == 200:
                        logger.info(f"vLLM server health check passed after {time.monotonic() - start_time:.1f} seconds.")
                        server_ready = True
                        break
                    else:
                        logger.debug(f"vLLM health check failed with status {response.status_code}. Retrying...")
                except requests.exceptions.Timeout:
                    logger.debug(f"vLLM health check timed out. Retrying...")
                except requests.exceptions.ConnectionError:
                    logger.debug("vLLM health check connection error. Server likely not up yet. Retrying...")
                except Exception as e:
                    logger.warning(f"Unexpected error during vLLM health check: {e}. Retrying...")

                time.sleep(10)

            if not server_ready:
                error_msg = f"vLLM server did not become healthy within the timeout ({startup_timeout} seconds)."
                logger.error(error_msg)
                manager = VLLMServerManager(process)  # Create manager to terminate
                manager.terminate()
                raise RuntimeError(error_msg)

            return VLLMServerManager(process)

        except Exception as e:
            logger.exception(f"Failed to launch or monitor vLLM server process: {e}", exc_info=True)
            # Attempt cleanup if process started
            if process and process.poll() is None:
                try:
                    process.kill()  # Kill forcefully if launch failed badly
                except Exception as kill_e:
                    logger.error(f"Error killing failed vLLM process: {kill_e}")
            raise RuntimeError(f"Failed to launch vLLM server: {e}") from e


class VLLMBackendManager(BackendManager):
    """Backend manager for LLM API calls using vLLM's OpenAI-compatible API with built-in server management."""
    
    def __init__(self, 
                 model_name: str,
                 host: str = "localhost",
                 port: int = 8000,
                 api_key: Optional[str] = None,
                 launch_server: bool = False,
                 tensor_parallel_size: int = 1,
                 gpu_memory_utilization: float = 0.9,
                 chat_template: Optional[str] = None,
                 max_model_len: Optional[int] = None,
                 startup_timeout: int = 1500,
                 request_timeout: int = 300,
                 health_check_interval: int = 60,
                 disable_output: bool = False):
        """
        Initialize a VLLM backend manager.
        
        Args:
            model_name: Name or path of the model to use
            host: Host for the API server
            port: Port for the API server
            api_key: API key (optional for some vLLM deployments)
            launch_server: Whether to launch a vLLM server
            tensor_parallel_size: Tensor parallel size for launched server
            gpu_memory_utilization: GPU memory utilization for launched server
            chat_template: Path to chat template (if needed)
            max_model_len: Max model length override
            startup_timeout: Seconds to wait for server to start up
            request_timeout: Request timeout in seconds
            health_check_interval: How often to perform health checks (in seconds)
            disable_output: Redirect vllm output to /dev/null
        """
        self.model_name = model_name
        self.host = host
        self.port = port
        self.api_key = api_key
        self.request_timeout = request_timeout
        self.health_check_interval = health_check_interval
        self.api_url = f"http://{host}:{port}/v1"
        
        self.logger = logging.getLogger(__name__)
        self.last_health_check = 0
        self.last_health_status = False
        self.server_manager = None
        
        # Launch the vLLM server if requested
        if launch_server:
            try:
                self.server_manager = VLLMServerManager.launch_and_wait(
                    model_name=model_name,
                    host=host,
                    port=port,
                    tensor_parallel_size=tensor_parallel_size,
                    gpu_memory_utilization=gpu_memory_utilization,
                    api_key=api_key,
                    chat_template=chat_template,
                    max_model_len=max_model_len,
                    startup_timeout=startup_timeout,
                    disable_log_requests=True,
                    disable_output=disable_output,
                )
                self.logger.info(f"Launched vLLM server for model {model_name}")
            except Exception as e:
                self.logger.error(f"Failed to launch vLLM server: {e}")
                raise
        
        # Initialize OpenAI client
        try:
            self.client = OpenAI(
                base_url=self.api_url,
                api_key=self.api_key if self.api_key else "dummy_api_key",
                timeout=self.request_timeout,
                max_retries=0,
            )
            self.logger.info(f"Initialized OpenAI client for vLLM API at {self.api_url} with model {self.model_name}")
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenAI client: {e}")
            raise

    def process(self, request: Request) -> Response:
        """
        Process a request through the vLLM API.
        
        Args:
            request: The request to process
        
        Returns:
            Response object containing the API result or error information
        """
        try:
            content = request.content
            
            # Check if model is specified and if it matches our expected model
            if 'model' in content and content['model'] != self.model_name:
                raise ValueError(f"Request specifies model '{content['model']}' but this backend is configured for '{self.model_name}'")
            content['model'] = self.model_name

            # Determine the API endpoint to use based on content structure
            if 'messages' in content:
                # This is a chat completion request
                completion = self.client.chat.completions.create(**content)
                result = self._process_chat_completion(completion)
            elif 'prompt' in content:
                # This is a text completion request
                completion = self.client.completions.create(**content)
                result = self._process_text_completion(completion)
            else:
                raise ValueError("Request must contain either 'messages' for chat or 'prompt' for text completion")
            
            return Response(request=request, content=result, model_name=self.model_name)
            
        except Exception as e:
            self.logger.error(f"Error calling vLLM API: {str(e)}")
            return Response.from_error(request, e, model_name=self.model_name)
    
    def _process_chat_completion(self, completion: ChatCompletion) -> Dict[str, Any]:
        """Process a chat completion response from the API."""
        result = {
            'id': completion.id,
            'choices': [],
            'model': completion.model,
            'created': completion.created
        }
        
        # Process choices
        if completion.choices and len(completion.choices) > 0:
            for choice in completion.choices:
                choice_data = {
                    'index': choice.index,
                    'finish_reason': choice.finish_reason,
                }
                
                if choice.message:
                    choice_data['message'] = {
                        'role': choice.message.role,
                        'content': choice.message.content
                    }
                
                result['choices'].append(choice_data)
        
        # Add usage information if available
        if hasattr(completion, 'usage') and completion.usage:
            result['usage'] = {
                'prompt_tokens': completion.usage.prompt_tokens,
                'completion_tokens': completion.usage.completion_tokens,
                'total_tokens': completion.usage.total_tokens
            }
        
        return result
    
    def _process_text_completion(self, completion: Completion) -> Dict[str, Any]:
        """Process a text completion response from the API."""
        result = {
            'id': completion.id,
            'choices': [],
            'model': completion.model,
            'created': completion.created
        }
        
        # Process choices
        if completion.choices and len(completion.choices) > 0:
            for choice in completion.choices:
                choice_data = {
                    'text': choice.text,
                    'index': choice.index,
                    'finish_reason': choice.finish_reason,
                }
                result['choices'].append(choice_data)
        
        # Add usage information if available
        if hasattr(completion, 'usage') and completion.usage:
            result['usage'] = {
                'prompt_tokens': completion.usage.prompt_tokens,
                'completion_tokens': completion.usage.completion_tokens,
                'total_tokens': completion.usage.total_tokens
            }
        
        return result
    
    def is_healthy(self) -> bool:
        """
        Check if the backend is healthy by making a simple API call.
        
        We cache the result to avoid excessive health checks.
        
        Returns:
            True if the backend is healthy, False otherwise
        """
        current_time = time.time()
        
        # Return cached result if we checked recently
        if current_time - self.last_health_check < self.health_check_interval:
            return self.last_health_status
        
        try:
            # Try to access the health endpoint directly
            response = requests.get(f"http://{self.host}:{self.port}/health", timeout=5)
            is_healthy = response.status_code == 200
            
            # Update health status
            self.last_health_check = current_time
            self.last_health_status = is_healthy
            return is_healthy
            
        except Exception as e:
            self.logger.warning(f"Health check failed: {e}")
            
            # Update health status
            self.last_health_check = current_time
            self.last_health_status = False
            return False
    
    def __del__(self):
        """Ensure the server is terminated when the object is deleted."""
        self.close()
    
    def close(self):
        """Terminate the vLLM server if we launched it."""
        if self.server_manager:
            self.logger.info("Terminating vLLM server...")
            self.server_manager.terminate()
            self.server_manager = None
