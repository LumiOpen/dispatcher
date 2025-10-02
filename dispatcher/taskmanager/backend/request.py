"""
This file defines the Request and Response classes used for communication
between the TaskManager and a backend processing service.
"""
import copy
from typing import Any, Dict, Optional, Union, List

class Request:
    """
    Represents a request to be processed by a backend.
    
    Contains all necessary information for the backend to process the request.
    """
    def __init__(self, content: Dict[str, Any], context: Optional[Any] = None):
        """
        Initialize a request.
        
        Args:
            content: Dictionary containing all parameters for the backend request
            context: Optional context that will be passed through to the response
        """
        # Deep copy the content to ensure the original isn't modified elsewhere
        self.content = copy.deepcopy(content)
        self.context = context


class Response:
    """
    Represents a response from a backend.
    
    Contains the result of processing a request, along with any error information
    and the original request.
    """
    def __init__(self, 
                 request: Request,
                 content: Optional[Dict[str, Any]] = None, 
                 error: Optional[Exception] = None,
                 model_name: Optional[str] = None):
        """
        Initialize a response.
        
        Args:
            request: The original request that generated this response
            content: Dictionary containing the response data
            error: Exception if an error occurred during processing
            model_name: The name of the model that processed the request.
        """
        self.request = request
        self.content = content
        self.error = error
        self.model_name = model_name
    
    @property
    def is_success(self) -> bool:
        """Check if the response represents a successful processing."""
        return self.error is None and self.content is not None
    
    @classmethod
    def from_error(cls, request: Request, error: Exception, model_name: Optional[str] = None) -> 'Response':
        """Create a response representing an error."""
        return cls(request=request, content=None, error=error, model_name=model_name)

    def get_text(self, n: Optional[int] = None) -> Optional[Union[str, List[str]]]:
        """Extracts model response text from standard response formats. Extracts multiple texts if n is specified.

        Works for both *chat* and *text* completion payloads.  Returns *None*
        if extraction fails or ``self.content`` is not a dict.
        """
        if not isinstance(self.content, dict):
            return None
        try:
            # Chat completion schema
            if n is not None:
                return [choice["message"]["content"] for choice in self.content["choices"][:n]]
            else:
                return self.content["choices"][0]["message"]["content"]
        except Exception:
            try:
                # Text completion schema
                return self.content["choices"][0]["text"]
            except Exception:
                return None
