import copy
from typing import Any, Dict, Optional

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
                 error: Optional[Exception] = None):
        """
        Initialize a response.
        
        Args:
            request: The original request that generated this response
            content: Dictionary containing the response data
            error: Exception if an error occurred during processing
        """
        self.request = request
        self.content = content
        self.error = error
    
    @property
    def is_success(self) -> bool:
        """Check if the response represents a successful processing."""
        return self.error is None and self.content is not None
    
    @classmethod
    def from_error(cls, request: Request, error: Exception) -> 'Response':
        """Create a response representing an error."""
        return cls(request=request, content=None, error=error)

    def get_text(self) -> Optional[str]:
        """Extracts model response text from standard response formats.

        Works for both *chat* and *text* completion payloads.  Returns *None*
        if extraction fails or ``self.content`` is not a dict.
        """
        if not isinstance(self.content, dict):
            return None
        try:
            # Chat completion schema
            return self.content["choices"][0]["message"]["content"]
        except Exception:
            try:
                # Text completion schema
                return self.content["choices"][0]["text"]
            except Exception:
                return None
