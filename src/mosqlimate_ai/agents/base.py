"""Base agent class for Karl DBot integration."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import logging

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class AgentConfig(BaseModel):
    """Configuration for agents."""
    name: str
    description: str
    model: str = "gpt-4"
    temperature: float = 0.3
    max_tokens: int = 4000


class BaseAgent(ABC):
    """Base class for all Karl DBot agents.
    
    This class provides the foundation for specialized agents
    in the dengue forecasting pipeline.
    
    Attributes:
        config: Agent configuration
        memory: Agent's working memory
        tools: Available tools for the agent
    """
    
    def __init__(self, config: AgentConfig):
        """Initialize base agent.
        
        Args:
            config: Agent configuration
        """
        self.config = config
        self.memory: Dict[str, Any] = {}
        self.tools: Dict[str, Any] = {}
        logger.info(f"Initialized {config.name} agent")
    
    @abstractmethod
    def run(self, task: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute agent's main task.
        
        Args:
            task: Task description or instructions
            context: Additional context for the task
            
        Returns:
            Dictionary with results and metadata
        """
        pass
    
    def add_to_memory(self, key: str, value: Any) -> None:
        """Store information in agent memory.
        
        Args:
            key: Memory key
            value: Memory value
        """
        self.memory[key] = value
        logger.debug(f"{self.config.name}: Stored {key} in memory")
    
    def get_from_memory(self, key: str) -> Any:
        """Retrieve information from agent memory.
        
        Args:
            key: Memory key
            
        Returns:
            Stored value or None
        """
        return self.memory.get(key)
    
    def register_tool(self, name: str, tool: Any) -> None:
        """Register a tool for the agent to use.
        
        Args:
            name: Tool name
            tool: Tool function or class
        """
        self.tools[name] = tool
        logger.debug(f"{self.config.name}: Registered tool {name}")
    
    def use_tool(self, name: str, **kwargs) -> Any:
        """Execute a registered tool.
        
        Args:
            name: Tool name
            **kwargs: Tool arguments
            
        Returns:
            Tool output
            
        Raises:
            ValueError: If tool not found
        """
        if name not in self.tools:
            raise ValueError(f"Tool {name} not registered")
        
        tool = self.tools[name]
        return tool(**kwargs)
    
    def communicate(self, message: str, to_agent: Optional[str] = None) -> Dict[str, Any]:
        """Send message to another agent or orchestrator.
        
        Args:
            message: Message content
            to_agent: Target agent name (None for orchestrator)
            
        Returns:
            Response dictionary
        """
        return {
            "from": self.config.name,
            "to": to_agent,
            "message": message,
            "status": "pending"
        }
