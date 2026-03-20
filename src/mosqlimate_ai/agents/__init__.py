"""Multi-agent system using Karl DBot."""

from mosqlimate_ai.agents.base import AgentConfig, BaseAgent
from mosqlimate_ai.agents.communication import (
    AgentCommunicationBus,
    AgentMessage,
    MemoryManager,
    MessagePriority,
    MessageType,
)
from mosqlimate_ai.agents.data_agent import DataEngineerAgent
from mosqlimate_ai.agents.ensemble_agent import EnsembleAgent
from mosqlimate_ai.agents.forecast_agent import ForecastAgent
from mosqlimate_ai.agents.knowledge_base import (
    CrossStateKnowledgeBase,
    HyperparameterConfig,
    StateProfile,
    ValidationResult,
)
from mosqlimate_ai.agents.model_agent import ModelArchitectAgent
from mosqlimate_ai.agents.orchestrator import AgentOrchestrator, Task, Workflow
from mosqlimate_ai.agents.prompts import AGENT_PROMPTS, get_prompt, list_agents
from mosqlimate_ai.agents.selection_agent import TopNModelSelectionAgent
from mosqlimate_ai.agents.tuner_agent import EfficientHyperparameterTuner
from mosqlimate_ai.agents.validator_agent import ValidatorAgent

__all__ = [
    # Base
    "BaseAgent",
    "AgentConfig",
    # Communication
    "AgentMessage",
    "MessageType",
    "MessagePriority",
    "AgentCommunicationBus",
    "MemoryManager",
    # Knowledge
    "CrossStateKnowledgeBase",
    "StateProfile",
    "ValidationResult",
    "HyperparameterConfig",
    # Tuning & Selection
    "EfficientHyperparameterTuner",
    "TopNModelSelectionAgent",
    # Existing agents
    "DataEngineerAgent",
    "ModelArchitectAgent",
    "ForecastAgent",
    "ValidatorAgent",
    "EnsembleAgent",
    "AgentOrchestrator",
    "Task",
    "Workflow",
    "AGENT_PROMPTS",
    "get_prompt",
    "list_agents",
]
