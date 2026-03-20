"""Agent communication framework for validation pipeline.

Provides message passing infrastructure and audit logging for multi-agent system.
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """Types of agent messages."""

    COMMAND = "command"
    RESULT = "result"
    SUGGESTION = "suggestion"
    ALERT = "alert"
    QUERY = "query"
    RESPONSE = "response"
    DECISION = "decision"


class MessagePriority(Enum):
    """Priority levels for messages."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AgentMessage:
    """Message structure for agent communication.

    Attributes:
        timestamp: When the message was created
        sender: Name of the sending agent
        receiver: Name of the receiving agent (None for broadcast)
        message_type: Type of message
        priority: Message priority level
        content: Message payload
        validation_test: Which validation test this relates to (1, 2, 3, or None)
        state: Which state UF this relates to (or "all")
        session_id: Unique session identifier
        message_id: Unique message identifier
    """

    sender: str
    receiver: Optional[str]
    message_type: MessageType
    content: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    priority: MessagePriority = MessagePriority.NORMAL
    validation_test: Optional[int] = None
    state: Optional[str] = None
    session_id: str = "default"
    message_id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S_%f"))

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        return {
            "message_id": self.message_id,
            "timestamp": self.timestamp.isoformat(),
            "sender": self.sender,
            "receiver": self.receiver,
            "message_type": self.message_type.value,
            "priority": self.priority.value,
            "content": self.content,
            "validation_test": self.validation_test,
            "state": self.state,
            "session_id": self.session_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentMessage":
        """Create message from dictionary."""
        return cls(
            sender=data["sender"],
            receiver=data.get("receiver"),
            message_type=MessageType(data["message_type"]),
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            priority=MessagePriority(data.get("priority", "normal")),
            validation_test=data.get("validation_test"),
            state=data.get("state"),
            session_id=data.get("session_id", "default"),
            message_id=data.get("message_id", ""),
        )


class AgentCommunicationBus:
    """Central message bus for agent communication.

    Handles message routing between agents and maintains audit logs.
    """

    def __init__(self, log_dir: Optional[Path] = None, session_id: Optional[str] = None):
        """Initialize communication bus.

        Args:
            log_dir: Directory for audit logs
            session_id: Unique session identifier
        """
        self.log_dir = Path(log_dir) if log_dir else Path("logs/agent_communications")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.messages: List[AgentMessage] = []
        self.subscribers: Dict[str, List[str]] = {}  # agent_name -> list of message types

        # Setup logging
        self.log_file = self.log_dir / f"communications_{self.session_id}.jsonl"
        logger.info(f"AgentCommunicationBus initialized with session {self.session_id}")

    def send_message(self, message: AgentMessage) -> None:
        """Send a message and log it.

        Args:
            message: Message to send
        """
        # Add session ID if not set
        if not message.session_id or message.session_id == "default":
            message.session_id = self.session_id

        self.messages.append(message)

        # Write to audit log
        self._write_to_log(message)

        # Log at appropriate level
        log_msg = (
            f"[{message.sender} -> {message.receiver or 'BROADCAST'}] {message.message_type.value}"
        )
        if message.state:
            log_msg += f" [State: {message.state}]"
        if message.validation_test:
            log_msg += f" [Test: {message.validation_test}]"

        if message.priority == MessagePriority.CRITICAL:
            logger.critical(log_msg)
        elif message.priority == MessagePriority.HIGH:
            logger.warning(log_msg)
        else:
            logger.info(log_msg)

    def _write_to_log(self, message: AgentMessage) -> None:
        """Write message to audit log file."""
        try:
            with open(self.log_file, "a") as f:
                f.write(json.dumps(message.to_dict()) + "\n")
        except Exception as e:
            logger.error(f"Failed to write message to log: {e}")

    def subscribe(self, agent_name: str, message_types: List[MessageType]) -> None:
        """Subscribe an agent to specific message types.

        Args:
            agent_name: Name of the agent
            message_types: List of message types to subscribe to
        """
        if agent_name not in self.subscribers:
            self.subscribers[agent_name] = []
        self.subscribers[agent_name].extend([mt.value for mt in message_types])
        logger.debug(f"{agent_name} subscribed to {message_types}")

    def get_messages_for_agent(
        self,
        agent_name: str,
        message_types: Optional[List[MessageType]] = None,
        validation_test: Optional[int] = None,
        state: Optional[str] = None,
        since: Optional[datetime] = None,
    ) -> List[AgentMessage]:
        """Get messages relevant to a specific agent.

        Args:
            agent_name: Name of the agent
            message_types: Filter by message types
            validation_test: Filter by validation test number
            state: Filter by state
            since: Only messages after this time

        Returns:
            List of matching messages
        """
        filtered = self.messages

        # Filter by receiver or broadcast
        filtered = [
            m
            for m in filtered
            if m.receiver == agent_name or m.receiver is None or m.sender == agent_name
        ]

        if message_types:
            type_values = [mt.value for mt in message_types]
            filtered = [m for m in filtered if m.message_type.value in type_values]

        if validation_test is not None:
            filtered = [m for m in filtered if m.validation_test == validation_test]

        if state is not None:
            filtered = [m for m in filtered if m.state == state or m.state is None]

        if since is not None:
            filtered = [m for m in filtered if m.timestamp >= since]

        return filtered

    def get_conversation_history(
        self,
        agent1: Optional[str] = None,
        agent2: Optional[str] = None,
        validation_test: Optional[int] = None,
        state: Optional[str] = None,
    ) -> List[AgentMessage]:
        """Get conversation history between agents.

        Args:
            agent1: First agent name
            agent2: Second agent name (None for all conversations with agent1)
            validation_test: Filter by validation test
            state: Filter by state

        Returns:
            List of messages in chronological order
        """
        filtered = self.messages

        if agent1 and agent2:
            # Conversation between two specific agents
            filtered = [
                m
                for m in filtered
                if (m.sender == agent1 and m.receiver == agent2)
                or (m.sender == agent2 and m.receiver == agent1)
            ]
        elif agent1:
            # All messages involving agent1
            filtered = [m for m in filtered if m.sender == agent1 or m.receiver == agent1]

        if validation_test is not None:
            filtered = [m for m in filtered if m.validation_test == validation_test]

        if state is not None:
            filtered = [m for m in filtered if m.state == state or m.state is None]

        return sorted(filtered, key=lambda m: m.timestamp)

    def export_audit_log(self, output_path: Path, format: str = "jsonl") -> None:
        """Export complete audit trail.

        Args:
            output_path: Where to save the audit log
            format: Export format ("jsonl" or "markdown")
        """
        if format == "jsonl":
            with open(output_path, "w") as f:
                for message in self.messages:
                    f.write(json.dumps(message.to_dict()) + "\n")
        elif format == "markdown":
            self._export_markdown(output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Audit log exported to {output_path}")

    def _export_markdown(self, output_path: Path) -> None:
        """Export audit log as human-readable markdown."""
        lines = [
            "# Agent Communication Audit Log",
            "",
            f"**Session ID:** {self.session_id}",
            f"**Generated:** {datetime.now().isoformat()}",
            f"**Total Messages:** {len(self.messages)}",
            "",
            "---",
            "",
        ]

        # Group by validation test
        for test_num in [1, 2, 3, None]:
            test_messages = [m for m in self.messages if m.validation_test == test_num]
            if not test_messages:
                continue

            if test_num is None:
                lines.extend(["## Final Forecast", ""])
            else:
                lines.extend([f"## Validation Test {test_num}", ""])

            # Group by state
            states = set(m.state for m in test_messages if m.state)
            for state in sorted(states):
                state_messages = [m for m in test_messages if m.state == state]
                if not state_messages:
                    continue

                lines.extend([f"### State: {state}", ""])

                for msg in sorted(state_messages, key=lambda m: m.timestamp):
                    lines.extend(
                        [
                            f"**{msg.timestamp.strftime('%H:%M:%S')}** - "
                            f"{msg.sender} → {msg.receiver or 'All'}",
                            f"",
                            f"**Type:** {msg.message_type.value}",
                            f"",
                            f"**Content:**",
                            f"```json",
                            f"{json.dumps(msg.content, indent=2)}",
                            f"```",
                            f"",
                            "---",
                            "",
                        ]
                    )

        with open(output_path, "w") as f:
            f.write("\n".join(lines))

    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of current session."""
        return {
            "session_id": self.session_id,
            "total_messages": len(self.messages),
            "agents": list(set(m.sender for m in self.messages)),
            "validation_tests": list(
                set(m.validation_test for m in self.messages if m.validation_test)
            ),
            "states": list(set(m.state for m in self.messages if m.state)),
            "log_file": str(self.log_file),
        }


class MemoryManager:
    """Manages shared memory between agents."""

    def __init__(self):
        """Initialize memory manager."""
        self.global_memory: Dict[str, Any] = {}
        self.agent_memories: Dict[str, Dict[str, Any]] = {}

    def store_global(self, key: str, value: Any) -> None:
        """Store value in global memory."""
        self.global_memory[key] = value
        logger.debug(f"Stored {key} in global memory")

    def get_global(self, key: str) -> Any:
        """Retrieve value from global memory."""
        return self.global_memory.get(key)

    def store_agent_memory(self, agent_name: str, key: str, value: Any) -> None:
        """Store value in agent-specific memory."""
        if agent_name not in self.agent_memories:
            self.agent_memories[agent_name] = {}
        self.agent_memories[agent_name][key] = value
        logger.debug(f"Stored {key} in {agent_name} memory")

    def get_agent_memory(self, agent_name: str, key: str) -> Any:
        """Retrieve value from agent-specific memory."""
        return self.agent_memories.get(agent_name, {}).get(key)

    def get_all_agent_memories(self, agent_name: str) -> Dict[str, Any]:
        """Get all memories for an agent."""
        return self.agent_memories.get(agent_name, {}).copy()
