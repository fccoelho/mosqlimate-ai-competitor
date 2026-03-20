"""ValidationOrchestrator for managing the complete validation pipeline.

Coordinates StateValidationAgents for all states, managing bounded parallelism
and aggregating results across the 4-run validation pipeline.
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from mosqlimate_ai.agents.communication import AgentCommunicationBus
from mosqlimate_ai.agents.knowledge_base import CrossStateKnowledgeBase
from mosqlimate_ai.validation.config import (
    ValidationPipelineConfig,
    get_validation_config,
)

logger = logging.getLogger(__name__)


class ValidationOrchestrator:
    """Orchestrates the multi-agent validation pipeline for all states.

    This orchestrator:
    1. Manages StateValidationAgents for each state
    2. Enforces bounded parallelism (max 5 concurrent states)
    3. Monitors memory usage
    4. Aggregates results across all states
    5. Generates reports and audit logs

    Example:
        >>> orchestrator = ValidationOrchestrator()
        >>> results = orchestrator.run_full_pipeline(states=["SP", "RJ"])
    """

    def __init__(
        self,
        config: Optional[ValidationPipelineConfig] = None,
        output_dir: Path = Path("validation_results"),
    ):
        """Initialize validation orchestrator.

        Args:
            config: Validation pipeline configuration
            output_dir: Directory for validation outputs
        """
        self.config = config or get_validation_config()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize shared components
        self.message_bus = AgentCommunicationBus()
        self.knowledge_base = CrossStateKnowledgeBase()

        # State agents
        self.agents: Dict[str, Any] = {}

        # Results storage
        self.results: Dict[str, Dict[str, Any]] = {}

        # Concurrency control
        self.semaphore = asyncio.Semaphore(self.config.max_concurrent_states)

        logger.info(
            f"Initialized ValidationOrchestrator "
            f"(max_concurrent={self.config.max_concurrent_states}, "
            f"max_memory={self.config.max_memory_gb}GB)"
        )

    def run_full_pipeline(
        self,
        states: Optional[List[str]] = None,
        test_numbers: Optional[List[int]] = None,
        run_final: bool = True,
    ) -> Dict[str, Any]:
        """Run complete validation pipeline.

        Args:
            states: List of state UFs (None = all states)
            test_numbers: List of test numbers to run (None = all 3 tests)
            run_final: Whether to run final forecast

        Returns:
            Dictionary with all validation results
        """
        states = states or self.config.states
        test_numbers = test_numbers or [1, 2, 3]

        logger.info(f"Starting validation pipeline for {len(states)} states")
        logger.info(f"Tests: {test_numbers}, Final: {run_final}")

        start_time = datetime.now()

        # Run validation for each state
        if len(states) == 1:
            # Sequential for single state
            self._run_state_validation(states[0], test_numbers, run_final)
        else:
            # Parallel for multiple states
            asyncio.run(self._run_parallel_validation(states, test_numbers, run_final))

        # Generate summary report
        elapsed = (datetime.now() - start_time).total_seconds()
        summary = self._generate_summary(elapsed)

        logger.info(f"Validation pipeline completed in {elapsed:.1f}s")
        return summary

    async def _run_parallel_validation(
        self,
        states: List[str],
        test_numbers: List[int],
        run_final: bool,
    ) -> None:
        """Run validation for multiple states in parallel."""
        tasks = [
            self._run_state_validation_async(state, test_numbers, run_final) for state in states
        ]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _run_state_validation_async(
        self,
        state: str,
        test_numbers: List[int],
        run_final: bool,
    ) -> None:
        """Async wrapper for state validation with semaphore."""
        async with self.semaphore:
            # Run synchronous validation in thread
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self._run_state_validation,
                state,
                test_numbers,
                run_final,
            )

    def _run_state_validation(
        self,
        state: str,
        test_numbers: List[int],
        run_final: bool,
    ) -> None:
        """Run validation for a single state."""
        logger.info(f"Starting validation for {state}")

        try:
            # Lazy import to avoid circular dependency
            from mosqlimate_ai.agents.state_validation_agent import StateValidationAgent

            # Create or get agent
            if state not in self.agents:
                self.agents[state] = StateValidationAgent(
                    state=state,
                    config=self.config,
                    message_bus=self.message_bus,
                    knowledge_base=self.knowledge_base,
                    output_dir=self.output_dir,
                )

            agent = self.agents[state]

            # Run specified tests
            results = {"state": state, "tests": {}, "final": None}

            for test_num in test_numbers:
                result = agent.run(f"run_validation_test_{test_num}")
                results["tests"][test_num] = result

            # Run final forecast if requested
            if run_final:
                final_result = agent.run("run_final_forecast")
                results["final"] = final_result

            self.results[state] = results

            # Save individual state results
            self._save_state_results(state, results)

            logger.info(f"Completed validation for {state}")

        except Exception as e:
            logger.error(f"Validation failed for {state}: {e}")
            self.results[state] = {"state": state, "error": str(e), "status": "failed"}

    def _save_state_results(self, state: str, results: Dict[str, Any]) -> None:
        """Save validation results for a single state.

        Args:
            state: State UF code
            results: Validation results dictionary
        """
        import json

        # Create state directory
        state_dir = self.output_dir / state
        state_dir.mkdir(parents=True, exist_ok=True)

        # Format results for validation report
        formatted_results = {
            "state": state,
            "timestamp": datetime.now().isoformat(),
            "validation_tests": {},
            "top_models": [],
        }

        # Convert test results to expected format
        for test_num, test_result in results.get("tests", {}).items():
            if test_result and isinstance(test_result, dict):
                formatted_test = {
                    "test_number": test_num,
                    "state": state,
                    "status": test_result.get("status", "unknown"),
                    "metrics": {},
                }

                # Extract metrics if available
                if "metrics" in test_result:
                    formatted_test["metrics"] = test_result["metrics"]

                formatted_results["validation_tests"][str(test_num)] = formatted_test

        # Add top models if available
        if "top_models" in results:
            formatted_results["top_models"] = results["top_models"]

        # Save to file
        results_file = state_dir / "validation_results.json"
        with open(results_file, "w") as f:
            json.dump(formatted_results, f, indent=2, default=str)

        logger.info(f"Saved validation results for {state} to {results_file}")

    def _generate_summary(self, elapsed_seconds: float) -> Dict[str, Any]:
        """Generate validation summary report."""
        import json

        total_states = len(self.results)
        successful_states = sum(1 for r in self.results.values() if "error" not in r)
        failed_states = total_states - successful_states

        summary = {
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": elapsed_seconds,
            "total_states": total_states,
            "successful_states": successful_states,
            "failed_states": failed_states,
            "results": self.results,
            "config": {
                "max_concurrent_states": self.config.max_concurrent_states,
                "max_memory_gb": self.config.max_memory_gb,
                "n_top_models": self.config.n_top_models,
                "tuning_iterations": self.config.tuning_iterations,
            },
        }

        # Save summary
        summary_file = self.output_dir / "validation_summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info(f"Saved validation summary to {summary_file}")
        return summary

    def receive_message(self, message: Any) -> None:
        """Receive and process messages from state agents.

        Args:
            message: AgentMessage from a state agent
        """
        from mosqlimate_ai.agents.communication import AgentMessage

        if isinstance(message, AgentMessage):
            logger.info(
                f"[Orchestrator] Received {message.message_type.value} " f"from {message.sender}"
            )
            if message.state:
                logger.info(f"  State: {message.state}")
            if message.content:
                logger.info(f"  Content: {message.content}")
        else:
            logger.warning(f"[Orchestrator] Received unknown message type: {type(message)}")
