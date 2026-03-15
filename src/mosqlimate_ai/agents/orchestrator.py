"""Agent orchestrator for Karl DBot multi-agent system."""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging

from mosqlimate_ai.agents.base import BaseAgent

logger = logging.getLogger(__name__)


@dataclass
class Task:
    """Represents a task in the workflow."""
    id: str
    name: str
    agent: str
    description: str
    dependencies: List[str] = field(default_factory=list)
    status: str = "pending"  # pending, running, completed, failed
    result: Optional[Dict[str, Any]] = None
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None


@dataclass
class Workflow:
    """Represents a complete forecasting workflow."""
    id: str
    name: str
    tasks: Dict[str, Task] = field(default_factory=dict)
    status: str = "pending"
    created_at: datetime = field(default_factory=datetime.now)


class AgentOrchestrator:
    """Orchestrates multi-agent workflow for dengue forecasting.
    
    This is the central coordinator for Karl DBot agents,
    managing task execution, inter-agent communication,
    and workflow optimization.
    
    Example:
        >>> orchestrator = AgentOrchestrator()
        >>> orchestrator.register_agent(DataEngineerAgent(config))
        >>> result = orchestrator.run_forecast_workflow(uf="SP")
    """
    
    def __init__(self):
        """Initialize orchestrator."""
        self.agents: Dict[str, BaseAgent] = {}
        self.workflows: Dict[str, Workflow] = {}
        self.message_queue: List[Dict[str, Any]] = []
        logger.info("AgentOrchestrator initialized")
    
    def register_agent(self, agent: BaseAgent) -> None:
        """Register an agent with the orchestrator.
        
        Args:
            agent: Agent instance to register
        """
        self.agents[agent.config.name] = agent
        logger.info(f"Registered agent: {agent.config.name}")
    
    def create_workflow(self, name: str, tasks: List[Task]) -> Workflow:
        """Create a new workflow.
        
        Args:
            name: Workflow name
            tasks: List of tasks
            
        Returns:
            Created workflow
        """
        workflow_id = f"wf_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        workflow = Workflow(
            id=workflow_id,
            name=name,
            tasks={t.id: t for t in tasks}
        )
        self.workflows[workflow_id] = workflow
        logger.info(f"Created workflow: {workflow_id}")
        return workflow
    
    def run_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """Execute a workflow.
        
        Args:
            workflow_id: Workflow identifier
            
        Returns:
            Workflow results
        """
        workflow = self.workflows.get(workflow_id)
        if not workflow:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        workflow.status = "running"
        logger.info(f"Starting workflow: {workflow_id}")
        
        # Execute tasks in dependency order
        completed = set()
        failed = []
        
        while len(completed) < len(workflow.tasks):
            # Find ready tasks (dependencies met)
            ready = [
                t for t in workflow.tasks.values()
                if t.status == "pending"
                and all(dep in completed for dep in t.dependencies)
            ]
            
            if not ready:
                if len(completed) + len(failed) < len(workflow.tasks):
                    logger.error("Dependency deadlock detected")
                    break
                break
            
            # Execute ready tasks
            for task in ready:
                try:
                    result = self._execute_task(task)
                    task.result = result
                    task.status = "completed"
                    task.completed_at = datetime.now()
                    completed.add(task.id)
                    logger.info(f"Task completed: {task.name}")
                except Exception as e:
                    task.status = "failed"
                    failed.append(task.id)
                    logger.error(f"Task failed: {task.name} - {e}")
        
        workflow.status = "completed" if not failed else "partial"
        
        return {
            "workflow_id": workflow_id,
            "status": workflow.status,
            "completed_tasks": len(completed),
            "failed_tasks": len(failed),
            "results": {t.id: t.result for t in workflow.tasks.values() if t.result}
        }
    
    def _execute_task(self, task: Task) -> Dict[str, Any]:
        """Execute a single task.
        
        Args:
            task: Task to execute
            
        Returns:
            Task results
        """
        agent = self.agents.get(task.agent)
        if not agent:
            raise ValueError(f"Agent {task.agent} not found")
        
        task.status = "running"
        logger.info(f"Executing task: {task.name} with {task.agent}")
        
        # Get context from dependencies
        context = {}
        for dep_id in task.dependencies:
            dep_task = self._find_task_by_id(dep_id)
            if dep_task and dep_task.result:
                context.update(dep_task.result)
        
        # Run agent
        return agent.run(task.description, context)
    
    def _find_task_by_id(self, task_id: str) -> Optional[Task]:
        """Find task across all workflows."""
        for workflow in self.workflows.values():
            if task_id in workflow.tasks:
                return workflow.tasks[task_id]
        return None
    
    def run_forecast_workflow(
        self,
        uf: str,
        start_date: str,
        end_date: str,
        model_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Run complete forecasting workflow.
        
        Args:
            uf: State abbreviation
            start_date: Forecast start date
            end_date: Forecast end date
            model_types: List of model types to use
            
        Returns:
            Forecast results with ensemble prediction
        """
        model_types = model_types or ["xgboost", "lstm", "prophet"]
        
        # Define workflow tasks
        tasks = [
            Task(
                id="t1",
                name="data_collection",
                agent="data_engineer",
                description=f"Collect dengue data for {uf} from Mosqlimate API"
            ),
            Task(
                id="t2",
                name="feature_engineering",
                agent="data_engineer",
                description="Create lag features and climate variables",
                dependencies=["t1"]
            ),
            Task(
                id="t3",
                name="model_training",
                agent="model_architect",
                description=f"Train {', '.join(model_types)} models",
                dependencies=["t2"]
            ),
            Task(
                id="t4",
                name="forecast_generation",
                agent="forecaster",
                description=f"Generate forecasts from {start_date} to {end_date}",
                dependencies=["t3"]
            ),
            Task(
                id="t5",
                name="validation",
                agent="validator",
                description="Validate forecast quality and uncertainty",
                dependencies=["t4"]
            ),
            Task(
                id="t6",
                name="ensemble",
                agent="ensembler",
                description="Combine models into ensemble prediction",
                dependencies=["t4", "t5"]
            ),
            Task(
                id="t7",
                name="submission_format",
                agent="ensembler",
                description="Format output for Mosqlimate submission",
                dependencies=["t6"]
            ),
        ]
        
        workflow = self.create_workflow(f"forecast_{uf}", tasks)
        return self.run_workflow(workflow.id)
