# Multi-Agent System Documentation

## Overview

The Mosqlimate AI Competitor uses a **multi-agent architecture** powered by **Karl DBot** to coordinate the dengue forecasting workflow. This document describes the agent system design, responsibilities, and interactions.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Karl DBot Orchestrator                        │
│                      (Coordination Layer)                        │
└──────────────────────┬──────────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┬──────────────┐
        │              │              │              │
        ▼              ▼              ▼              ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ Data Engineer │ │   Model      │ │  Forecaster  │ │  Validator   │
│    Agent      │ │  Architect   │ │    Agent     │ │    Agent     │
└───────┬───────┘ └───────┬───────┘ └───────┬───────┘ └───────┬───────┘
        │                 │                 │                 │
        │                 │                 │                 │
        └─────────────────┴─────────────────┴─────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │  Ensembler Agent │
                    └─────────┬────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │  Submission      │
                    │  Formatter       │
                    └──────────────────┘
```

## Agent Roles

### 1. Data Engineer Agent

**File**: `mosqlimate_ai/agents/data_agent.py`

**Purpose**: Handles all data-related tasks from collection to feature engineering.

**Responsibilities**:
- Download competition data from FTP
- Validate data integrity and schema
- Clean and preprocess raw data
- Handle missing values and outliers
- Engineer temporal and spatial features
- Create train/validation splits

**Key Methods**:
```python
class DataEngineerAgent:
    def download_data(self, force: bool = False) -> None
    def validate_data(self, df: pd.DataFrame) -> ValidationReport
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame
    def create_splits(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]
```

**Usage**:
```python
from mosqlimate_ai.agents.data_agent import DataEngineerAgent

agent = DataEngineerAgent()
agent.download_data()
df = agent.load_state_data("SP")
df_processed = agent.preprocess(df)
features = agent.engineer_features(df_processed)
```

### 2. Model Architect Agent

**File**: `mosqlimate_ai/agents/model_agent.py`

**Purpose**: Designs and trains machine learning models.

**Responsibilities**:
- Select appropriate model architectures
- Configure hyperparameters
- Train models with cross-validation
- Perform hyperparameter tuning
- Save trained models
- Generate model cards

**Supported Models**:
- XGBoost (quantile regression)
- LSTM (with MC Dropout)
- Prophet (time series)
- LightGBM (gradient boosting)

**Key Methods**:
```python
class ModelArchitectAgent:
    def select_architecture(self, data_profile: dict) -> str
    def configure_model(self, model_type: str) -> dict
    def train(self, df: pd.DataFrame, model_type: str) -> TrainedModel
    def tune_hyperparameters(self, df: pd.DataFrame) -> dict
    def save_model(self, model: TrainedModel, path: str) -> None
```

**Usage**:
```python
from mosqlimate_ai.agents.model_agent import ModelArchitectAgent

agent = ModelArchitectAgent()
model_type = agent.select_architecture(data_profile)
model = agent.train(df, model_type="xgboost")
agent.save_model(model, "models/SP/xgboost")
```

### 3. Forecaster Agent

**File**: `mosqlimate_ai/agents/forecast_agent.py`

**Purpose**: Generates predictions and uncertainty quantification.

**Responsibilities**:
- Load trained models
- Generate point forecasts
- Create prediction intervals
- Handle missing future features
- Format predictions for evaluation

**Key Methods**:
```python
class ForecasterAgent:
    def load_model(self, path: str) -> Model
    def forecast(self, model: Model, steps: int) -> pd.DataFrame
    def generate_intervals(self, forecasts: pd.DataFrame, 
                          levels: List[float]) -> pd.DataFrame
    def handle_missing_features(self, df: pd.DataFrame) -> pd.DataFrame
```

**Usage**:
```python
from mosqlimate_ai.agents.forecast_agent import ForecasterAgent

agent = ForecasterAgent()
model = agent.load_model("models/SP/xgboost")
forecasts = agent.forecast(model, steps=52)
intervals = agent.generate_intervals(forecasts, levels=[0.5, 0.8, 0.95])
```

### 4. Validator Agent

**File**: `mosqlimate_ai/agents/validator_agent.py`

**Purpose**: Evaluates model performance and validates predictions.

**Responsibilities**:
- Calculate evaluation metrics (CRPS, WIS, RMSE, MAE)
- Check prediction interval coverage
- Perform backtesting
- Validate forecast format
- Generate evaluation reports

**Key Methods**:
```python
class ValidatorAgent:
    def evaluate(self, y_true: np.ndarray, 
                 predictions: pd.DataFrame) -> dict
    def check_coverage(self, predictions: pd.DataFrame, 
                       y_true: np.ndarray) -> CoverageReport
    def backtest(self, model: Model, df: pd.DataFrame, 
                 n_splits: int = 5) -> BacktestResults
    def validate_format(self, predictions: pd.DataFrame) -> bool
    def generate_report(self, results: dict) -> str
```

**Usage**:
```python
from mosqlimate_ai.agents.validator_agent import ValidatorAgent

agent = ValidatorAgent()
metrics = agent.evaluate(y_true, predictions)
coverage = agent.check_coverage(predictions, y_true)
report = agent.generate_report(metrics)
```

### 5. Ensembler Agent

**File**: `mosqlimate_ai/agents/ensemble_agent.py`

**Purpose**: Combines multiple models for robust predictions.

**Responsibilities**:
- Aggregate predictions from multiple models
- Optimize ensemble weights
- Handle model diversity
- Apply conformal calibration
- Generate final ensemble forecast

**Ensemble Methods**:
- Weighted Average (CRPS-optimized)
- Median Ensemble
- Stacking (meta-learner)
- Bayesian Model Averaging

**Key Methods**:
```python
class EnsemblerAgent:
    def add_model(self, name: str, predictions: pd.DataFrame) -> None
    def optimize_weights(self, y_val: np.ndarray) -> dict
    def ensemble(self, method: str = "weighted_average") -> pd.DataFrame
    def calibrate_intervals(self, predictions: pd.DataFrame) -> pd.DataFrame
```

**Usage**:
```python
from mosqlimate_ai.agents.ensemble_agent import EnsemblerAgent

agent = EnsemblerAgent()
agent.add_model("xgboost", xgb_predictions)
agent.add_model("lstm", lstm_predictions)
agent.optimize_weights(y_validation)
ensemble_pred = agent.ensemble()
```

## Agent Communication

### Message Protocol

Agents communicate through a message-passing system:

```python
class AgentMessage:
    def __init__(self, sender: str, receiver: str, 
                 message_type: str, payload: dict):
        self.sender = sender
        self.receiver = receiver
        self.message_type = message_type
        self.payload = payload
        self.timestamp = datetime.now()
```

### Workflow Orchestration

**File**: `mosqlimate_ai/agents/orchestrator.py`

The orchestrator coordinates the complete workflow:

```python
class WorkflowOrchestrator:
    def run_full_pipeline(self, config: dict) -> Results:
        # 1. Data Engineering
        data_agent = DataEngineerAgent()
        data = data_agent.load_and_process(config["state"])
        
        # 2. Model Training
        model_agent = ModelArchitectAgent()
        models = {}
        for model_type in config["models"]:
            models[model_type] = model_agent.train(data, model_type)
        
        # 3. Forecasting
        forecast_agent = ForecasterAgent()
        predictions = {}
        for name, model in models.items():
            predictions[name] = forecast_agent.forecast(model, steps=52)
        
        # 4. Validation
        validator_agent = ValidatorAgent()
        metrics = {}
        for name, pred in predictions.items():
            metrics[name] = validator_agent.evaluate(data["target"], pred)
        
        # 5. Ensemble
        ensemble_agent = EnsemblerAgent()
        for name, pred in predictions.items():
            ensemble_agent.add_model(name, pred)
        final_pred = ensemble_agent.ensemble()
        
        return Results(predictions=final_pred, metrics=metrics)
```

## LLM Integration

### Karl DBot

The agents use **Karl DBot** for intelligent decision-making:

**Capabilities**:
- Natural language task descriptions
- Automatic code generation
- Model selection recommendations
- Error diagnosis and debugging

**Configuration**:
```yaml
# config.yaml
agents:
  llm:
    provider: gemini
    model: gemini-2.5
    temperature: 0.7
    max_tokens: 2048
```

**Example Usage**:
```python
from karldbot import KarlDBot

bot = KarlDBot(model="gemini-2.5")

# Ask for model recommendations
recommendation = bot.ask(
    "Given this data profile: {profile}, "
    "which model architecture would be best for dengue forecasting?"
)

# Generate preprocessing code
code = bot.generate_code(
    "Create a function to handle missing values in time series data "
    "using seasonal decomposition"
)
```

## Agent Collaboration Patterns

### Pattern 1: Sequential Pipeline

```
Data Engineer → Model Architect → Forecaster → Validator
```

Use case: Standard model training and evaluation workflow

### Pattern 2: Parallel Training

```
Data Engineer → [Model Architect × N] → Ensembler → Validator
```

Use case: Training multiple models simultaneously

### Pattern 3: Iterative Refinement

```
Validator → Model Architect → Forecaster → Validator
     ↑_________________________________________|
```

Use case: Hyperparameter tuning based on validation feedback

### Pattern 4: Real-time Monitoring

```
Forecaster → Validator → [Alert if metrics degrade]
```

Use case: Production monitoring and drift detection

## Configuration

### Agent Configuration

```yaml
# config.yaml
agents:
  data_engineer:
    validation_threshold: 0.95
    outlier_method: "iqr"
    imputation_strategy: "seasonal"
    
  model_architect:
    default_model: "xgboost"
    cross_validation_folds: 5
    early_stopping_patience: 50
    
  forecaster:
    confidence_levels: [0.5, 0.8, 0.9, 0.95]
    max_forecast_horizon: 52
    
  validator:
    primary_metric: "crps"
    coverage_tolerance: 0.05
    backtesting_splits: 5
    
  ensembler:
    method: "weighted_average"
    weight_metric: "crps"
    calibrate_intervals: true
```

## Error Handling

### Agent-Level Errors

Each agent has specific error handling:

```python
class DataEngineerAgent:
    def handle_error(self, error: Exception) -> RecoveryAction:
        if isinstance(error, DataValidationError):
            return self.attempt_repair()
        elif isinstance(error, DownloadError):
            return self.retry_with_backoff()
        else:
            return self.escalate_to_orchestrator()
```

### Orchestrator Error Recovery

```python
class WorkflowOrchestrator:
    def handle_agent_failure(self, agent: Agent, error: Exception):
        # Log the error
        self.logger.error(f"Agent {agent.name} failed: {error}")
        
        # Attempt recovery
        if self.can_retry(agent):
            return self.retry_agent(agent)
        
        # Fallback to alternative
        if self.has_fallback(agent):
            return self.use_fallback(agent)
        
        # Abort workflow
        raise WorkflowAbortError(f"Critical failure in {agent.name}")
```

## Monitoring & Logging

### Agent Metrics

Track agent performance:

```python
@dataclass
class AgentMetrics:
    agent_name: str
    tasks_completed: int
    tasks_failed: int
    avg_execution_time: float
    error_rate: float
    last_active: datetime
```

### Logging

```python
import logging

logger = logging.getLogger("mosqlimate_ai.agents")

# Agent activity logging
logger.info(f"Agent {agent.name} started task: {task.id}")
logger.debug(f"Agent {agent.name} intermediate result: {result}")
logger.error(f"Agent {agent.name} failed: {error}")
```

## Best Practices

### 1. Agent Design
- Single responsibility per agent
- Clear input/output contracts
- Idempotent operations where possible
- Graceful degradation

### 2. Communication
- Use typed messages
- Include timestamps
- Handle timeouts
- Implement retries

### 3. Error Handling
- Fail fast for configuration errors
- Retry for transient failures
- Degrade gracefully for non-critical errors
- Log everything

### 4. Testing
- Unit test each agent independently
- Mock agent dependencies
- Test communication protocols
- Integration test full workflows

## Future Enhancements

### Planned Features

1. **AutoML Agent**: Automatically search model architectures
2. **Drift Detector**: Monitor data and model drift
3. **Explainability Agent**: Generate SHAP values and explanations
4. **A/B Testing Agent**: Compare model variants
5. **Deployment Agent**: Manage model deployment

### Research Directions

- Reinforcement learning for agent coordination
- Multi-objective optimization
- Federated learning across states
- Causal inference for feature importance

## References

- [Karl DBot Documentation](https://github.com/Deeplearn-PeD/KarlDBot)
- [Multi-Agent Systems](https://en.wikipedia.org/wiki/Multi-agent_system)
- [AutoML Survey](https://arxiv.org/abs/1908.00709)
