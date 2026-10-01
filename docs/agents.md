# Multi-Agent System Documentation

> **Status: legacy / experimental.** The production validation path is
> the deterministic harness in `validation/backtest.py`
> (`mosqlimate-ai validate`); it does **not** use agents. The CLI's
> `--show-logs` / `--export-audit` flags are deprecated no-ops
> ("the deterministic pipeline has no agent logs"). The agent system
> described here is importable (`mosqlimate_ai.agents`) but not
> invoked by any CLI command or script, and has no test coverage.

## Overview

The multi-agent layer is built around `BaseAgent` subclasses coordinated
by `AgentOrchestrator`, with inter-agent messaging, shared memory, and a
cross-state knowledge base. The design references **Karl DBot**
(`karldbot` is a Git dependency in `pyproject.toml`, and `AgentConfig`
carries LLM-ish fields such as `model="gemini-2.5"`), but **no agent in
`src/` imports or calls an LLM** — every agent executes deterministic
Python over the shared data/model utilities.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                   AgentOrchestrator                           │
│        (workflows, tasks, fine-tuning loops)                  │
└───────────────┬──────────────────────────────────────────────┘
                │ AgentCommunicationBus / MemoryManager
        ┌───────┼───────────────┬───────────────────┐
        ▼       ▼               ▼                   ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────────┐
│ State-       │ │ Efficient-   │ │ TopNModel-   │ │ Classic agents   │
│ Validation   │ │ Hyperparam-  │ │ Selection    │ │ (Data/Model/     │
│ Agent (×UF)  │ │ eterTuner    │ │ Agent        │ │  Forecast/       │
└──────┬───────┘ └──────────────┘ └──────────────┘ │  Validator/      │
       │                                       │   │  Ensemble)       │
       └───────────── CrossStateKnowledgeBase ─┴───┴──────────────────┘
```

## Modules and Public API

Everything below is exported from `mosqlimate_ai.agents`.

### Base (`agents/base.py`)

```python
class AgentConfig(BaseModel):
    name: str
    description: str
    model: str = "gemini-2.5"      # descriptive only; no LLM calls
    temperature: float = 0.3
    max_tokens: int = 4000

class BaseAgent(ABC):
    def run(self, task: str, context: dict | None = None) -> dict  # abstract
    def add_to_memory(self, key: str, value) -> None
    def get_from_memory(self, key: str)
    def register_tool(self, name: str, tool) -> None    # tools are plain callables
    def use_tool(self, name: str, **kwargs)
    def communicate(self, message: str, to_agent: str | None = None) -> dict
```

### Communication (`agents/communication.py`)

- `AgentMessage` (+ `MessageType`, `MessagePriority`) with
  `to_dict()` / `from_dict()`
- `AgentCommunicationBus`: `send_message`, `subscribe`,
  `get_messages_for_agent`, `get_conversation_history`,
  `export_audit_log(path, format="jsonl")`, `get_session_summary`
- `MemoryManager`: global plus per-agent key/value memory

### Cross-state knowledge (`agents/knowledge_base.py`)

- `CrossStateKnowledgeBase`: `get_similar_states`, `share_results`,
  `get_best_params`, `get_tuning_recommendations`,
  `get_aggregate_insights`, `save`/`load`
- Dataclasses: `StateProfile`, `ValidationResult`,
  `HyperparameterConfig`

### State validation (`agents/state_validation_agent.py`)

`StateValidationAgent(uf, config, message_bus, knowledge_base)` —
per-state validation loop that combines the tuner, model pre-selection
and knowledge sharing:

```python
def run(self, task, context=None) -> dict
def run_full_validation(self) -> dict
```

The per-state agents are wired together by
`mosqlimate_ai.validation.orchestrator.ValidationOrchestrator`
(`run_full_pipeline(states=[...])`), which owns the message bus and the
knowledge base.

### Tuning & selection

- `EfficientHyperparameterTuner` (`agents/tuner_agent.py`):
  `tune(...)`, `suggest_focus_areas(...)`, `get_supported_models()`;
  with `ConvergenceTracker` (early-stop bookkeeping)
- `TopNModelSelectionAgent` (`agents/selection_agent.py`):
  `select_top_models(...)`, `generate_model_report(...)`
- `ModelPreSelector` / `ModelRecommendation`
  (`agents/model_selector_agent.py`): cheap per-state model shortlist
  over the registry's model classes (not re-exported by
  `__init__.py`)

### Classic agents

All follow the `run(task, context)` contract and wrap the ordinary
data/model utilities (loader, preprocessor, legacy feature engineer,
model registry):

| Agent | File | Notable methods |
|-------|------|-----------------|
| `DataEngineerAgent` | `data_agent.py` | `run` (loads state data via `CompetitionDataLoader`, preprocesses, builds features); registered tools: `load_state_data`, `load_all_states`, `load_ocean_data` |
| `ModelArchitectAgent` | `model_agent.py` | `run`, `train_xgboost`, `train_lstm`, `load_models`, `cross_validate` |
| `ForecastAgent` | `forecast_agent.py` | `run`, `generate_future_dates`, `recursive_forecast`, `get_forecast`, `save_forecasts` / `load_forecasts`, `combine_forecasts` |
| `ValidatorAgent` | `validator_agent.py` | `run`, `cross_validate_time_series`, `validate_prediction_intervals`, `check_overfitting`, `generate_report`, `save_results` |
| `EnsembleAgent` | `ensemble_agent.py` | `run`, `add_model`, `fit_weights`, `predict`, `calibrate`, `format_submission`, `compare_methods`, `save_ensemble` / `load_ensemble`, `generate_report` |

### Orchestrator (`agents/orchestrator.py`)

```python
orchestrator = AgentOrchestrator()
orchestrator.register_agent(agent)                 # by agent.config.name
workflow = orchestrator.create_workflow("wf", tasks=[...])
orchestrator.run_workflow(workflow.id)             # dependency-ordered
orchestrator.run_forecast_workflow(uf="SP", start_date=..., end_date=...)
orchestrator.run_fine_tuning_workflow(...)         # iterative refinement
```

Supporting types: `Task`, `Workflow`, `FineTuningConfig`,
`FineTuningResult`, `PerformanceTracker`.

### Prompts (`agents/prompts.py`)

`AGENT_PROMPTS` (per-agent system prompts), `get_prompt(name)`,
`list_agents()`. Prompts are stored on each agent
(`agent.system_prompt`) for provenance; they are **not** sent to an
LLM anywhere in this codebase.

## Relationship to the production pipeline

| Concern | Production path (deterministic) | Agent layer (legacy) |
|---------|--------------------------------|----------------------|
| Validation / backtests | `validation/backtest.py` | `StateValidationAgent` + `ValidationOrchestrator` |
| Hyperparameter tuning | `validation/tuning.py` (seeded random search, WIS-scored, cached under `validation_results/backtest/hyperparams/`) | `EfficientHyperparameterTuner` |
| Model selection | `validation/selection.py` (skill-gated per state) | `TopNModelSelectionAgent`, `ModelPreSelector` |
| Ensembles | `ens_qavg` / `ens_median` in `backtest.py` | `EnsembleAgent` |

When extending the system, prefer the deterministic modules: they are
what the CLI (`mosqlimate-ai validate`), the scripts
(`scripts/run_backtests.py`, `generate_forecast.py`,
`make_submission.py`) and the test suite actually exercise.

## References

- [Karl DBot](https://github.com/Deeplearn-PeD/KarlDBot) (declared
  dependency; not imported by `src/`)
- `docs/data_pipeline.md` — data loading actually used by both paths
- `VALIDATION_PIPELINE_SUMMARY.md` — historical design notes for the
  agent system
