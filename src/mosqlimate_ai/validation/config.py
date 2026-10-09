"""Validation pipeline configuration for Mosqlimate 3rd IMDC 2026.

Defines the 4-run validation structure according to competition rules:
- 4 validation tests (2022-2023, 2023-2024, 2024-2025, 2025-2026)
- 1 final forecast target (2026-2027 season, EW41 2026 - EW40 2027)
"""

from dataclasses import dataclass


@dataclass
class ValidationTestConfig:
    """Configuration for a single validation test."""

    test_number: int  # 1, 2, 3, or 4
    season: str  # e.g., "2022-2023"
    train_end: str  # EW 25 of training year
    target_start: str  # EW 41 of forecast year
    target_end: str  # EW 40 of following year
    description: str


@dataclass
class ValidationPipelineConfig:
    """Complete validation pipeline configuration."""

    # Validation tests
    validation_tests: list[ValidationTestConfig]

    # Final forecast
    final_forecast_train_end: str
    final_forecast_target_start: str
    final_forecast_target_end: str

    # States to validate
    states: list[str]

    # Parallelization limits
    max_concurrent_states: int
    max_memory_gb: float

    # Model selection
    n_top_models: int
    min_coverage_threshold: float
    max_bias_threshold: float

    # Hyperparameter tuning with convergence-based stopping
    tuning_iterations: int
    convergence_patience: int
    min_improvement_rate: float

    # Model pre-selection
    preselect_models: bool
    max_models_per_state: int

    # Agent LLM settings
    llm_model: str
    llm_temperature: float


# Default configuration according to competition rules
DEFAULT_VALIDATION_CONFIG = ValidationPipelineConfig(
    validation_tests=[
        ValidationTestConfig(
            test_number=1,
            season="2022-2023",
            train_end="2022-06-26",  # EW 25 2022
            target_start="2022-10-09",  # EW 41 2022
            target_end="2023-10-08",  # EW 40 2023
            description="Validation Test 1: 2022-2023 season",
        ),
        ValidationTestConfig(
            test_number=2,
            season="2023-2024",
            train_end="2023-06-25",  # EW 25 2023
            target_start="2023-10-08",  # EW 41 2023
            target_end="2024-10-06",  # EW 40 2024
            description="Validation Test 2: 2023-2024 season",
        ),
        ValidationTestConfig(
            test_number=3,
            season="2024-2025",
            train_end="2024-06-23",  # EW 25 2024
            target_start="2024-10-06",  # EW 41 2024
            target_end="2025-10-05",  # EW 40 2025
            description="Validation Test 3: 2024-2025 season",
        ),
        ValidationTestConfig(
            test_number=4,
            season="2025-2026",
            train_end="2025-06-22",  # EW 25 2025
            target_start="2025-10-05",  # EW 41 2025
            target_end="2026-10-04",  # EW 40 2026
            description="Validation Test 4: 2025-2026 season",
        ),
    ],
    final_forecast_train_end="2026-06-21",  # EW 25 2026
    final_forecast_target_start="2026-10-11",  # EW 41 2026
    final_forecast_target_end="2027-10-03",  # EW 40 2027
    states=[
        "AC",
        "AL",
        "AP",
        "AM",
        "BA",
        "CE",
        "DF",
        # ES (Espírito Santo) excluded by competition rules
        "GO",
        "MA",
        "MT",
        "MS",
        "MG",
        "PA",
        "PB",
        "PR",
        "PE",
        "PI",
        "RJ",
        "RN",
        "RS",
        "RO",
        "RR",
        "SC",
        "SP",
        "SE",
        "TO",
    ],
    max_concurrent_states=5,
    max_memory_gb=16.0,
    n_top_models=3,
    min_coverage_threshold=0.85,
    max_bias_threshold=500.0,
    tuning_iterations=30,
    convergence_patience=5,
    min_improvement_rate=0.001,
    preselect_models=True,
    max_models_per_state=3,
    llm_model="gemini-2.5",
    llm_temperature=0.3,
)


def get_validation_config() -> ValidationPipelineConfig:
    """Get default validation configuration."""
    return DEFAULT_VALIDATION_CONFIG
