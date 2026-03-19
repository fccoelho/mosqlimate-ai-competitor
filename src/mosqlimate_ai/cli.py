"""Command-line interface for mosqlimate-ai."""

import logging
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from mosqlimate_ai.data.downloader import DataDownloader
from mosqlimate_ai.data.features import FeatureEngineer
from mosqlimate_ai.data.loader import CompetitionDataLoader
from mosqlimate_ai.data.preprocessor import DataPreprocessor
from mosqlimate_ai.evaluation.metrics import evaluate_forecast
from mosqlimate_ai.models.ensemble import EnsembleForecaster
from mosqlimate_ai.models.lstm_model import LSTMForecaster
from mosqlimate_ai.models.xgboost_model import XGBoostForecaster
from mosqlimate_ai.submission.api_client import MosqlimateClient, get_git_commit_hash
from mosqlimate_ai.submission.formatter import SubmissionFormatter

app = typer.Typer(
    name="mosqlimate-ai",
    help="AI-powered dengue forecasting for Mosqlimate-Infodengue Challenge",
)
console = Console()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@app.command("download-data")
def download_data_cmd(
    cache_dir: Optional[Path] = typer.Option(
        None,
        "--cache-dir",
        "-c",
        help="Directory to store downloaded data (default: project data/ folder)",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Force re-download even if files exist",
    ),
    clear: bool = typer.Option(
        False,
        "--clear",
        help="Clear cache before downloading",
    ),
) -> None:
    """Download and cache Mosqlimate competition data."""
    downloader = DataDownloader(cache_dir=cache_dir)

    if clear:
        downloader.clear_cache()

    console.print(f"[cyan]Cache directory: {downloader.cache_dir}[/cyan]\n")
    downloader.download_all(force=force)


@app.command("cache-info")
def cache_info_cmd(
    cache_dir: Optional[Path] = typer.Option(
        None,
        "--cache-dir",
        "-c",
        help="Directory where data is cached",
    ),
) -> None:
    """Show information about cached data files."""
    downloader = DataDownloader(cache_dir=cache_dir)
    info = downloader.get_cache_info()

    table = Table(title="Cached Data Files")
    table.add_column("File", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Size", style="yellow")
    table.add_column("Description")

    for filename, data in info.items():
        status = "✓ Cached" if data["cached"] else "✗ Not cached"
        status_style = "green" if data["cached"] else "red"
        table.add_row(
            filename,
            f"[{status_style}]{status}[/{status_style}]",
            data["size_formatted"],
            data["description"],
        )

    console.print(table)

    cached_count = sum(1 for d in info.values() if d["cached"])
    total_size = sum(d["size"] for d in info.values())
    console.print(
        f"\n[cyan]{cached_count}/{len(info)} files cached "
        f"({downloader._format_size(total_size)} total)[/cyan]"
    )


@app.command("clear-cache")
def clear_cache_cmd(
    cache_dir: Optional[Path] = typer.Option(
        None,
        "--cache-dir",
        "-c",
        help="Directory where data is cached",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip confirmation prompt",
    ),
) -> None:
    """Clear all cached data files."""
    downloader = DataDownloader(cache_dir=cache_dir)

    if not yes:
        confirm = typer.confirm(f"Are you sure you want to clear cache at {downloader.cache_dir}?")
        if not confirm:
            console.print("[yellow]Cancelled.[/yellow]")
            raise typer.Exit()

    downloader.clear_cache()


@app.command("train")
def train_cmd(
    output_dir: Path = typer.Option(
        Path("models"),
        "--output",
        "-o",
        help="Directory to save trained models",
    ),
    states: Optional[str] = typer.Option(
        None,
        "--states",
        "-s",
        help="Comma-separated state UFs to train (default: all)",
    ),
    models: str = typer.Option(
        "xgboost,lstm",
        "--models",
        "-m",
        help="Comma-separated models to train (xgboost, lstm)",
    ),
    validation_size: float = typer.Option(
        0.1,
        "--val-size",
        help="Fraction of data for validation",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed training output",
    ),
) -> None:
    """Train forecasting models on competition data.

    Trains XGBoost and/or LSTM models for dengue forecasting.
    Models are saved to the output directory for later use.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_list = [m.strip() for m in models.split(",")]
    state_list = [s.strip() for s in states.split(",")] if states else None

    console.print("[cyan]Loading competition data...[/cyan]")

    loader = CompetitionDataLoader()
    preprocessor = DataPreprocessor()
    feature_engineer = FeatureEngineer()

    states_data = loader.load_all_states(aggregate=True)
    ocean_df = loader.load_ocean_data()

    if state_list:
        states_data = {uf: df for uf, df in states_data.items() if uf in state_list}

    console.print(f"[cyan]Training models for {len(states_data)} states...[/cyan]")

    for uf, df in states_data.items():
        console.print(f"\n[green]Processing {uf}...[/green]")

        df = preprocessor.clean(df)
        df = preprocessor.impute_missing(df)
        df = preprocessor.add_epidemiological_features(df)
        df = feature_engineer.build_feature_set(
            df, target_col="casos", ocean_df=ocean_df if not ocean_df.empty else None
        )

        state_output = output_dir / uf
        state_output.mkdir(parents=True, exist_ok=True)

        if "xgboost" in model_list:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task(f"Training XGBoost for {uf}...", total=None)

                xgb = XGBoostForecaster(
                    target_col="casos",
                    n_estimators=500,
                    max_depth=6,
                    learning_rate=0.05,
                )
                xgb.fit(df, validation_size=validation_size, verbose=verbose)
                xgb.save(state_output / "xgboost")

                progress.remove_task(task)
            console.print(f"  [green]✓[/green] XGBoost saved to {state_output / 'xgboost'}")

        if "lstm" in model_list:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task(f"Training LSTM for {uf}...", total=None)

                lstm = LSTMForecaster(
                    target_col="casos",
                    hidden_size=128,
                    num_layers=2,
                    dropout=0.2,
                    epochs=100,
                )
                lstm.fit(df, validation_size=validation_size, verbose=verbose)
                lstm.save(state_output / "lstm")

                progress.remove_task(task)
            console.print(f"  [green]✓[/green] LSTM saved to {state_output / 'lstm'}")

    console.print(f"\n[green]Training complete! Models saved to {output_dir}[/green]")


@app.command("forecast")
def forecast_cmd(
    model_dir: Path = typer.Option(
        Path("models"),
        "--model-dir",
        "-m",
        help="Directory containing trained models",
    ),
    output_dir: Path = typer.Option(
        Path("forecasts"),
        "--output",
        "-o",
        help="Directory to save forecasts",
    ),
    states: Optional[str] = typer.Option(
        None,
        "--states",
        "-s",
        help="Comma-separated state UFs to forecast (default: all)",
    ),
    start_date: Optional[str] = typer.Option(
        None,
        "--start-date",
        help="Start date for forecast (YYYY-MM-DD)",
    ),
    n_weeks: int = typer.Option(
        52,
        "--weeks",
        "-w",
        help="Number of weeks to forecast",
    ),
    ensemble: bool = typer.Option(
        True,
        "--ensemble/--no-ensemble",
        help="Create ensemble from multiple models",
    ),
) -> None:
    """Generate forecasts using trained models.

    Loads trained models and generates forecasts with prediction intervals
    for the specified forecast period.
    """
    model_dir = Path(model_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    state_list = [s.strip() for s in states.split(",")] if states else None

    if not model_dir.exists():
        console.print(f"[red]Model directory not found: {model_dir}[/red]")
        console.print("[yellow]Run 'mosqlimate-ai train' first to train models.[/yellow]")
        raise typer.Exit(1)

    state_dirs = [d for d in model_dir.iterdir() if d.is_dir()]
    if state_list:
        state_dirs = [d for d in state_dirs if d.name in state_list]

    if not state_dirs:
        console.print(f"[red]No trained models found in {model_dir}[/red]")
        raise typer.Exit(1)

    console.print(f"[cyan]Generating forecasts for {len(state_dirs)} states...[/cyan]")

    loader = CompetitionDataLoader()
    preprocessor = DataPreprocessor()
    feature_engineer = FeatureEngineer()
    ocean_df = loader.load_ocean_data()

    for state_dir in state_dirs:
        uf = state_dir.name
        console.print(f"\n[green]Forecasting {uf}...[/green]")

        df = loader.load_state_data(uf=uf, aggregate=True)
        df = preprocessor.clean(df)
        df = preprocessor.impute_missing(df)
        df = preprocessor.add_epidemiological_features(df)
        df = feature_engineer.build_feature_set(
            df, target_col="casos", ocean_df=ocean_df if not ocean_df.empty else None
        )

        forecasts = {}

        if (state_dir / "xgboost").exists():
            xgb = XGBoostForecaster()
            xgb.load(state_dir / "xgboost")
            xgb_forecast = xgb.predict(df, levels=[0.50, 0.80, 0.90, 0.95])
            forecasts["xgboost"] = xgb_forecast
            console.print("  [green]✓[/green] XGBoost forecast generated")

        if (state_dir / "lstm").exists():
            lstm = LSTMForecaster()
            lstm.load(state_dir / "lstm")
            lstm_forecast = lstm.predict(df, n_mc_samples=100)
            forecasts["lstm"] = lstm_forecast
            console.print("  [green]✓[/green] LSTM forecast generated")

        if ensemble and len(forecasts) > 1:
            ensemble_forecaster = EnsembleForecaster(method="weighted_average")
            for name, forecast in forecasts.items():
                ensemble_forecaster.add_model(name, forecast)

            final_forecast = ensemble_forecaster.predict()
            console.print("  [green]✓[/green] Ensemble forecast created")
        else:
            final_forecast = list(forecasts.values())[0]

        forecast_output = output_dir / f"{uf}_forecast.csv"
        final_forecast.to_csv(forecast_output, index=False)
        console.print(f"  [green]✓[/green] Saved to {forecast_output}")

    console.print(f"\n[green]Forecasting complete! Results saved to {output_dir}[/green]")


@app.command("submit")
def submit_cmd(
    forecast_dir: Path = typer.Option(
        Path("forecasts"),
        "--forecast-dir",
        "-f",
        help="Directory containing forecast files",
    ),
    model_id: int = typer.Option(
        ...,
        "--model-id",
        "-m",
        help="Registered model ID from Mosqlimate",
    ),
    predict_date: Optional[str] = typer.Option(
        None,
        "--predict-date",
        help="Prediction date (YYYY-MM-DD, default: today)",
    ),
    description: Optional[str] = typer.Option(
        None,
        "--description",
        "-d",
        help="Prediction description",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Prepare submissions without sending to API",
    ),
    output_json: Optional[Path] = typer.Option(
        None,
        "--output-json",
        help="Save submissions to JSON file (for dry-run)",
    ),
) -> None:
    """Submit forecasts to Mosqlimate API.

    Formats forecasts according to API specification and submits
    to the Mosqlimate platform. Requires MOSQLIMATE_API_KEY environment variable.
    """
    forecast_dir = Path(forecast_dir)
    predict_date = predict_date or date.today().isoformat()
    description = description or "Dengue forecast for EW 41 2025 - EW 40 2026"

    if not forecast_dir.exists():
        console.print(f"[red]Forecast directory not found: {forecast_dir}[/red]")
        console.print("[yellow]Run 'mosqlimate-ai forecast' first to generate forecasts.[/yellow]")
        raise typer.Exit(1)

    forecast_files = list(forecast_dir.glob("*_forecast.csv"))
    if not forecast_files:
        console.print(f"[red]No forecast files found in {forecast_dir}[/red]")
        raise typer.Exit(1)

    console.print(f"[cyan]Preparing submissions for {len(forecast_files)} states...[/cyan]")

    formatter = SubmissionFormatter(
        model_id=model_id,
        predict_date=predict_date,
        description=description,
        commit=get_git_commit_hash(),
    )

    submissions = []
    for forecast_file in sorted(forecast_files):
        uf = forecast_file.stem.replace("_forecast", "")

        import pandas as pd

        forecast_df = pd.read_csv(forecast_file)

        if "date" not in forecast_df.columns:
            forecast_df["date"] = pd.date_range(
                start="2025-10-05", periods=len(forecast_df), freq="W-SUN"
            )

        submission = formatter.format_state_forecast(forecast_df, uf)
        submissions.append(submission)
        console.print(f"  [green]✓[/green] Prepared submission for {uf}")

    issues = formatter.validate_submissions()
    if issues:
        console.print(f"\n[yellow]Warning: Found {len(issues)} validation issues:[/yellow]")
        for issue in issues[:5]:
            console.print(f"  - {issue}")

    if dry_run:
        console.print("\n[yellow]Dry run mode - not submitting to API[/yellow]")

        if output_json:
            import json

            with open(output_json, "w") as f:
                json.dump(submissions, f, indent=2)
            console.print(f"[green]Submissions saved to {output_json}[/green]")
        return

    api_key = (
        typer.prompt("Enter your Mosqlimate API key", hide_input=True) if not dry_run else None
    )

    client = MosqlimateClient(api_key=api_key)

    if not client.test_connection():
        console.print("[red]Failed to connect to Mosqlimate API. Check your API key.[/red]")
        raise typer.Exit(1)

    console.print(f"\n[cyan]Submitting {len(submissions)} forecasts to Mosqlimate...[/cyan]")

    try:
        responses = client.submit_all_predictions(model_id, submissions)
        console.print(f"\n[green]Successfully submitted {len(responses)} forecasts![/green]")
    except Exception as e:
        console.print(f"\n[red]Error submitting forecasts: {e}[/red]")
        raise typer.Exit(1) from None


@app.command("evaluate")
def evaluate_cmd(
    forecast_dir: Path = typer.Option(
        Path("forecasts"),
        "--forecast-dir",
        "-f",
        help="Directory containing forecast files",
    ),
    state: str = typer.Option(
        ...,
        "--state",
        "-s",
        help="State UF to evaluate",
    ),
    end_date: Optional[str] = typer.Option(
        None,
        "--end-date",
        help="End date for evaluation data (for backtesting)",
    ),
) -> None:
    """Evaluate forecasts against historical data.

    Computes CRPS, WIS, coverage, and other metrics for forecasts.
    Useful for backtesting and model comparison.
    """
    import pandas as pd

    forecast_file = forecast_dir / f"{state}_forecast.csv"
    if not forecast_file.exists():
        console.print(f"[red]Forecast file not found: {forecast_file}[/red]")
        raise typer.Exit(1)

    console.print("[cyan]Loading data for evaluation...[/cyan]")

    loader = CompetitionDataLoader()
    df = loader.load_state_data(uf=state, aggregate=True, end_date=end_date)

    forecast_df = pd.read_csv(forecast_file)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    if "date" in forecast_df.columns:
        forecast_df["date"] = pd.to_datetime(forecast_df["date"])

    common_dates = set(df["date"].values) & set(forecast_df["date"].values)

    if not common_dates:
        console.print(
            "[yellow]Warning: No overlapping dates between forecast and historical data[/yellow]"
        )
        console.print("[yellow]Cannot evaluate - showing forecast summary only[/yellow]")

        table = Table(title=f"Forecast Summary for {state}")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Total weeks", str(len(forecast_df)))
        table.add_row(
            "Median range", f"{forecast_df['median'].min():.1f} - {forecast_df['median'].max():.1f}"
        )
        table.add_row("Median mean", f"{forecast_df['median'].mean():.1f}")

        console.print(table)
        return

    common_dates_list = list(common_dates)
    df_eval = df[df["date"].isin(common_dates_list)].sort_values("date")
    forecast_eval = forecast_df[forecast_df["date"].isin(common_dates_list)].sort_values("date")

    y_true = np.asarray(df_eval["casos"].values)

    metrics = evaluate_forecast(y_true, forecast_eval)

    table = Table(title=f"Evaluation Results for {state}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    for metric, value in sorted(metrics.items()):
        if isinstance(value, float):
            table.add_row(metric, f"{value:.4f}")
        else:
            table.add_row(metric, str(value))

    console.print(table)


@app.command("report")
def report_cmd(
    model_dir: Path = typer.Option(
        Path("models"),
        "--model-dir",
        "-m",
        help="Directory containing trained models",
    ),
    forecast_dir: Path = typer.Option(
        Path("forecasts"),
        "--forecast-dir",
        "-f",
        help="Directory containing forecast files",
    ),
    output: Path = typer.Option(
        Path("forecast_report.md"),
        "--output",
        "-o",
        help="Output markdown file path",
    ),
    states: Optional[str] = typer.Option(
        None,
        "--states",
        "-s",
        help="Comma-separated state UFs to include (default: all)",
    ),
    end_date: Optional[str] = typer.Option(
        None,
        "--end-date",
        help="End date for evaluation data (for backtesting)",
    ),
) -> None:
    """Generate a markdown report of forecast performance by model.

    Evaluates all trained models across all states and generates a comprehensive
    markdown report with performance metrics, rankings, and comparisons.
    """
    from datetime import datetime

    import pandas as pd

    model_dir = Path(model_dir)
    forecast_dir = Path(forecast_dir)

    if not model_dir.exists():
        console.print(f"[red]Model directory not found: {model_dir}[/red]")
        raise typer.Exit(1)

    if not forecast_dir.exists():
        console.print(f"[red]Forecast directory not found: {forecast_dir}[/red]")
        raise typer.Exit(1)

    state_list = [s.strip() for s in states.split(",")] if states else None

    state_dirs = [d for d in model_dir.iterdir() if d.is_dir()]
    if state_list:
        state_dirs = [d for d in state_dirs if d.name in state_list]

    if not state_dirs:
        console.print(f"[red]No trained models found in {model_dir}[/red]")
        raise typer.Exit(1)

    console.print(f"[cyan]Generating performance report for {len(state_dirs)} states...[/cyan]")

    loader = CompetitionDataLoader()
    preprocessor = DataPreprocessor()
    feature_engineer = FeatureEngineer()
    ocean_df = loader.load_ocean_data()

    all_results: dict[str, dict[str, dict[str, float]]] = {}
    model_names = ["xgboost", "lstm", "ensemble"]

    for state_dir in state_dirs:
        uf = state_dir.name
        console.print(f"[green]Evaluating {uf}...[/green]")

        df = loader.load_state_data(uf=uf, aggregate=True, end_date=end_date)
        df = preprocessor.clean(df)
        df = preprocessor.impute_missing(df)
        df = preprocessor.add_epidemiological_features(df)
        df = feature_engineer.build_feature_set(
            df, target_col="casos", ocean_df=ocean_df if not ocean_df.empty else None
        )

        forecast_file = forecast_dir / f"{uf}_forecast.csv"
        if not forecast_file.exists():
            console.print(f"  [yellow]⚠ No forecast file for {uf}[/yellow]")
            continue

        ensemble_forecast_df = pd.read_csv(forecast_file)
        if "date" in ensemble_forecast_df.columns:
            ensemble_forecast_df["date"] = pd.to_datetime(ensemble_forecast_df["date"])

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])

        common_dates_list = list(set(df["date"].values) & set(ensemble_forecast_df["date"].values))

        if not common_dates_list:
            console.print(f"  [yellow]⚠ No overlapping dates for {uf}[/yellow]")
            continue

        df_eval = df[df["date"].isin(common_dates_list)].sort_values("date")  # type: ignore[call-overload]
        y_true = np.asarray(df_eval["casos"].values)

        state_results: dict[str, dict[str, float]] = {}

        for model_name in model_names[:-1]:
            model_path = state_dir / model_name
            if not model_path.exists():
                continue

            if model_name == "xgboost":
                model = XGBoostForecaster()
            elif model_name == "lstm":
                model = LSTMForecaster()
            else:
                continue

            try:
                model.load(model_path)
                if model_name == "xgboost":
                    forecast = model.predict(df, levels=[0.50, 0.80, 0.95])  # type: ignore[call-arg]
                else:
                    forecast = model.predict(df, n_mc_samples=100)  # type: ignore[call-arg]
                if "date" in forecast.columns:
                    forecast["date"] = pd.to_datetime(forecast["date"])
                common_dates_list_local = list(forecast["date"].values)
                forecast_eval = forecast[
                    forecast["date"].isin(common_dates_list_local)
                ].sort_values("date")  # type: ignore[call-overload]
                metrics = evaluate_forecast(np.asarray(y_true), forecast_eval)
                state_results[model_name] = metrics
            except Exception as e:
                console.print(f"  [yellow]⚠ Failed to evaluate {model_name} for {uf}: {e}[/yellow]")

        if len(state_results) >= 2:
            ensemble_forecaster = EnsembleForecaster(method="weighted_average")
            for name in state_results:
                if name == "xgboost":
                    model = XGBoostForecaster()
                    model.load(state_dir / name)
                    forecast = model.predict(df, levels=[0.50, 0.80, 0.95])
                elif name == "lstm":
                    model = LSTMForecaster()
                    model.load(state_dir / name)
                    forecast = model.predict(df, n_mc_samples=100)
                else:
                    continue
                if "date" in forecast.columns:
                    forecast["date"] = pd.to_datetime(forecast["date"])
                ensemble_forecaster.add_model(name, forecast)
            ensemble_pred = ensemble_forecaster.predict()
            if "date" in ensemble_pred.columns:  # type: ignore[union-attr]
                ensemble_pred["date"] = pd.to_datetime(ensemble_pred["date"])  # type: ignore[index]
            ensemble_eval = ensemble_pred[
                ensemble_pred["date"].isin(common_dates_list)  # type: ignore[index, union-attr]
            ].sort_values("date")  # type: ignore[call-overload]
            state_results["ensemble"] = evaluate_forecast(np.asarray(y_true), ensemble_eval)  # type: ignore[arg-type]

        if state_results:
            all_results[uf] = state_results

    if not all_results:
        console.print("[red]No evaluation results generated[/red]")
        raise typer.Exit(1)

    lines = [
        "# Forecast Performance Report",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**States Evaluated:** {len(all_results)}",
        f"**Models Compared:** {', '.join(model_names)}",
        "",
        "## Executive Summary",
        "",
    ]

    aggregated: dict[str, dict[str, list[float]]] = {m: {} for m in model_names}
    for _uf, state_res in all_results.items():
        for model_name, metrics in state_res.items():
            for metric, value in metrics.items():
                if metric not in aggregated[model_name]:
                    aggregated[model_name][metric] = []
                aggregated[model_name][metric].append(value)

    avg_metrics: dict[str, dict[str, float]] = {}
    for model_name, metrics_dict in aggregated.items():
        avg_metrics[model_name] = {}
        for metric, values in metrics_dict.items():
            avg_metrics[model_name][metric] = float(np.mean(values))

    lines.append("### Average Performance Across All States")
    lines.append("")
    lines.append(
        "| Model | CRPS | WIS Total | RMSE | MAE | MAPE | Bias | Coverage 95% | Coverage 50% |"
    )
    lines.append(
        "|-------|------|-----------|------|-----|------|------|--------------|--------------|"
    )

    for model_name in model_names:
        if model_name not in avg_metrics:
            continue
        m = avg_metrics[model_name]
        lines.append(
            f"| {model_name.upper()} | "
            f"{m.get('crps', float('nan')):.4f} | "
            f"{m.get('wis_total', float('nan')):.4f} | "
            f"{m.get('rmse', float('nan')):.4f} | "
            f"{m.get('mae', float('nan')):.4f} | "
            f"{m.get('mape', float('nan')):.2f}% | "
            f"{m.get('bias', float('nan')):.4f} | "
            f"{m.get('coverage_95', 0):.2%} | "
            f"{m.get('coverage_50', 0):.2%} |"
        )

    lines.append("")

    best_by_metric: dict[str, tuple[str, float]] = {}
    metrics_to_rank = ["crps", "wis_total", "rmse", "mae", "mape"]
    for metric in metrics_to_rank:
        best_model = None
        best_value = float("inf")
        for model_name in model_names:
            if model_name in avg_metrics and metric in avg_metrics[model_name]:
                val = avg_metrics[model_name][metric]
                if val < best_value:
                    best_value = val
                    best_model = model_name
        if best_model:
            best_by_metric[metric] = (best_model, best_value)

    lines.append("### Best Model by Metric")
    lines.append("")
    lines.append("| Metric | Best Model | Value |")
    lines.append("|--------|------------|-------|")
    for metric, (best_model, value) in best_by_metric.items():
        lines.append(f"| {metric.upper()} | {best_model.upper()} | {value:.4f} |")
    lines.append("")

    lines.append("## Detailed Results by State")
    lines.append("")

    for uf in sorted(all_results.keys()):
        state_res = all_results[uf]
        lines.append(f"### {uf}")
        lines.append("")
        lines.append(
            "| Model | CRPS | WIS Total | RMSE | MAE | MAPE | Bias | Coverage 95% | Coverage 50% |"
        )
        lines.append(
            "|-------|------|-----------|------|-----|------|------|--------------|--------------|"
        )

        for model_name in model_names:
            if model_name not in state_res:
                continue
            m = state_res[model_name]
            lines.append(
                f"| {model_name.upper()} | "
                f"{m.get('crps', float('nan')):.4f} | "
                f"{m.get('wis_total', float('nan')):.4f} | "
                f"{m.get('rmse', float('nan')):.4f} | "
                f"{m.get('mae', float('nan')):.4f} | "
                f"{m.get('mape', float('nan')):.2f}% | "
                f"{m.get('bias', float('nan')):.4f} | "
                f"{m.get('coverage_95', 0):.2%} | "
                f"{m.get('coverage_50', 0):.2%} |"
            )
        lines.append("")

    lines.append("## Metric Definitions")
    lines.append("")
    lines.append("| Metric | Description |")
    lines.append("|--------|-------------|")
    lines.append(
        "| **CRPS** | Continuous Ranked Probability Score - measures the integrated squared difference between the empirical CDF and the predicted CDF. Lower is better. |"
    )
    lines.append(
        "| **WIS Total** | Weighted Interval Score - combines interval sharpness with penalty for misses across multiple confidence levels. Lower is better. |"
    )
    lines.append(
        "| **RMSE** | Root Mean Square Error - square root of the average of squared errors. Lower is better. |"
    )
    lines.append(
        "| **MAE** | Mean Absolute Error - average of absolute prediction errors. Lower is better. |"
    )
    lines.append(
        "| **MAPE** | Mean Absolute Percentage Error - percentage error relative to actual values. Lower is better. |"
    )
    lines.append(
        "| **Bias** | Systematic error (mean of predictions minus actual). Positive = overestimation, Negative = underestimation. |"
    )
    lines.append(
        "| **Coverage 95%** | Percentage of true values within the 95% prediction interval. Ideal: ~95%. |"
    )
    lines.append(
        "| **Coverage 50%** | Percentage of true values within the 50% prediction interval. Ideal: ~50%. |"
    )
    lines.append("")

    output = Path(output)
    output.write_text("\n".join(lines))
    console.print(f"\n[green]✓ Report generated: {output}[/green]")


def main() -> None:
    """Entry point for CLI."""
    app()
