"""LLM demonstration with Gemini API.

This script mirrors the artifact contract of train.py and train_deep.py:
- response.txt (full response text)
- metadata.json (token counts, latency, timestamps)
- params.json (config parameters used)
- bundle_info.json (run metadata)
- runs/summary.csv (append row)

Usage:
    python llm_demo.py --config configs/llm/gemini.yaml \
                       --prompt-config configs/prompts/titanic_classification.yaml \
                       --query "Female, Age 28, Ticket Class 1"

Environment:
    Set GEMINI_API_KEY environment variable or pass via config.
    Get free key at: https://ai.google.dev/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.models.gemini_client import GeminiClient
from src.utils.common import ensure_dir, load_yaml


def load_api_key_from_dotenv(env_name: str) -> str | None:
    """Load API key from project-level .env file when available."""
    dotenv_path = Path(__file__).resolve().parent / ".env"
    if not dotenv_path.exists():
        return None

    for line in dotenv_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith(f"{env_name}="):
            value = stripped.split("=", 1)[1].strip().strip('"').strip("'")
            return value or None
    return None


def main() -> None:
    """Main entry point for LLM demonstration."""
    parser = argparse.ArgumentParser(
        description="Demonstrate Gemini LLM with YAML-driven config",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Classification demo
  python llm_demo.py --config configs/llm/gemini.yaml \\
                     --prompt-config configs/prompts/titanic_classification.yaml \\
                     --query "Male, Age 45, Ticket Class 3"

  # Model explanation
  python llm_demo.py --config configs/llm/gemini.yaml \\
                     --prompt-config configs/prompts/model_explanation.yaml \\
                     --query "random forest classifier"

  # Code review
  python llm_demo.py --config configs/llm/gemini.yaml \\
                     --prompt-config configs/prompts/code_review.yaml \\
                     --query "def train(X, y):\\n    model = LogisticRegression()\\n    return model.fit(X, y)"
        """,
    )
    parser.add_argument("--config", required=True, help="Path to configs/llm/*.yaml")
    parser.add_argument(
        "--prompt-config",
        required=True,
        help="Path to configs/prompts/*.yaml",
    )
    parser.add_argument(
        "--query",
        required=True,
        help="User query to send to Gemini",
    )
    parser.add_argument(
        "--run-name",
        help="Custom run name (auto-generated if not provided)",
    )
    args = parser.parse_args()

    # Load configurations
    llm_cfg = load_yaml(args.config)
    prompt_cfg = load_yaml(args.prompt_config)

    # Generate run name if not provided
    if args.run_name:
        run_name = args.run_name
    else:
        task_name = prompt_cfg.get("task_name", "llm_run")
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_name = f"{task_name}_{timestamp}"

    # Initialize Gemini client
    env_name = llm_cfg["paths"]["api_key_env"]
    api_key = os.getenv(env_name) or load_api_key_from_dotenv(env_name)
    if not api_key:
        print(
            f"ERROR: {env_name} environment variable not set.\n"
            f"Tip: set {env_name} in your shell or add it to a local .env file.\n"
            "Get free key at: https://ai.google.dev/"
        )
        sys.exit(1)

    # Get generation parameters with potential overrides from prompt config
    gen_params = llm_cfg.get("generation_params", {})
    if "generation_override" in prompt_cfg:
        gen_params.update(prompt_cfg["generation_override"])

    client = GeminiClient(
        api_key=api_key,
        model_name=llm_cfg.get("model_name", "gemini-2.0-flash"),
        temperature=gen_params.get("temperature", 0.7),
        max_tokens=gen_params.get("max_tokens", 1024),
    )

    # Build user prompt from template
    user_prompt = prompt_cfg.get("user_prompt_template", "{query}").format(
        query=args.query,
        concept=args.query,  # Support both placeholders
        passenger_info=args.query,
        code=args.query,
    )

    system_prompt = prompt_cfg.get("system_prompt", None)

    print(f"[{run_name}] Calling Gemini API...")
    print(f"  Model: {llm_cfg['model_name']}")
    print(f"  Task: {prompt_cfg['task_name']}")
    print(f"  Temperature: {gen_params.get('temperature', 0.7)}")
    print()

    # Call Gemini
    response = client.generate(
        prompt=user_prompt,
        system_prompt=system_prompt,
        temperature=gen_params.get("temperature"),
        max_tokens=gen_params.get("max_tokens"),
    )

    # Create run directory
    run_dir = ensure_dir(Path(llm_cfg["paths"]["runs_dir"]) / run_name)

    # Save response text
    response_file = llm_cfg["artifacts"]["response_file"]
    (run_dir / response_file).write_text(response.text, encoding="utf-8")

    # Save metadata (token counts, latency)
    metadata_file = llm_cfg["artifacts"]["metadata_file"]
    metadata = {
        "model": response.model,
        "timestamp": response.timestamp,
        "latency_ms": response.latency_ms,
        "prompt_tokens": response.prompt_tokens,
        "response_tokens": response.response_tokens,
        "total_tokens": response.total_tokens,
        "task": prompt_cfg.get("task_name"),
        "query": args.query,
    }
    (run_dir / metadata_file).write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # Save parameters
    params_file = llm_cfg["artifacts"]["params_file"]
    params = {
        "llm_config": args.config,
        "prompt_config": args.prompt_config,
        "model_name": llm_cfg["model_name"],
        "temperature": gen_params.get("temperature", 0.7),
        "max_tokens": gen_params.get("max_tokens", 1024),
    }
    (run_dir / params_file).write_text(json.dumps(params, indent=2), encoding="utf-8")

    # Save bundle info
    bundle_info = {
        "run_name": run_name,
        "task_name": prompt_cfg.get("task_name"),
        "model": llm_cfg["model_name"],
        "prompt_config": args.prompt_config,
        "llm_config": args.config,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "files": {
            "response": response_file,
            "metadata": metadata_file,
            "params": params_file,
        },
    }
    bundle_info_file = llm_cfg["artifacts"]["bundle_info_file"]
    (run_dir / bundle_info_file).write_text(
        json.dumps(bundle_info, indent=2),
        encoding="utf-8",
    )

    # Update runs/summary.csv
    summary_path = Path(llm_cfg["paths"]["runs_dir"]) / "summary.csv"
    summary_row = {
        "run_name": run_name,
        "task": prompt_cfg.get("task_name"),
        "model": llm_cfg["model_name"],
        "timestamp": response.timestamp,
        "latency_ms": response.latency_ms,
        "status": "success" if "Error" not in response.text else "error",
    }

    if summary_path.exists():
        df = pd.read_csv(summary_path)
        df = pd.concat([df, pd.DataFrame([summary_row])], ignore_index=True)
    else:
        df = pd.DataFrame([summary_row])
    df.to_csv(summary_path, index=False)

    # Print results
    print("=" * 70)
    print(f"Response ({response.latency_ms} ms):")
    print("=" * 70)
    print(response.text)
    print("=" * 70)
    print(f"\nArtifacts saved to: {run_dir}")
    print(f"  - {response_file}")
    print(f"  - {metadata_file}")
    print(f"  - {params_file}")
    print(f"  - {bundle_info_file}")


if __name__ == "__main__":
    main()
