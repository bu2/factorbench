import os
import sys
import json
import argparse
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

try:
    import openai  # type: ignore
except Exception as e:  # pragma: no cover
    sys.exit("openai is required for token counting. Install with: pip install openai")

try:
    import tiktoken  # type: ignore
except Exception as e:  # pragma: no cover
    sys.exit("tiktoken is required for token counting. Install with: pip install tiktoken")

# Ensure we can import from repo root (for check.py)
sys.path.append('.')
from check import evaluate_answers, _extract_json_payload  # reuse existing logic


# Default models for OpenAI-compatible servers
MODELS = [
    "gpt-4o-mini"
]

NTRIES = 1


# ANSI styles for terminal output
ANSI_RESET = "\033[0m"
ANSI_BOLD = "\033[1m"
ANSI_GREY = "\033[90m"


def _get_encoding(name: str):
    try:
        return tiktoken.get_encoding(name)
    except Exception:
        return tiktoken.get_encoding("cl100k_base")


def _count_tokens(encoding, text: str) -> int:
    try:
        return len(encoding.encode(text))
    except Exception:
        return 0


def run_model(
    model: str,
    prompt: str,
    temperature: float | None,
    seed: int | None,
    num_ctx: int | None,
    stream: bool = True,
    high: bool = False,
    low: bool = False,
    encoding_name: str = "cl100k_base",
) -> dict:
    """Send prompt to an OpenAI-compatible Chat Completions API and return a result dict.

    Streams tokens to stdout when stream=True and collects the full response.

    Returns a dict with: model, response, latency_s, error.
    """
    print(f"\n=== Running model '{model}' (temperature={temperature}, seed={seed}) ===")
    enc = _get_encoding(encoding_name)
    prompt_tokens = _count_tokens(enc, prompt)
    print(f"Prompt tokens: {prompt_tokens}")

    # Configure OpenAI client (supports custom base URL)
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_TOKEN") or "EMPTY"

    client = openai.OpenAI(base_url=base_url, api_key=api_key)

    t0 = time.perf_counter()
    error = None
    content = ""
    t_first = None
    gen_tokens = 0

    messages = [{"role": "user", "content": prompt}]

    # Build request kwargs (only include provided params)
    req_kwargs: dict = {"model": model, "messages": messages}
    if temperature is not None:
        req_kwargs["temperature"] = float(temperature)
    if seed is not None:
        # Supported in newer OpenAI-compatible servers; ignored otherwise
        req_kwargs["seed"] = int(seed)
    if num_ctx is not None:
        # Many OpenAI-compatible servers use max_tokens to cap output length
        req_kwargs["max_tokens"] = int(num_ctx)
    if high:
        req_kwargs["reasoning_effort"] = "high"
    if low:
        # If both are set, low overrides high by assignment order
        req_kwargs["reasoning_effort"] = "low"

    try:
        if stream:
            parts: list[str] = []
            all_parts: list[str] = []  # accumulate reasoning + final for token counting
            use_color = sys.stdout.isatty()
            first_real_token = True
            for chunk in client.chat.completions.create(stream=True, **req_kwargs):
                choice = chunk.choices[0]
                delta = getattr(choice, "delta", None)
                reasoning = getattr(delta, "reasoning", None) if delta is not None else None
                if reasoning:
                    if t_first is None:
                        t_first = time.perf_counter() - t0
                    if use_color:
                        print(f"{ANSI_GREY}{reasoning}{ANSI_RESET}", end="", flush=True)
                    else:
                        print(str(reasoning), end="", flush=True)
                    all_parts.append(str(reasoning))
                token = getattr(delta, "content", None) if delta is not None else None
                if token:
                    if t_first is None:
                        t_first = time.perf_counter() - t0
                    if first_real_token:
                        first_real_token = False
                        print()
                    if use_color:
                        print(f"{ANSI_BOLD}{token}{ANSI_RESET}", end="", flush=True)
                    else:
                        print(token, end="", flush=True)
                    parts.append(token)
                    all_parts.append(str(reasoning))
            print()
            content = "".join(parts)
            gen_text = "".join(all_parts)
            gen_tokens = _count_tokens(enc, gen_text)
        else:
            resp = client.chat.completions.create(stream=False, **req_kwargs)
            try:
                reasoning = resp.choices[0].message.reasoning or ""
            except Exception:
                reasoning = ""
            try:
                content = resp.choices[0].message.content or ""
            except Exception:
                content = ""
            t_first = None
            gen_tokens = _count_tokens(enc, reasoning + content)
    except Exception as exc:  # pragma: no cover
        error = f"{type(exc).__name__}: {exc}"

    latency = time.perf_counter() - t0
    t_first_token_s = float(t_first) if t_first is not None else None
    if t_first_token_s is not None and latency > t_first_token_s:
        gen_time = max(1e-9, latency - t_first_token_s)
        tok_per_s = gen_tokens / gen_time if gen_tokens else 0.0
    else:
        tok_per_s = None
    print(
        f"--- Completed in {latency:.2f}s | prompt_tokens={prompt_tokens} | generated_tokens={gen_tokens} | first_token={t_first_token_s if t_first_token_s is not None else 'n/a'}s | speed={tok_per_s if tok_per_s is not None else 'n/a'} tok/s ---\n"
    )
    return {
        "model": model,
        "response": content,
        "latency_s": latency,
        "error": error,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": gen_tokens,
        "first_token_s": t_first_token_s,
        "tok_per_s": tok_per_s,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate OpenAI-compatible models on factorization task.")
    parser.add_argument("--models", type=str, default=','.join(MODELS), help=f"Comma-separated list of model names (default: {MODELS})")
    parser.add_argument("--ntries", type=int, default=NTRIES, help=f"Max retries per model when accuracy < 1.0 (default: {NTRIES})")
    parser.add_argument("--prompt", type=str, default="prompt.txt", help="Path to prompt.txt (default: prompt.txt)")
    parser.add_argument("--polys", type=str, default="polynomials.json", help="Path to polynomials.json (default: polynomials.json)")
    parser.add_argument("--csv", type=str, default="results.csv", help="Path to write CSV results (default: results.csv)")
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature (optional)")
    parser.add_argument("--seed", type=int, default=None, help="Sampling seed (optional)")
    parser.add_argument("--num-ctx", type=int, dest="num_ctx", default=None, help="If set, used as max_tokens for output (optional)")
    parser.add_argument("--encoding", type=str, default="cl100k_base", help="tiktoken encoding for token counts (default: cl100k_base)")
    parser.add_argument("--high", action="store_true", help="Enable high reasoning effort for models that support it")
    parser.add_argument("--low", action="store_true", help="Enable low reasoning effort for models that support it")
    args = parser.parse_args()

    prompt_path = Path(args.prompt)
    polys_path = Path(args.polys)
    csv_path = Path(args.csv)

    if not prompt_path.exists():
        sys.exit(f"Prompt file not found: {prompt_path}")
    if not polys_path.exists():
        sys.exit(f"Ground-truth file not found: {polys_path}")

    prompt = prompt_path.read_text(encoding="utf-8")
    polys = json.loads(polys_path.read_text(encoding="utf-8"))

    models = [m.strip() for m in args.models.split(',') if m.strip()]
    if not models:
        sys.exit("No models specified via --models")

    records: list[dict] = []
    now_iso = datetime.now().isoformat() + "Z"

    for model in models:
        max_retries = max(1, int(args.ntries))
        for attempt in range(1, max_retries + 1):
            run = run_model(
                model,
                prompt,
                args.temperature,
                args.seed,
                args.num_ctx,
                stream=True,
                high=args.high,
                low=args.low,
                encoding_name=args.encoding,
            )
            status = "ok" if not run["error"] else "error"

            if status == "ok":
                raw = run["response"]
                try:
                    answers_obj = _extract_json_payload(raw)
                    report = evaluate_answers(answers_obj, polys)
                    summary = report["summary"]
                    total = max(1, int(summary.get("total_expected", 0)))
                    accuracy = (int(summary.get("exact", 0)) + int(summary.get("equivalent", 0))) / total
                except json.JSONDecodeError as exc:
                    status = "invalid_json"
                    summary = {"total_expected": len(polys), "total_received": None, "exact": 0, "equivalent": 0, "wrong": len(polys)}
                    accuracy = 0.0
                    run["error"] = f"{type(exc).__name__}: {exc}"
                except Exception as exc:  # pragma: no cover
                    status = "eval_error"
                    summary = {"total_expected": len(polys), "total_received": None, "exact": 0, "equivalent": 0, "wrong": len(polys)}
                    accuracy = 0.0
                    run["error"] = f"{type(exc).__name__}: {exc}"
            else:
                summary = {"total_expected": len(polys), "total_received": None, "exact": 0, "equivalent": 0, "wrong": len(polys)}
                accuracy = 0.0

            record = {
                "timestamp": now_iso,
                "model": model,
                "attempt": attempt,
                "status": status,
                "latency_s": round(float(run["latency_s"]), 3),
                "prompt_tokens": run.get("prompt_tokens"),
                "generated_tokens": run.get("generated_tokens"),
                "first_token_s": run.get("first_token_s"),
                "tok_per_s": run.get("tok_per_s"),
                "total_expected": summary.get("total_expected"),
                "total_received": summary.get("total_received"),
                "exact": summary.get("exact"),
                "equivalent": summary.get("equivalent"),
                "wrong": summary.get("wrong"),
                "accuracy": round(float(accuracy), 6),
                "response": run["response"],
                "error": run["error"],
            }
            print(f"Attempt {attempt}/{max_retries} accuracy={record['accuracy']:.3f} status={status}")
            records.append(record)

    df = pd.DataFrame.from_records(records)
    df.to_csv(csv_path, index=False)
    # Also print a compact summary
    cols = [
        "model",
        "attempt",
        "status",
        "exact",
        "equivalent",
        "wrong",
        "accuracy",
        "latency_s",
        "prompt_tokens",
        "generated_tokens",
        "first_token_s",
        "tok_per_s",
    ]
    present_cols = [c for c in cols if c in df.columns]
    print(df[present_cols].to_string(index=False))


if __name__ == "__main__":
    main()
