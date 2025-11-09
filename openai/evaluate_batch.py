import os
import sys
import json
import argparse
import time
import asyncio
from datetime import datetime
from pathlib import Path

import pandas as pd

try:
    import openai  # type: ignore
except Exception as e:  # pragma: no cover
    sys.exit("openai is required. Install with: pip install openai")

try:
    import tiktoken  # type: ignore
except Exception as e:  # pragma: no cover
    sys.exit("tiktoken is required for token counting. Install with: pip install tiktoken")

# Ensure we can import from repo root (for check.py)
sys.path.append('.')
from check import evaluate_answers, _extract_json_payload  # reuse existing logic


# Default models for OpenAI-compatible servers
MODELS = [
    "gpt-4o-mini",
]

NTRIES = 1




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


def build_prompt_from_polys(polys: list[dict]) -> str:
    """Construct the exact prompt text used in generate.py for a list of polynomials.

    This mirrors generate.py's prompt format so the model sees the same instructions.
    """
    lines = [
        "Provide the canonical factored form for each expanded polynomial below.\n",
    ]
    for idx, entry in enumerate(polys, start=1):
        lines.append(f"{idx}) {entry['shuffled']}")

    lines += (
        "\nDO NOT USE ANY TOOL! Use Python expressions for factored forms, use the same symbols from the expanded form, and answer with valid JSON only like:",
        "```",
        '["<<<factored form>>>", ...]',
        "```\n",
    )

    return "\n".join(lines) + "\n"


async def run_model_async(
    client: "openai.AsyncOpenAI",
    model: str,
    prompt: str,
    temperature: float | None,
    seed: int | None,
    num_ctx: int | None,
    encoding_name: str = "cl100k_base",
) -> dict:
    """Send a single prompt concurrently to an OpenAI-compatible Chat Completions API.

    Streaming is disabled for batch evaluation; collects full response.
    Returns a dict with: model, response, latency_s, error, token counts.
    """
    enc = _get_encoding(encoding_name)
    prompt_tokens = _count_tokens(enc, prompt)

    t0 = time.perf_counter()
    error = None
    content = ""
    gen_tokens = 0

    messages = [{"role": "user", "content": prompt}]

    # Build request kwargs (only include provided params)
    req_kwargs: dict = {"model": model, "messages": messages}
    if temperature is not None:
        req_kwargs["temperature"] = float(temperature)
    if seed is not None:
        req_kwargs["seed"] = int(seed)
    if num_ctx is not None:
        req_kwargs["max_tokens"] = int(num_ctx)

    try:
        resp = await client.chat.completions.create(stream=False, **req_kwargs)
        try:
            reasoning = resp.choices[0].message.reasoning or ""
        except Exception:
            reasoning = ""
        try:
            content = resp.choices[0].message.content or ""
        except Exception:
            content = ""
        gen_tokens = _count_tokens(enc, reasoning + content)
    except Exception as exc:  # pragma: no cover
        error = f"{type(exc).__name__}: {exc}"

    latency = time.perf_counter() - t0

    return {
        "model": model,
        "response": content,
        "latency_s": latency,
        "error": error,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": gen_tokens,
        "first_token_s": None,
        "tok_per_s": None,
    }


async def main_async():
    parser = argparse.ArgumentParser(description="Evaluate OpenAI-compatible models on factorization task (per-polynomial batch calls).")
    parser.add_argument("--models", type=str, default=','.join(MODELS), help=f"Comma-separated list of model names (default: {MODELS})")
    parser.add_argument("--ntries", type=int, default=NTRIES, help=f"Max retries per model when accuracy < 1.0 (default: {NTRIES})")
    # Intentionally no --prompt here; we rebuild prompts per polynomial
    parser.add_argument("--polys", type=str, default="polynomials.json", help="Path to polynomials.json (default: polynomials.json)")
    parser.add_argument("--csv", type=str, default="results.csv", help="Path to write CSV results (default: results.csv)")
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature (optional)")
    parser.add_argument("--seed", type=int, default=None, help="Sampling seed (optional)")
    parser.add_argument("--num-ctx", type=int, dest="num_ctx", default=None, help="If set, used as max_tokens for output (optional)")
    parser.add_argument("--encoding", type=str, default="cl100k_base", help="tiktoken encoding for token counts (default: cl100k_base)")
    args = parser.parse_args()

    polys_path = Path(args.polys)
    csv_path = Path(args.csv)

    if not polys_path.exists():
        sys.exit(f"Ground-truth file not found: {polys_path}")

    polys = json.loads(polys_path.read_text(encoding="utf-8"))
    if not isinstance(polys, list):
        sys.exit("polynomials.json must contain a JSON array")

    models = [m.strip() for m in args.models.split(',') if m.strip()]
    if not models:
        sys.exit("No models specified via --models")

    records: list[dict] = []
    now_iso = datetime.now().isoformat() + "Z"

    # Pre-construct the per-polynomial prompts so each attempt reuses identical inputs
    per_prompts: list[str] = [build_prompt_from_polys([p]) for p in polys]

    # Configure AsyncOpenAI client (supports custom base URL)
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_TOKEN") or "EMPTY"
    aclient = openai.AsyncOpenAI(base_url=base_url, api_key=api_key)

    for model in models:
        max_retries = max(1, int(args.ntries))
        for attempt in range(1, max_retries + 1):
            # Submit the batch of prompts concurrently (one API call per polynomial), no streaming
            tasks = [
                run_model_async(
                    aclient,
                    model,
                    prompt,
                    args.temperature,
                    args.seed,
                    args.num_ctx,
                    encoding_name=args.encoding,
                )
                for prompt in per_prompts
            ]

            runs = await asyncio.gather(*tasks, return_exceptions=False)

            answers: list[str] = []
            total_latency = 0.0
            total_prompt_tokens = 0
            total_generated_tokens = 0
            per_errors: list[str] = []

            for idx, run in enumerate(runs):
                total_latency += float(run.get("latency_s", 0.0) or 0.0)
                total_prompt_tokens += int(run.get("prompt_tokens", 0) or 0)
                total_generated_tokens += int(run.get("generated_tokens", 0) or 0)

                if run.get("error"):
                    per_errors.append(f"i={idx}: {run['error']}")
                    continue

                raw = run.get("response", "")
                try:
                    payload = _extract_json_payload(raw)
                    # Accept either ["ans"] or {answers:["ans"]} or "ans"
                    if isinstance(payload, list):
                        if len(payload) == 0:
                            per_errors.append(f"i={idx}: empty list answer")
                            continue
                        ans = payload[0]
                    elif isinstance(payload, dict):
                        arr = payload.get("answers") or payload.get("result") or payload.get("results") or payload.get("polynomials") or payload.get("data")
                        if isinstance(arr, list) and arr:
                            ans = arr[0]
                        else:
                            per_errors.append(f"i={idx}: invalid object answer")
                            continue
                    elif isinstance(payload, str):
                        ans = payload
                    else:
                        per_errors.append(f"i={idx}: unsupported payload type {type(payload).__name__}")
                        continue

                    if not isinstance(ans, (str, dict)):
                        per_errors.append(f"i={idx}: unsupported answer type {type(ans).__name__}")
                        continue

                    # If dict, expect {"factored": "..."}
                    if isinstance(ans, dict):
                        ans_val = str(ans.get("factored", "")).strip()
                    else:
                        ans_val = str(ans).strip()

                    answers.append(ans_val)
                except json.JSONDecodeError as exc:
                    per_errors.append(f"i={idx}: {type(exc).__name__}: {exc}")
                except Exception as exc:  # pragma: no cover
                    per_errors.append(f"i={idx}: {type(exc).__name__}: {exc}")

            # Build combined raw response as JSON array string (for parity with single-shot mode)
            combined_response = json.dumps(answers, ensure_ascii=False)

            # Evaluate answers against ground truth
            try:
                report = evaluate_answers(answers, polys)
                summary = report["summary"]
                total = max(1, int(summary.get("total_expected", 0)))
                accuracy = (int(summary.get("exact", 0)) + int(summary.get("equivalent", 0))) / total
                status = "ok" if not per_errors else "partial"
                err_str = "; ".join(per_errors) if per_errors else None
            except Exception as exc:  # pragma: no cover
                status = "eval_error"
                summary = {"total_expected": len(polys), "total_received": None, "exact": 0, "equivalent": 0, "wrong": len(polys)}
                accuracy = 0.0
                err_str = f"{type(exc).__name__}: {exc}"

            # Aggregate metrics across per-polynomial calls
            tok_per_s = (float(total_generated_tokens) / total_latency) if (total_latency > 0 and total_generated_tokens > 0) else None

            record = {
                "timestamp": now_iso,
                "model": model,
                "attempt": attempt,
                "status": status,
                "latency_s": round(float(total_latency), 3),
                "prompt_tokens": int(total_prompt_tokens),
                "generated_tokens": int(total_generated_tokens),
                "first_token_s": None,
                "tok_per_s": tok_per_s,
                "total_expected": summary.get("total_expected"),
                "total_received": summary.get("total_received"),
                "exact": summary.get("exact"),
                "equivalent": summary.get("equivalent"),
                "wrong": summary.get("wrong"),
                "accuracy": round(float(accuracy), 6),
                "response": combined_response,
                "error": err_str,
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


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
