import argparse
import json
import logging
from pathlib import Path
import re
import sys

import sympy as sp



def _read_stdin_text() -> str:
    return sys.stdin.read()


def _extract_json_payload(text: str):
    """Try to parse JSON from raw text; falls back to fenced code block parsing."""
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    m = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        inner = m.group(1)
        return json.loads(inner)

    # Last resort: raise
    raise json.JSONDecodeError("Failed to parse JSON from stdin. Provide a JSON array of answers or wrap it in ```json fences.")


def _answers_list(obj):
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        # Common containers
        for key in ("answers", "result", "results", "polynomials", "data"):
            if key in obj and isinstance(obj[key], list):
                return obj[key]
    raise json.JSONDecodeError("Input JSON must be a list of answers or a dict containing a list under 'answers'.")


def evaluate_answers(answers_obj, polys):
    """Evaluate answers against ground-truth polynomials.

    Parameters
    - answers_obj: list of answers or a dict containing a list (via _answers_list)
                   Each answer can be a string (factored expression) or an object
                   with a 'factored' field.
    - polys: list loaded from polynomials.json

    Returns a dict with keys: 'summary' and 'results'.
    """
    answers = _answers_list(answers_obj)

    n_expected = len(polys)
    n_given = len(answers)
    n = min(n_expected, n_given)

    def _parse(expr_str: str):
        return sp.parse_expr(expr_str)

    def _mul_shape(expr):
        """Top-level multiplication shape (no factoring):
        - ignore overall numeric coefficient
        - compare factor strings ignoring order
        Non-product expressions return a single-item shape.
        """
        expr = sp.simplify(expr)
        if expr.is_Mul:
            factors = []
            for a in expr.args:
                if getattr(a, 'is_Number', False):
                    continue
                factors.append(str(a))
            return tuple(sorted(factors))
        else:
            return (str(expr),)

    results = []
    counts = {
        "exact": 0,
        "equivalent": 0,
        "partially_factored": 0,
        "wrong": 0,
    }

    for i in range(n):
        answer_ = answers[i]
        if isinstance(answer_, str):
            answer = answer_.strip()
        elif isinstance(answer_, dict):
            answer = str(answer_.get("factored", "")).strip()
        else:
            answer = ""

        truth = polys[i]["factored"]

        status = "wrong"
        detail = None
        parsed_answer = None
        parsed_truth = None

        try:
            # 1) exact string match of SymPy representations
            if str(answer) == str(truth):
                status = "equal"
            else:
                parsed_truth = _parse(truth)
                parsed_answer = _parse(answer)

                # 2) same top-level product shape (same factors, any order)
                if _mul_shape(parsed_answer) == _mul_shape(parsed_truth):
                    status = "equivalent"
                else:
                    # 3) mathematically equivalent but not in factored form
                    same_value = sp.simplify(sp.expand(parsed_answer - parsed_truth)) == 0
                    if same_value:
                        status = "partially_factored"
                    else:
                        status = "wrong"
        except Exception as exc:
            status = "wrong"
            detail = f"parse_error: {type(exc).__name__}: {exc}"
            logging.exception(detail)

        # Count
        if status == "equal":
            counts["exact"] += 1
        elif status == "equivalent":
            counts["equivalent"] += 1
        elif status == "partially_factored":
            counts["partially_factored"] += 1
        else:
            counts["wrong"] += 1

        results.append({
            "index": i,
            "answer": answer,
            "expected": truth,
            "status": status,
            **({"detail": detail} if detail else {}),
        })

    summary = {
        "total_expected": n_expected,
        "total_received": n_given,
        "exact": counts["exact"],
        "equivalent": counts["equivalent"],
        # "wrong" should include both wrong kinds for aggregate metrics
        "partially_factored": counts["partially_factored"],
        "wrong": counts["wrong"] + counts["partially_factored"],
        "ok": counts["exact"] + counts["equivalent"],
    }

    return {"summary": summary, "results": results}

def main():
    parser = argparse.ArgumentParser(description="Check JSON answers against polynomials.json factored forms.")
    parser.add_argument("--polys", default="polynomials.json", help="Path to the ground-truth polynomials JSON (default: polynomials.json)")
    args = parser.parse_args()

    poly_path = Path(args.polys)
    if not poly_path.exists():
        raise SystemExit(f"Ground-truth file not found: {poly_path}")

    polys = json.loads(poly_path.read_text(encoding="utf-8"))
    if not isinstance(polys, list):
        raise SystemExit("polynomials.json must contain a JSON array")

    raw = _read_stdin_text()
    payload = _extract_json_payload(raw)
    # Produce report
    report = evaluate_answers(payload, polys)

    # Emit warning to stderr if counts mismatch (preserve script behavior)
    if report["summary"]["total_expected"] != report["summary"]["total_received"]:
        print(
            f"Warning: expected {report['summary']['total_expected']} answers, received {report['summary']['total_received']}",
            file=sys.stderr,
        )

    print(json.dumps(report, ensure_ascii=False, indent=2))

    # Exit non-zero if there are any wrong items
    sys.exit(0 if report["summary"]["wrong"] == 0 else 1)


if __name__ == "__main__":
    main()
