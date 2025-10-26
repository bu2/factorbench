import sys
import json
import argparse
from pathlib import Path

import sympy as sp
import re
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
)


TRANSFORMS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)


def _read_stdin_text() -> str:
    return sys.stdin.read()


def _extract_json_payload(text: str):
    """Try to parse JSON from raw text; falls back to fenced code block parsing."""
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    # Try to extract from a code fence ``` ... ``` optionally tagged as json
    import re

    m = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        inner = m.group(1)
        return json.loads(inner)

    # Last resort: raise
    raise SystemExit("Failed to parse JSON from stdin. Provide a JSON array of answers or wrap it in ```json fences.")


def _answers_list(obj):
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        # Common containers
        for key in ("answers", "result", "results", "polynomials", "data"):
            if key in obj and isinstance(obj[key], list):
                return obj[key]
    raise SystemExit("Input JSON must be a list of answers or a dict containing a list under 'answers'.")


def _parse_expr_safe(expr_str: str, local_map: dict):
    # First try a direct parse using the provided locals
    try:
        return parse_expr(expr_str, local_dict=local_map, transformations=TRANSFORMS, evaluate=True)
    except Exception:
        pass

    # Fallback: substitute any non-identifier variable names (e.g., emojis)
    # with temporary valid identifiers, then parse and remap to the original symbols.
    try:
        names = list(local_map.keys())
        if not names:
            return None
        # Map each original name to a safe temporary identifier
        safe_map = {name: f"v{i}" for i, name in enumerate(names)}
        # Build a regex that matches any of the variable names, longest-first to avoid partial matches
        # Match names only when not adjacent to identifier characters
        alternation = "|".join(re.escape(n) for n in sorted(names, key=len, reverse=True))
        pattern = re.compile(rf"(?<!\w)(?:{alternation})(?!\w)")
        replaced = pattern.sub(lambda m: safe_map[m.group(0)], expr_str)
        safe_locals = {safe_map[name]: local_map[name] for name in names}
        return parse_expr(replaced, local_dict=safe_locals, transformations=TRANSFORMS, evaluate=True)
    except Exception:
        return None


def _factor_signature(expr: sp.Expr, symbols: list[sp.Symbol]):
    """Return a canonical factorization signature: (coeff, [(monic_factor_srepr, exp), ...]).

    - Uses polynomial factorization over rationals.
    - Factors are made monic and sorted; all scalar content goes into coeff.
    - Returns None if the expression cannot be treated as a polynomial.
    """
    try:
        poly = sp.Poly(sp.expand(expr), *symbols, domain=sp.QQ)
    except Exception:
        return None

    try:
        coeff, factors = sp.factor_list(poly)
    except Exception:
        return None

    total_coeff = sp.Rational(coeff)
    pairs = []
    for f_expr, exp in factors:
        pf = sp.Poly(f_expr, *symbols, domain=sp.QQ)
        lc = pf.LC()
        # Make the factor monic; push its leading coefficient into the scalar coefficient
        pf_monic = pf.monic()
        total_coeff *= lc**int(exp)
        pairs.append((sp.srepr(pf_monic.as_expr()), int(exp)))

    pairs.sort(key=lambda t: (t[0], t[1]))
    return (sp.together(total_coeff), tuple(pairs))


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
    answers = _answers_list(payload)

    n_expected = len(polys)
    n_given = len(answers)
    if n_given != n_expected:
        print(f"Warning: expected {n_expected} answers, received {n_given}", file=sys.stderr)

    n = min(n_expected, n_given)

    results = []
    exact_cnt = 0
    equiv_cnt = 0
    wrong_cnt = 0

    for i in range(n):
        ans_item = answers[i]
        if isinstance(ans_item, str):
            ans_factored_str = ans_item
        elif isinstance(ans_item, dict):
            ans_factored_str = ans_item.get("factored", "")
        else:
            ans_factored_str = ""

        truth = polys[i]
        truth_factored_str = truth.get("factored", "")
        var_names = truth.get("variables", [])
        # Build symbol table for parsing
        local_map = {name: sp.Symbol(name) for name in var_names}

        strict_equal = ans_factored_str.strip() == truth_factored_str.strip()

        equivalent = False
        parse_error = None
        eq_mode = None
        if not strict_equal:
            # Parse both sides and check equivalence
            lhs = _parse_expr_safe(ans_factored_str, local_map)
            rhs = _parse_expr_safe(truth_factored_str, local_map)
            if lhs is None or rhs is None:
                parse_error = True
                equivalent = False
            else:
                # Prefer factor-multiset comparison (ignoring order)
                sig_lhs = _factor_signature(lhs, list(local_map.values()))
                sig_rhs = _factor_signature(rhs, list(local_map.values()))
                if sig_lhs is not None and sig_rhs is not None and sig_lhs == sig_rhs:
                    equivalent = True
                    eq_mode = "factor_multiset"
                else:
                    # Fall back to algebraic equivalence via expand/simplify
                    try:
                        equivalent = (sp.simplify(sp.expand(lhs - rhs)) == 0)
                        if equivalent:
                            eq_mode = "algebraic"
                    except Exception:
                        parse_error = True
                        equivalent = False

        status = "exact" if strict_equal else ("equivalent" if equivalent else "mismatch")
        exact_cnt += 1 if status == "exact" else 0
        equiv_cnt += 1 if status == "equivalent" else 0
        wrong_cnt += 1 if status == "mismatch" else 0

        results.append(
            {
                "index": i + 1,
                "strict_equal": strict_equal,
                "equivalent": True if strict_equal else equivalent,
                "status": status,
                **({"equivalence_mode": eq_mode} if (eq_mode and not strict_equal) else {}),
                # Include the raw strings for reference only when mismatched
                **(
                    {
                        "expected": truth_factored_str,
                        "got": ans_factored_str,
                    }
                    if status == "mismatch"
                    else {}
                ),
                **({"parse_error": True} if parse_error else {}),
            }
        )

    # Handle any extra answers provided beyond expected
    for j in range(n, n_given):
        wrong_cnt += 1
        results.append(
            {
                "index": j + 1,
                "strict_equal": False,
                "equivalent": False,
                "status": "extra_answer_no_ground_truth",
            }
        )

    # Handle any missing answers
    for j in range(n, n_expected):
        wrong_cnt += 1
        results.append(
            {
                "index": j + 1,
                "strict_equal": False,
                "equivalent": False,
                "status": "missing_answer",
            }
        )

    summary = {
        "total_expected": n_expected,
        "total_received": n_given,
        "exact": exact_cnt,
        "equivalent": equiv_cnt,
        "wrong": wrong_cnt,
    }

    print(json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2))

    # Exit non-zero if there are any wrong items
    sys.exit(0 if wrong_cnt == 0 else 1)


if __name__ == "__main__":
    main()
