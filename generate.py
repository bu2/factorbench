# random_factored_poly.py
import random
import string
import json
import argparse
from pathlib import Path
import sympy as sp


N = 3

NVARS = 1
NFACTORS = 2
DEGREES = 1
COEFF = 2


def random_factored_poly(
    n_vars=1,          # number of distinct variables
    n_factors=2,       # how many factors to multiply
    d=1,               # per-factor degree is chosen uniformly from 1..d
    coeff_range=(-3, 3),
    seed=None,
    alphabet=string.ascii_lowercase,     # sequence of allowed symbol names; defaults to ascii lowercase
):
    """
    Returns (symbols_tuple, factored_expr, expanded_expr).

    Each factor is a random univariate polynomial of exact degree k,
    where k ~ Uniform{1, ..., d}, in a randomly chosen variable.

    Parameters
    - n_vars: number of distinct variables to create.
    - alphabet: the pool of symbol names to choose from. Can be a string
      (characters are treated as individual symbols) or an iterable of
      strings (each entry is taken as a symbol name, e.g., ["α", "β", "γ"]).
      Defaults to lowercase English letters.
    """
    assert n_vars >= 1 and n_factors >= 1 and d >= 1
    rng = random.Random(seed)

    # Build the variable symbols from a configurable alphabet
    if alphabet is None:
        pool = list(string.ascii_lowercase)
    elif isinstance(alphabet, str):
        pool = list(alphabet)
    else:
        pool = list(alphabet)

    # Deduplicate while preserving order to ensure enough unique symbols
    pool_unique = list(dict.fromkeys(pool))
    if len(pool_unique) < n_vars:
        raise ValueError(
            f"alphabet must contain at least {n_vars} unique symbols; got {len(pool_unique)}"
        )

    chosen_names = rng.sample(pool_unique, k=n_vars)
    xs = tuple(sp.Symbol(name) for name in chosen_names)

    # Make sure every variable appears at least once when possible
    base = list(range(min(n_vars, n_factors)))
    rng.shuffle(base)
    rest = [rng.randrange(n_vars) for _ in range(n_factors - len(base))]
    var_indices = base + rest
    rng.shuffle(var_indices)

    c_lo, c_hi = coeff_range

    def rand_poly(v, deg):
        # exact-degree polynomial: ensure nonzero leading coefficient
        lead = 0
        while lead == 0:
            lead = rng.randint(c_lo, c_hi)
        coeffs = [rng.randint(c_lo, c_hi) for _ in range(deg)]  # c_0..c_{deg-1}
        poly = sp.Integer(lead) * v**deg + sum(sp.Integer(coeffs[j]) * v**j for j in range(deg))
        return sp.expand(poly)

    factors = [rand_poly(xs[var_indices[i]], rng.randint(1, d)) for i in range(n_factors)]

    # Canonical factored form (sorted, merged powers), plus expanded version if needed
    factored = sp.factor(sp.expand(sp.Mul(*factors)))
    expanded = sp.expand(factored)
    return xs, factored, expanded


if __name__ == "__main__":
    # Parse CLI options to override defaults
    parser = argparse.ArgumentParser(description="Generate random factored polynomials and prompts.")
    parser.add_argument("-n", type=int, default=N, help="Number of polynomials to generate (default: %(default)s)")
    parser.add_argument("--nvars", type=int, default=NVARS, help="Number of distinct variables (default: %(default)s)")
    parser.add_argument("--nfactors", type=int, default=NFACTORS, help="Number of factors to multiply (default: %(default)s)")
    parser.add_argument("--degrees", type=int, default=DEGREES, help="Max per-factor degree bound d; each factor uses degree in [1,d] (default: %(default)s)")
    parser.add_argument("--coeff", type=int, default=COEFF, help="Coefficient magnitude bound; samples from [-COEFF, COEFF] (default: %(default)s)")
    parser.add_argument(
        "--alphabet",
        type=str,
        default=None,
        help=(
            "Alphabet of symbols to sample variables from. "
            "Either a raw string of characters (e.g. 'xyz'), or a comma-separated list of names "
            "(e.g. 'α,β,γ'). If omitted, uses ascii lowercase."
        ),
    )
    args = parser.parse_args()

    N_val = args.n
    NVARS_val = args.nvars
    NFACTORS_val = args.nfactors
    DEGREES_val = args.degrees
    COEFF_val = args.coeff
    ALPHABET_arg = args.alphabet

    # Normalize the alphabet argument
    if ALPHABET_arg is None:
        alphabet_param = string.ascii_lowercase
    else:
        if "," in ALPHABET_arg:
            # Treat as comma-separated list of explicit names
            alphabet_param = [tok.strip() for tok in ALPHABET_arg.split(',') if tok.strip()]
        else:
            # Allow shortcuts for common sets
            if ALPHABET_arg.lower() in {"ascii_lowercase", "lower", "letters", "latin_lower"}:
                alphabet_param = string.ascii_lowercase
            elif ALPHABET_arg.lower() in {"ascii_uppercase", "upper", "latin_upper"}:
                alphabet_param = string.ascii_uppercase
            else:
                # Use raw string as the character pool
                alphabet_param = ALPHABET_arg

    if N_val < 1:
        raise SystemExit("N must be >= 1")
    if NVARS_val < 1:
        raise SystemExit("NVARS must be >= 1")
    if NFACTORS_val < 1:
        raise SystemExit("NFACTORS must be >= 1")
    if DEGREES_val < 1:
        raise SystemExit("DEGREES must be >= 1")
    if COEFF_val < 1:
        raise SystemExit("COEFF must be >= 1")

    polys = []
    for i in range(N_val):
        xs, F, P = random_factored_poly(
            n_vars=NVARS_val,
            n_factors=NFACTORS_val,
            d=DEGREES_val,
            coeff_range=(-COEFF_val, COEFF_val),
            alphabet=alphabet_param,
        )
        # Create a shuffled expanded-form by shuffling the addends of P
        addends = list(sp.Add.make_args(P))
        rng = random.Random(i)
        rng.shuffle(addends)

        # Build a textual representation that preserves the shuffled order
        addend_strs = [str(t) for t in addends]
        if addend_strs:
            shuffled_str = addend_strs[0]
            for term in addend_strs[1:]:
                if term.startswith('-'):
                    shuffled_str += ' - ' + term[1:]
                else:
                    shuffled_str += ' + ' + term
        else:
            shuffled_str = '0'

        # Verify equivalence by parsing the shuffled string back into a SymPy expr
        # using the same variable symbols, then compare to P
        local_map = {str(v): v for v in xs}
        try:
            parsed_from_str = sp.sympify(shuffled_str, locals=local_map)
            equivalent = (sp.simplify(sp.expand(parsed_from_str - P)) == 0)
        except Exception:
            equivalent = False

        # Safety net: if the string isn't equivalent, fall back to exact expanded
        if not equivalent:
            shuffled_str = str(P)

        polys.append({
            "variables": [str(v) for v in xs],
            "factored": str(F),
            "expanded": str(P),
            "shuffled": shuffled_str,
        })

    # Save the generated polynomials to JSON
    out_path = Path("polynomials.json")
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(polys, f, ensure_ascii=False, indent=2)

    # Output a prompt asking for canonical factored forms
    lines = [
        "Provide the canonical factored form for each expanded polynomial below.\n",
    ]
    for idx, entry in enumerate(polys, start=1):
        lines.append(f"{idx}) {entry['shuffled']}")

    lines += (
        "\nDO NOT USE ANY TOOL! Use Python expressions for factored forms, use the same symbols from the expanded form, and answer in JSON like:",
        "```",
        '["<<<factored form>>>", ...]',
        "```\n"
    )

    prompt = "\n".join(lines) + "\n"

    # Save the full prompt to a file before printing
    prompt_path = Path("prompt.txt")
    prompt_path.write_text(prompt, encoding="utf-8")

    # Print to stdout
    print(prompt, end="")
