#!/usr/bin/env python3
"""Static checker for syntax and runtime errors in the simopt codebase.

Checks for:
  1. Syntax errors (via ast.parse)
  2. Cross-module import errors (names imported from simopt.* that don't exist)
  3. Type annotation errors (undefined names in annotations)
  4. Default value / annotation mismatches
  5. Duplicate function/class definitions
  6. Common runtime pitfalls (mutable default arguments, etc.)

Usage:
    python check_errors.py                  # check all simopt/**/*.py files
    python check_errors.py simopt/base.py   # check specific files
    python check_errors.py --verbose        # show per-file progress
"""

from __future__ import annotations

import ast
import sys
import textwrap
from collections import defaultdict
from fnmatch import fnmatch
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent
SIMOPT_ROOT = REPO_ROOT / "simopt"

# Patterns to skip (mirrors ruff.toml excludes + test/notebook files)
EXCLUDE_PATTERNS = [
    "simopt/gui/*",
    "simopt/models/ermexample.py",
    "simopt/solvers/active_subspaces/*",
    "simopt/solvers/Equadratures/*",
    "simopt/solvers/TrustRegion/*",
    "simopt/solvers/ASTROMoRF_old.py",
    "simopt/solvers/ASTROMoRF_OMoRF_Geo.py",
    "simopt/solvers/OMoRF.py",
    "simopt/solvers/GeometryImprovement.py",
    "simopt/linear_algebra_base.py",
    "simopt/HPC_code/*",
    "simopt/online_bandit/*",
]

# Well-known third-party and stdlib modules we won't try to resolve
EXTERNAL_MODULES = {
    "mrg32k3a",
    "numpy",
    "np",
    "scipy",
    "pandas",
    "pd",
    "matplotlib",
    "plt",
    "seaborn",
    "sns",
    "pydantic",
    "sklearn",
    "joblib",
    "boltons",
    "cvxpy",
    "PIL",
    "pillow",
    "tkinter",
    "tk",
    "typing",
    "typing_extensions",
    "collections",
    "abc",
    "enum",
    "dataclasses",
    "functools",
    "itertools",
    "math",
    "os",
    "sys",
    "pathlib",
    "json",
    "csv",
    "pickle",
    "copy",
    "re",
    "warnings",
    "logging",
    "traceback",
    "contextlib",
    "textwrap",
    "io",
    "time",
    "datetime",
    "hashlib",
    "inspect",
    "importlib",
    "subprocess",
    "shutil",
    "tempfile",
    "glob",
    "fnmatch",
    "argparse",
    "unittest",
    "pytest",
    "multiprocessing",
    "threading",
    "concurrent",
    "queue",
    "random",
    "statistics",
    "bisect",
    "heapq",
    "operator",
    "string",
    "struct",
    "ctypes",
    "zlib",
    "gzip",
    "zipfile",
    "tarfile",
    "configparser",
    "socket",
    "http",
    "urllib",
    "xml",
    "html",
    "weakref",
    "decimal",
    "fractions",
    "numbers",
    "cmath",
    "array",
    "types",
    "ast",
    "dis",
    "code",
    "pdb",
    "profile",
    "timeit",
    "platform",
    "signal",
    "site",
    "sysconfig",
    "builtins",
    "__future__",
}

# Python built-in names that are valid in annotations
# if isinstance(__builtins__, dict) :
#     BUILTIN_NAMES = set(dir(__builtins__))
# else :
#     set(dir(__builtins__))
BUILTIN_NAMES = set(dir(__builtins__))

# Standard typing names
TYPING_NAMES = {
    "Any",
    "Union",
    "Optional",
    "Literal",
    "Final",
    "ClassVar",
    "TypeVar",
    "TypeAlias",
    "Protocol",
    "runtime_checkable",
    "overload",
    "Self",
    "Never",
    "NoReturn",
    "Annotated",
    "TypeGuard",
    "TypedDict",
    "NamedTuple",
    "Generic",
    "Callable",
    "Iterator",
    "Generator",
    "Iterable",
    "Sequence",
    "Mapping",
    "MutableMapping",
    "MutableSequence",
    "Set",
    "FrozenSet",
    "List",
    "Dict",
    "Tuple",
    "Type",
    "Deque",
    "DefaultDict",
    "OrderedDict",
    "Counter",
    "ChainMap",
    "IO",
    "TextIO",
    "BinaryIO",
    "Pattern",
    "Match",
    "Awaitable",
    "Coroutine",
    "AsyncIterator",
    "AsyncGenerator",
    "SupportsInt",
    "SupportsFloat",
    "SupportsComplex",
    "SupportsBytes",
    "SupportsAbs",
    "SupportsRound",
    "Reversible",
    "Container",
    "Collection",
    "Hashable",
    "Sized",
    "TYPE_CHECKING",
    "cast",
    "assert_type",
    "reveal_type",
    "get_type_hints",
    "get_origin",
    "get_args",
    "is_typeddict",
    "dataclass_transform",
    "ParamSpec",
    "Concatenate",
    "Unpack",
    "TypeVarTuple",
    "Required",
    "NotRequired",
    "ReadOnly",
    "LiteralString",
}


# ── Helpers ────────────────────────────────────────────────────────────


class Issue:
    """Represents a single error/warning found in the codebase."""

    def __init__(  # noqa: D107
        self,
        file: Path,
        line: int,
        col: int,
        category: str,
        message: str,
    ) -> None:
        self.file = file
        self.line = line
        self.col = col
        self.category = category
        self.message = message

    def __str__(self) -> str:  # noqa: D105
        rel = self.file.relative_to(REPO_ROOT)
        return f"{rel}:{self.line}:{self.col}: [{self.category}] {self.message}"


def is_excluded(path: Path) -> bool:
    """Check if a path matches any exclusion pattern."""
    rel = str(path.relative_to(REPO_ROOT))
    return any(fnmatch(rel, pat) for pat in EXCLUDE_PATTERNS)


def collect_files(targets: list[Path] | None = None) -> list[Path]:
    """Collect all .py files to check."""
    if targets:
        return [p for p in targets if p.suffix == ".py" and p.exists()]
    files = sorted(SIMOPT_ROOT.rglob("*.py"))
    return [f for f in files if not is_excluded(f)]


# ── 1. Syntax Check ───────────────────────────────────────────────────


def check_syntax(path: Path) -> tuple[ast.Module | None, list[Issue]]:
    """Parse a file and return (AST, issues). AST is None on failure."""
    source = path.read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(source, filename=str(path))
        return tree, []
    except SyntaxError as e:
        issue = Issue(
            path,
            e.lineno or 0,
            e.offset or 0,
            "SyntaxError",
            e.msg,
        )
        return None, [issue]


# ── 2. Build module export index ───────────────────────────────────────


def build_export_index(
    files: list[Path],
    trees: dict[Path, ast.Module],
) -> dict[str, set[str]]:
    """Build a mapping of simopt module dotpath -> set of exported names.

    Scans each AST for top-level class, function, and assignment definitions.
    """
    index: dict[str, set[str]] = {}

    for fpath in files:
        tree = trees.get(fpath)
        if tree is None:
            continue

        # Determine dotted module path
        rel = fpath.relative_to(REPO_ROOT)
        parts = list(rel.with_suffix("").parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        modname = ".".join(parts)
        names: set[str] = set()

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                names.add(node.name)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        names.add(target.id)
                    elif isinstance(target, ast.Tuple):
                        for elt in target.elts:
                            if isinstance(elt, ast.Name):
                                names.add(elt.id)
            elif isinstance(node, ast.AnnAssign):
                if isinstance(node.target, ast.Name):
                    names.add(node.target.id)
            elif isinstance(node, ast.ImportFrom) and node.names:
                # Re-exports: from X import Y makes Y available
                for alias in node.names:
                    exported_name = alias.asname if alias.asname else alias.name
                    if exported_name != "*":
                        names.add(exported_name)

        # Also handle __all__ if present
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if (
                        isinstance(target, ast.Name)
                        and target.id == "__all__"
                        and isinstance(node.value, ast.List | ast.Tuple)
                    ):
                        for elt in node.value.elts:
                            if isinstance(elt, ast.Constant) and isinstance(
                                elt.value, str
                            ):
                                names.add(elt.value)

        index[modname] = names

    return index


# ── 3. Check imports ──────────────────────────────────────────────────


def check_imports(
    path: Path,
    tree: ast.Module,
    export_index: dict[str, set[str]],
) -> list[Issue]:
    """Check that all 'from simopt.X import Y' references exist."""
    issues: list[Issue] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        module = node.module or ""

        # Only check simopt internal imports
        if not module.startswith("simopt"):
            continue

        # Skip TYPE_CHECKING-guarded imports (they don't run)
        # We do a simple heuristic: check if parent is an If with
        # test == TYPE_CHECKING. We'll handle this below more carefully.

        available = export_index.get(module)
        if available is None:
            # Module might be excluded or external
            continue

        for alias in node.names:
            name = alias.name
            if name == "*":
                continue
            if name in available:
                continue

            # Check if the name is a submodule (e.g. `from simopt.experiment
            # import run_solver` where run_solver is a .py file or package)
            submod_dotpath = f"{module}.{name}"
            if submod_dotpath in export_index:
                continue  # It's a valid submodule import

            # Also check if a .py file exists for it (might be excluded
            # from our index)
            submod_path = REPO_ROOT / submod_dotpath.replace(".", "/")
            if (
                submod_path.with_suffix(".py").exists()
                or (submod_path / "__init__.py").exists()
            ):
                continue

            issues.append(
                Issue(
                    path,
                    node.lineno,
                    node.col_offset,
                    "ImportError",
                    f"'{name}' is not defined in '{module}' "
                    f"(available: {_suggest(name, available)})",
                )
            )

    return issues


def _suggest(name: str, available: set[str], n: int = 3) -> str:
    """Suggest similar names from the available set."""
    if not available:
        return "module is empty"
    # Simple prefix/substring matching
    suggestions = sorted(
        [
            a
            for a in available
            if name.lower() in a.lower() or a.lower() in name.lower()
        ],
    )[:n]
    if suggestions:
        return f"did you mean: {', '.join(suggestions)}?"
    # Fallback: show some names
    sample = sorted(available)[:5]
    suffix = ", ..." if len(available) > 5 else ""
    return f"exports include: {', '.join(sample)}{suffix}"


# ── 4. Check annotations ──────────────────────────────────────────────


def _collect_defined_names(tree: ast.Module) -> set[str]:
    """Collect all names defined/imported at module level."""
    names: set[str] = set()

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name):
                names.add(node.target.id)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name != "*":
                    names.add(alias.asname or alias.name)

    return names


def _extract_annotation_names(
    node: ast.expr,
    inside_literal: bool = False,
) -> list[tuple[str, ast.expr]]:
    """Extract all Name references from a type annotation AST node."""
    results: list[tuple[str, ast.expr]] = []

    if isinstance(node, ast.Name):
        results.append((node.id, node))
    elif isinstance(node, ast.Attribute):
        # e.g. np.ndarray -- skip, these are qualified
        pass
    elif isinstance(node, ast.Subscript):
        # Check if this is Literal[...] — skip string values inside it
        is_literal = isinstance(node.value, ast.Name) and node.value.id == "Literal"
        results.extend(_extract_annotation_names(node.value, inside_literal))
        results.extend(_extract_annotation_names(node.slice, inside_literal=is_literal))
    elif isinstance(node, ast.BinOp):
        # X | Y union syntax
        results.extend(_extract_annotation_names(node.left, inside_literal))
        results.extend(_extract_annotation_names(node.right, inside_literal))
    elif isinstance(node, ast.Tuple | ast.List):
        for elt in node.elts:
            results.extend(_extract_annotation_names(elt, inside_literal))
    elif (
        isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and not inside_literal
    ):
        # String annotations -- parse them, unless inside Literal[]
        try:
            inner = ast.parse(node.value, mode="eval")
            results.extend(_extract_annotation_names(inner.body))
        except SyntaxError:
            pass

    return results


def check_annotations(
    path: Path,
    tree: ast.Module,
) -> list[Issue]:
    """Check for undefined names in type annotations."""
    issues: list[Issue] = []
    defined = _collect_defined_names(tree)

    # If file uses `from __future__ import annotations`, all annotations are
    # strings and won't cause NameError at runtime. We still check them for
    # correctness since they'd fail at type-checking time.
    has_future_annotations = any(
        isinstance(node, ast.ImportFrom)
        and node.module == "__future__"
        and any(alias.name == "annotations" for alias in node.names)
        for node in ast.iter_child_nodes(tree)
    )

    # Collect annotations from function signatures and variable annotations
    for node in ast.walk(tree):
        annotations_to_check: list[ast.expr] = []

        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            # Return annotation
            if node.returns:
                annotations_to_check.append(node.returns)
            # Argument annotations
            for arg in node.args.args + node.args.posonlyargs + node.args.kwonlyargs:
                if arg.annotation:
                    annotations_to_check.append(arg.annotation)
            if node.args.vararg and node.args.vararg.annotation:
                annotations_to_check.append(node.args.vararg.annotation)
            if node.args.kwarg and node.args.kwarg.annotation:
                annotations_to_check.append(node.args.kwarg.annotation)

        elif isinstance(node, ast.AnnAssign):
            annotations_to_check.append(node.annotation)

        for ann in annotations_to_check:
            for name, name_node in _extract_annotation_names(ann):
                if (
                    name in defined
                    or name in BUILTIN_NAMES
                    or name in TYPING_NAMES
                    or name in EXTERNAL_MODULES
                    or name
                    in {
                        "int",
                        "float",
                        "str",
                        "bool",
                        "bytes",
                        "list",
                        "dict",
                        "set",
                        "tuple",
                        "type",
                        "None",
                        "object",
                        "complex",
                        "frozenset",
                        "bytearray",
                        "memoryview",
                        "range",
                        "slice",
                        "property",
                        "classmethod",
                        "staticmethod",
                        "super",
                        "ndarray",
                        "DataFrame",
                        "Series",
                        "Axes",
                        "Figure",
                        "Path",
                    }
                ):
                    continue
                # Skip if it looks like a forward reference in a
                # future-annotations file (very common, usually fine)
                if has_future_annotations:
                    continue
                issues.append(
                    Issue(
                        path,
                        getattr(name_node, "lineno", 0),
                        getattr(name_node, "col_offset", 0),
                        "UndefinedAnnotation",
                        f"Name '{name}' used in type annotation"
                        " is not defined in this file",
                    )
                )

    return issues


# ── 5. Check default value mismatches ─────────────────────────────────


_TYPE_MAP = {
    "int": (int,),
    "float": (int, float),
    "str": (str,),
    "bool": (bool,),
    "bytes": (bytes,),
}


def check_default_mismatches(
    path: Path,
    tree: ast.Module,
) -> list[Issue]:
    """Check for obvious type annotation / default value mismatches."""
    issues: list[Issue] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue

        defaults_map: list[tuple[ast.arg, ast.expr | None]] = []

        # Positional args with defaults
        n_args = len(node.args.args)
        n_defaults = len(node.args.defaults)
        for i, arg in enumerate(node.args.args):
            default_idx = i - (n_args - n_defaults)
            default = node.args.defaults[default_idx] if default_idx >= 0 else None
            defaults_map.append((arg, default))

        # Keyword-only args
        for arg, default in zip(
            node.args.kwonlyargs, node.args.kw_defaults, strict=False
        ):
            defaults_map.append((arg, default))

        for arg, default in defaults_map:
            if arg.annotation is None or default is None:
                continue
            if not isinstance(arg.annotation, ast.Name):
                continue
            if not isinstance(default, ast.Constant):
                continue

            ann_name = arg.annotation.id
            expected_types = _TYPE_MAP.get(ann_name)
            if expected_types is None:
                continue

            actual_value = default.value
            if actual_value is None:
                continue  # None is fine for any type (Optional pattern)

            if not isinstance(actual_value, expected_types):
                # Special case: bool is subclass of int, so True/False matching
                # int is fine in some contexts, but float annotated with bool
                # default is definitely wrong
                actual_type = type(actual_value).__name__
                issues.append(
                    Issue(
                        path,
                        arg.lineno,
                        arg.col_offset,
                        "TypeMismatch",
                        f"Parameter '{arg.arg}' annotated as '{ann_name}' "
                        f"but default value is {actual_value!r} ({actual_type})",
                    )
                )

    return issues


# ── 6. Check mutable defaults ─────────────────────────────────────────


def check_mutable_defaults(
    path: Path,
    tree: ast.Module,
) -> list[Issue]:
    """Check for mutable default arguments (list, dict, set literals)."""
    issues: list[Issue] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue

        all_defaults = list(node.args.defaults) + [
            d for d in node.args.kw_defaults if d is not None
        ]

        for default in all_defaults:
            if isinstance(default, ast.List):
                issues.append(
                    Issue(
                        path,
                        default.lineno,
                        default.col_offset,
                        "MutableDefault",
                        "Mutable default argument (list). Use None and "
                        "assign inside the function body instead.",
                    )
                )
            elif isinstance(default, ast.Dict):
                issues.append(
                    Issue(
                        path,
                        default.lineno,
                        default.col_offset,
                        "MutableDefault",
                        "Mutable default argument (dict). Use None and "
                        "assign inside the function body instead.",
                    )
                )
            elif isinstance(default, ast.Set):
                issues.append(
                    Issue(
                        path,
                        default.lineno,
                        default.col_offset,
                        "MutableDefault",
                        "Mutable default argument (set). Use None and "
                        "assign inside the function body instead.",
                    )
                )

    return issues


# ── 7. Check duplicate definitions ────────────────────────────────────


def check_duplicate_defs(
    path: Path,
    tree: ast.Module,
) -> list[Issue]:
    """Check for duplicate top-level function or class definitions."""
    issues: list[Issue] = []
    seen: dict[str, int] = {}

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            name = node.name
            if name in seen:
                issues.append(
                    Issue(
                        path,
                        node.lineno,
                        node.col_offset,
                        "DuplicateDefinition",
                        f"'{name}' is already defined at line {seen[name]}",
                    )
                )
            else:
                seen[name] = node.lineno

    return issues


# ── Main ──────────────────────────────────────────────────────────────


def main() -> int:
    """Run all checks and print results."""
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    targets_args = [
        Path(a).resolve() for a in sys.argv[1:] if a not in ("--verbose", "-v")
    ]

    files = collect_files(targets_args or None)
    if not files:
        print("No files to check.")
        return 0

    if verbose:
        print(f"Checking {len(files)} files...\n")

    # Phase 1: Syntax check all files
    trees: dict[Path, ast.Module] = {}
    all_issues: list[Issue] = []

    for fpath in files:
        tree, syntax_issues = check_syntax(fpath)
        all_issues.extend(syntax_issues)
        if tree is not None:
            trees[fpath] = tree

    # Phase 2: Build export index
    export_index = build_export_index(files, trees)

    # Phase 3: Run all checks on each file
    for fpath in files:
        tree = trees.get(fpath)
        if tree is None:
            continue

        if verbose:
            rel = fpath.relative_to(REPO_ROOT)
            print(f"  Checking {rel}...")

        all_issues.extend(check_imports(fpath, tree, export_index))
        all_issues.extend(check_annotations(fpath, tree))
        all_issues.extend(check_default_mismatches(fpath, tree))
        all_issues.extend(check_mutable_defaults(fpath, tree))
        all_issues.extend(check_duplicate_defs(fpath, tree))

    # ── Report ─────────────────────────────────────────────────────
    if not all_issues:
        print(f"\n  All {len(files)} files passed with no issues.")
        return 0

    # Group by file
    by_file: dict[Path, list[Issue]] = defaultdict(list)
    for issue in all_issues:
        by_file[issue.file].append(issue)

    # Group by category for summary
    by_category: dict[str, int] = defaultdict(int)
    for issue in all_issues:
        by_category[issue.category] += 1

    print("=" * 70)
    print(f"  ERRORS FOUND: {len(all_issues)} issues in {len(by_file)} files")
    print("=" * 70)

    for fpath in sorted(by_file):
        rel = fpath.relative_to(REPO_ROOT)
        file_issues = sorted(by_file[fpath], key=lambda i: (i.line, i.col))
        print(f"\n{rel} ({len(file_issues)} issues):")
        for issue in file_issues:
            print(f"  Line {issue.line}:{issue.col}  [{issue.category}]")
            for line in textwrap.wrap(issue.message, width=64):
                print(f"    {line}")

    print("\n" + "-" * 70)
    print("Summary by category:")
    for cat, count in sorted(by_category.items(), key=lambda x: -x[1]):
        print(f"  {cat:.<30} {count}")
    print(f"  {'TOTAL':.<30} {len(all_issues)}")
    print("-" * 70)

    return 1


if __name__ == "__main__":
    sys.exit(main())
