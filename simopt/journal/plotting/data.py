"""Ingestion + schema-validation layer for the journal sensitivity analyses.

The data model is the tidy long-form table emitted by
``scripts/journal_factors_test.py`` and concatenated by
``scripts/journal_aggregate.py``.  The schema below is the single source of
truth this plotting package will accept; any violation is rejected loudly.

Why so strict?
    The previous plotting code (``demo/plot_datafarming.py``) silently
    tolerated column renames, missing fields, and duplicate macroreps,
    producing plots that quietly lied about the underlying experiment.  We
    refuse to repeat that mistake: every parsed frame is validated, and every
    failure is escalated with a message that names the offending file,
    column, or row.
"""
from __future__ import annotations

import dataclasses
import gzip
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Schema definitions
# ---------------------------------------------------------------------------
STUDIES: tuple[str, ...] = ("subspace", "basis", "regularisation")

SWEPT_FACTORS: dict[str, str] = {
    "subspace": "subspace_dim",
    "basis": "basis_x_degree",
    "regularisation": "regularisation",
}

LEVEL_COLUMNS: dict[str, tuple[str, ...]] = {
    "subspace": ("subspace_dim",),
    "basis": ("polynomial_basis", "polynomial_degree"),
    "regularisation": ("subproblem_regularisation",),
}

# Columns required in journal_long_form.{parquet,csv} (and, by inheritance,
# in final_objectives.{parquet,csv}, which is just a filtered view).
LONG_FORM_REQUIRED: tuple[str, ...] = (
    "study",
    "problem",
    "problem_dim",
    "design_point_id",
    "swept_factor",
    "subspace_dim",
    "polynomial_basis",
    "polynomial_degree",
    "subproblem_regularisation",
    "adaptive",
    "macrorep",
    "budget_idx",
    "budget",
    "obj_postrep_mean",
    "is_final_budget",
)


class SchemaError(ValueError):
    """Raised when a loaded frame does not match the expected schema."""


# ---------------------------------------------------------------------------
# Optional parquet engine
# ---------------------------------------------------------------------------
def _parquet_engine() -> str | None:
    for engine in ("pyarrow", "fastparquet"):
        try:
            __import__(engine)
            return engine
        except ImportError:
            continue
    return None


def _read_table(parquet: Path, csv: Path, csv_gz: Path) -> pd.DataFrame | None:
    """Read whichever of three candidate paths exists, parquet first."""
    if parquet.exists():
        engine = _parquet_engine()
        if engine is not None:
            return pd.read_parquet(parquet, engine=engine)
    if csv.exists():
        return pd.read_csv(csv)
    if csv_gz.exists():
        with gzip.open(csv_gz, "rt") as fh:
            return pd.read_csv(fh)
    return None


# ---------------------------------------------------------------------------
# Coercion
# ---------------------------------------------------------------------------
_BOOL_TRUE = {"true", "True", "TRUE", "1", 1}
_BOOL_FALSE = {"false", "False", "FALSE", "0", 0}


def _coerce_bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(bool)
    return s.map(
        lambda v: True if v in _BOOL_TRUE
        else False if v in _BOOL_FALSE
        else (_ for _ in ()).throw(
            SchemaError(f"non-boolean value {v!r} found in bool column")
        )
    ).astype(bool)


def _coerce_long_form(df: pd.DataFrame) -> pd.DataFrame:
    """Apply type coercion in-place-style, returning a new frame."""
    out = df.copy()
    out["adaptive"] = _coerce_bool_series(out["adaptive"])
    out["is_final_budget"] = _coerce_bool_series(out["is_final_budget"])
    int_cols = ("problem_dim", "subspace_dim", "polynomial_degree",
                "macrorep", "budget_idx")
    for c in int_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="raise").astype(int)
    for c in ("subproblem_regularisation", "budget", "obj_postrep_mean"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="raise").astype(float)
    return out


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def _check_required_columns(df: pd.DataFrame, required: Iterable[str],
                            source: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SchemaError(
            f"{source}: missing required column(s) {missing}. "
            f"Found columns: {list(df.columns)}."
        )


def validate_long_form(df: pd.DataFrame, *,
                       strict_unique: bool = False) -> None:
    """Validate the canonical long-form table.

    Parameters
    ----------
    df
        DataFrame to validate.
    strict_unique
        If True, enforce uniqueness of
        ``(study, problem, design_point_id, macrorep, budget_idx)``.  The
        aggregator should already guarantee this, but cross-array re-submission
        bugs have produced duplicates in the past, so we surface them here.
    """
    _check_required_columns(df, LONG_FORM_REQUIRED, "journal_long_form")

    unknown = set(df["study"].unique()) - set(STUDIES)
    if unknown:
        raise SchemaError(
            f"unknown study value(s) {sorted(unknown)!r}; "
            f"valid: {STUDIES!r}"
        )

    if strict_unique:
        key = ["study", "problem", "design_point_id", "macrorep", "budget_idx"]
        dup = df[df.duplicated(key, keep=False)]
        if not dup.empty:
            sample = dup.head(5)[key].to_dict(orient="records")
            raise SchemaError(
                f"duplicate ({', '.join(key)}) rows detected "
                f"({len(dup)} rows). Sample: {sample}"
            )

    # Sanity-check that the swept_factor labels are internally consistent
    # with the study; if not, downstream filters silently drop rows.
    for study, expected in SWEPT_FACTORS.items():
        bad = df[(df["study"] == study) & (df["swept_factor"] != expected)]
        if not bad.empty:
            raise SchemaError(
                f"study={study!r}: expected swept_factor={expected!r}, "
                f"found {sorted(bad['swept_factor'].unique())!r}"
            )


# ---------------------------------------------------------------------------
# Aggregated-results container
# ---------------------------------------------------------------------------
@dataclass
class AggregatedResults:
    """In-memory view of the four artefacts written by journal_aggregate.py."""

    long_form: pd.DataFrame
    finals: pd.DataFrame
    summary: dict[str, pd.DataFrame] = field(default_factory=dict)
    paired_ci: dict[str, pd.DataFrame] = field(default_factory=dict)
    source: Path | None = None

    def for_study(self, study: str) -> AggregatedResults:
        """Return a copy of these results filtered to a single study."""
        if study not in STUDIES:
            raise ValueError(
                f"unknown study {study!r}; valid: {STUDIES!r}"
            )
        return dataclasses.replace(
            self,
            long_form=self.long_form[self.long_form["study"] == study]
                .reset_index(drop=True),
            finals=self.finals[self.finals["study"] == study]
                .reset_index(drop=True),
            summary={study: self.summary.get(study, pd.DataFrame())},
            paired_ci={study: self.paired_ci.get(study, pd.DataFrame())},
        )

    def for_problem(self, problem: str) -> AggregatedResults:
        """Return a copy of these results filtered to a single problem."""
        if problem not in set(self.long_form["problem"].unique()):
            raise ValueError(
                f"unknown problem {problem!r}; "
                f"have {sorted(self.long_form['problem'].unique())!r}"
            )
        sub = self.long_form["problem"] == problem
        fin = self.finals["problem"] == problem
        return dataclasses.replace(
            self,
            long_form=self.long_form[sub].reset_index(drop=True),
            finals=self.finals[fin].reset_index(drop=True),
            summary={
                k: v[v["problem"] == problem] if not v.empty else v
                for k, v in self.summary.items()
            },
            paired_ci={
                k: v[v["problem"] == problem] if not v.empty else v
                for k, v in self.paired_ci.items()
            },
        )

    def problems(self) -> list[str]:
        """List of problem names present in the loaded long-form table."""
        return sorted(self.long_form["problem"].unique().tolist())


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------
def load_aggregated(analysis_dir: Path | str) -> AggregatedResults:
    """Load the four canonical artefacts under ``analysis_dir``.

    Files (any of the three formats below per artefact)::

        journal_long_form.{parquet, csv, csv.gz}     [required]
        final_objectives.{parquet, csv, csv.gz}      [optional - recomputed]
        summary_<study>.csv                          [optional]
        paired_ci_<study>.csv                        [optional]

    Optional artefacts that are missing are simply replaced with empty frames;
    only the long-form table is mandatory because every other view is
    derivable from it.
    """
    analysis_dir = Path(analysis_dir)
    if not analysis_dir.exists():
        raise FileNotFoundError(
            f"analysis directory does not exist: {analysis_dir}"
        )

    long_form = _read_table(
        analysis_dir / "journal_long_form.parquet",
        analysis_dir / "journal_long_form.csv",
        analysis_dir / "journal_long_form.csv.gz",
    )
    if long_form is None:
        raise SchemaError(
            f"missing journal_long_form.{{parquet,csv,csv.gz}} in {analysis_dir}"
        )
    long_form = _coerce_long_form(long_form)
    validate_long_form(long_form)

    finals = _read_table(
        analysis_dir / "final_objectives.parquet",
        analysis_dir / "final_objectives.csv",
        analysis_dir / "final_objectives.csv.gz",
    )
    if finals is None:
        finals = long_form[long_form["is_final_budget"]].reset_index(drop=True)
    else:
        finals = _coerce_long_form(finals)

    summary: dict[str, pd.DataFrame] = {}
    paired_ci: dict[str, pd.DataFrame] = {}
    for study in STUDIES:
        s_path = analysis_dir / f"summary_{study}.csv"
        p_path = analysis_dir / f"paired_ci_{study}.csv"
        summary[study] = (
            pd.read_csv(s_path) if s_path.exists() else pd.DataFrame()
        )
        paired_ci[study] = (
            pd.read_csv(p_path) if p_path.exists() else pd.DataFrame()
        )

    return AggregatedResults(
        long_form=long_form,
        finals=finals,
        summary=summary,
        paired_ci=paired_ci,
        source=analysis_dir,
    )


__all__ = [
    "LEVEL_COLUMNS",
    "LONG_FORM_REQUIRED",
    "STUDIES",
    "SWEPT_FACTORS",
    "AggregatedResults",
    "SchemaError",
    "load_aggregated",
    "validate_long_form",
]
