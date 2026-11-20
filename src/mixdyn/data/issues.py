"""What reading a panel found: each finding names its code, its severity, the geo and the dates."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum


class Severity(Enum):
    """Whether a finding stops reading."""

    ERROR = "error"
    """Reading fails."""
    WARNING = "warning"
    """Reading succeeds; the panel carries the finding in `issues`."""


class IssueCode(Enum):
    """What a finding is about; its value is the code a document of `ISSUES` writes."""

    MISSING_COLUMN = "missing_column"
    BAD_SPEC = "bad_spec"
    """The reading instructions contradict themselves or the data: a column in two roles, unequal
    column lengths, no rows."""
    BAD_DATE = "bad_date"
    BAD_GEO = "bad_geo"
    NOT_NUMERIC = "not_numeric"
    NON_FINITE = "non_finite"
    NEGATIVE_SPEND = "negative_spend"
    MISSING_KPI = "missing_kpi"
    OFF_GRID = "off_grid"
    """A date is not on the cadence's grid that the other dates share."""
    DUPLICATE = "duplicate"
    MISSING_PERIOD = "missing_period"
    """A geo has no row for a period inside the panel's calendar."""
    CONSTANT = "constant"
    UNIT = "unit"
    HIERARCHY = "hierarchy"
    POPULATION = "population"
    CADENCE = "cadence"
    """The dates are spaced finer or coarser than the cadence named: reading does not resample."""
    FEW_OBSERVATIONS = "few_observations"
    """A geo has fewer observed periods per effect, channel or control, than an effect needs."""
    ENDOGENOUS_SPEND = "endogenous_spend"
    """A channel's spend reads the KPI's last period beyond its own last period and the calendar
    (`mixdyn.data.feedback.spend_feedback`): a model that reads the spend as set apart from demand
    credits the channel with the demand."""


@dataclass(frozen=True, slots=True)
class Issue:
    """One finding of `read_panel`: what it is, how severe, and where.

    `geo` and `column` are None when the finding is not about a single one. `dates` holds every
    date involved; `message` lists only the first few. `channel` names the channel whose spend the
    finding is about, by the panel's name for it, which need not be its column's: `read_panel`'s
    `spend` maps a channel to a column of any name. It is None for every other finding, a spend
    column missing or given two roles among them, which name the column alone.
    """

    code: IssueCode
    severity: Severity
    message: str
    geo: str | None = None
    column: str | None = None
    dates: tuple[date, ...] = ()
    channel: str | None = None


class PanelError(ValueError):
    """The data cannot be read as a panel. `issues` holds every error found, and the warnings."""

    MAX_LISTED = 40

    def __init__(self, issues: tuple[Issue, ...]) -> None:
        self.issues = issues
        errors = [issue.message for issue in issues if issue.severity is Severity.ERROR]
        listed = errors[: self.MAX_LISTED]
        if len(errors) > len(listed):
            listed.append(f"... and {len(errors) - len(listed)} more in `.issues`")
        noun = "error" if len(errors) == 1 else "errors"
        super().__init__(f"{len(errors)} {noun} reading the panel:\n- " + "\n- ".join(listed))
