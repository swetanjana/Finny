"""Guard for computed metrics that combine data across sensitivity tags.

A role can individually be allowed `public_financial` and still be forbidden from
seeing "revenue per employee", because that computation also depends on
`headcount_compensation`. Refuse the computation explicitly rather than letting an
agent attempt it from whatever partial context it can fetch.
"""

from typing import FrozenSet

DERIVED_METRIC_REQUIREMENTS: dict[str, FrozenSet[str]] = {
    "revenue_per_employee": frozenset({"public_financial", "headcount_compensation"}),
    "compensation_ratio": frozenset({"public_financial", "headcount_compensation"}),
}


class DerivedMetricAccessDenied(PermissionError):
    def __init__(self, metric_key: str, missing: FrozenSet[str], role: str):
        self.metric_key = metric_key
        self.missing = missing
        super().__init__(
            f"Role '{role}' cannot compute '{metric_key}': missing access to "
            f"sensitivity tag(s) {sorted(missing)}. Refusing rather than computing "
            f"from partial data."
        )


def check_derived_metric_access(metric_key: str, context) -> None:
    """Raises DerivedMetricAccessDenied if `context`'s role lacks any tag this
    computation depends on. No-op for metric_key not in the dependency map."""
    required = DERIVED_METRIC_REQUIREMENTS.get(metric_key)
    if required is None:
        return
    missing = required - context.allowed_sensitivities
    if missing:
        raise DerivedMetricAccessDenied(metric_key, missing, context.role)
