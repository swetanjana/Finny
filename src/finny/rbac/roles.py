"""Role -> allowed sensitivity-tag table. Hardcoded dict is fine at this scope
(per CLAUDE.md) — roles are a CLI flag, not authenticated identities."""

from typing import FrozenSet

ROLE_PERMISSIONS: dict[str, FrozenSet[str]] = {
    "ceo": frozenset({"public_narrative", "public_financial", "headcount_compensation"}),
    "cto": frozenset({"public_narrative", "public_financial"}),
    "analyst": frozenset({"public_financial"}),
}


class UnknownRoleError(ValueError):
    pass


def allowed_sensitivities(role: str) -> FrozenSet[str]:
    key = role.lower().strip()
    if key not in ROLE_PERMISSIONS:
        raise UnknownRoleError(f"Unknown role '{role}'. Valid roles: {sorted(ROLE_PERMISSIONS)}")
    return ROLE_PERMISSIONS[key]
