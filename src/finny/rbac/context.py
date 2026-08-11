"""PermissionContext — the single object every tool call receives to know what
sensitivity tags the caller may see. Enforcement happens where this context is
consumed (SQL WHERE / Chroma where), never after the fact."""

from typing import FrozenSet

from finny.rbac.roles import allowed_sensitivities


class PermissionContext:
    def __init__(self, role: str):
        self.role = role.lower().strip()
        self.allowed_sensitivities: FrozenSet[str] = allowed_sensitivities(self.role)

    def can_access(self, sensitivity: str) -> bool:
        return sensitivity in self.allowed_sensitivities

    def __repr__(self) -> str:
        return f"PermissionContext(role={self.role!r}, allowed={sorted(self.allowed_sensitivities)})"
