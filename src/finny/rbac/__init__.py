"""RBAC package: role -> allowed-tags table, PermissionContext, and the
RBAC-enforced data access tools used by the agent layer."""

from finny.rbac.context import PermissionContext
from finny.rbac.roles import ROLE_PERMISSIONS, UnknownRoleError, allowed_sensitivities

__all__ = ["PermissionContext", "ROLE_PERMISSIONS", "UnknownRoleError", "allowed_sensitivities"]
