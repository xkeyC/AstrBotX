"""Per-user permission rules (replaces the DynamicPersona plugin's bindings).

Rules are evaluated in order; the first enabled rule whose match conditions
hit the event wins. Matching syntax (compatible with DynamicPersona):

- ``<group_id>/<sender_id>``: that sender in that group
- ``p_<sender_id>``: that sender anywhere
- ``g_<group_id>``: everyone in that group
- ``role:admin`` / ``role:member``: by AstrBot role
- ``*``: everyone

A rule is a permission group. It may inherit from another rule (``inherits``
names that rule's ``id``): every field it leaves unset -- an empty list, an
empty persona / model / reply, an ``inherit`` switch or rate limit -- comes
from the parent, and so on up the chain. Match conditions and the enabled
switch are never inherited, so a rule with no match conditions (or a disabled
one) still serves as a template.

``rate_limit`` caps how many requests an account may send to the agent within
sliding windows; see ``astrbot/core/permission_rate_limit.py``.

``cron_max_tasks`` and ``cron_min_interval_hours`` limit the scheduled tasks
(``future_task``) an account keeps and how often each may run; 0 is no
limit. Left unset along the whole chain (and for a sender no rule matches),
members get one task at most every 6 hours and admins are not limited.

Tool permissions are always enforced when a tool is called. Under code mode
the denied tools are also left out of the set the model can see, which costs
nothing because deferred tool specs never enter the prompt prefix; without
deferral they stay listed, so the prefix (and its cache) is the same for
everyone.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from typing import Any

CONFIG_KEY = "permission_rules"
EVENT_EXTRA_KEY = "_permission_policy"
# Longest inheritance chain followed; deeper ones are cut off there.
MAX_INHERIT_DEPTH = 16
# Longest rate-limit window, in seconds (30 days).
MAX_WINDOW_S = 30 * 24 * 3600
# Permission scopes: capabilities sent with each Codex turn and checked by
# Codex's own tools when they run, so a thread shared by many people (a group)
# still acts on the rights of whoever sent the current turn.
MEMORY_WRITE_GLOBAL_SCOPE = "memory.write_global"
MEMORY_DELETE_SCOPE = "memory.delete"
# Scheduled-task limits of a member whose rules leave them unset.
MEMBER_CRON_MAX_TASKS = 1
MEMBER_CRON_MIN_INTERVAL_S = 6 * 3600


@dataclass(frozen=True)
class PermissionPolicy:
    rule_name: str = ""
    tools_allow: tuple[str, ...] = ()
    tools_deny: tuple[str, ...] = ()
    mcp_allow: tuple[str, ...] = ()
    mcp_deny: tuple[str, ...] = ()
    persona_id: str = ""
    model: str = ""
    # Consumed by the scoped memory module (may_write_global).
    global_memory: bool | None = None
    # (window seconds, max requests) pairs; every one must hold. Empty: no limit.
    rate_limits: tuple[tuple[int, int], ...] = ()
    rate_limit_reply: str = ""
    # Scheduled-task limits: None for the role's default, 0 for no limit.
    cron_max_tasks: int | None = None
    cron_min_interval_s: int | None = None
    # Ids of the rule that matched and the rules it inherited from, in order.
    chain: tuple[str, ...] = ()

    @property
    def scopes(self) -> list[str]:
        """Permission scopes of the sender, sent with each of their turns."""
        scopes = []
        if self.global_memory is True:
            scopes += [MEMORY_WRITE_GLOBAL_SCOPE, MEMORY_DELETE_SCOPE]
        return scopes

    def cron_limits(self, role: str) -> tuple[int, int]:
        """The sender's scheduled-task limits, 0 meaning none.

        Args:
            role: The sender's AstrBot role (``admin`` or ``member``).

        Returns:
            (most tasks the sender may keep, least seconds between two runs
            of a recurring task).
        """
        admin = role == "admin"
        tasks = self.cron_max_tasks
        if tasks is None:
            tasks = 0 if admin else MEMBER_CRON_MAX_TASKS
        interval = self.cron_min_interval_s
        if interval is None:
            interval = 0 if admin else MEMBER_CRON_MIN_INTERVAL_S
        return tasks, interval

    @property
    def is_default(self) -> bool:
        return not (
            self.tools_allow or self.tools_deny or self.mcp_allow or self.mcp_deny
        )

    def allows_tool(self, tool_name: str, mcp_server: str | None = None) -> bool:
        """Deny lists always win. MCP tools are decided by the MCP lists when
        either is set; otherwise, like plugin tools, by the tool lists."""
        if any(fnmatch.fnmatchcase(tool_name, p) for p in self.tools_deny):
            return False
        if mcp_server and (self.mcp_allow or self.mcp_deny):
            if any(fnmatch.fnmatchcase(mcp_server, p) for p in self.mcp_deny):
                return False
            if self.mcp_allow:
                return any(fnmatch.fnmatchcase(mcp_server, p) for p in self.mcp_allow)
            return True
        if self.tools_allow:
            return any(fnmatch.fnmatchcase(tool_name, p) for p in self.tools_allow)
        return True

    def summary(self) -> str:
        """Short, model-facing description of the sender's restrictions.

        Returns:
            One line, or an empty string when nothing is restricted.
        """
        parts = []
        if self.tools_allow:
            parts.append("allowed tools: " + ", ".join(self.tools_allow))
        if self.tools_deny:
            parts.append("denied tools: " + ", ".join(self.tools_deny))
        if self.mcp_allow:
            parts.append("allowed MCP servers: " + ", ".join(self.mcp_allow))
        if self.mcp_deny:
            parts.append("denied MCP servers: " + ", ".join(self.mcp_deny))
        return "; ".join(parts)


DEFAULT_POLICY = PermissionPolicy()


@dataclass
class EventFacts:
    sender_id: str
    group_id: str
    role: str


def _as_list(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        items = value.replace(",", "\n").splitlines()
    elif isinstance(value, list):
        items = [str(v) for v in value]
    else:
        items = []
    return tuple(i.strip() for i in items if i and i.strip())


def _as_opt_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in ("true", "false"):
        return value.lower() == "true"
    return None


def _as_opt_number(value: Any) -> float | None:
    """A non-negative number, or None (unset: empty, ``inherit``, invalid)."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if 0 <= number < float("inf") else None


def condition_matches(condition: str, facts: EventFacts) -> bool:
    cond = condition.strip()
    if not cond:
        return False
    if cond == "*":
        return True
    if cond.startswith("role:"):
        return facts.role == cond[len("role:") :]
    if cond.startswith("p_"):
        return facts.sender_id == cond[2:]
    if cond.startswith("g_"):
        return bool(facts.group_id) and facts.group_id == cond[2:]
    if "/" in cond:
        group, sender = cond.split("/", 1)
        return facts.group_id == group and facts.sender_id == sender
    return False


def parse_rate_limits(value: Any) -> tuple[tuple[int, int], ...] | None:
    """A rule's rate limits: None to inherit, () for none, else the windows.

    Accepts ``"inherit"`` / missing, ``"unlimited"``, or a list of
    ``{"window": seconds, "count": n}``. Malformed or out-of-range entries are
    skipped; a list left empty that way means no limit.
    """
    if value is None or value == "inherit":
        return None
    if not isinstance(value, list):
        return ()
    limits = []
    for item in value:
        if not isinstance(item, dict):
            continue
        try:
            window = int(float(item.get("window")))
            count = int(float(item.get("count")))
        except (TypeError, ValueError, OverflowError):
            continue
        if 0 < window <= MAX_WINDOW_S and count > 0:
            limits.append((window, count))
    return tuple(sorted(set(limits)))


def rule_id(rule: dict) -> str:
    return str(rule.get("id") or "").strip()


def inheritance_chain(rule: dict, rules: list[dict] | None) -> list[dict]:
    """The rule followed by its ancestors, nearest first.

    A missing parent ends the chain; a cycle is cut where it closes.
    """
    by_id: dict[str, dict] = {}
    for r in rules or []:
        # The first rule with an id owns it, as the WebUI shows.
        if isinstance(r, dict) and rule_id(r):
            by_id.setdefault(rule_id(r), r)
    chain = [rule]
    seen = {id(rule)}
    current = rule
    while len(chain) < MAX_INHERIT_DEPTH:
        parent = by_id.get(str(current.get("inherits") or "").strip())
        if parent is None or id(parent) in seen:
            break
        chain.append(parent)
        seen.add(id(parent))
        current = parent
    return chain


def _nearest(chain: list[dict], parse: Any, unset: Any) -> Any:
    """The nearest value in the chain that is set, else ``unset``."""
    for rule in chain:
        value = parse(rule)
        if value is not None and value != unset:
            return value
    return unset


def policy_from_rule(rule: dict, rules: list[dict] | None) -> PermissionPolicy:
    """The effective policy of one rule, with everything it inherits."""
    chain = inheritance_chain(rule, rules)

    def names(key: str) -> tuple[str, ...]:
        return _nearest(chain, lambda r: _as_list(r.get(key)), ())

    def text(key: str) -> str:
        return _nearest(chain, lambda r: str(r.get(key) or "").strip(), "")

    def switch(key: str) -> bool | None:
        return _nearest(chain, lambda r: _as_opt_bool(r.get(key)), None)

    # "unlimited" on a nearer rule is set, not unset: it ends the search.
    rate_limits = _nearest(
        chain, lambda r: parse_rate_limits(r.get("rate_limit")), None
    )
    max_tasks = _nearest(chain, lambda r: _as_opt_number(r.get("cron_max_tasks")), None)
    interval_h = _nearest(
        chain, lambda r: _as_opt_number(r.get("cron_min_interval_hours")), None
    )
    return PermissionPolicy(
        rule_name=str(rule.get("name") or ""),
        tools_allow=names("tools_allow"),
        tools_deny=names("tools_deny"),
        mcp_allow=names("mcp_allow"),
        mcp_deny=names("mcp_deny"),
        persona_id=text("persona_id"),
        model=text("model"),
        global_memory=switch("global_memory"),
        rate_limits=rate_limits or (),
        rate_limit_reply=text("rate_limit_reply"),
        cron_max_tasks=None if max_tasks is None else int(max_tasks),
        cron_min_interval_s=None if interval_h is None else round(interval_h * 3600),
        chain=tuple(rule_id(r) for r in chain),
    )


def resolve_policy(rules: list[dict] | None, facts: EventFacts) -> PermissionPolicy:
    for rule in rules or []:
        if (
            not isinstance(rule, dict)
            or _as_opt_bool(rule.get("enabled", True)) is False
        ):
            continue
        conditions = _as_list(rule.get("match"))
        if not any(condition_matches(c, facts) for c in conditions):
            continue
        return policy_from_rule(rule, rules)
    return DEFAULT_POLICY


def event_facts(event: Any) -> EventFacts:
    return EventFacts(
        sender_id=str(event.get_sender_id() or ""),
        group_id=str(event.get_group_id() or ""),
        role=str(getattr(event, "role", "") or "member"),
    )


def policy_for_event(event: Any, rules: list[dict] | None) -> PermissionPolicy:
    """Resolve and cache the policy on the event."""
    cached = event.get_extra(EVENT_EXTRA_KEY)
    if isinstance(cached, PermissionPolicy):
        return cached
    policy = resolve_policy(rules, event_facts(event))
    event.set_extra(EVENT_EXTRA_KEY, policy)
    return policy


def tool_mcp_server(tool: Any) -> str | None:
    """MCP server name of an AstrBot MCP tool, if it is one."""
    for attr in ("mcp_server_name", "server_name"):
        value = getattr(tool, attr, None)
        if isinstance(value, str) and value:
            return value
    return None


DYNAMIC_PERSONA_CONFIG = "astrbot_plugin_DynamicPersona_config.json"


def rules_from_dynamic_persona(plugin_conf: Any) -> list[dict]:
    """Convert DynamicPersona ``persona_bindings`` into permission rules.

    Match lines and persona carry over; the plugin's per-rule chat provider
    has no Codex equivalent and is dropped (set ``model`` on the rule instead).
    """
    if not isinstance(plugin_conf, dict):
        return []
    rules: list[dict] = []
    for binding in plugin_conf.get("persona_bindings") or []:
        if not isinstance(binding, dict):
            continue
        match = [
            line.strip()
            for line in str(binding.get("match_conditions") or "").splitlines()
            if line.strip()
        ]
        if not match:
            continue
        rules.append(
            {
                "name": str(binding.get("rule_name") or "") or "DynamicPersona",
                "enabled": bool(binding.get("rule_enabled", True))
                and bool(plugin_conf.get("enabled", True)),
                "match": match,
                "persona_id": str(binding.get("persona_id") or ""),
            }
        )
    return rules
