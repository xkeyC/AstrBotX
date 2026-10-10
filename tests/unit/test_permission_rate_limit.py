"""Permission groups: inheritance and sliding-window rate limits per account."""

from types import SimpleNamespace

import pytest

from astrbot.core.permission_rate_limit import (
    ACTIVE_REPLY_EXTRA,
    RateLimiter,
    check_rate_limit,
    format_duration,
    keep_rate_limit_use,
    refund_rate_limit,
)
from astrbot.core.permission_rules import (
    EVENT_EXTRA_KEY,
    MAX_WINDOW_S,
    EventFacts,
    parse_rate_limits,
    resolve_policy,
)


def facts(sender="9", group="", role="member"):
    return EventFacts(sender_id=sender, group_id=group, role=role)


# ------------------------------------------------------------ inheritance

BASE = {
    "id": "base",
    "name": "base",
    "match": [],
    "tools_deny": ["shell_*"],
    "persona_id": "p-base",
    "global_memory": "false",
    "rate_limit": [{"window": 60, "count": 5}],
    "rate_limit_reply": "slow down",
}


def test_unset_fields_come_from_the_parent():
    rules = [BASE, {"id": "grp", "name": "grp", "match": ["g_1"], "inherits": "base"}]

    policy = resolve_policy(rules, facts(group="1"))

    assert policy.rule_name == "grp"
    assert policy.tools_deny == ("shell_*",)
    assert policy.persona_id == "p-base"
    assert policy.global_memory is False
    assert policy.rate_limits == ((60, 5),)
    assert policy.rate_limit_reply == "slow down"
    assert policy.chain == ("grp", "base")


def test_set_fields_override_the_parent():
    child = {
        "id": "vip",
        "match": ["p_7"],
        "inherits": "base",
        "tools_deny": ["rm"],
        "global_memory": True,
        "rate_limit": "unlimited",
    }

    policy = resolve_policy([BASE, child], facts(sender="7"))

    assert policy.tools_deny == ("rm",)
    assert policy.global_memory is True
    assert policy.rate_limits == ()
    # Still inherited where the child says nothing.
    assert policy.persona_id == "p-base"


def test_inheritance_runs_through_several_levels():
    mid = {"id": "mid", "match": [], "inherits": "base", "model": "gpt-5.5"}
    leaf = {"id": "leaf", "match": ["*"], "inherits": "mid"}

    policy = resolve_policy([BASE, mid, leaf], facts())

    assert policy.model == "gpt-5.5"
    assert policy.rate_limits == ((60, 5),)
    assert policy.chain == ("leaf", "mid", "base")


def test_a_disabled_parent_still_serves_as_a_template():
    rules = [
        {**BASE, "enabled": False, "match": ["*"]},
        {"match": ["*"], "inherits": "base"},
    ]

    policy = resolve_policy(rules, facts())

    assert policy.persona_id == "p-base"


def test_cycles_and_missing_parents_end_the_chain():
    a = {"id": "a", "match": ["*"], "inherits": "b", "model": "m"}
    b = {"id": "b", "match": [], "inherits": "a", "persona_id": "pb"}
    orphan = {"id": "o", "match": ["p_1"], "inherits": "nope", "model": "x"}

    looped = resolve_policy([a, b], facts())
    alone = resolve_policy([orphan, a, b], facts(sender="1"))

    assert looped.chain == ("a", "b")
    assert (looped.model, looped.persona_id) == ("m", "pb")
    assert alone.chain == ("o",)


def test_match_conditions_are_not_inherited():
    rules = [
        {**BASE, "match": ["*"], "enabled": False},
        {"id": "c", "inherits": "base"},
    ]

    assert resolve_policy(rules, facts()).rule_name == ""


@pytest.mark.parametrize(
    ("value", "parsed"),
    [
        (None, None),
        ("inherit", None),
        ("unlimited", ()),
        ([], ()),
        ([{"window": 60, "count": 3}, {"window": 0, "count": 3}], ((60, 3),)),
        ([{"window": "3600", "count": "10"}, {"window": 60}], ((3600, 10),)),
    ],
)
def test_rate_limit_values(value, parsed):
    assert parse_rate_limits(value) == parsed


# ------------------------------------------------------------ the limiter


@pytest.fixture
def limiter(temp_db):
    return RateLimiter(temp_db)


async def _take(limiter, account, limits, now):
    """True when the use was recorded, False when it was refused."""
    use_id, refused = await limiter.acquire(account, limits, now=now)
    assert (use_id is None) == (refused is not None)
    return refused is None


@pytest.mark.asyncio
async def test_a_sliding_window_frees_slots_as_uses_age_out(limiter):
    limits = ((60, 2),)

    assert await _take(limiter, "qq:1", limits, 1000)
    assert await _take(limiter, "qq:1", limits, 1030)
    _, (wait, limit) = await limiter.acquire("qq:1", limits, now=1040)

    # The first use leaves the window at 1060, not at a fixed boundary.
    assert wait == pytest.approx(20)
    assert limit == (60, 2)
    assert await _take(limiter, "qq:1", limits, 1061)
    # The use at 1030 is still inside: full again.
    assert not await _take(limiter, "qq:1", limits, 1062)


@pytest.mark.asyncio
async def test_every_window_must_hold(limiter):
    limits = ((60, 10), (3600, 3))
    for t in (0, 100, 200):
        assert await _take(limiter, "qq:1", limits, t)

    _, (wait, limit) = await limiter.acquire("qq:1", limits, now=300)

    assert limit == (3600, 3)
    assert wait == pytest.approx(3300)


@pytest.mark.asyncio
async def test_accounts_are_counted_apart(limiter):
    limits = ((60, 1),)

    assert await _take(limiter, "qq:1", limits, 0)
    assert await _take(limiter, "qq:2", limits, 1)
    assert await _take(limiter, "tg:1", limits, 2)
    assert not await _take(limiter, "qq:1", limits, 3)


@pytest.mark.asyncio
async def test_a_refused_use_is_not_counted(limiter):
    limits = ((60, 1),)
    await limiter.acquire("qq:1", limits, now=0)
    for t in range(1, 50):
        await limiter.acquire("qq:1", limits, now=t)

    assert await _take(limiter, "qq:1", limits, 61)


@pytest.mark.asyncio
async def test_a_refunded_use_frees_its_slot(limiter):
    limits = ((60, 1),)
    use_id, _ = await limiter.acquire("qq:1", limits, now=0)

    await limiter.refund(use_id)

    assert await _take(limiter, "qq:1", limits, 1)


@pytest.mark.asyncio
async def test_uses_survive_a_new_limiter(temp_db):
    # Stored in SQLite: a restart does not reset a daily quota.
    await RateLimiter(temp_db).acquire("qq:1", ((86400, 1),), now=0)

    assert not await _take(RateLimiter(temp_db), "qq:1", ((86400, 1),), 10)


@pytest.mark.asyncio
async def test_pruning_keeps_whatever_any_rule_may_count(limiter):
    # Another config file may count this account over a long window, so a
    # short-window request must not prune its uses.
    await limiter.acquire("qq:1", ((MAX_WINDOW_S, 1),), now=0)
    limiter._last_prune = -1e9
    await limiter.acquire("qq:2", ((60, 5),), now=MAX_WINDOW_S - 10)

    assert not await _take(limiter, "qq:1", ((MAX_WINDOW_S, 1),), MAX_WINDOW_S - 5)


def test_a_refusal_is_announced_once_per_period(temp_db):
    limiter = RateLimiter(temp_db)
    minute = (60, 1)

    assert limiter.should_notify("qq:1", minute, 30, now=0) is True
    assert limiter.should_notify("qq:1", minute, 30, now=10) is False
    assert limiter.should_notify("qq:2", minute, 30, now=10) is True
    assert limiter.should_notify("qq:1", minute, 30, now=31) is True


def test_a_longer_limit_met_elsewhere_is_still_announced(temp_db):
    limiter = RateLimiter(temp_db)

    assert limiter.should_notify("qq:1", (60, 1), 30, now=0) is True
    assert limiter.should_notify("qq:1", (86400, 5), 36000, now=10) is True


def test_durations_read_naturally():
    assert format_duration(20) == "20 秒"
    assert format_duration(90) == "1 分钟 30 秒"
    assert format_duration(3700) == "1 小时 1 分钟"
    assert format_duration(90000) == "1 天 1 小时"


# ------------------------------------------------------------ per event


class _Event:
    def __init__(self, sender="1", group="", platform="qq", role="member"):
        self.role = role
        self._extras = {}
        self.sender = sender
        self.group = group
        self.platform = platform

    def get_sender_id(self):
        return self.sender

    def get_group_id(self):
        return self.group

    def get_platform_id(self):
        return self.platform

    def get_extra(self, key, default=None):
        return self._extras.get(key, default)

    def set_extra(self, key, value):
        self._extras[key] = value


LIMITED = {
    "permission_rules": [
        {
            "id": "admin",
            "match": ["role:admin"],
            "rate_limit": "unlimited",
        },
        {
            "id": "all",
            "match": ["*"],
            "rate_limit": [{"window": 60, "count": 1}],
            "rate_limit_reply": "wait {wait}; {limit}",
        },
    ]
}


@pytest.mark.asyncio
async def test_a_limited_group_refuses_and_replies_once(limiter):
    assert await check_rate_limit(_Event(), LIMITED, limiter) is None

    first = await check_rate_limit(_Event(), LIMITED, limiter)
    second = await check_rate_limit(_Event(), LIMITED, limiter)

    assert first.startswith("wait ") and first.endswith("每 1 分钟 1 次")
    assert second == ""


@pytest.mark.asyncio
async def test_one_account_shares_its_count_across_chats(limiter):
    assert await check_rate_limit(_Event(group="g1"), LIMITED, limiter) is None

    assert await check_rate_limit(_Event(group="g2"), LIMITED, limiter) is not None


@pytest.mark.asyncio
async def test_a_group_without_limits_is_not_counted(limiter):
    admin = _Event(role="admin")
    for _ in range(5):
        assert await check_rate_limit(admin, LIMITED, limiter) is None
        admin._extras.clear()

    # Nothing was recorded, so the member group starts from zero.
    assert await check_rate_limit(_Event(), LIMITED, limiter) is None


@pytest.mark.asyncio
async def test_scheduled_wakeups_are_not_counted(limiter, monkeypatch):
    from astrbot.core.cron import events

    class Wake(_Event):
        pass

    monkeypatch.setattr(events, "CronMessageEvent", Wake)
    for _ in range(3):
        assert await check_rate_limit(Wake(), LIMITED, limiter) is None


@pytest.mark.asyncio
async def test_active_replies_are_not_counted(limiter):
    # The bot chimed in on chatter nobody addressed to it.
    for _ in range(3):
        chatter = _Event()
        chatter.set_extra(ACTIVE_REPLY_EXTRA, True)
        assert await check_rate_limit(chatter, LIMITED, limiter) is None

    assert await check_rate_limit(_Event(), LIMITED, limiter) is None


@pytest.mark.asyncio
async def test_a_request_stopped_before_the_agent_is_given_back(limiter):
    event = _Event()
    assert await check_rate_limit(event, LIMITED, limiter) is None

    await refund_rate_limit(event)
    await refund_rate_limit(event)  # a second call is a no-op

    assert await check_rate_limit(_Event(), LIMITED, limiter) is None
    assert await check_rate_limit(_Event(), LIMITED, limiter) is not None


@pytest.mark.asyncio
async def test_the_policy_is_resolved_once_per_event(limiter):
    event = _Event()
    await check_rate_limit(event, LIMITED, limiter)

    assert event.get_extra(EVENT_EXTRA_KEY).chain == ("all",)


@pytest.mark.asyncio
async def test_a_counter_failure_lets_the_request_through():
    broken = RateLimiter(SimpleNamespace(get_db=None))

    assert await check_rate_limit(_Event(), LIMITED, broken) is None


@pytest.mark.asyncio
async def test_a_use_the_agent_accepted_is_not_given_back(limiter):
    event = _Event()
    await check_rate_limit(event, LIMITED, limiter)

    keep_rate_limit_use(event)
    await refund_rate_limit(event)

    assert await check_rate_limit(_Event(), LIMITED, limiter) is not None


@pytest.mark.asyncio
async def test_a_broken_reply_template_is_sent_as_is(limiter):
    config = {
        "permission_rules": [
            {
                "match": ["*"],
                "rate_limit": [{"window": 60, "count": 1}],
                "rate_limit_reply": "slow {wait.x}",
            }
        ]
    }
    await check_rate_limit(_Event(), config, limiter)

    assert await check_rate_limit(_Event(), config, limiter) == "slow {wait.x}"


def test_whole_number_strings_are_read_as_the_webui_shows_them():
    assert parse_rate_limits([{"window": "60.0", "count": "2.0"}]) == ((60, 2),)
    assert parse_rate_limits([{"window": "inf", "count": 1}]) == ()
