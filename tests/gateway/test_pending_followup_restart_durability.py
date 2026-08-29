import asyncio
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.platforms.base import merge_pending_message_event
from gateway.run import (
    _merge_queued_followup_completion,
    _prepare_resume_pending_message,
)
from gateway.session import SessionEntry, SessionStore
from tests.gateway.restart_test_helpers import make_restart_runner, make_restart_source


@pytest.fixture(autouse=True)
def _isolate_restart_loop_guard(monkeypatch):
    """Durability scheduler tests must never mutate the live restart marker."""
    check = MagicMock(return_value=False)
    monkeypatch.setattr("gateway.restart_loop_guard.check_and_record", check)
    return check


def _store(tmp_path):
    return SessionStore(sessions_dir=tmp_path, config=GatewayConfig())


def test_mark_resume_pending_persists_followup_across_reload(tmp_path) -> None:
    store = _store(tmp_path)
    source = make_restart_source(chat_id="durable-chat")
    entry = store.get_or_create_session(source)

    assert (
        store.mark_resume_pending(
            entry.session_key,
            reason="shutdown_pending_followup",
            pending_followup_text="  test  ",
        )
        is True
    )

    reloaded = _store(tmp_path)
    restored = reloaded.get_or_create_session(source)
    assert restored.session_id == entry.session_id
    assert restored.resume_pending is True
    assert restored.resume_reason == "shutdown_pending_followup"
    assert restored.pending_followup_text == "  test  "


def test_complete_event_queue_persists_across_primary_store_reload(tmp_path) -> None:
    store = _store(tmp_path)
    runner, _ = make_restart_runner()
    source = make_restart_source(chat_id="durable-chat")
    entry = store.get_or_create_session(source)
    event = MessageEvent(
        text="caption",
        message_type=MessageType.PHOTO,
        source=source,
        media_urls=["/tmp/photo.jpg"],
        media_types=["image/jpeg"],
    )
    payload = runner._serialize_pending_followup_event(event)
    assert payload is not None

    assert store.mark_resume_pending(
        entry.session_key,
        reason="shutdown_pending_followup",
        pending_followup_text="caption",
        pending_followup_events=[payload],
    )

    restored = _store(tmp_path).get_or_create_session(source)
    assert restored.pending_followup_events == [payload]


def test_different_existing_followup_is_not_overwritten(tmp_path) -> None:
    store = _store(tmp_path)
    source = make_restart_source(chat_id="durable-chat")
    entry = store.get_or_create_session(source)
    assert (
        store.mark_resume_pending(
            entry.session_key,
            reason="shutdown_pending_followup",
            pending_followup_text="first",
        )
        is True
    )

    accepted = store.mark_resume_pending(
        entry.session_key,
        reason="shutdown_pending_followup",
        pending_followup_text="second",
    )

    assert accepted is False
    assert store._entries[entry.session_key].pending_followup_text == "first"


def test_distinct_same_text_events_do_not_collapse(tmp_path) -> None:
    store = _store(tmp_path)
    runner, _ = make_restart_runner()
    source = make_restart_source(chat_id="durable-chat")
    entry = store.get_or_create_session(source)
    first = runner._serialize_pending_followup_event(
        MessageEvent(text="same", source=source)
    )
    second = runner._serialize_pending_followup_event(
        MessageEvent(text="same", source=source)
    )
    assert first is not None and second is not None
    assert first["event_id"] != second["event_id"]
    assert store.mark_resume_pending(
        entry.session_key,
        reason="shutdown_pending_followup",
        pending_followup_text="same",
        pending_followup_events=[first],
    )

    assert not store.mark_resume_pending(
        entry.session_key,
        reason="shutdown_pending_followup",
        pending_followup_text="same",
        pending_followup_events=[second],
    )
    assert store._entries[entry.session_key].pending_followup_events == [first]


def test_failed_mark_does_not_publish_in_memory_ownership(
    tmp_path, monkeypatch
) -> None:
    store = _store(tmp_path)
    source = make_restart_source(chat_id="durable-chat")
    entry = store.get_or_create_session(source)

    def fail_save_entry(*args, **kwargs):
        raise OSError("forced routing write failure")

    monkeypatch.setattr(store, "_save", fail_save_entry)
    monkeypatch.setattr(store, "_save_entry", fail_save_entry)

    with pytest.raises(OSError, match="forced routing write failure"):
        store.mark_resume_pending(
            entry.session_key,
            reason="shutdown_pending_followup",
            pending_followup_text="test",
        )

    live = store._entries[entry.session_key]
    assert live.resume_pending is False
    assert live.resume_reason is None
    assert live.pending_followup_text is None


def test_failed_clear_keeps_in_memory_ownership_retryable(
    tmp_path, monkeypatch
) -> None:
    store = _store(tmp_path)
    source = make_restart_source(chat_id="durable-chat")
    entry = store.get_or_create_session(source)
    store.mark_resume_pending(
        entry.session_key,
        reason="shutdown_pending_followup",
        pending_followup_text="test",
    )

    def fail_save_entry(*args, **kwargs):
        raise OSError("forced routing write failure")

    monkeypatch.setattr(store, "_save", fail_save_entry)
    monkeypatch.setattr(store, "_save_entry", fail_save_entry)

    with pytest.raises(OSError, match="forced routing write failure"):
        store.clear_resume_pending(
            entry.session_key,
            acknowledge_pending_followup=True,
            expected_pending_followup_text="test",
        )

    live = store._entries[entry.session_key]
    assert live.resume_pending is True
    assert live.resume_reason == "shutdown_pending_followup"
    assert live.pending_followup_text == "test"


def test_append_to_transcript_reports_missing_db_as_not_durable(tmp_path) -> None:
    store = _store(tmp_path)
    store._db = None

    assert not store.append_to_transcript("sid", {"role": "user", "content": "x"})


def test_append_to_transcript_reports_retry_queue_as_not_durable(
    tmp_path, monkeypatch
) -> None:
    store = _store(tmp_path)

    def fail_append(*args, **kwargs):
        raise OSError("forced transcript write failure")

    monkeypatch.setattr(store, "_append_transcript_message", fail_append)

    assert not store.append_to_transcript("sid", {"role": "user", "content": "x"})
    assert store._dirty_transcripts["sid"]


def test_failed_head_is_not_reclassified_by_successful_tail() -> None:
    merged = _merge_queued_followup_completion(
        {"failed": True, "final_response": "failed"},
        {
            "failed": False,
            "final_response": "tail succeeded",
            "_direct_turn_succeeded": True,
        },
        "tail-event",
    )

    assert merged["_direct_turn_succeeded"] is False
    assert merged["_completed_pending_followup_event_ids"] == ["tail-event"]


@pytest.mark.asyncio
async def test_nondurable_transcript_receipt_does_not_ack_followup(
    monkeypatch, tmp_path
) -> None:
    from tests.gateway.test_internal_notification_marker_82888 import (
        _bootstrap,
        _event,
    )

    runner = _bootstrap(monkeypatch, tmp_path)
    runner._session_db = None
    event = _event(internal=True, text="durable replay")
    setattr(event, "_pending_followup_event_id", "event-1")
    runner._run_agent = AsyncMock(
        return_value={
            "final_response": "done",
            "messages": [],
            "tools": [],
            "history_offset": 0,
            "last_prompt_tokens": 0,
            "agent_persisted": False,
        }
    )
    append_to_transcript = MagicMock(return_value=False)
    clear_resume_pending = MagicMock(return_value=True)
    runner.session_store.append_to_transcript = append_to_transcript
    runner.session_store.clear_resume_pending = clear_resume_pending

    await runner._handle_message_with_agent(
        event,
        event.source,
        "agent:main:telegram:group:-1001:12345",
        1,
    )

    ack_calls = [
        call
        for call in clear_resume_pending.call_args_list
        if call.kwargs.get("acknowledge_pending_followup") is True
    ]
    assert ack_calls == []


@pytest.mark.asyncio
async def test_owned_but_nondurable_agent_write_does_not_ack_followup(
    monkeypatch, tmp_path
) -> None:
    from tests.gateway.test_internal_notification_marker_82888 import (
        _bootstrap,
        _event,
    )

    runner = _bootstrap(monkeypatch, tmp_path)
    event = _event(internal=True, text="durable replay")
    setattr(event, "_pending_followup_event_id", "event-1")
    runner._run_agent = AsyncMock(
        return_value={
            "final_response": "done",
            "messages": [],
            "tools": [],
            "history_offset": 0,
            "last_prompt_tokens": 0,
            "agent_persistence_owned": True,
            "agent_persisted": False,
        }
    )
    runner.session_store.append_to_transcript = MagicMock(return_value=True)
    clear_resume_pending = MagicMock(return_value=True)
    runner.session_store.clear_resume_pending = clear_resume_pending

    await runner._handle_message_with_agent(
        event,
        event.source,
        "agent:main:telegram:group:-1001:12345",
        1,
    )

    assert not [
        call
        for call in clear_resume_pending.call_args_list
        if call.kwargs.get("acknowledge_pending_followup") is True
    ]


@pytest.mark.asyncio
async def test_startup_resume_replays_original_followup_text() -> None:
    runner, adapter = make_restart_runner()
    source = make_restart_source(chat_id="durable-chat")
    pending_entry = SessionEntry(
        session_key="agent:main:telegram:dm:durable-chat",
        session_id="sid",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        origin=source,
        platform=Platform.TELEGRAM,
        chat_type="dm",
        resume_pending=True,
        resume_reason="shutdown_pending_followup",
        last_resume_marked_at=datetime.now(),
        pending_followup_text="test",
    )
    runner.session_store._entries = {pending_entry.session_key: pending_entry}
    adapter.handle_message = AsyncMock()

    scheduled = runner._schedule_resume_pending_sessions()
    await asyncio_sleep_once()

    assert scheduled == 1
    event = adapter.handle_message.await_args.args[0]
    assert event.internal is True
    assert event.text == "test"


@pytest.mark.asyncio
async def test_startup_resume_rehydrates_media_head_and_fifo_tail() -> None:
    runner, adapter = make_restart_runner()
    source = make_restart_source(chat_id="durable-chat")
    first = MessageEvent(
        text="caption",
        message_type=MessageType.PHOTO,
        source=source,
        media_urls=["/tmp/photo.jpg"],
        media_types=["image/jpeg"],
    )
    second = MessageEvent(text="second", source=source)
    payloads = [
        runner._serialize_pending_followup_event(first),
        runner._serialize_pending_followup_event(second),
    ]
    assert all(payload is not None for payload in payloads)
    pending_entry = SessionEntry(
        session_key="agent:main:telegram:dm:durable-chat",
        session_id="sid",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        origin=source,
        platform=Platform.TELEGRAM,
        chat_type="dm",
        resume_pending=True,
        resume_reason="shutdown_pending_followup",
        last_resume_marked_at=datetime.now(),
        pending_followup_text="caption",
        pending_followup_events=[
            payload for payload in payloads if payload is not None
        ],
    )
    runner.session_store._entries = {pending_entry.session_key: pending_entry}
    adapter.handle_message = AsyncMock()

    assert runner._schedule_resume_pending_sessions() == 1
    await asyncio_sleep_once()

    call = adapter.handle_message.await_args
    assert call is not None
    head = call.args[0]
    assert head.message_type is MessageType.PHOTO
    assert head.media_urls == ["/tmp/photo.jpg"]
    assert adapter._pending_messages[pending_entry.session_key].text == "second"


@pytest.mark.asyncio
async def test_startup_reauthorizes_every_durable_event_source() -> None:
    runner, adapter = make_restart_runner()
    owner = make_restart_source(chat_id="durable-chat")
    owner.user_id = "allowed"
    blocked = make_restart_source(chat_id="durable-chat")
    blocked.user_id = "blocked"
    first = runner._serialize_pending_followup_event(
        MessageEvent(text="first", source=owner)
    )
    second = runner._serialize_pending_followup_event(
        MessageEvent(text="second", source=blocked)
    )
    assert first is not None and second is not None
    pending_entry = SessionEntry(
        session_key="agent:main:telegram:dm:durable-chat",
        session_id="sid",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        origin=owner,
        platform=Platform.TELEGRAM,
        chat_type="dm",
        resume_pending=True,
        resume_reason="shutdown_pending_followup",
        last_resume_marked_at=datetime.now(),
        pending_followup_text="first",
        pending_followup_events=[first, second],
    )
    runner.session_store._entries = {pending_entry.session_key: pending_entry}
    adapter.handle_message = AsyncMock()
    setattr(
        runner,
        "_is_user_authorized",
        lambda source, **kwargs: source.user_id == "allowed",
    )

    assert runner._schedule_resume_pending_sessions() == 0
    await asyncio_sleep_once()
    adapter.handle_message.assert_not_awaited()


def test_malformed_durable_event_does_not_claim_startup_slot() -> None:
    runner, _ = make_restart_runner()
    source = make_restart_source(chat_id="durable-chat")
    pending_entry = SessionEntry(
        session_key="agent:main:telegram:dm:durable-chat",
        session_id="sid",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        origin=source,
        platform=Platform.TELEGRAM,
        chat_type="dm",
        resume_pending=True,
        resume_reason="shutdown_pending_followup",
        last_resume_marked_at=datetime.now(),
        pending_followup_events=[{"message_type": "text"}],
    )
    runner.session_store._entries = {pending_entry.session_key: pending_entry}

    assert runner._schedule_resume_pending_sessions() == 0
    assert runner._peek_session_state(pending_entry.session_key) is None


async def asyncio_sleep_once() -> None:
    import asyncio

    await asyncio.sleep(0)


def test_resume_note_uses_followup_but_persists_raw_user_text() -> None:
    recovery_message, persist_message = _prepare_resume_pending_message(
        "shutdown_pending_followup",
        "test",
    )

    assert "test" in recovery_message
    assert persist_message == "test"


def test_successful_clear_acknowledges_followup_payload(tmp_path) -> None:
    store = _store(tmp_path)
    source = make_restart_source(chat_id="durable-chat")
    entry = store.get_or_create_session(source)
    store.mark_resume_pending(
        entry.session_key,
        reason="shutdown_pending_followup",
        pending_followup_text="test",
    )

    assert (
        store.clear_resume_pending(
            entry.session_key,
            acknowledge_pending_followup=True,
            expected_pending_followup_text="test",
        )
        is True
    )
    restored = store._entries[entry.session_key]
    assert restored.resume_pending is False
    assert restored.pending_followup_text is None


def test_non_ack_clear_preserves_pending_followup(tmp_path) -> None:
    store = _store(tmp_path)
    source = make_restart_source(chat_id="durable-chat")
    entry = store.get_or_create_session(source)
    store.mark_resume_pending(
        entry.session_key,
        reason="shutdown_pending_followup",
        pending_followup_text="test",
    )

    assert (
        store.clear_resume_pending(
            entry.session_key,
            acknowledge_pending_followup=False,
        )
        is False
    )
    restored = store._entries[entry.session_key]
    assert restored.resume_pending is True
    assert restored.resume_reason == "shutdown_pending_followup"
    assert restored.pending_followup_text == "test"


@pytest.mark.asyncio
async def test_failed_durable_handoff_requeues_dequeued_event() -> None:
    runner, adapter = make_restart_runner()
    setattr(
        runner.async_session_store,
        "mark_resume_pending",
        AsyncMock(return_value=False),
    )
    source = make_restart_source(chat_id="durable-chat")
    event = MessageEvent(
        text="test",
        message_type=MessageType.TEXT,
        source=source,
    )

    preserved = await runner._preserve_draining_followup(
        "agent:main:telegram:dm:durable-chat",
        "test",
        pending_event=event,
        adapter=adapter,
    )

    assert preserved is False
    assert adapter._pending_messages["agent:main:telegram:dm:durable-chat"] is event


@pytest.mark.asyncio
async def test_successful_durable_handoff_keeps_adapter_slot_empty() -> None:
    runner, adapter = make_restart_runner()
    mark_resume_pending = AsyncMock(return_value=True)
    setattr(
        runner.async_session_store,
        "mark_resume_pending",
        mark_resume_pending,
    )
    source = make_restart_source(chat_id="durable-chat")
    event = MessageEvent(
        text="test",
        message_type=MessageType.TEXT,
        source=source,
    )

    preserved = await runner._preserve_draining_followup(
        "agent:main:telegram:dm:durable-chat",
        "test",
        pending_event=event,
        adapter=adapter,
    )

    assert preserved is True
    assert "agent:main:telegram:dm:durable-chat" not in adapter._pending_messages
    mark_resume_pending.assert_awaited_once()
    call = mark_resume_pending.await_args
    assert call is not None
    assert call.args == ("agent:main:telegram:dm:durable-chat",)
    assert call.kwargs["reason"] == "shutdown_pending_followup"
    assert call.kwargs["pending_followup_text"] == "test"
    assert [item["text"] for item in call.kwargs["pending_followup_events"]] == ["test"]


@pytest.mark.asyncio
async def test_failed_handoff_restores_dequeued_event_ahead_of_promoted_fifo() -> None:
    runner, adapter = make_restart_runner()
    setattr(
        runner.async_session_store,
        "mark_resume_pending",
        AsyncMock(return_value=False),
    )
    session_key = "agent:main:telegram:dm:durable-chat"
    source = make_restart_source(chat_id="durable-chat")
    dequeued = MessageEvent(
        text="first",
        message_type=MessageType.TEXT,
        source=source,
    )
    promoted = MessageEvent(
        text="second",
        message_type=MessageType.TEXT,
        source=source,
    )
    adapter._pending_messages[session_key] = promoted

    preserved = await runner._preserve_draining_followup(
        session_key,
        "first",
        pending_event=dequeued,
        adapter=adapter,
    )

    assert preserved is False
    assert adapter._pending_messages[session_key] is dequeued
    assert runner._session_state(session_key).conversation.queued_events == [promoted]


@pytest.mark.asyncio
async def test_successful_handoff_persists_complete_fifo_and_releases_local_copies() -> (
    None
):
    runner, adapter = make_restart_runner()
    mark_resume_pending = AsyncMock(return_value=True)
    setattr(runner.async_session_store, "mark_resume_pending", mark_resume_pending)
    session_key = "agent:main:telegram:dm:durable-chat"
    source = make_restart_source(chat_id="durable-chat")
    first = MessageEvent(text="first", source=source)
    second = MessageEvent(text="second", source=source)
    third = MessageEvent(text="third", source=source)
    adapter._pending_messages[session_key] = second
    runner._session_state(session_key).conversation.queued_events = [third]

    assert await runner._preserve_draining_followup(
        session_key,
        "first",
        pending_event=first,
        adapter=adapter,
        source=source,
    )

    call = mark_resume_pending.await_args
    assert call is not None
    payloads = call.kwargs["pending_followup_events"]
    assert [item["text"] for item in payloads] == ["first", "second", "third"]
    assert session_key not in adapter._pending_messages
    assert runner._session_state(session_key).conversation.queued_events == []


@pytest.mark.asyncio
async def test_handoff_does_not_delete_event_enqueued_during_durable_write() -> None:
    runner, adapter = make_restart_runner()
    session_key = "agent:main:telegram:dm:durable-chat"
    source = make_restart_source(chat_id="durable-chat")
    head = MessageEvent(text="head", source=source)
    late = MessageEvent(text="late-during-save", source=source)

    async def save_then_enqueue(*args, **kwargs):
        adapter._pending_messages[session_key] = late
        return True

    setattr(
        runner.async_session_store,
        "mark_resume_pending",
        AsyncMock(side_effect=save_then_enqueue),
    )

    assert await runner._preserve_draining_followup(
        session_key,
        head.text,
        pending_event=head,
        adapter=adapter,
        source=source,
    )
    assert adapter._pending_messages[session_key] is late


@pytest.mark.asyncio
async def test_drain_busy_path_persists_new_event_before_accepting_it() -> None:
    runner, adapter = make_restart_runner()
    session_key = "agent:main:telegram:dm:durable-chat"
    source = make_restart_source(chat_id="durable-chat")
    event = MessageEvent(text="late", source=source)
    runner._draining = True
    runner._restart_requested = True
    runner._busy_input_mode = "queue"
    setattr(runner, "_is_user_authorized", lambda *_args, **_kwargs: True)
    setattr(runner, "_adapter_for_source", lambda _source: adapter)
    adapter._send_with_retry = AsyncMock()
    append = AsyncMock(return_value=True)
    setattr(runner, "_append_draining_followup_event", append)

    assert await runner._handle_active_session_busy_message(event, session_key)

    append.assert_awaited_once_with(session_key, event)
    assert session_key not in adapter._pending_messages
    assert runner._session_state(session_key).conversation.queued_events == []
    assert "queued for the next turn" in adapter._send_with_retry.await_args.kwargs[
        "content"
    ]


@pytest.mark.asyncio
async def test_late_drain_arrival_waits_then_appends_to_durable_fifo() -> None:
    runner, adapter = make_restart_runner()
    session_key = "agent:main:telegram:dm:durable-chat"
    source = make_restart_source(chat_id="durable-chat")
    head = MessageEvent(text="head", source=source)
    late = MessageEvent(text="late", source=source)
    mark_started = asyncio.Event()
    allow_mark = asyncio.Event()
    durable_payloads = []

    async def get_entry(_session_key):
        return SimpleNamespace(pending_followup_events=list(durable_payloads))

    async def mark(*_args, **kwargs):
        mark_started.set()
        await allow_mark.wait()
        durable_payloads.extend(kwargs["pending_followup_events"])
        return True

    async def append(*_args, **kwargs):
        assert kwargs["expected_existing_event_ids"] == [
            durable_payloads[0]["event_id"]
        ]
        durable_payloads.extend(_args[1])
        return True

    runner.async_session_store.get = AsyncMock(side_effect=get_entry)
    runner.async_session_store.mark_resume_pending = AsyncMock(side_effect=mark)
    runner.async_session_store.append_pending_followup_events = AsyncMock(
        side_effect=append
    )
    setattr(runner, "_adapter_for_source", lambda _source: adapter)

    initial = asyncio.create_task(
        runner._preserve_draining_followup(
            session_key,
            head.text,
            pending_event=head,
            adapter=adapter,
            source=source,
        )
    )
    await mark_started.wait()
    late_task = asyncio.create_task(
        runner._append_draining_followup_event(session_key, late)
    )
    await asyncio.sleep(0)
    assert not late_task.done()
    allow_mark.set()

    assert await initial
    assert await late_task
    assert [item["text"] for item in durable_payloads] == ["head", "late"]
    assert session_key not in adapter._pending_messages
    assert runner._session_state(session_key).conversation.queued_events == []


@pytest.mark.asyncio
async def test_handoff_isolates_snapshot_from_in_place_merge_during_write() -> None:
    runner, adapter = make_restart_runner()
    session_key = "agent:main:telegram:dm:durable-chat"
    source = make_restart_source(chat_id="durable-chat")
    head = MessageEvent(text="head", source=source)
    staged = MessageEvent(text="second", source=source)
    adapter._pending_messages[session_key] = staged

    async def save_then_merge(*args, **kwargs):
        merge_pending_message_event(
            adapter._pending_messages,
            session_key,
            MessageEvent(text="late-merged", source=source),
            merge_text=True,
        )
        return True

    setattr(
        runner.async_session_store,
        "mark_resume_pending",
        AsyncMock(side_effect=save_then_merge),
    )

    assert await runner._preserve_draining_followup(
        session_key,
        head.text,
        pending_event=head,
        adapter=adapter,
        source=source,
    )
    assert adapter._pending_messages[session_key].text == "late-merged"


@pytest.mark.asyncio
async def test_serializer_exception_restores_complete_local_fifo() -> None:
    runner, adapter = make_restart_runner()
    session_key = "agent:main:telegram:dm:durable-chat"
    source = make_restart_source(chat_id="durable-chat")
    head = MessageEvent(text="head", source=source)
    staged = MessageEvent(text="second", source=source)
    adapter._pending_messages[session_key] = staged

    def fail_to_dict():
        raise RuntimeError("forced source serialization failure")

    source.to_dict = fail_to_dict

    assert not await runner._preserve_draining_followup(
        session_key,
        head.text,
        pending_event=head,
        adapter=adapter,
        source=source,
    )
    assert adapter._pending_messages[session_key] is head
    assert runner._session_state(session_key).conversation.queued_events == [staged]


@pytest.mark.asyncio
async def test_media_only_first_drain_event_uses_nonblank_legacy_head() -> None:
    runner, adapter = make_restart_runner()
    session_key = "agent:main:telegram:dm:durable-chat"
    source = make_restart_source(chat_id="durable-chat")
    event = MessageEvent(
        text="",
        message_type=MessageType.PHOTO,
        source=source,
        media_urls=["/tmp/photo.jpg"],
        media_types=["image/jpeg"],
    )
    mark = AsyncMock(return_value=True)
    runner.async_session_store.mark_resume_pending = mark

    assert await runner._preserve_draining_followup(
        session_key,
        "",
        pending_event=event,
        adapter=adapter,
        source=source,
    )

    legacy_head = mark.await_args.kwargs["pending_followup_text"]
    assert isinstance(legacy_head, str)
    assert legacy_head.strip()
    assert "/tmp/photo.jpg" in legacy_head


def test_media_event_roundtrips_through_durable_payload() -> None:
    runner, _ = make_restart_runner()
    source = make_restart_source(chat_id="durable-chat")
    event = MessageEvent(
        text="caption",
        message_type=MessageType.PHOTO,
        source=source,
        media_urls=["/tmp/photo.jpg"],
        media_types=["image/jpeg"],
        reply_to_message_id="reply-1",
        channel_prompt="channel instructions",
        metadata={"album": "a1"},
    )

    payload = runner._serialize_pending_followup_event(event)
    assert payload is not None
    restored = runner._deserialize_pending_followup_event(payload)

    assert restored is not None
    assert restored.message_type is MessageType.PHOTO
    assert restored.media_urls == ["/tmp/photo.jpg"]
    assert restored.media_types == ["image/jpeg"]
    assert restored.reply_to_message_id == "reply-1"
    assert restored.channel_prompt == "channel instructions"
    assert restored.metadata == {"album": "a1"}


@pytest.mark.parametrize("timestamp", [None, 7, "not-a-datetime"])
def test_irregular_timestamp_fails_serialization_closed(timestamp) -> None:
    runner, _ = make_restart_runner()
    source = make_restart_source(chat_id="durable-chat")
    event = MessageEvent(text="test", source=source)
    event.timestamp = timestamp

    assert runner._serialize_pending_followup_event(event) is None


@pytest.mark.asyncio
async def test_failed_text_only_handoff_restores_real_message_event() -> None:
    runner, adapter = make_restart_runner()
    setattr(
        runner.async_session_store,
        "mark_resume_pending",
        AsyncMock(return_value=False),
    )
    session_key = "agent:main:telegram:dm:durable-chat"
    source = make_restart_source(chat_id="durable-chat")

    preserved = await runner._preserve_draining_followup(
        session_key,
        "leftover steer",
        pending_event=None,
        adapter=adapter,
        source=source,
    )

    assert preserved is False
    restored = adapter._pending_messages[session_key]
    assert isinstance(restored, MessageEvent)
    assert restored.text == "leftover steer"
    assert restored.source is source


@pytest.mark.asyncio
async def test_role_only_authorization_is_not_persisted_as_restart_trust() -> None:
    runner, adapter = make_restart_runner()
    mark_resume_pending = AsyncMock(return_value=True)
    setattr(runner.async_session_store, "mark_resume_pending", mark_resume_pending)
    setattr(runner, "_is_user_authorized", lambda *args, **kwargs: False)
    session_key = "agent:main:discord:channel:durable-chat"
    source = make_restart_source(chat_id="durable-chat")
    source.platform = Platform.DISCORD
    source.role_authorized = True
    event = MessageEvent(text="role-authorized", source=source)

    preserved = await runner._preserve_draining_followup(
        session_key,
        event.text,
        pending_event=event,
        adapter=adapter,
        source=source,
    )

    assert preserved is False
    assert adapter._pending_messages[session_key] is event
    mark_resume_pending.assert_not_awaited()


def test_acknowledging_head_keeps_remaining_fifo_durable(tmp_path) -> None:
    store = _store(tmp_path)
    source = make_restart_source(chat_id="durable-chat")
    entry = store.get_or_create_session(source)
    payloads = [
        {"event_id": "event-1", "text": "first", "message_type": "text"},
        {"event_id": "event-2", "text": "second", "message_type": "text"},
    ]
    assert store.mark_resume_pending(
        entry.session_key,
        reason="shutdown_pending_followup",
        pending_followup_text="first",
        pending_followup_events=payloads,
    )

    assert store.clear_resume_pending(
        entry.session_key,
        acknowledge_pending_followup=True,
        expected_followup_event_ids=["event-1"],
    )

    live = store._entries[entry.session_key]
    assert live.resume_pending is True
    assert live.pending_followup_text == "second"
    assert live.pending_followup_events == [payloads[1]]


def test_append_pending_followup_events_extends_existing_fifo_atomically(tmp_path) -> None:
    store = _store(tmp_path)
    source = make_restart_source(chat_id="durable-chat")
    entry = store.get_or_create_session(source)
    first = {"event_id": "event-1", "text": "first", "message_type": "text"}
    second = {"event_id": "event-2", "text": "second", "message_type": "text"}
    assert store.mark_resume_pending(
        entry.session_key,
        reason="shutdown_pending_followup",
        pending_followup_text="first",
        pending_followup_events=[first],
    )

    assert store.append_pending_followup_events(
        entry.session_key,
        [second],
        expected_existing_event_ids=["event-1"],
    )

    live = store._entries[entry.session_key]
    assert [item["event_id"] for item in live.pending_followup_events] == [
        "event-1",
        "event-2",
    ]
    assert live.pending_followup_text == "first"


def test_append_pending_followup_events_is_failure_atomic(tmp_path) -> None:
    store = _store(tmp_path)
    source = make_restart_source(chat_id="durable-chat")
    entry = store.get_or_create_session(source)
    first = {"event_id": "event-1", "text": "first", "message_type": "text"}
    second = {"event_id": "event-2", "text": "second", "message_type": "text"}
    assert store.mark_resume_pending(
        entry.session_key,
        reason="shutdown_pending_followup",
        pending_followup_text="first",
        pending_followup_events=[first],
    )

    with patch.object(store, "_save_entry", side_effect=OSError("forced write failure")):
        with pytest.raises(OSError, match="forced write failure"):
            store.append_pending_followup_events(
                entry.session_key,
                [second],
                expected_existing_event_ids=["event-1"],
            )

    live = store._entries[entry.session_key]
    assert live.pending_followup_events == [first]


def test_wrong_turn_cannot_acknowledge_durable_head(tmp_path) -> None:
    store = _store(tmp_path)
    source = make_restart_source(chat_id="durable-chat")
    entry = store.get_or_create_session(source)
    payload = {"event_id": "durable-head", "text": "newer", "message_type": "text"}
    assert store.mark_resume_pending(
        entry.session_key,
        reason="shutdown_pending_followup",
        pending_followup_text="newer",
        pending_followup_events=[payload],
    )

    assert not store.clear_resume_pending(
        entry.session_key,
        acknowledge_pending_followup=True,
        expected_followup_event_ids=["different-turn"],
    )
    live = store._entries[entry.session_key]
    assert live.resume_pending is True
    assert live.pending_followup_events == [payload]
