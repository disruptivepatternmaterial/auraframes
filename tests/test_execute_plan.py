"""Offline tests for `auraframes.sync.execute_plan` (Phase 8 Plan 02, Task 2;
rewritten for the batched write-path semantics, quick task 260708-fyr).

`execute_plan` is exercised against the REST layer mocked via
`offline_aura()` (`httpx.MockTransport`), while S3/SQS are duck-typed fakes
injected as `s3_client=`/`sqs_client=` (RESEARCH.md Don't-Hand-Roll: inject
fakes, not a botocore Stubber). Zero network access, zero AWS credentials --
no real `S3Client`/`SQSClient` is ever constructed here.

A batched upload chunk issues exactly ONE `select_asset` call (carrying every
prepped file's local_identifier) and ONE `batch_update` call (carrying every
prepped file's `AssetPartial`) -- not one round-trip per file. Per-file
attribution is recovered from `batch_update`'s `successes[].local_identifier`.
A delete chunk issues exactly ONE `remove_asset` call carrying every asset id
in the chunk.
"""
import httpx
import pytest
from loguru import logger
from PIL import Image

from auraframes.models.asset import Asset
from auraframes.sync import SyncPlan, execute_plan
from tests.offline import offline_aura

FRAME_ID = 'frame-fake-0001'
SELECT_ASSET_PATH = f'/v5/frames/{FRAME_ID}/select_asset.json'
REMOVE_ASSET_PATH = f'/v5/frames/{FRAME_ID}/remove_asset.json'
# No `.json` suffix -- deliberate, matches the app and is live-confirmed.
EXCLUDE_ASSET_PATH = f'/v5/frames/{FRAME_ID}/exclude_asset'
BATCH_UPDATE_PATH = '/v5/assets/batch_update.json'


@pytest.fixture(autouse=True)
def _reset_loguru():
    # loguru's `logger` is a process-global singleton; execute_plan's
    # trailing debug log could otherwise fire against a sink torn down by a
    # prior test (see tests/test_cli_sync.py's identical rationale).
    logger.remove()
    yield
    logger.remove()


def _write_jpeg(path, color=(255, 0, 0)):
    Image.new('RGB', (4, 4), color).save(path, format='JPEG')


def _asset(id_):
    return Asset.model_construct(id=id_, md5_hash='deadbeef', taken_at='2024-03-11T12:00:00.000Z')


def _default_overrides():
    """Canned success responses for every write endpoint execute_plan can
    reach -- select_asset/remove_asset report zero failures, batch_update
    acknowledges every local_identifier it was sent. MockTransport routes
    purely on path, so `_AckAllBatchUpdate` (installed per-test where needed)
    is what actually inspects the payload; this default response is only
    used by tests that don't care about per-file attribution."""
    return {
        SELECT_ASSET_PATH: httpx.Response(200, json={'number_failed': 0}),
        REMOVE_ASSET_PATH: httpx.Response(200, json={'number_failed': 0}),
        EXCLUDE_ASSET_PATH: httpx.Response(200, json={'number_failed': 0}),
        BATCH_UPDATE_PATH: httpx.Response(200, json={
            'ids': ['local-id'],
            'successes': [{'id': 'new-asset-id', 'local_identifier': 'local-id'}],
        }),
    }


class _FakeS3Client:
    """Duck-typed S3 fake -- records every upload_file call (and, if given a
    shared list, appends to it for cross-fake ordering proof)."""

    def __init__(self, call_order=None):
        self.upload_calls = []
        self.call_order = call_order if call_order is not None else []

    def upload_file(self, data: bytes, extension: str):
        self.upload_calls.append((data, extension))
        self.call_order.append('upload')
        return f'uploaded-{len(self.upload_calls)}{extension}', 'fake-md5-hash'


class _FakeSQSClient:
    """Duck-typed SQS fake -- get_queue_url records the frame_id it was
    queried with; receive_message is a no-op (execute_plan's chunk poll is
    best-effort/observational only, never used to gate success)."""

    def __init__(self):
        self.queue_url_requests = []

    def get_queue_url(self, frame_id: str):
        self.queue_url_requests.append(frame_id)
        return f'https://sqs.fake/{frame_id}'

    def receive_message(self, queue_url, wait_time_seconds=5):
        return {}


def _install_ack_all_batch_update(aura):
    """Monkeypatch `aura.asset_api.batch_update` to acknowledge every
    local_identifier it is sent, recording each call's list of sent
    `AssetPartial`s so a test can assert call counts / call shapes.

    Established pattern (per PLAN context): the offline `MockTransport`
    router discriminates only by path, so per-payload behavior (echoing
    back exactly what was sent) requires monkeypatching the wrapper
    directly rather than a canned httpx.Response.
    """
    from auraframes.models.asset import AssetPartialId

    calls: list = []

    def _fake_batch_update(assets):
        items = assets if isinstance(assets, list) else [assets]
        calls.append(items)
        ids = [item.local_identifier for item in items]
        successes = [{'id': f'new-{lid}', 'local_identifier': lid} for lid in ids]
        return ids, [AssetPartialId(**s) for s in successes]

    aura.asset_api.batch_update = _fake_batch_update
    return calls


def _install_select_asset_recorder(aura):
    """Monkeypatch `aura.frame_api.select_asset` to record each call's list
    of sent `AssetPartialId`s (as local_identifiers) while preserving the
    real success (`number_failed=0`) behavior, so call-count/shape can be
    asserted without relying on MockTransport's path-only routing."""
    calls: list = []

    def _fake_select_asset(frame_id, asset_partial_ids):
        items = asset_partial_ids if isinstance(asset_partial_ids, list) else [asset_partial_ids]
        calls.append([item.local_identifier for item in items])
        return 0

    aura.frame_api.select_asset = _fake_select_asset
    return calls


def test_execute_plan_happy_path_uploads_and_deletes(tmp_path):
    path_a = tmp_path / 'a.jpg'
    path_b = tmp_path / 'b.jpg'
    _write_jpeg(path_a)
    _write_jpeg(path_b)
    plan = SyncPlan(to_upload=[path_a, path_b], to_delete=[_asset('asset-to-delete')])

    aura = offline_aura(overrides=_default_overrides())
    select_calls = _install_select_asset_recorder(aura)
    batch_calls = _install_ack_all_batch_update(aura)
    s3 = _FakeS3Client()
    sqs = _FakeSQSClient()

    result = execute_plan(plan, aura, FRAME_ID, s3_client=s3, sqs_client=sqs, sleep=lambda *_: None)

    assert result.upload_succeeded == 2
    assert result.delete_succeeded == 1
    assert result.upload_failures == []
    assert result.delete_failures == []
    assert len(s3.upload_calls) == 2
    assert sqs.queue_url_requests == [FRAME_ID]

    # Exactly ONE select_asset + ONE batch_update call for the whole
    # (single-chunk) upload batch -- not one round-trip per file.
    assert len(select_calls) == 1
    assert len(select_calls[0]) == 2
    assert len(batch_calls) == 1
    assert {path for path, _ in result.upload_ids} == {path_a, path_b}
    assert all(asset_id.startswith('new-') for _, asset_id in result.upload_ids)
    assert len({asset_id for _, asset_id in result.upload_ids}) == 2


def test_execute_plan_records_upload_ids_only_from_success_ids(tmp_path):
    """Manifest ownership must come from successes[].id, never from md5."""
    path_a = tmp_path / 'a.jpg'
    path_b = tmp_path / 'b.jpg'
    _write_jpeg(path_a)
    _write_jpeg(path_b)
    plan = SyncPlan(to_upload=[path_a, path_b], to_delete=[])

    aura = offline_aura(overrides=_default_overrides())

    def _ids_from_successes(assets):
        from auraframes.models.asset import AssetPartialId
        items = assets if isinstance(assets, list) else [assets]
        ids = [item.local_identifier for item in items]
        successes = [
            AssetPartialId(id=f'asset-{lid}', local_identifier=lid)
            for lid in ids
        ]
        return ids, successes

    aura.asset_api.batch_update = _ids_from_successes
    result = execute_plan(
        plan, aura, FRAME_ID,
        s3_client=_FakeS3Client(), sqs_client=_FakeSQSClient(), sleep=lambda *_: None,
    )

    assert result.upload_succeeded == 2
    assert {path for path, _ in result.upload_ids} == {path_a, path_b}
    assert all(asset_id.startswith('asset-') for _, asset_id in result.upload_ids)


def test_prep_upload_fails_closed_on_png(tmp_path):
    from auraframes.sync import _prep_upload

    path = tmp_path / 'x.png'
    Image.new('RGB', (4, 4), (0, 255, 0)).save(path, format='PNG')
    with pytest.raises(ValueError, match='Unsupported upload extension'):
        _prep_upload(path, _FakeS3Client())


def test_execute_plan_partial_batch_update_splits_upload_succeeded_and_failures(tmp_path):
    # A PARTIAL batch_update successes response (some local_identifiers
    # acknowledged, some absent) must yield a correct mixed
    # upload_succeeded/upload_failures split, naming the right Paths.
    path_a = tmp_path / 'a.jpg'
    path_b = tmp_path / 'b.jpg'
    path_c = tmp_path / 'c.jpg'
    _write_jpeg(path_a)
    _write_jpeg(path_b)
    _write_jpeg(path_c)
    plan = SyncPlan(to_upload=[path_a, path_b, path_c], to_delete=[])

    aura = offline_aura(overrides=_default_overrides())

    def _partial_batch_update(assets):
        from auraframes.models.asset import AssetPartialId
        items = assets if isinstance(assets, list) else [assets]
        ids = [item.local_identifier for item in items]
        # Acknowledge only the FIRST and LAST sent local_identifier --
        # sorted(plan.to_upload) processes a, b, c in that order, so this
        # acks a.jpg and c.jpg but drops b.jpg.
        successes = [
            {'id': f'new-{lid}', 'local_identifier': lid}
            for lid in (ids[0], ids[-1])
        ]
        return ids, [AssetPartialId(**s) for s in successes]

    aura.asset_api.batch_update = _partial_batch_update

    result = execute_plan(
        plan, aura, FRAME_ID,
        s3_client=_FakeS3Client(), sqs_client=_FakeSQSClient(), sleep=lambda *_: None,
    )

    assert result.upload_succeeded == 2
    assert len(result.upload_failures) == 1
    failed_path, message = result.upload_failures[0]
    assert failed_path == path_b
    assert 'not acknowledged' in message
    assert {path for path, _ in result.upload_ids} == {path_a, path_c}


def test_execute_plan_chunks_uploads_past_batch_size(tmp_path):
    # More than batch_size files must split into multiple chunks, each
    # issuing its own select_asset + batch_update call pair, paced by the
    # injected throttle.
    paths = []
    for i in range(5):
        p = tmp_path / f'{i:02d}.jpg'
        _write_jpeg(p)
        paths.append(p)
    plan = SyncPlan(to_upload=paths, to_delete=[])

    aura = offline_aura(overrides=_default_overrides())
    select_calls = _install_select_asset_recorder(aura)
    batch_calls = _install_ack_all_batch_update(aura)

    sleeps: list = []

    result = execute_plan(
        plan, aura, FRAME_ID,
        s3_client=_FakeS3Client(), sqs_client=_FakeSQSClient(),
        sleep=lambda s: sleeps.append(s),
        batch_size=2,
        chunk_delay_seconds=0,  # isolate this test to the per-call throttle only
    )

    assert result.upload_succeeded == 5
    assert result.upload_failures == []

    # 5 files at batch_size=2 -> chunks of [2, 2, 1] -> 3 chunks.
    assert len(select_calls) == 3
    assert len(batch_calls) == 3
    assert [len(c) for c in select_calls] == [2, 2, 1]
    assert [len(c) for c in batch_calls] == [2, 2, 1]

    # Each chunk paces 2 throttled write calls (select_asset + batch_update).
    assert len(sleeps) == 3 * 2


def test_execute_plan_delete_chunk_failure_records_whole_chunk(monkeypatch):
    # remove_asset returns only a count, so a raised chunk-level failure
    # attributes ALL deletes in that chunk as failed (coarse per-chunk
    # attribution -- documented tradeoff, unlike upload's per-file
    # attribution via batch_update successes).
    plan = SyncPlan(to_upload=[], to_delete=[_asset('asset-good'), _asset('asset-bad')])

    aura = offline_aura(overrides=_default_overrides())

    def _failing_remove_asset(frame_id, asset_partial_ids):
        raise RuntimeError('simulated remove_asset chunk failure')

    monkeypatch.setattr(aura.frame_api, 'remove_asset', _failing_remove_asset)

    result = execute_plan(
        plan, aura, FRAME_ID,
        s3_client=_FakeS3Client(), sqs_client=_FakeSQSClient(), sleep=lambda *_: None,
        # Pinned to the mode whose primitive this test fakes; the coarse
        # per-chunk attribution it asserts is mode-agnostic.
        removal_mode='delete',
    )

    assert result.delete_succeeded == 0
    assert len(result.delete_failures) == 2
    assert {aid for aid, _ in result.delete_failures} == {'asset-good', 'asset-bad'}
    assert all('simulated remove_asset chunk failure' in msg for _, msg in result.delete_failures)


def test_execute_plan_reports_progress_per_item(tmp_path):
    path_a = tmp_path / 'a.jpg'
    path_b = tmp_path / 'b.jpg'
    _write_jpeg(path_a)
    _write_jpeg(path_b)
    plan = SyncPlan(to_upload=[path_a, path_b], to_delete=[_asset('asset-good')])

    aura = offline_aura(overrides=_default_overrides())

    def _partial_batch_update(assets):
        from auraframes.models.asset import AssetPartialId
        items = assets if isinstance(assets, list) else [assets]
        # sorted(plan.to_upload) processes a.jpg first -- ack only b.jpg's
        # local_identifier (the second sent item), dropping a.jpg's.
        acked = items[1].local_identifier
        return [item.local_identifier for item in items], [
            AssetPartialId(id='new-asset', local_identifier=acked)
        ]

    aura.asset_api.batch_update = _partial_batch_update

    recorded: list = []

    def _recording_progress(kind, identifier, ok):
        recorded.append((kind, identifier, ok))

    result = execute_plan(
        plan, aura, FRAME_ID, s3_client=_FakeS3Client(), sqs_client=_FakeSQSClient(), sleep=lambda *_: None,
        progress=_recording_progress,
    )

    assert result.upload_succeeded == 1
    assert len(result.upload_failures) == 1

    assert len(recorded) == len(plan.to_upload) + len(plan.to_delete)

    failed_upload = [r for r in recorded if r[0] == 'upload' and r[2] is False]
    assert failed_upload == [('upload', path_a, False)]

    succeeded_upload = [r for r in recorded if r[0] == 'upload' and r[2] is True]
    assert succeeded_upload == [('upload', path_b, True)]

    delete_entries = [r for r in recorded if r[0] == 'delete']
    assert delete_entries == [('delete', 'asset-good', True)]

    # Every 'upload' entry appears before every 'delete' entry (D-09).
    upload_indices = [i for i, r in enumerate(recorded) if r[0] == 'upload']
    delete_indices = [i for i, r in enumerate(recorded) if r[0] == 'delete']
    assert max(upload_indices) < min(delete_indices)


def test_execute_plan_all_uploads_precede_all_deletes(tmp_path, monkeypatch):
    path_a = tmp_path / 'a.jpg'
    _write_jpeg(path_a)
    plan = SyncPlan(to_upload=[path_a], to_delete=[_asset('asset-1'), _asset('asset-2')])

    aura = offline_aura(overrides=_default_overrides())
    _install_ack_all_batch_update(aura)
    call_order: list = []
    s3 = _FakeS3Client(call_order=call_order)
    sqs = _FakeSQSClient()

    original_remove_asset = aura.frame_api.remove_asset

    def _recording_remove_asset(frame_id, asset_partial_ids):
        call_order.append('delete')
        return original_remove_asset(frame_id, asset_partial_ids)

    monkeypatch.setattr(aura.frame_api, 'remove_asset', _recording_remove_asset)

    result = execute_plan(plan, aura, FRAME_ID, s3_client=s3, sqs_client=sqs, sleep=lambda *_: None,
                          removal_mode='delete')  # pinned: this test records remove_asset

    assert result.upload_succeeded == 1
    assert result.delete_succeeded == 2
    # Both deletes are chunked into a SINGLE remove_asset call, so 'delete'
    # appears exactly once in call_order -- after the single 'upload'.
    assert call_order == ['upload', 'delete']


# --- 3-tier removal + always-runs re-show (HIDE-03/HIDE-04, D-01/D-03/D-05) --

class _RecordingPrimitives:
    """Replaces every mutating primitive execute_plan can reach for removal or
    re-show, recording the asset ids each was called with. Lets a test assert
    which single primitive a removal_mode selected -- and, just as important,
    which ones it left alone."""

    def __init__(self, aura, fail=None):
        self.exclude, self.remove, self.select, self.hard_delete = [], [], [], []
        self._fail = fail or set()
        aura.frame_api.exclude_asset = self._record('exclude', self.exclude)
        aura.frame_api.remove_asset = self._record('remove', self.remove)
        aura.frame_api.select_asset = self._record('select', self.select)

        def _delete_asset(asset):
            if 'hard_delete' in self._fail:
                raise RuntimeError('fake hard_delete failure')
            self.hard_delete.append(asset.id)

        aura.asset_api.delete_asset = _delete_asset

    def _record(self, name, sink):
        def call(frame_id, asset_partial_ids):
            if name in self._fail:
                raise RuntimeError(f'fake {name} failure')
            items = asset_partial_ids if isinstance(asset_partial_ids, list) else [asset_partial_ids]
            sink.append([item.id for item in items])
            return 0
        return call


def _run(plan, aura, **kwargs):
    return execute_plan(plan, aura, FRAME_ID, s3_client=_FakeS3Client(),
                        sqs_client=_FakeSQSClient(), sleep=lambda *_: None, **kwargs)


def test_hide_mode_excludes_and_never_removes_or_deletes():
    """The default mode must reach ONLY the non-destructive primitive."""
    plan = SyncPlan(to_upload=[], to_delete=[_asset('a1'), _asset('a2')])
    aura = offline_aura(overrides=_default_overrides())
    spy = _RecordingPrimitives(aura)

    result = _run(plan, aura)  # removal_mode defaults to 'hide'

    assert spy.exclude == [['a1', 'a2']], "hide must batch both ids into one exclude_asset call"
    assert spy.remove == [] and spy.hard_delete == []
    assert result.delete_succeeded == 2


def test_delete_mode_uses_remove_asset():
    plan = SyncPlan(to_upload=[], to_delete=[_asset('a1')])
    aura = offline_aura(overrides=_default_overrides())
    spy = _RecordingPrimitives(aura)

    _run(plan, aura, removal_mode='delete')

    assert spy.remove == [['a1']]
    assert spy.exclude == [] and spy.hard_delete == []


def test_hard_delete_mode_calls_delete_asset_once_per_asset():
    """delete_asset is asset-scoped -- there is no batch form, so it is one
    call per asset (Pitfall 3)."""
    plan = SyncPlan(to_upload=[], to_delete=[_asset('a1'), _asset('a2')])
    aura = offline_aura(overrides=_default_overrides())
    spy = _RecordingPrimitives(aura)

    _run(plan, aura, removal_mode='hard_delete')

    assert spy.hard_delete == ['a1', 'a2']
    assert spy.exclude == [] and spy.remove == []


@pytest.mark.parametrize('removal_mode', ['hide', 'delete', 'hard_delete'])
def test_reshow_loop_runs_under_every_removal_mode(removal_mode):
    """Re-showing is not a removal concern -- a photo the user brought back
    must be un-hidden regardless of how removals are being handled (D-05)."""
    plan = SyncPlan(to_upload=[], to_delete=[], to_reshow=[_asset('r1'), _asset('r2')])
    aura = offline_aura(overrides=_default_overrides())
    spy = _RecordingPrimitives(aura)

    result = _run(plan, aura, removal_mode=removal_mode)

    assert spy.select == [['r1', 'r2']]
    assert result.reshow_succeeded == 2
    assert result.reshow_failures == []


def test_reshow_failure_is_attributed_to_every_asset_in_the_chunk():
    plan = SyncPlan(to_upload=[], to_delete=[], to_reshow=[_asset('r1'), _asset('r2')])
    aura = offline_aura(overrides=_default_overrides())
    _RecordingPrimitives(aura, fail={'select'})

    result = _run(plan, aura)

    assert result.reshow_succeeded == 0
    assert [aid for aid, _ in result.reshow_failures] == ['r1', 'r2']
    assert all('fake select failure' in err for _, err in result.reshow_failures)


def test_reshow_precedes_removal():
    """Non-destructive work first: a budget or lockout that stops the run
    part-way should have already restored the photos the user wants back."""
    order = []
    plan = SyncPlan(to_upload=[], to_delete=[_asset('a1')], to_reshow=[_asset('r1')])
    aura = offline_aura(overrides=_default_overrides())

    def _select(frame_id, ids):
        order.append('reshow')
        return 0

    def _exclude(frame_id, ids):
        order.append('removal')
        return 0

    aura.frame_api.select_asset = _select
    aura.frame_api.exclude_asset = _exclude

    _run(plan, aura)

    assert order == ['reshow', 'removal']
