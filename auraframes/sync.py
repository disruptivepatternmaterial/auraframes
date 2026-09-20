"""Sync engine: pure dry-run core (Phase 7) plus the single mutating
execute path (Phase 8, batched in quick task 260708-fyr).

Contains a recursive local-directory scanner + content hasher
(`scan_directory`) and a pure diff function (`compute_plan`) that classifies
a frame's assets into upload / delete / unchanged against local hashes --
both remain pure, no I/O beyond reading local file bytes, no mutation.

`execute_plan()` is this module's ONLY mutating function -- the sole place
that calls select_asset/S3 upload/batch_update (for `to_upload`) and the
chosen removal primitive (for `to_delete`). Default `removal_mode='hide'`
calls `exclude_asset`. `'delete'` calls `remove_asset`. `'hard_delete'`
calls `delete_asset`. The destructive primitives are unreachable unless a
caller names them.
Uploads are attempted before any delete (D-09), and a single item's
failure is caught, recorded with its identity, and the loop continues
rather than aborting (D-08), with results reported back as a separated
`ExecutionResult` (D-10).

Both `select_asset` and `batch_update` are native Pushd BATCH endpoints
(the official app sends a whole collection in one `{"assets":[...]}` call
rather than one call per asset -- see
`.planning/debug/resolved/select-asset-401-unauthorized.md`). `execute_plan`
chunks `to_upload`/`to_delete` at `WRITE_BATCH_SIZE` and, per upload chunk,
performs per-file S3 prep (S3 is AWS, not the Pushd anti-abuse surface, so
those uploads stay per-file) followed by exactly ONE `select_asset` call and
ONE `batch_update` call carrying every prepped file in the chunk -- not the
double `select_asset` + per-file round-trip the original single-item flow
used. Per-file attribution is recovered from `batch_update`'s
`successes[].local_identifier`: any prepped file whose local_identifier is
absent from `successes` is attributed a failure. A delete chunk issues one
`remove_asset` call for every asset id in the chunk; since `remove_asset`
returns only a failure COUNT (never per-item), a chunk-level failure
attributes ALL deletes in that chunk (coarser than upload attribution --
documented tradeoff, T-fyr-01).
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image
from loguru import logger

from auraframes.aws.s3client import get_md5
from auraframes.client import RateLimitError
from auraframes.models.asset import AssetPartial, AssetPartialId
from auraframes.utils.dt import format_dt_to_aura, get_utc_now

# Only these extensions are eligible for content-hash diffing (D-02). Phase 6
# confirmed md5_hash is populated for photo assets but null for video assets,
# so videos/non-images are excluded here rather than diffed unsafely.
ELIGIBLE_EXTENSIONS = frozenset({'.jpg', '.jpeg', '.png', '.heic'})

# Maps a local file's suffix to the Apple UTI the API expects in
# `data_uti`. Deliberately narrower than ELIGIBLE_EXTENSIONS: '.heic'
# has no registered Pillow decoder in this environment (no pillow-heif
# installed) so `Image.open()` on a `.heic` path always raises before a
# UTI would even be used, and '.png' has no verified-correct UTI value
# yet -- both are left unmapped so `_execute_upload` fails closed with a
# named reason instead of mislabeling the upload server-side.
_DATA_UTI_BY_SUFFIX = {'.jpg': 'public.jpeg', '.jpeg': 'public.jpeg'}

# Seconds to pause before each write network call (select_asset /
# batch_update / remove_asset) so a bulk apply is paced rather than fired as
# a rapid burst. Root cause of the select-asset-401-unauthorized session:
# the Pushd anti-abuse layer trips on burst write volume (a 120-file apply
# fired ~300ms apart, each upload issuing 2x select_asset + 1x batch_update,
# got 401 on every call then escalated to a login lockout) while the
# human-paced phone app is never flagged. Batching (WRITE_BATCH_SIZE) fixes
# the root cause -- call COUNT -- by collapsing ~3N Pushd write calls to ~2
# per chunk; this throttle remains as a second, independent layer of
# defense that paces whatever write calls remain (2 per upload chunk, 1 per
# delete chunk) so even a batched apply is never fired as a rapid burst. It
# is deliberately conservative (a full apply is a rare, batch operation, not
# a latency-sensitive path). Callers can override via `throttle_seconds` (0
# disables) and inject `sleep` for tests.
WRITE_THROTTLE_SECONDS = 0.5

# Maximum number of items (uploads or deletes) batched into a single
# select_asset/batch_update/remove_asset call. The official Aura Android app
# has no observed hard client-side cap and does not appear to chunk at all in
# normal use, but a single call carrying hundreds of assets is untested
# territory -- it could re-trip the anti-abuse layer on payload size/shape
# rather than call volume, or hit an undocumented server-side limit. 50 is a
# conservative chunk size: small enough to stay well within any plausible
# server limit, large enough that the ~3N-to-~2-per-chunk call-count
# reduction is realized in practice. Injectable via `batch_size` so offline
# tests can exercise the chunk-boundary logic with a small size.
WRITE_BATCH_SIZE = 50

# Seconds to pause BETWEEN write chunks (separate from WRITE_THROTTLE_SECONDS,
# which paces individual network calls). This spaces out the per-chunk write
# bursts over the whole apply so the run looks human-paced rather than firing
# every chunk back-to-back -- a third layer of defense on top of batching
# (call count) and the per-call throttle. Motivated live: after the batched
# fix, a first 50-file chunk succeeded but the immediately-following second
# chunk still tripped the anti-abuse layer, consistent with a cumulative
# write-volume-over-time limit, not a per-call one. Realized via the injected
# `sleep` in 1s steps and surfaced through `on_wait` so the CLI can show a
# live countdown instead of a frozen progress bar. Injectable via
# `chunk_delay_seconds` (0 disables). Skipped before the first write chunk.
WRITE_CHUNK_DELAY_SECONDS = 5.0

# Number of write items that must fail in an unbroken run before execute_plan
# aborts the whole batch. Root cause of this session's REOPENED gap: the Pushd
# anti-abuse trip does NOT reliably announce itself with the 429/475 that
# `RateLimitError` catches. The live regression showed it as a plain HTTP 401
# on every write after the 7th succeeded, so execute_plan caught each of the
# 103 post-trip failures per-item (D-08) and kept hammering the throttling
# server ~103 more times instead of backing off. A RUN of consecutive failures
# is the reliable, STATUS-CODE-AGNOSTIC signature of a systemic cut-off, while a
# genuine isolated failure (one bad/expired asset ref, one permissions edge) is
# a single failure surrounded by successes and never reaches the threshold.
# Deliberately counts ANY caught per-item write failure (plain 401, 5xx,
# network error, ValueError) rather than sniffing the status -- the whole
# lesson of this session is that the trip cannot be recognised from the HTTP
# status alone. This is a BACKSTOP below the RateLimitError fast-path: a 429/475
# still aborts on its FIRST occurrence via its own branch; this catches trips
# the status code hides. N=5 balances catching the lockout early (5 wasted calls
# vs 103) against tolerating a small unlucky cluster of independent failures.
MAX_CONSECUTIVE_WRITE_FAILURES = 5


class ConsecutiveWriteFailureError(Exception):
    """Raised by `execute_plan` when `max_consecutive_failures` write items
    fail in an unbroken run -- the signature of an account lockout / systemic
    cut-off that did NOT announce itself with a 429/475 (`RateLimitError`),
    e.g. the plain-HTTP-401 form of the Pushd anti-abuse trip seen in the
    select-asset-401-unauthorized session's live regression.

    Distinct from `RateLimitError` so the CLI surfaces a separate "run of
    failures -- probable lockout, stop and investigate" message rather than the
    throttle-specific Retry-After wording. Carries the run length (`count`), the
    last per-item error string (`last_error`), and the partial `ExecutionResult`
    accumulated before the abort (`result`) so the caller can report what
    succeeded first.
    """

    def __init__(self, count: int, last_error: str, result: "ExecutionResult"):
        self.count = count
        self.last_error = last_error
        self.result = result
        super().__init__(
            f'Aborted after {count} consecutive write failures -- this is the '
            f'signature of an account lockout or a systemic cut-off (the Pushd '
            f'anti-abuse trip can surface as a run of plain HTTP 401s with no '
            f'Retry-After), not isolated per-item errors. Stop and investigate '
            f'before retrying; continued calls may extend a lockout. '
            f'Last error: {last_error}'
        )


@dataclass
class ScanResult:
    local_hashes: dict[str, list[Path]]
    skipped_non_image: int


def scan_directory(root: Path) -> ScanResult:
    """Recursively walk `root` (D-01) and content-hash every eligible image
    file, grouping local paths by base64-MD5 hash (D-05: byte-identical
    files collapse to one hash key/logical want).

    Non-eligible files (videos, dotfiles, arbitrary junk) are counted in
    `skipped_non_image` and never error (D-02, D-03). `Path.rglob` does not
    recurse into symlinked directories, and symlinks are excluded
    explicitly below (in addition to the `is_file()` check) -- bounding
    traversal to real files under the user's own directory (T-07-01,
    WR-01: a symlink to a file directly inside the scanned root would
    otherwise still be matched by `rglob('*')` and `is_file()` follows the
    symlink, silently reading and hashing content from outside `root`).

    Raises `NotADirectoryError` if `root` does not exist or is not a
    directory (CR-01): `Path.rglob` silently yields nothing for a missing
    or non-directory path, which would otherwise be indistinguishable from
    a genuinely empty directory and produce a misleading "delete everything"
    plan downstream.
    """
    if not root.is_dir():
        raise NotADirectoryError(f'{root} is not an existing directory')

    local_hashes: dict[str, list[Path]] = {}
    skipped_non_image = 0

    for p in root.rglob('*'):
        if p.is_symlink() or not p.is_file():
            continue

        if p.suffix.lower() not in ELIGIBLE_EXTENSIONS:
            skipped_non_image += 1
            continue

        data = p.read_bytes()
        h = get_md5(data)
        local_hashes.setdefault(h, []).append(p)

    return ScanResult(local_hashes, skipped_non_image)


@dataclass
class SyncPlan:
    to_upload: list[Path] = field(default_factory=list)
    # Mode-agnostic REMOVAL-CANDIDATE list (D-07): frame assets no longer
    # wanted locally. It deliberately keeps the name `to_delete` -- which
    # primitive actually acts on it (hide vs. hard delete) is the executor's
    # choice, not the diff's.
    to_delete: list = field(default_factory=list)
    # Frame assets that ARE wanted locally but are currently hidden on the
    # frame -- the re-show candidates (D-05).
    to_reshow: list = field(default_factory=list)
    unchanged: int = 0
    # Frame assets no longer wanted locally that are ALREADY hidden: nothing
    # left to do, so they must never re-enter to_delete on later runs (D-06).
    already_hidden: int = 0
    skipped_non_image: int = 0
    frame_no_hash: int = 0


def compute_plan(local_hashes: dict[str, list[Path]], frame_assets: list, skipped_non_image: int = 0) -> SyncPlan:
    """Diff local content hashes against a frame's assets (SYNC-01).

    Pure function -- no I/O, no network, no mutation of its inputs --
    mirroring `resolve_frame`'s pure dataclass-result shape. There is
    deliberately no execute/mutating counterpart here; the dry-run
    guarantee is structural.

    Local duplicates (per `scan_directory`'s dedup) already collapse to one
    logical want per hash (D-05), so each unique hash demands exactly one
    frame copy. Frame-side assets are matched count-for-count against that
    demand (D-06 multiset asymmetry): surplus frame copies beyond local
    demand become delete candidates, they are NOT deduped as a group.
    Hashless frame assets (e.g. videos) are excluded from both unchanged
    and delete, and counted separately in `frame_no_hash`.

    Every hash-bearing frame asset is classified two ways at once -- whether
    the local directory still wants it, and whether it is currently visible on
    the frame (`asset.selected`, which `FrameApi.get_assets` joins from
    `asset_settings` so it means THIS FRAME's visibility):

    | local     | visible | outcome                                  |
    |-----------|---------|------------------------------------------|
    | present   | hidden  | `to_reshow` -- bring it back (D-05)      |
    | present   | visible | `unchanged`                              |
    | gone      | visible | `to_delete` -- removal candidate         |
    | gone      | hidden  | `already_hidden` -- no-op (D-06)         |

    A hidden asset consumes local demand exactly like a visible one, so a
    photo that is merely hidden is re-shown rather than uploaded a second
    time (D-06).

    This function stays PURE and MODE-AGNOSTIC: it never consults the removal
    mode. It reports what each asset's state *is*; choosing hide-vs-delete for
    `to_delete` belongs to the executor.
    """
    demand = {h: 1 for h in local_hashes}
    to_delete: list = []
    to_reshow: list = []
    unchanged = 0
    already_hidden = 0
    frame_no_hash = 0

    for asset in frame_assets:
        if not asset.md5_hash:
            frame_no_hash += 1
            continue

        if demand.get(asset.md5_hash, 0) > 0:
            # Wanted locally. Consume the demand either way (D-06 dedup) --
            # a hidden copy still counts as present, so it is never
            # re-uploaded, only re-shown.
            demand[asset.md5_hash] -= 1
            if asset.selected:
                unchanged += 1
            else:
                to_reshow.append(asset)
        else:
            # No longer wanted locally. Already hidden means there is nothing
            # left to do; re-listing it would re-issue the same hide forever.
            if asset.selected:
                to_delete.append(asset)
            else:
                already_hidden += 1

    to_upload = [local_hashes[h][0] for h, remaining in demand.items() if remaining > 0]

    return SyncPlan(
        to_upload=to_upload,
        to_delete=to_delete,
        to_reshow=to_reshow,
        unchanged=unchanged,
        already_hidden=already_hidden,
        skipped_non_image=skipped_non_image,
        frame_no_hash=frame_no_hash,
    )


@dataclass
class ExecutionResult:
    upload_succeeded: int = 0
    delete_succeeded: int = 0
    reshow_succeeded: int = 0
    upload_failures: list = field(default_factory=list)  # list[tuple[Path, str]]
    delete_failures: list = field(default_factory=list)  # list[tuple[str, str]]
    reshow_failures: list = field(default_factory=list)  # list[tuple[str, str]]
    # Ids this call actually created, from batch_update successes[].id.
    # Never infer by md5 — that would adopt a pre-existing photo with the
    # same bytes.
    upload_ids: list = field(default_factory=list)  # list[tuple[Path, str]]


# The three tiers of "this photo is no longer wanted locally" (D-01/D-03).
# execute_plan applies exactly ONE of these per run, chosen by `removal_mode`,
# so the destructive primitive is unreachable unless a caller names it:
#
#   hide        exclude_asset  reversible, non-destructive -- the DEFAULT
#   delete      remove_asset   disassociates from this frame only
#   hard_delete delete_asset   irreversible, destroys the asset account-wide
#
# `hide` and `delete` are native batch endpoints (one request per chunk);
# `hard_delete` is asset-scoped with no batch form, hence the per-asset loop
# -- which is also why it is charged the budget per asset (Pitfall 3).
_REMOVAL_PRIMITIVE = {
    'hide': lambda aura, frame_id, assets: aura.frame_api.exclude_asset(
        frame_id, [AssetPartialId(id=asset.id) for asset in assets]),
    'delete': lambda aura, frame_id, assets: aura.frame_api.remove_asset(
        frame_id, [AssetPartialId(id=asset.id) for asset in assets]),
    'hard_delete': lambda aura, frame_id, assets: [
        aura.asset_api.delete_asset(asset) for asset in assets],
}

# Requests a removal chunk actually costs, per mode -- batch endpoints are one
# call regardless of chunk size; hard_delete is one call per asset.
_REMOVAL_REQUEST_COST = {
    'hide': lambda chunk: 1,
    'delete': lambda chunk: 1,
    'hard_delete': lambda chunk: len(chunk),
}


def _chunked(items: list, size: int):
    """Yield successive `size`-length slices of `items` (the final slice may
    be shorter). `size` is expected to be a positive int (`WRITE_BATCH_SIZE`
    or an injected override); a chunk boundary is where the Pushd write-call
    count collapses from ~3N to ~2 per chunk."""
    for i in range(0, len(items), size):
        yield items[i:i + size]


def _prep_upload(path: Path, s3_client) -> AssetPartial:
    """Per-file S3 prep phase for a single new local file: resolve the
    Apple UTI, read image dimensions, upload the raw bytes to S3, and build
    the `AssetPartial` that will be sent in the chunk's batched
    `batch_update` call. Raises (fails closed) if the extension is
    unmapped, `Image.open` fails, or the S3 upload fails -- the caller
    catches this per file so one bad file never blocks the rest of the
    chunk.
    """
    data_uti = _DATA_UTI_BY_SUFFIX.get(path.suffix.lower())
    if data_uti is None:
        raise ValueError(f'Unsupported upload extension: {path.suffix}')

    local_identifier = str(uuid.uuid4())
    with Image.open(path) as image:
        width, height = image.size

    filename, md5 = s3_client.upload_file(path.read_bytes(), path.suffix)

    return local_identifier, AssetPartial(
        local_identifier=local_identifier,
        file_name=filename,
        md5_hash=md5,
        height=height,
        width=width,
        taken_at=format_dt_to_aura(get_utc_now()),
        data_uti=data_uti,
        selected=True,
        upload_priority=0,
    )


def execute_plan(plan: SyncPlan, aura, frame_id: str, *, s3_client, sqs_client,
                 throttle_seconds: float = WRITE_THROTTLE_SECONDS, sleep=time.sleep,
                 max_consecutive_failures: int = MAX_CONSECUTIVE_WRITE_FAILURES,
                 progress=lambda *args: None,
                 batch_size: int = WRITE_BATCH_SIZE,
                 chunk_delay_seconds: float = WRITE_CHUNK_DELAY_SECONDS,
                 on_wait=lambda *args: None,
                 budget=None, geo_check=None, wait_on_budget: bool = True,
                 max_wait_seconds: float = 3600.0, clock=get_utc_now,
                 removal_mode: str = 'hide') -> ExecutionResult:
    """Execute a `SyncPlan` against a live frame -- the module's only
    mutating entry point (D-06/D-08/D-09/D-10), batched (quick task
    260708-fyr) to collapse ~3N Pushd write calls to ~2 per chunk.

    `plan.to_upload` is processed in sorted-path chunks of up to
    `batch_size`: each chunk performs per-file S3 uploads (S3 is AWS, not
    the Pushd anti-abuse surface, so those stay per-file and precede the
    chunk's batched writes), then exactly ONE `select_asset` call and ONE
    `batch_update` call carrying every successfully-prepped file in the
    chunk. Per-file attribution is recovered from `batch_update`'s
    `successes[].local_identifier`: a prepped file's local_identifier
    absent from `successes` is recorded as a per-file failure. `plan.to_delete`
    is processed in chunks the same way, issuing one `remove_asset` call per
    chunk; since `remove_asset` returns only a failure COUNT (never
    per-item), a chunk-level failure attributes ALL deletes in that chunk
    (coarser than upload attribution -- documented tradeoff, T-fyr-01). All
    uploads are attempted before any delete is attempted (D-09) -- on
    interruption mid-run this leaves the safer partial state (content added,
    nothing removed).

    Work runs least-destructive-first: uploads, then re-shows, then removals.
    The re-show loop calls `select_asset` on `plan.to_reshow` and ALWAYS runs,
    whatever `removal_mode` is -- bringing back a photo the user restored
    locally is not a removal concern (D-05).

    The removal loop applies exactly ONE primitive, chosen by `removal_mode`
    from `_REMOVAL_PRIMITIVE` (D-01/D-03). It defaults to `'hide'`, so the
    destructive tiers are unreachable unless a caller names one.

    `s3_client`/`sqs_client` are injected by the caller (never constructed
    in this module) so this function is offline-testable with fakes --
    no AWS client construction call appears here at all.

    :param removal_mode: Which primitive acts on `plan.to_delete` -- `'hide'`
        (default, reversible `exclude_asset`), `'delete'` (`remove_asset`,
        frame-scoped) or `'hard_delete'` (`delete_asset`, irreversible and
        account-wide). Preservation is the default because a mistaken sync
        must never destroy photos (D-01/D-03).
    :param plan: The `SyncPlan` (from `compute_plan`) to execute.
    :param aura: An authenticated `Aura` instance.
    :param frame_id: The frame to upload to / delete from.
    :param s3_client: An object providing `upload_file(data, extension) -> (filename, md5)`.
    :param sqs_client: An object providing `get_queue_url(frame_id)` and `receive_message(...)`.
    :param throttle_seconds: Seconds to pause before each write network call
        so a bulk apply is paced rather than fired as a burst -- a second,
        independent layer of defense on top of batching itself (root-cause
        mitigation for the select-asset-401-unauthorized session). 0 disables.
    :param sleep: The sleep function to call (injectable for offline tests
        so they pace-check without real delays).
    :param max_consecutive_failures: Abort the whole batch once this many
        write items fail in an unbroken run (reset on any success) -- the
        backstop for an anti-abuse trip that surfaces as plain HTTP 401s
        rather than the 429/475 `RateLimitError` catches (see
        `MAX_CONSECUTIVE_WRITE_FAILURES`). Status-code agnostic: any caught
        per-item write failure counts. 0 disables the backstop entirely
        (restoring the pure unbounded per-item D-08 behaviour).
    :param progress: Optional reporter called exactly once per RESOLVED
        item -- an upload from `plan.to_upload` or a delete from
        `plan.to_delete` -- as `progress(kind, identifier, ok)`, where
        `kind` is `'upload'` or `'delete'`, `identifier` is the `Path` for
        an upload or the asset id string for a delete, and `ok` is True on
        success / False on a caught per-item failure. Deliberately NOT
        called on the `RateLimitError` abort path below -- that item never
        resolves, the whole batch stops there, so the number of reporter
        calls always equals the number of attempted-and-resolved items.
        Defaults to a no-op so offline tests and any caller that doesn't
        care about live feedback are unaffected.
    :param batch_size: Maximum number of items batched into a single
        select_asset/batch_update/remove_asset call (see
        `WRITE_BATCH_SIZE`). Injectable so offline tests can exercise the
        chunk-boundary logic with a small size.
    :param chunk_delay_seconds: Seconds to pause BETWEEN write chunks (see
        `WRITE_CHUNK_DELAY_SECONDS`) -- distinct from `throttle_seconds`
        (which paces individual calls). Skipped before the first write chunk
        of the run; applied before every subsequent chunk (uploads then
        deletes are paced as one sequence). Realized via `sleep` in 1s steps.
        0 disables. Injectable for offline tests.
    :param on_wait: Optional callback invoked once per second during an
        inter-chunk pause as `on_wait(remaining_seconds)`, so a CLI can render
        a live countdown instead of a frozen bar. Defaults to a no-op.
    :param budget: Optional `auraframes.ratelimit.WriteBudget` (Phase 09,
        ANTI-03/04) gating each write chunk with a client-side token-bucket
        request budget -- a proactive defense making the anti-abuse
        write-lockout structurally hard to hit. When `None` (the default),
        every budget-related touch point below is skipped entirely: this is
        a true byte-for-byte no-op, matching this function's pre-Phase-09
        behavior exactly.
    :param geo_check: Optional zero-arg callable (Phase 09, ANTI-03) invoked
        exactly once, before any write (including for a delete-only plan),
        raising `auraframes.ratelimit.GeoMismatchError` on a VPN/exit-IP
        country mismatch. When `None` (the default), skipped entirely.
    :param wait_on_budget: Forwarded to `budget.acquire(..., wait=...)` --
        when True (the default), a chunk waits for enough tokens to refill
        rather than raising `BudgetExhausted` immediately. Ignored when
        `budget` is `None`.
    :param max_wait_seconds: Forwarded to `budget.acquire(..., max_wait=...)`
        -- the longest a chunk will wait for tokens before raising
        `BudgetExhausted` anyway (default 3600s/1hr). Ignored when `budget`
        is `None`.
    :param clock: Zero-arg function returning the current instant (Phase 09),
        injected so tests can control time deterministically -- mirrors the
        `sleep` seam. Its return VALUE is passed as `budget.acquire(...,
        now=clock())` and `budget.reconcile_tripped(clock())`; defaults to
        `auraframes.utils.dt.get_utc_now`. Only ever called when `budget` is
        not `None` (or `geo_check`, which never touches `clock`).
    :return: An `ExecutionResult` with separated upload/delete success counts and named failures.

    Raises `RateLimitError` (from the client layer) WITHOUT catching it:
    a 429/475 throttle or lockout aborts the entire batch immediately
    rather than being recorded as one of N per-item failures -- the whole
    point being to stop hammering a throttling server and surface a single
    "back off" message instead of dozens of confusing per-item errors.
    Ordinary per-item/per-chunk failures are still caught and recorded (D-08).

    Raises `ConsecutiveWriteFailureError` when `max_consecutive_failures`
    write items fail in an unbroken run -- the backstop for an anti-abuse
    trip that surfaces as plain HTTP 401s (indistinguishable at the status
    level from an isolated per-item auth failure) rather than the 429/475
    `RateLimitError` catches. The counter spans BOTH loops and resets on any
    success, so an isolated failure never trips it. The final failing item is
    still recorded in `result` and reported via `progress` before the abort
    (so the reporter-call count still equals the attempted-and-resolved item
    count); items after the abort are never attempted.
    """
    # Geo pre-flight (Phase 09, ANTI-03) -- the VERY FIRST executable
    # statement of the body, before ExecutionResult is even constructed, so
    # a delete-only plan is gated too: no S3 upload, no select_asset, no
    # remove_asset call has happened yet at this point regardless of the
    # plan's shape.
    if geo_check is not None:
        geo_check()

    result = ExecutionResult()
    consecutive_failures = 0

    def throttle() -> None:
        if throttle_seconds > 0:
            sleep(throttle_seconds)

    def note_failure(last_error: str) -> None:
        # Bump the shared consecutive-failure run and abort the whole batch
        # once it crosses the threshold. Status-code agnostic on purpose: the
        # trip cannot be recognised from the HTTP status alone (this session's
        # core lesson), so ANY caught per-item write failure counts toward the
        # run. RateLimitError never reaches here -- it is re-raised above and
        # aborts on its first occurrence.
        nonlocal consecutive_failures
        consecutive_failures += 1
        if 0 < max_consecutive_failures <= consecutive_failures:
            if budget is not None:
                # Phase 09 ANTI-04: a systemic cut-off just tripped (the
                # backstop for the plain-401 form of the anti-abuse lockout
                # that RateLimitError's 429/475 branch can't catch) -- force
                # the local budget to reflect reality even though it wasn't
                # what ran the bucket dry.
                budget.reconcile_tripped(clock())
                budget.save()
            raise ConsecutiveWriteFailureError(consecutive_failures, last_error, result)

    first_write_chunk = True

    def interchunk_pause() -> None:
        # Human-pacing pause BETWEEN write chunks -- separate from throttle()
        # (which paces individual network calls). Skipped before the very
        # first write chunk of the run, so uploads then deletes are spaced as
        # one continuous sequence. Realized via the injected `sleep` in 1s
        # steps, calling on_wait(remaining) each second so the CLI can show a
        # live countdown rather than a frozen bar. chunk_delay_seconds=0
        # disables the pause entirely.
        nonlocal first_write_chunk
        if first_write_chunk:
            first_write_chunk = False
            return
        remaining = chunk_delay_seconds
        while remaining > 0:
            on_wait(remaining)
            step = 1.0 if remaining >= 1.0 else remaining
            sleep(step)
            remaining -= step

    queue_url = sqs_client.get_queue_url(frame_id) if plan.to_upload else None

    for chunk in _chunked(sorted(plan.to_upload), batch_size):
        interchunk_pause()
        if budget is not None:
            # 2 requests per upload chunk (select_asset + batch_update) --
            # acquired BEFORE the per-file S3-prep loop so a BudgetExhausted
            # stop never wastes an S3 upload on a chunk that won't be written.
            budget.acquire(2, wait=wait_on_budget, max_wait=max_wait_seconds,
                            now=clock(), sleep=sleep, on_wait=on_wait)
        prepped: list = []  # list[tuple[Path, local_identifier, AssetPartial]]
        for path in chunk:
            try:
                local_identifier, partial = _prep_upload(path, s3_client)
                prepped.append((path, local_identifier, partial))
            except Exception as e:
                result.upload_failures.append((path, str(e)))
                progress('upload', path, False)
                note_failure(str(e))

        if not prepped:
            continue

        try:
            throttle()
            aura.frame_api.select_asset(
                frame_id, [AssetPartialId(local_identifier=lid) for (_, lid, _) in prepped]
            )
            # Best-effort/observational poll only -- never gates success or
            # failure (Pitfall 3); at most once per chunk, not per file.
            message = sqs_client.receive_message(queue_url, wait_time_seconds=5)
            logger.debug(f'Best-effort SQS poll after chunk select_asset: {message}')

            throttle()
            _, successes = aura.asset_api.batch_update([partial for (_, _, partial) in prepped])
            succeeded = {
                s.local_identifier: s.id
                for s in successes
                if s.local_identifier
            }

            for path, local_identifier, _ in prepped:
                asset_id = succeeded.get(local_identifier)
                if asset_id:
                    result.upload_succeeded += 1
                    result.upload_ids.append((path, asset_id))
                    consecutive_failures = 0
                    progress('upload', path, True)
                else:
                    reason = (
                        'file not acknowledged in batch_update successes'
                        if local_identifier not in succeeded
                        else 'batch_update success had no asset id'
                    )
                    result.upload_failures.append((path, reason))
                    progress('upload', path, False)
                    note_failure(reason)
        except RateLimitError:
            # Anti-abuse throttle/lockout: abort the whole batch (do not
            # mask it as one per-item failure and keep hammering).
            if budget is not None:
                budget.reconcile_tripped(clock())
                budget.save()
            raise
        except ConsecutiveWriteFailureError:
            # note_failure() above can raise this from WITHIN the per-file
            # attribution loop (all prepped files already individually
            # appended/reported there) -- it must propagate as-is, NOT be
            # re-caught by the generic Exception branch below, which would
            # otherwise mistake the abort for a whole-chunk Pushd failure and
            # double-attribute every prepped file a second time. (Reconcile
            # already happened inside note_failure() before this raised.)
            raise
        except Exception as e:
            # A whole-chunk Pushd failure (select_asset or batch_update
            # raising) attributes ALL prepped files in this chunk as failed
            # -- there is no per-item signal to fall back on.
            for path, _, _ in prepped:
                result.upload_failures.append((path, str(e)))
                progress('upload', path, False)
                note_failure(str(e))

        if budget is not None:
            # Save after every chunk that returns normally (success OR
            # ordinary caught failure, Phase 09 ANTI-04) -- acquire() already
            # decremented tokens in-memory for the attempted requests
            # regardless of per-item outcome, so this keeps the on-disk
            # estimate from drifting optimistic after an interrupt.
            budget.save()

    for chunk in _chunked(plan.to_reshow, batch_size):
        interchunk_pause()
        if budget is not None:
            # 1 request per chunk (select_asset is a batch endpoint).
            budget.acquire(1, wait=wait_on_budget, max_wait=max_wait_seconds,
                            now=clock(), sleep=sleep, on_wait=on_wait)
        try:
            throttle()
            aura.frame_api.select_asset(frame_id, [AssetPartialId(id=asset.id) for asset in chunk])
            for asset in chunk:
                result.reshow_succeeded += 1
                consecutive_failures = 0
                progress('reshow', asset.id, True)
        except RateLimitError:
            if budget is not None:
                budget.reconcile_tripped(clock())
                budget.save()
            raise
        except Exception as e:
            # select_asset returns only a count, never per-item -- a raised
            # chunk-level failure attributes ALL re-shows in this chunk, the
            # same coarser attribution the removal loop documents.
            for asset in chunk:
                result.reshow_failures.append((asset.id, str(e)))
                progress('reshow', asset.id, False)
                note_failure(str(e))

        if budget is not None:
            budget.save()

    for chunk in _chunked(plan.to_delete, batch_size):
        interchunk_pause()
        if budget is not None:
            # 1 request per chunk for the batch endpoints; one per asset for
            # hard_delete, which has no batch form (Pitfall 3) -- charging it
            # 1 would let a hard delete run the budget dry unnoticed and
            # re-trip the anti-abuse lockout.
            budget.acquire(_REMOVAL_REQUEST_COST[removal_mode](chunk),
                            wait=wait_on_budget, max_wait=max_wait_seconds,
                            now=clock(), sleep=sleep, on_wait=on_wait)
        try:
            throttle()
            _REMOVAL_PRIMITIVE[removal_mode](aura, frame_id, chunk)
            for asset in chunk:
                result.delete_succeeded += 1
                consecutive_failures = 0
                progress('delete', asset.id, True)
        except RateLimitError:
            if budget is not None:
                budget.reconcile_tripped(clock())
                budget.save()
            raise
        except Exception as e:
            # remove_asset returns only a count, never per-item -- a raised
            # chunk-level failure attributes ALL deletes in this chunk
            # (coarser than upload attribution, T-fyr-01).
            for asset in chunk:
                result.delete_failures.append((asset.id, str(e)))
                progress('delete', asset.id, False)
                note_failure(str(e))

        if budget is not None:
            budget.save()

    return result
