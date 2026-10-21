# SPDX-License-Identifier: Apache-2.0
"""Rank zero's Lightning MTP decisions, replayed by every other rank.

Only rank zero holds the MTP head, so only its drafts and samples count. A
verify cycle still runs the target forward on every rank, so the other ranks
need rank zero's choices at two points: the tokens the verify forward reads,
and how many of them were accepted. A token sampled outside a cycle (MTP
activation, a handoff back to the standard step) travels the same way. All of
it goes over the rank-control channel, which keeps it in the same strict order
as request sharing.

Drafting and acceptance draw from rank zero's RNG only. Each message therefore
carries a seed from rank zero that every rank applies once rank zero has built
its last random draw, so a standard step that follows samples the same token
on every rank again.
"""

from __future__ import annotations

import os
from typing import Any


def _new_seed() -> int:
    return int.from_bytes(os.urandom(8), "little") >> 1


class ClusterMtpSync:
    """The decision channel one rank's batch generator reads during MTP."""

    def __init__(self, *, rank: int, control_plane: Any | None = None) -> None:
        self.rank = int(rank)
        self.control_plane = control_plane

    @property
    def is_leader(self) -> bool:
        return self.rank == 0

    def _broadcast(self, kind: str, *fields: Any) -> tuple[Any, ...]:
        plane = self.control_plane
        if plane is None:
            raise RuntimeError("cluster Lightning MTP needs the rank-control channel")
        message = plane.broadcast_object((kind, *fields) if self.is_leader else None)
        if not isinstance(message, tuple) or not message or message[0] != kind:
            got = message[0] if isinstance(message, tuple) and message else message
            raise RuntimeError(f"cluster MTP expected {kind!r}, got {got!r}")
        return message[1:]

    def share_tokens(self, token_ids: list[int] | None) -> tuple[list[int], int]:
        """Rank zero's sampled tokens, plus a seed."""

        ids, seed = self._broadcast("tokens", list(token_ids or ()), _new_seed())
        return [int(token) for token in ids], int(seed)

    def share_cycle_inputs(
        self,
        next_main: list[int] | None,
        drafts: list[int] | None,
        *,
        fallback: str | None = None,
    ) -> tuple[list[int], list[int], str | None]:
        """Rank zero's verify inputs, or the reason it leaves MTP instead."""

        main_ids, draft_ids, reason = self._broadcast(
            "inputs", list(next_main or ()), list(drafts or ()), fallback
        )
        return list(main_ids), list(draft_ids), reason

    def share_cycle_result(self, accepted: int, emit_id: int) -> tuple[int, int, int]:
        """Rank zero's accepted draft count and final token, plus a seed."""

        accepted, emit_id, seed = self._broadcast(
            "result", int(accepted), int(emit_id), _new_seed()
        )
        return int(accepted), int(emit_id), int(seed)
