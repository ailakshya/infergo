"""infergo Shared Memory client — zero-copy, zero-syscall inference."""

import mmap
import struct
import time
import os
from typing import Optional


# Must match cpp/server/shm_transport.hpp
SHM_SLOT_SIZE = 65536  # 64KB
SHM_MAX_SLOTS = 16
MAGIC = 0x494E4647  # "INFG"

# SlotState
FREE = 0
READY = 1
BUSY = 2
DONE = 3

# Offsets in SHMRegion
HEADER_SIZE = 64
SLOT_STATE_OFFSET = 0    # uint32 atomic
SLOT_REQ_LEN_OFFSET = 4  # uint32
SLOT_RESP_LEN_OFFSET = 8 # uint32
SLOT_CLIENT_ID_OFFSET = 12 # uint32
SLOT_REQUEST_OFFSET = 16
SLOT_RESPONSE_OFFSET = 16 + SHM_SLOT_SIZE
SLOT_TOTAL_SIZE = 16 + SHM_SLOT_SIZE * 2 + 16  # request + response + padding


class SHMClient:
    """Connect to infergo via shared memory for lowest possible latency.

    Usage:
        client = SHMClient("infergo_shm")
        client.connect()
        response = client.generate("Hello", max_tokens=32)
        client.disconnect()
    """

    def __init__(self, name: str = "infergo_shm"):
        self.name = name
        self._mm: Optional[mmap.mmap] = None
        self._fd: Optional[int] = None
        self._n_slots = 0
        self._pid = os.getpid()

    def connect(self):
        path = f"/dev/shm/{self.name}"
        if not os.path.exists(path):
            raise FileNotFoundError(f"Shared memory not found: {path}")

        self._fd = os.open(path, os.O_RDWR)
        size = os.fstat(self._fd).st_size
        self._mm = mmap.mmap(self._fd, size)

        # Verify magic
        magic = struct.unpack_from("<I", self._mm, 0)[0]
        if magic != MAGIC:
            raise RuntimeError(f"Invalid magic: 0x{magic:08X}")

        # Check server alive
        alive = struct.unpack_from("<I", self._mm, 16)[0]
        if not alive:
            raise RuntimeError("Server not alive")

        self._n_slots = struct.unpack_from("<I", self._mm, 8)[0]

    def disconnect(self):
        if self._mm:
            self._mm.close()
            self._mm = None
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.disconnect()

    def _slot_offset(self, slot: int) -> int:
        return HEADER_SIZE + slot * SLOT_TOTAL_SIZE

    def _acquire_slot(self) -> int:
        for i in range(self._n_slots):
            off = self._slot_offset(i)
            state = struct.unpack_from("<I", self._mm, off + SLOT_STATE_OFFSET)[0]
            if state == FREE:
                # Mark as ours (still FREE, client writes then sets READY)
                struct.pack_into("<I", self._mm, off + SLOT_CLIENT_ID_OFFSET, self._pid)
                return i
        return -1

    def _send_request(self, slot: int, data: bytes):
        off = self._slot_offset(slot)
        data_len = min(len(data), SHM_SLOT_SIZE - 1)

        # Write request data
        self._mm[off + SLOT_REQUEST_OFFSET:off + SLOT_REQUEST_OFFSET + data_len] = data[:data_len]
        struct.pack_into("<I", self._mm, off + SLOT_REQ_LEN_OFFSET, data_len)

        # Signal: READY
        struct.pack_into("<I", self._mm, off + SLOT_STATE_OFFSET, READY)

    def _wait_response(self, slot: int, timeout_ms: int = 5000) -> bytes:
        off = self._slot_offset(slot)
        deadline = time.monotonic() + timeout_ms / 1000.0
        spins = 0

        while time.monotonic() < deadline:
            state = struct.unpack_from("<I", self._mm, off + SLOT_STATE_OFFSET)[0]
            if state == DONE:
                resp_len = struct.unpack_from("<I", self._mm, off + SLOT_RESP_LEN_OFFSET)[0]
                data = bytes(self._mm[off + SLOT_RESPONSE_OFFSET:off + SLOT_RESPONSE_OFFSET + resp_len])
                # Release slot
                struct.pack_into("<I", self._mm, off + SLOT_STATE_OFFSET, FREE)
                return data
            spins += 1
            if spins > 1000:
                time.sleep(0.0001)  # 100us yield

        raise TimeoutError(f"SHM response timeout after {timeout_ms}ms")

    def generate(self, prompt: str, max_tokens: int = 32,
                 temperature: float = 0.7, timeout_ms: int = 5000) -> str:
        """Generate text via shared memory — lowest possible latency."""
        slot = self._acquire_slot()
        if slot < 0:
            raise RuntimeError("No free SHM slots")

        import json
        payload = json.dumps({
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }).encode()

        self._send_request(slot, payload)
        response = self._wait_response(slot, timeout_ms)
        return response.decode("utf-8", errors="replace")
