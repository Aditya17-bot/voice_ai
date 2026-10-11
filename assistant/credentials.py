"""Login tokens for Gmail and GitHub, encrypted for the current Windows user.

Stored under %APPDATA%\\voice_assistant\\ (outside the project, so never
committed) and encrypted with Windows DPAPI: only this Windows account on
this computer can decrypt them. Nothing is ever printed or logged.
"""

import ctypes
import ctypes.wintypes
import json
import os
from pathlib import Path

ENTROPY = b"offline-voice-assistant"  # ties the blobs to this app


class _Blob(ctypes.Structure):
    _fields_ = [("size", ctypes.wintypes.DWORD), ("data", ctypes.POINTER(ctypes.c_char))]


def _blob(data: bytes) -> _Blob:
    buffer = ctypes.create_string_buffer(data, len(data))
    return _Blob(len(data), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_char)))


def _crypt(data: bytes, protect: bool) -> bytes:
    crypt32 = ctypes.windll.crypt32
    source, entropy, out = _blob(data), _blob(ENTROPY), _Blob()
    call = crypt32.CryptProtectData if protect else crypt32.CryptUnprotectData
    if not call(ctypes.byref(source), None, ctypes.byref(entropy), None, None, 0, ctypes.byref(out)):
        raise OSError("Windows could not " + ("encrypt" if protect else "decrypt") + " the secret")
    try:
        return ctypes.string_at(out.data, out.size)
    finally:
        ctypes.windll.kernel32.LocalFree(out.data)


def _path(name: str) -> Path:
    base = Path(os.environ.get("APPDATA", Path.home())) / "voice_assistant"
    return base / f"{name}.bin"


def save(name: str, value: dict) -> None:
    path = _path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_bytes(_crypt(json.dumps(value).encode(), protect=True))
    temp.replace(path)


def load(name: str) -> dict | None:
    """The stored value, or None if it was never saved or can't be decrypted."""
    try:
        return json.loads(_crypt(_path(name).read_bytes(), protect=False))
    except (OSError, ValueError, AttributeError):
        return None


def delete(name: str) -> bool:
    try:
        _path(name).unlink()
        return True
    except FileNotFoundError:
        return False
