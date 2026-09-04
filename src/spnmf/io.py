"""Audio file I/O.

Uses `soundfile <https://pypi.org/project/soundfile/>`_ when it is installed,
which brings FLAC, OGG and friends.  Falls back to the standard library's
``wave`` module so plain PCM WAV works with NumPy alone.
"""

from __future__ import annotations

import wave

import numpy as np

__all__ = ["read_audio", "write_audio"]


def _soundfile():
    try:
        import soundfile  # noqa: PLC0415
    except ImportError:
        return None
    return soundfile


def _read_wav_stdlib(path):
    with wave.open(str(path), "rb") as fh:
        n_channels = fh.getnchannels()
        width = fh.getsampwidth()
        sample_rate = fh.getframerate()
        raw = fh.readframes(fh.getnframes())

    if width == 1:  # 8-bit PCM is unsigned
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float64) - 128.0) / 128.0
    elif width == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    elif width == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        packed = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        packed = np.where(packed & 0x800000, packed - 0x1000000, packed)
        data = packed.astype(np.float64) / 8388608.0
    elif width == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float64) / 2147483648.0
    else:
        raise ValueError(f"unsupported WAV sample width: {width} bytes")

    if n_channels > 1:
        data = data.reshape(-1, n_channels)
    return data, sample_rate


def _write_wav_stdlib(path, data, sample_rate):
    data = np.asarray(data, dtype=np.float64)
    if data.ndim == 1:
        data = data[:, None]
    n_channels = data.shape[1]
    clipped = np.clip(data, -1.0, 1.0)
    pcm = np.round(clipped * 32767.0).astype("<i2")

    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(n_channels)
        fh.setsampwidth(2)
        fh.setframerate(int(sample_rate))
        fh.writeframes(pcm.tobytes())


def read_audio(path):
    """Read an audio file as ``(data, sample_rate)``.

    ``data`` is float64 in ``[-1, 1]``, shaped ``(n_samples,)`` for mono or
    ``(n_samples, n_channels)`` otherwise.
    """
    sf = _soundfile()
    if sf is not None:
        data, sample_rate = sf.read(str(path), dtype="float64", always_2d=False)
        return np.asarray(data, dtype=np.float64), int(sample_rate)

    if str(path).lower().endswith(".wav"):
        return _read_wav_stdlib(path)
    raise RuntimeError(
        f"cannot read {path!r}: only PCM WAV is supported without soundfile. "
        "Install it with `pip install 'spnmf[audio]'`."
    )


def write_audio(path, data, sample_rate):
    """Write ``data`` to ``path``.  Mirrors :func:`read_audio`'s conventions."""
    sf = _soundfile()
    if sf is not None:
        sf.write(str(path), np.asarray(data, dtype=np.float64), int(sample_rate))
        return

    if str(path).lower().endswith(".wav"):
        _write_wav_stdlib(path, data, sample_rate)
        return
    raise RuntimeError(
        f"cannot write {path!r}: only PCM WAV is supported without soundfile. "
        "Install it with `pip install 'spnmf[audio]'`."
    )
