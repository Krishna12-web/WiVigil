"""Shared helpers: parsing ESP32 CSI lines, windowing, features."""
import csv
import io
import json
import numpy as np

WINDOW = 100   # packets per window (about 1 second at 100 packets/s)
STEP = 50      # 50 percent overlap
N_SUB = 64     # subcarriers kept (pairs per packet is padded or cut to this)


def parse_csi_line(line):
    """Parse one 'CSI_DATA,...' line from the esp-csi receiver.

    Returns (rssi, amplitude_array) or None if the line is not valid CSI.
    The last field is a quoted list of ints: imag, real, imag, real, ...
    """
    line = line.strip()
    if not line.startswith("CSI_DATA"):
        return None
    try:
        fields = next(csv.reader(io.StringIO(line)))
        raw = json.loads(fields[-1])
        rssi = float(fields[3])
    except (ValueError, IndexError, StopIteration, json.JSONDecodeError):
        return None
    if len(raw) < 2 or len(raw) % 2 != 0:
        return None
    arr = np.asarray(raw, dtype=np.float32).reshape(-1, 2)
    amp = np.sqrt(arr[:, 0] ** 2 + arr[:, 1] ** 2)
    out = np.zeros(N_SUB, dtype=np.float32)
    n = min(len(amp), N_SUB)
    out[:n] = amp[:n]
    return rssi, out


def clean(amp):
    """amp: (T, N_SUB). Hampel-style outlier clip, then smooth over time."""
    amp = amp.astype(np.float32).copy()
    med = np.median(amp, axis=0)
    mad = np.median(np.abs(amp - med), axis=0) + 1e-6
    lo, hi = med - 6 * 1.4826 * mad, med + 6 * 1.4826 * mad
    amp = np.clip(amp, lo, hi)
    k = 5
    kernel = np.ones(k, dtype=np.float32) / k
    sm = np.apply_along_axis(lambda c: np.convolve(c, kernel, mode="same"), 0, amp)
    return sm


def window_features(win):
    """win: (WINDOW, N_SUB) cleaned amplitudes -> 1D feature vector."""
    valid = win.mean(axis=0) > 1e-3          # drop null subcarriers
    w = win[:, valid] if valid.any() else win
    var_t = w.var(axis=0)                     # temporal variance per subcarrier
    diff = np.abs(np.diff(w, axis=0))
    total = w.sum(axis=1)                     # overall energy over time
    spec = np.abs(np.fft.rfft(total - total.mean()))
    spec = spec / (spec.sum() + 1e-9)
    # coarse spectral bands (motion lives at higher frequency than breathing)
    bands = [spec[1:4].sum(), spec[4:10].sum(), spec[10:25].sum(), spec[25:].sum()]
    feats = [
        var_t.mean(), var_t.std(), var_t.max(),
        diff.mean(), diff.std(),
        w.mean(), w.std(),
        total.std() / (total.mean() + 1e-9),
        float(-(spec * np.log(spec + 1e-12)).sum()),   # spectral entropy
        *bands,
    ]
    return np.asarray(feats, dtype=np.float32)


def make_windows(amp):
    """amp: (T, N_SUB) raw amplitudes -> (n_windows, n_features)."""
    amp = clean(amp)
    out = []
    for s in range(0, len(amp) - WINDOW + 1, STEP):
        out.append(window_features(amp[s:s + WINDOW]))
    return np.asarray(out, dtype=np.float32)
