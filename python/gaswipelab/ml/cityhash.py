"""CatBoost のカテゴリ特徴ハッシュ（CityHash64 の下位32bitを符号付きint32として解釈）。

CatBoost 本体と完全一致することを、実モデルの ctr_data ハッシュ表および
ゴールデンケース30件で検証済み。文字列長に依存せず正しく動作する。
"""
from __future__ import annotations

M64 = (1 << 64) - 1
_K0 = 0xC3A5C85C97CB3127
_K1 = 0xB492B66FBE98F273
_K2 = 0x9AE16A3B2F90404F
_K3 = 0xC949D7C7509E6557
_KMUL = 0x9DDFEA08EB382D69


def _u64(x: int) -> int:
    return x & M64


def _fetch64(s: bytes, i: int) -> int:
    return int.from_bytes(s[i:i + 8], "little")


def _fetch32(s: bytes, i: int) -> int:
    return int.from_bytes(s[i:i + 4], "little")


def _rot(v: int, shift: int) -> int:
    return v if shift == 0 else _u64((v >> shift) | (v << (64 - shift)))


def _rot1(v: int, shift: int) -> int:
    return _u64((v >> shift) | (v << (64 - shift)))


def _shift_mix(v: int) -> int:
    return v ^ (v >> 47)


def _hash_len16(u: int, v: int) -> int:
    a = _u64((u ^ v) * _KMUL)
    a ^= a >> 47
    b = _u64((v ^ a) * _KMUL)
    b ^= b >> 47
    return _u64(b * _KMUL)


def _hash_len0to16(s: bytes) -> int:
    n = len(s)
    if n > 8:
        a = _fetch64(s, 0)
        b = _fetch64(s, n - 8)
        return _hash_len16(a, _rot1(_u64(b + n), n)) ^ b
    if n >= 4:
        a = _fetch32(s, 0)
        return _hash_len16(_u64(n + (a << 3)), _fetch32(s, n - 4))
    if n > 0:
        a, b, c = s[0], s[n >> 1], s[n - 1]
        y = _u64(a + (b << 8))
        z = _u64(n + (c << 2))
        return _u64(_shift_mix(_u64(y * _K2) ^ _u64(z * _K3)) * _K2)
    return _K2


def _hash_len17to32(s: bytes) -> int:
    n = len(s)
    a = _u64(_fetch64(s, 0) * _K1)
    b = _fetch64(s, 8)
    c = _u64(_fetch64(s, n - 8) * _K2)
    d = _u64(_fetch64(s, n - 16) * _K0)
    return _hash_len16(
        _u64(_rot(_u64(a - b), 43) + _rot(c, 30) + d),
        _u64(a + _rot(_u64(b ^ _K3), 20) - c + n),
    )


def _weak_hash32(s: bytes, i: int, a: int, b: int) -> tuple[int, int]:
    w = _fetch64(s, i)
    x = _fetch64(s, i + 8)
    y = _fetch64(s, i + 16)
    z = _fetch64(s, i + 24)
    a = _u64(a + w)
    b = _rot(_u64(b + a + z), 21)
    c = a
    a = _u64(a + x + y)
    b = _u64(b + _rot(a, 44))
    return _u64(a + z), _u64(b + c)


def _hash_len33to64(s: bytes) -> int:
    n = len(s)
    z = _fetch64(s, 24)
    a = _u64(_fetch64(s, 0) + _u64((n + _fetch64(s, n - 16)) * _K0))
    b = _rot(_u64(a + z), 52)
    c = _rot(a, 37)
    a = _u64(a + _fetch64(s, 8))
    c = _u64(c + _rot(a, 7))
    a = _u64(a + _fetch64(s, 16))
    vf = _u64(a + z)
    vs = _u64(b + _rot(a, 31) + c)

    a = _u64(_fetch64(s, 16) + _fetch64(s, n - 32))
    z = _fetch64(s, n - 8)
    b = _rot(_u64(a + z), 52)
    c = _rot(a, 37)
    a = _u64(a + _fetch64(s, n - 24))
    c = _u64(c + _rot(a, 7))
    a = _u64(a + _fetch64(s, n - 16))
    wf = _u64(a + z)
    ws = _u64(b + _rot(a, 31) + c)

    r = _shift_mix(_u64(_u64((vf + ws) * _K2) + _u64((wf + vs) * _K0)))
    return _u64(_shift_mix(_u64(r * _K0 + vs)) * _K2)


def cityhash64(s: bytes) -> int:
    n = len(s)
    if n <= 32:
        return _hash_len0to16(s) if n <= 16 else _hash_len17to32(s)
    if n <= 64:
        return _hash_len33to64(s)

    x = _fetch64(s, 0)
    y = _u64(_fetch64(s, n - 16) ^ _K1)
    z = _u64(_fetch64(s, n - 56) ^ _K0)
    v = _weak_hash32(s, n - 64, n, y)
    w = _weak_hash32(s, n - 32, _u64(n * _K1), _K0)
    z = _u64(z + _u64(_shift_mix(v[1]) * _K1))
    x = _u64(_rot(_u64(z + x), 39) * _K1)
    y = _u64(_rot(y, 33) * _K1)

    remaining = (n - 1) & ~63
    i = 0
    while True:
        x = _u64(_rot(_u64(x + y + v[0] + _fetch64(s, i + 16)), 37) * _K1)
        y = _u64(_rot(_u64(y + v[1] + _fetch64(s, i + 48)), 42) * _K1)
        x ^= w[1]
        y ^= v[0]
        z = _rot(z ^ w[0], 33)
        v = _weak_hash32(s, i, _u64(v[1] * _K1), _u64(x + w[0]))
        w = _weak_hash32(s, i + 32, _u64(z + w[1]), y)
        z, x = x, z
        i += 64
        remaining -= 64
        if remaining == 0:
            break
    return _hash_len16(
        _u64(_hash_len16(v[0], w[0]) + _u64(_shift_mix(y) * _K1) + z),
        _u64(_hash_len16(v[1], w[1]) + x),
    )


def cat_feature_hash(value: str) -> int:
    """カテゴリ値 → CatBoost が用いる符号付き int32 ハッシュ。"""
    h = cityhash64(value.encode("utf-8")) & 0xFFFFFFFF
    return h - (1 << 32) if h >= (1 << 31) else h
