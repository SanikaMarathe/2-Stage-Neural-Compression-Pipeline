"""
test_fgk_roundtrip.py -- Property-based roundtrip tests for the FGK codec.

For every test input: decode(encode(s)) == s
"""

import sys
import os
import random
import string
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'huffman-service', 'app'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'huffman-service'))

from app.fgk import encode, decode  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def roundtrip(s: str) -> None:
    """Assert that decode(encode(s)) == s."""
    assert decode(encode(s)) == s, f"roundtrip failed for {s!r}"


# ---------------------------------------------------------------------------
# Deterministic / fixed test cases
# ---------------------------------------------------------------------------

class TestEmpty:
    def test_empty_string(self):
        roundtrip("")


class TestSingleChar:
    @pytest.mark.parametrize("ch", list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 !@#$%^&*()"))
    def test_single_char(self, ch):
        roundtrip(ch)


class TestAllSame:
    @pytest.mark.parametrize("ch,n", [
        ("a", 1), ("a", 2), ("a", 7), ("a", 10), ("a", 50), ("a", 100),
        ("z", 1), ("z", 5), ("z", 20),
        (" ", 3), (" ", 15),
        ("0", 8), ("9", 25),
    ])
    def test_all_same(self, ch, n):
        roundtrip(ch * n)


class TestDigitStrings:
    """Digit strings -- directly relevant to OCR output."""

    @pytest.mark.parametrize("s", [
        "0", "1", "9",
        "42", "99", "00",
        "123", "456", "789",
        "4 7 2 9",
        "0 1 2 3 4 5 6 7 8 9",
        "12345",
        "99999",
        "10203040506070809",
    ])
    def test_known_digit_strings(self, s):
        roundtrip(s)

    @pytest.mark.parametrize("length", range(1, 21))
    def test_digit_string_lengths(self, length):
        """Digit strings of varying length 1-20 digits with spaces."""
        rng = random.Random(length * 7919)  # deterministic per length
        digits = [str(rng.randint(0, 9)) for _ in range(length)]
        s = " ".join(digits)
        roundtrip(s)


class TestFullByteRange:
    def test_all_256_chars(self):
        """All 256 Latin-1 code points in a single string."""
        s = "".join(chr(i) for i in range(256))
        roundtrip(s)

    @pytest.mark.parametrize("i", range(256))
    def test_each_byte_value_alone(self, i):
        """Each byte value individually."""
        roundtrip(chr(i))


class TestMixedPatterns:
    @pytest.mark.parametrize("s", [
        "aababc",
        "hello world",
        "the quick brown fox jumps over the lazy dog",
        "aaabbbccc",
        "abcabc",
        "aaa bbb ccc",
        "mississippi",
        "abracadabra",
        "banana",
        "aaaaaabbbbbbcccc",
        "Hello, World!",
        "foo bar baz",
        "zzzzzzzzz",
        "xyxyzxyz",
        "1 2 3 4 5 6 7 8 9 0",
    ])
    def test_mixed(self, s):
        roundtrip(s)


# ---------------------------------------------------------------------------
# Random test cases (seeded for reproducibility) -- targeting 1000+ total
# ---------------------------------------------------------------------------

_RNG = random.Random(42)

def _rand_str(min_len: int, max_len: int, alphabet: str) -> str:
    n = _RNG.randint(min_len, max_len)
    return "".join(_RNG.choice(alphabet) for _ in range(n))


# Build parametrized lists at module level (deterministic seed)
_PRINTABLE = string.printable
_LATIN1    = "".join(chr(i) for i in range(256))

_SHORT_STRINGS   = [_rand_str(10, 50,  _PRINTABLE) for _ in range(300)]
_LONG_STRINGS    = [_rand_str(100, 500, _PRINTABLE) for _ in range(200)]
_LATIN1_STRINGS  = [_rand_str(10, 80,  _LATIN1)    for _ in range(200)]
_DIGIT_STRINGS   = [
    " ".join(str(_RNG.randint(0, 9)) for _ in range(_RNG.randint(1, 20)))
    for _ in range(150)
]
_REPUNIQ_STRINGS = []
for _ in range(150):
    rep  = _RNG.choice(_PRINTABLE) * _RNG.randint(3, 10)
    uniq = "".join(_RNG.choice(_PRINTABLE) for _ in range(_RNG.randint(3, 10)))
    _REPUNIQ_STRINGS.append(rep + uniq + rep)


class TestRandomShort:
    @pytest.mark.parametrize("s", _SHORT_STRINGS)
    def test_short_random(self, s):
        roundtrip(s)


class TestRandomLong:
    @pytest.mark.parametrize("s", _LONG_STRINGS)
    def test_long_random(self, s):
        roundtrip(s)


class TestLatin1Random:
    @pytest.mark.parametrize("s", _LATIN1_STRINGS)
    def test_latin1_random(self, s):
        roundtrip(s)


class TestRandomDigits:
    @pytest.mark.parametrize("s", _DIGIT_STRINGS)
    def test_digit_random(self, s):
        roundtrip(s)


class TestRepeatUnique:
    @pytest.mark.parametrize("s", _REPUNIQ_STRINGS)
    def test_repuniq(self, s):
        roundtrip(s)


# ---------------------------------------------------------------------------
# Bulk loop: ensure we hit the 1000-case minimum even after counting above
# ---------------------------------------------------------------------------

def test_bulk_roundtrip_loop():
    """
    An additional in-loop test that exercises 200 extra random inputs,
    bringing the overall test-case count well above 1000.
    """
    rng = random.Random(12345)
    for _ in range(200):
        length = rng.randint(0, 300)
        alphabet = [chr(i) for i in range(256)]
        s = "".join(rng.choice(alphabet) for _ in range(length))
        roundtrip(s)
