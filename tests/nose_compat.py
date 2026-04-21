"""Drop-in replacements for nose.tools assertion functions, backed by plain asserts."""


def assert_equal(a, b, msg=None):
    assert a == b, msg or f"{a!r} != {b!r}"


def assert_true(expr, msg=None):
    assert expr, msg or f"Expected truthy, got {expr!r}"


def assert_false(expr, msg=None):
    assert not expr, msg or f"Expected falsy, got {expr!r}"


def assert_almost_equal(a, b, places=7, msg=None):
    assert round(a - b, places) == 0, msg or f"{a!r} != {b!r} within {places} places"


def nottest(func):
    """Mark a function as not a test (pytest ignores non-test-prefixed functions anyway)."""
    func.__test__ = False
    return func
