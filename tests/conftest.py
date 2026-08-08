"""Keep the arithmetic tests hermetic.

`cfas/waggle.py` holds its calibration in module state, because it is a property of a
deployment rather than of a request. That is defensible in the server and hazardous in
a test suite: loading a deployment's calibration file silently changes the numbers
every previously-written test is asserting against, and the order tests happen to run
in then decides whether they pass.

So every test starts from the documented reference calibration. Tests that want the
numbers a deployment actually ships ask for them explicitly, in test_calibration.py,
where the values come from the file rather than from whatever ran first.
"""
import pytest

from cfas import waggle

REFERENCE_TPR = 0.82
REFERENCE_FPR = 0.48


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "deployed_calibration: run against the calibration this deployment ships, "
        "rather than the documented reference. For integration tests, which should "
        "exercise the real numbers.")


@pytest.fixture(autouse=True)
def reference_calibration(request, monkeypatch):
    if request.node.get_closest_marker("deployed_calibration"):
        return
    monkeypatch.setattr(waggle, "TPR", REFERENCE_TPR)
    monkeypatch.setattr(waggle, "FPR", REFERENCE_FPR)
