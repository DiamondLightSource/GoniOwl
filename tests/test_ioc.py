"""Start the real IOC and drive it over Channel Access, as GDA does."""

import asyncio
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest

from conftest import free_port, raw_frame

pytestmark = [
    pytest.mark.slow,
    # aioca's connection callbacks can fire after asyncio.run() closes its loop.
    pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning"),
    pytest.mark.filterwarnings("ignore:Setting the shape on a NumPy array:DeprecationWarning"),
]

# Keep CA/PVA traffic on loopback and on ports no real IOC is using.
# This must happen before aioca loads libca.
CA_PORT = free_port()
os.environ.update(
    EPICS_CA_SERVER_PORT=str(CA_PORT),
    EPICS_CA_ADDR_LIST="127.0.0.1",
    EPICS_CA_AUTO_ADDR_LIST="NO",
    EPICS_CAS_INTF_ADDR_LIST="127.0.0.1",
    EPICS_PVA_SERVER_PORT=str(free_port()),
    EPICS_PVA_BROADCAST_PORT=str(free_port()),
    EPICS_PVA_ADDR_LIST="127.0.0.1",
    EPICS_PVA_AUTO_ADDR_LIST="NO",
)

aioca = pytest.importorskip("aioca")


@pytest.fixture
def ioc(camera, tmp_path):
    prefix = f"GONIOWL-PYTEST-{uuid.uuid4().hex[:6].upper()}"
    log_path = tmp_path / "controller.log"
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "GoniOwl", "--non-interactive",
            "--prefix", prefix,
            "--camera-url", camera.url,
            "--log-path", str(log_path),
            "--last-image", str(tmp_path / "tmp.jpg"),
        ],
        cwd=tmp_path,
        env=os.environ.copy(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    yield prefix, log_path
    proc.terminate()
    try:
        out, _ = proc.communicate(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        out, _ = proc.communicate()
    print(out)


async def wait_for_ioc(pv: str, timeout: float = 60.0):
    deadline = time.monotonic() + timeout
    while True:
        try:
            return await aioca.caget(pv, timeout=2)
        except aioca.CANothing:
            if time.monotonic() > deadline:
                raise


def test_ioc_serves_pvs_and_infers(ioc, camera, parity):
    prefix, log_path = ioc

    async def run():
        assert await wait_for_ioc(f"{prefix}:WHOAMI") == "GoniOwl Python IOC"
        assert await aioca.caget(f"{prefix}:HOSTNAME") == os.uname()[1]
        assert await aioca.caget(f"{prefix}:GONIOSTATUS") == 0

        statuses = []
        for path in parity["paths"]:
            camera.frame = raw_frame(path)
            await aioca.caput(f"{prefix}:INFER", 1, wait=True, timeout=30)
            statuses.append(int(await aioca.caget(f"{prefix}:GONIOSTATUS")))

        # A plain put (no callback, as GDA does today) also triggers inference.
        camera.frame = b"broken"
        await aioca.caput(f"{prefix}:INFER", 1)
        deadline = time.monotonic() + 10
        while int(await aioca.caget(f"{prefix}:GONIOSTATUS")) != 3 and time.monotonic() < deadline:
            await asyncio.sleep(0.05)
        statuses.append(int(await aioca.caget(f"{prefix}:GONIOSTATUS")))

        # HIGH=1 resets INFER to Disable after a second.
        await asyncio.sleep(1.5)
        assert await aioca.caget(f"{prefix}:INFER") == 0
        return statuses

    statuses = asyncio.run(run())
    assert statuses == [2, 2, 2, 1, 1, 1, 3, 3]
    log = Path(log_path).read_text()
    assert log.count("Status is ") == 7
    assert "Could not read camera image" in log
