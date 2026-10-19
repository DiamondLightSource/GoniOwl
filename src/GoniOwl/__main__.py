import asyncio
import os
from argparse import ArgumentParser
from typing import Optional

from softioc import asyncio_dispatcher, builder, softioc

from . import __version__
from .GoniOwl_controller import (
    DEFAULT_CAMERA_URL,
    DEFAULT_LOG_PATH,
    DEFAULT_MODEL,
    GoniOwlController,
)

__all__ = ["main"]


def parse_args(argv: Optional[list[str]] = None):
    parser = ArgumentParser(description="GoniOwl pin-detection soft IOC.")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--prefix", default="GONIOWL-TEST", help="PV prefix.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="PyTorch checkpoint (.pt) to serve.")
    parser.add_argument("--camera-url", default=DEFAULT_CAMERA_URL, help="URL of the latest camera JPEG.")
    parser.add_argument("--log-path", default=DEFAULT_LOG_PATH, help="Controller log file ('' to disable).")
    parser.add_argument("--last-image", default="tmp.jpg", help="Where to save the last cropped frame ('' to disable).")
    parser.add_argument("--device", default="cpu", help="Torch device for inference.")
    parser.add_argument("--non-interactive", action="store_true", help="Run without the interactive IOC shell.")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None):
    args = parse_args(argv)
    print("Hello I am setting up")
    goniowl_controller = GoniOwlController(
        model_path=args.model,
        camera_url=args.camera_url,
        log_path=args.log_path or None,
        device=args.device,
        last_image_path=args.last_image or None,
    )

    dispatcher = asyncio_dispatcher.AsyncioDispatcher()
    lock = asyncio.Lock()

    builder.SetDeviceName(args.prefix)
    builder.stringIn("WHOAMI", initial_value="GoniOwl Python IOC")
    builder.stringIn("HOSTNAME", initial_value=os.uname()[1])
    goniostatus = builder.aOut("GONIOSTATUS", initial_value=0, LOPR=0, HOPR=3, PINI="YES")

    async def on_infer(value):
        # INFER is reset to 0 by HIGH=1, which also calls back here; ignore that.
        if not value:
            return
        async with lock:
            status = await asyncio.get_running_loop().run_in_executor(None, goniowl_controller.infer)
            goniostatus.set(status)

    # always_update: every put of 1 runs inference, even while HIGH still holds INFER at 1.
    # blocking: a put with callback (caput -c) completes only once GONIOSTATUS is updated.
    builder.boolOut(
        "INFER", ZNAM="Disable", ONAM="Enable", HIGH=1,
        on_update=on_infer, always_update=True, blocking=True,
    )
    builder.LoadDatabase()
    softioc.iocInit(dispatcher)

    if args.non_interactive:
        softioc.non_interactive_ioc()
    else:
        softioc.interactive_ioc(globals() | locals())


if __name__ == "__main__":
    main()
