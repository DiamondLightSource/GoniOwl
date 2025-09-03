import cv2
import os
from datetime import date
import random
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import glob
import shutil
import argparse
import logging

# Logging
def setup_logging(level: str = "INFO", log_file: str | None = None) -> None:
    """Configure root logging with console and optional file handler."""
    numeric_level = getattr(logging, (level or "INFO").upper(), logging.INFO)
    fmt = "[%(asctime)s] [%(levelname)s] [%(processName)s] %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=numeric_level,
        format=fmt,
        datefmt=datefmt,
        handlers=handlers,
        force=True,
    )

logger = logging.getLogger(__name__)


# CLI
parser = argparse.ArgumentParser(description="Crop goniometer pin images and create a test split.")
parser.add_argument("--imgdir", default="/path/to/temporary/directory", help="Directory for output images")
parser.add_argument("--snapshots", default="/path/to/snapshots", help="Root directory of snapshots")
parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Logging verbosity")
parser.add_argument("--log-file", default=None, help="Optional path to a log file")
args = parser.parse_args()

setup_logging(args.log_level, args.log_file)


# Config
imgdir = args.imgdir
snapshots_location = args.snapshots

today = date.today()
random.seed(42)
now = today.strftime("%d%m%Y")
cwd = os.getcwd()
ON_folders = ["Pin_gripper_on_gonio"]  # Name of folder(s) with pin on images
OFF_folders = ["Before_sample_load"]   # Name of folder(s) with pin off images
folder_list = ["pinon", "pinoff"]
path = os.path.join(imgdir, f"goniopin_auto_{now}_binary")


def checkSetup():
    """Check if the snapshots directory exists and contains the required subdirectories."""
    logger.info("Validating input directory structure...")
    if not os.path.exists(snapshots_location):
        msg = f"Snapshots directory does not exist: {snapshots_location}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    for folder in ON_folders:
        on_path = os.path.join(snapshots_location, "pin_ON", folder)
        if not os.path.exists(on_path):
            msg = f"ON folder does not exist: {on_path}"
            logger.error(msg)
            raise FileNotFoundError(msg)
        logger.debug("Found ON folder: %s", on_path)

    for folder in OFF_folders:
        off_path = os.path.join(snapshots_location, "pin_OFF", folder)
        if not os.path.exists(off_path):
            msg = f"OFF folder does not exist: {off_path}"
            logger.error(msg)
            raise FileNotFoundError(msg)
        logger.debug("Found OFF folder: %s", off_path)

    logger.info("Input directories validated successfully.")


def croppit(filein, folderout):
    """Crop 100 pixels from each side of the image and save to output folder.

    Args:
        filein (os.path): file name in with path
        folderout (os.path): file name out with path
    """
    img = cv2.imread(filein)
    if img is None:
        logger.warning("Failed to load image: %s", filein)
        return

    h, w = img.shape[:2]
    if h <= 200 or w <= 200:
        logger.warning("Image too small to crop (h=%d, w=%d): %s", h, w, filein)
        return

    cropped_image = img[100:-100, 100:-100]
    os.makedirs(folderout, exist_ok=True)
    _, filename = os.path.split(filein)
    out_path = os.path.join(folderout, filename)

    ok = cv2.imwrite(out_path, cropped_image)
    if not ok:
        logger.error("Failed to write cropped image: %s", out_path)
    else:
        logger.debug("Wrote cropped image: %s", out_path)


def moveImagesToTest(path=path, percentage=10):
    """Move a percentage of images from each class folder to a test folder. Images moved are randomly selected.

    Args:
        path (os.path, optional): Parent directory to move test images to. Defaults to path (variable, set at beginning of script).
        percentage (float, optional): Percentage of images to move to test directory. Defaults to 10.
    """
    logger.info("Creating test split (percentage=%s%%)...", percentage)
    testdir = os.path.join(imgdir, f"test_{now}_binary")
    os.makedirs(testdir, exist_ok=True)

    total_moved = 0
    for folder in folder_list:
        classdir = os.path.join(path, folder)
        testclassdir = os.path.join(testdir, folder)
        os.makedirs(testclassdir, exist_ok=True)

        images = glob.glob(os.path.join(classdir, "*.jpg"))
        numImg = int(len(images) * (percentage / 100))

        logger.info(
            "Folder '%s': found %d images, will move %d to test: %s -> %s",
            folder, len(images), numImg, classdir, testclassdir
        )

        if numImg == 0:
            logger.info("No images to move from %s (too few images for requested percentage).", classdir)
            continue

        try:
            imgMov = random.sample(images, numImg)
        except ValueError as e:
            logger.error("Sampling error in '%s' (requested=%d, available=%d): %s", folder, numImg, len(images), e)
            continue

        for image in tqdm(imgMov, desc=f"Moving images from {folder}", leave=False):
            try:
                shutil.move(image, testclassdir)
                logger.debug("Moved %s -> %s", image, testclassdir)
                total_moved += 1
            except Exception as e:
                logger.exception("Failed to move %s -> %s: %s", image, testclassdir, e)

    logger.info("Test split created. Total images moved: %d", total_moved)


def run():
    """Main function to create directories, process images in parallel, and save cropped images."""
    logger.info("Starting run with imgdir='%s', snapshots='%s'", imgdir, snapshots_location)
    checkSetup()

    os.makedirs(path, exist_ok=True)
    for folder in folder_list:
        os.makedirs(os.path.join(path, folder), exist_ok=True)

    max_workers = max(1, multiprocessing.cpu_count() - 1)
    logger.info("Using up to %d worker processes.", max_workers)

    futures = []
    total_scheduled = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for pinon_image_dir in ON_folders:
            searchdir = os.path.join(snapshots_location, "pin_ON", pinon_image_dir)
            files = [f for f in os.listdir(searchdir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            logger.info("Scheduling %d ON images from %s", len(files), searchdir)
            for file in files:
                image = os.path.join(searchdir, file)
                futures.append(executor.submit(croppit, image, os.path.join(path, "pinon")))
                total_scheduled += 1

        for pinoff_image_dir in OFF_folders:
            searchdir = os.path.join(snapshots_location, "pin_OFF", pinoff_image_dir)
            files = [f for f in os.listdir(searchdir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            logger.info("Scheduling %d OFF images from %s", len(files), searchdir)
            for file in files:
                image = os.path.join(searchdir, file)
                futures.append(executor.submit(croppit, image, os.path.join(path, "pinoff")))
                total_scheduled += 1

        logger.info("Total images scheduled for cropping: %d", total_scheduled)

        failures = 0
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing images", leave=True):
            try:
                future.result()
            except Exception as e:
                failures += 1
                logger.exception("A worker task failed: %s", e)

        logger.info("Cropping complete. Success: %d, Failures: %d", total_scheduled - failures, failures)

    moveImagesToTest()


if __name__ == "__main__":
    run()