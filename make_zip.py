import os
import zipfile

SOURCE = "."

OUTPUT = "../my_project_clean.zip"

EXCLUDE_DIRS = {
    ".venv",
    "venv",
    "__pycache__",
    ".git",
    ".idea",
    ".vscode"
}

EXCLUDE_FILES = {
    "unet_baseline.pth",
    "prediction_result.png",
    "prediction_probability.png",
    "test_mask.png"
}

with zipfile.ZipFile(
    OUTPUT,
    "w",
    compression=zipfile.ZIP_DEFLATED
) as zip_file:

    for root, dirs, files in os.walk(SOURCE):

        # Don't include unwanted directories
        dirs[:] = [
            d for d in dirs
            if d not in EXCLUDE_DIRS
        ]

        for file in files:

            if file in EXCLUDE_FILES:
                continue

            # Don't include the ZIP itself
            if file.endswith(".zip"):
                continue

            full_path = os.path.join(root, file)

            # Path inside ZIP
            archive_path = os.path.relpath(
                full_path,
                SOURCE
            )

            zip_file.write(
                full_path,
                archive_path
            )

print("Clean ZIP created!")
print("Location:", os.path.abspath(OUTPUT))