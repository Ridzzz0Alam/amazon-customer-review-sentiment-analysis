"""Extra cells that turn the main notebook into a Google Colab ready notebook.

build_notebook.py inserts COLAB_PRE after the title cell and appends COLAB_POST at the end.
"""

from pathlib import Path

PREDICT_SOURCE = (Path(__file__).parent / "src" / "predict.py").read_text()

COLAB_PRE = [
    ("md", """
## 0. Colab setup

This version of the notebook is ready for Google Colab. It needs no GPU. A standard CPU runtime finishes in a few minutes.

**How to run it**

1. Click **Runtime** and then **Run all**.
2. The first cell checks the library versions. If Colab has different versions it installs the pinned ones and **restarts the session once**. Colab may show a message that the session crashed. That is expected.
3. Click **Runtime** and then **Run all** again. This time every cell runs to the end.

**Where the data comes from.** Set `DATA_SOURCE` in the second setup cell.

| Option | What it does |
|---|---|
| `"kaggle"` (default) | Downloads the dataset with `kagglehub`. Public datasets usually work without a login. If Kaggle asks for credentials add `KAGGLE_USERNAME` and `KAGGLE_KEY` as Colab secrets (key icon on the left) |
| `"drive"` | Reads `Amazon_Unlocked_Mobile.csv` from your Google Drive. Edit `DRIVE_CSV_PATH` to match where you put it |
| `"upload"` | Opens a file picker so you can upload the CSV from your computer. This is slow for a 130 MB file |
"""),
    ("code", r'''
# Setup 1 of 2. Pin the library versions used for the saved model and restart once if needed.
import importlib.metadata as md
import os
import subprocess
import sys

PINNED = {"scikit-learn": "1.9.1", "pandas": "3.0.5", "joblib": "1.6.0"}

def installed(pkg):
    try:
        return md.version(pkg)
    except md.PackageNotFoundError:
        return None

mismatch = {p: v for p, v in PINNED.items() if installed(p) != v}
if mismatch:
    print("Installing pinned versions:", mismatch)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "kagglehub",
                    *[f"{p}=={v}" for p, v in PINNED.items()]], check=True)
    print("Done. Restarting the session now. When it reconnects choose Runtime > Run all again.")
    sys.stdout.flush()
    os.kill(os.getpid(), 9)
else:
    print("Library versions OK:", {p: installed(p) for p in PINNED})
'''),
    ("code", r'''
# Setup 2 of 2. Get the data and create the project folders.
import os
from pathlib import Path

DATA_SOURCE = "kaggle"          # "kaggle" or "drive" or "upload"
DRIVE_CSV_PATH = "/content/drive/MyDrive/Amazon_Unlocked_Mobile.csv"
KAGGLE_DATASET = "PromptCloudHQ/amazon-reviews-unlocked-mobile-phones"
CSV_NAME = "Amazon_Unlocked_Mobile.csv"

try:
    import google.colab  # noqa: F401
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

WORK_DIR = Path("/content/amazon-review-sentiment") if IN_COLAB else Path.cwd()
WORK_DIR.mkdir(parents=True, exist_ok=True)
os.chdir(WORK_DIR)
for sub in ["data", "models", "reports/figures", "src"]:
    Path(sub).mkdir(parents=True, exist_ok=True)

local_csv = Path("data") / CSV_NAME
if local_csv.exists():
    COLAB_DATA_PATH = local_csv
elif DATA_SOURCE == "kaggle":
    import kagglehub
    try:
        folder = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    except Exception as exc:
        raise RuntimeError(
            "Kaggle download failed. Add KAGGLE_USERNAME and KAGGLE_KEY as Colab secrets "
            'or set DATA_SOURCE = "drive" and put the CSV in your Google Drive.'
        ) from exc
    COLAB_DATA_PATH = next(folder.rglob(CSV_NAME))
elif DATA_SOURCE == "drive":
    from google.colab import drive
    drive.mount("/content/drive")
    COLAB_DATA_PATH = Path(DRIVE_CSV_PATH)
elif DATA_SOURCE == "upload":
    from google.colab import files
    uploaded = files.upload()
    name = next(iter(uploaded))
    Path(name).replace(local_csv)
    COLAB_DATA_PATH = local_csv
else:
    raise ValueError('DATA_SOURCE must be "kaggle" or "drive" or "upload"')

if not Path(COLAB_DATA_PATH).exists():
    raise FileNotFoundError(f"Could not find the dataset at {COLAB_DATA_PATH}. Check DATA_SOURCE and the path.")

# The prediction helper used in section 8. Written here so the notebook needs no other files.
Path("src/predict.py").write_text(PREDICT_SOURCE)

print("Working folder:", WORK_DIR)
print("Dataset:", COLAB_DATA_PATH, f"({Path(COLAB_DATA_PATH).stat().st_size / 1e6:.0f} MB)")
'''.replace("PREDICT_SOURCE", repr(PREDICT_SOURCE))),
]

COLAB_POST = [
    ("md", """
## 10. Save your results

Colab deletes files when the session ends. The cell below packs the saved model and the figures into one zip file and downloads it. Set `COPY_TO_DRIVE = True` to also keep a copy in your Google Drive.
"""),
    ("code", r'''
import shutil

COPY_TO_DRIVE = False

bundle = Path("sentiment_results")
if bundle.exists():
    shutil.rmtree(bundle)
bundle.mkdir()
shutil.copytree("models", bundle / "models")
shutil.copytree("reports", bundle / "reports")
shutil.copy("src/predict.py", bundle / "predict.py")
zip_path = shutil.make_archive("sentiment_results", "zip", root_dir=bundle)
print("Created", zip_path)

if IN_COLAB:
    from google.colab import files
    files.download(zip_path)
    if COPY_TO_DRIVE:
        from google.colab import drive
        drive.mount("/content/drive")
        shutil.copy(zip_path, "/content/drive/MyDrive/sentiment_results.zip")
        print("Copied to MyDrive/sentiment_results.zip")
'''),
]
