from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from paths import path_key, resolve_data_path, resolve_project_path  # noqa: E402


def test_path_key_strips_authoring_root():
    p = r"C:\author\cfp9\dataset\_raw\Fundus10\images\original\A\x.jpg"
    assert path_key(p) == "_raw/Fundus10/images/original/A/x.jpg"
    assert path_key("dataset/01_MESSIDOR2/train/class_0/x.jpg") == "01_MESSIDOR2/train/class_0/x.jpg"


def test_path_key_outputs():
    p = r"C:\old\project\outputs\preprocessed\DDR\x.png"
    assert path_key(p) == "outputs/preprocessed/DDR/x.png"


def test_resolve_relative_data_path():
    got = resolve_data_path("01_MESSIDOR2/train/class_0/x.jpg")
    assert got == PROJECT_ROOT / "dataset" / "01_MESSIDOR2" / "train" / "class_0" / "x.jpg"


def test_resolve_relative_output_path():
    got = resolve_project_path("outputs/preprocessed/DDR/x.png")
    assert got == PROJECT_ROOT / "outputs" / "preprocessed" / "DDR" / "x.png"
