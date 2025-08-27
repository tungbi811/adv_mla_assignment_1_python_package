import pytest
import pandas as pd
from pathlib import Path

from amla_at1_25608516.data.sets import load_data  # adjust import to your package path


@pytest.fixture
def tmp_data_path(tmp_path):
    """Temporary folder to simulate data/processed structure."""
    return tmp_path


def make_csv(path: Path, name: str, df: pd.DataFrame):
    """Helper to write CSV in tmp dir."""
    file = path / name
    df.to_csv(file, index=False)
    return file


def test_load_data_all_files(tmp_data_path):
    # Create dummy splits
    X_train = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    y_train = pd.Series([0, 1], name="target")
    X_val = pd.DataFrame({"a": [5], "b": [6]})
    y_val = pd.Series([1], name="target")
    X_test = pd.DataFrame({"a": [7], "b": [8]})
    y_test = pd.Series([0], name="target")

    make_csv(tmp_data_path, "X_train.csv", X_train)
    make_csv(tmp_data_path, "y_train.csv", y_train.to_frame())
    make_csv(tmp_data_path, "X_val.csv", X_val)
    make_csv(tmp_data_path, "y_val.csv", y_val.to_frame())
    make_csv(tmp_data_path, "X_test.csv", X_test)
    make_csv(tmp_data_path, "y_test.csv", y_test.to_frame())

    Xtr, ytr, Xv, yv, Xte, yte = load_data(tmp_data_path)

    pd.testing.assert_frame_equal(Xtr, X_train)
    pd.testing.assert_series_equal(ytr, y_train)
    pd.testing.assert_frame_equal(Xv, X_val)
    pd.testing.assert_series_equal(yv, y_val)
    pd.testing.assert_frame_equal(Xte, X_test)
    pd.testing.assert_series_equal(yte, y_test)


def test_load_data_missing_y_test(tmp_data_path):
    # Only train + val + X_test
    X_train = pd.DataFrame({"a": [1], "b": [2]})
    y_train = pd.Series([1], name="target")
    X_val = pd.DataFrame({"a": [3], "b": [4]})
    y_val = pd.Series([0], name="target")
    X_test = pd.DataFrame({"a": [5], "b": [6]})

    make_csv(tmp_data_path, "X_train.csv", X_train)
    make_csv(tmp_data_path, "y_train.csv", y_train.to_frame())
    make_csv(tmp_data_path, "X_val.csv", X_val)
    make_csv(tmp_data_path, "y_val.csv", y_val.to_frame())
    make_csv(tmp_data_path, "X_test.csv", X_test)
    # y_test.csv not created

    Xtr, ytr, Xv, yv, Xte, yte = load_data(tmp_data_path)

    assert not Xtr.empty
    assert not ytr.empty
    assert not Xv.empty
    assert not yv.empty
    assert not Xte.empty
    assert isinstance(yte, pd.Series)
    assert yte.empty


def test_load_data_missing_validation(tmp_data_path):
    # Only train + test
    X_train = pd.DataFrame({"a": [1, 2]})
    y_train = pd.Series([0, 1], name="target")
    X_test = pd.DataFrame({"a": [3, 4]})
    y_test = pd.Series([1, 0], name="target")

    make_csv(tmp_data_path, "X_train.csv", X_train)
    make_csv(tmp_data_path, "y_train.csv", y_train.to_frame())
    make_csv(tmp_data_path, "X_test.csv", X_test)
    make_csv(tmp_data_path, "y_test.csv", y_test.to_frame())
    # X_val.csv and y_val.csv not created

    Xtr, ytr, Xv, yv, Xte, yte = load_data(tmp_data_path)

    pd.testing.assert_frame_equal(Xtr, X_train)
    pd.testing.assert_series_equal(ytr, y_train)
    assert isinstance(Xv, pd.DataFrame) and Xv.empty
    assert isinstance(yv, pd.Series) and yv.empty
    pd.testing.assert_frame_equal(Xte, X_test)
    pd.testing.assert_series_equal(yte, y_test)
