import os
import pytest
import matplotlib.pyplot as plt
from collections import Counter

# =========================
# DATASET CLASS DISTRIBUTION LOGIC
# =========================
def class_distribution(data_directory):
    """Scans the dataset directory and returns a dictionary of class distributions."""
    class_counts = {}

    for category in os.listdir(data_directory):
        category_path = os.path.join(data_directory, category)

        if os.path.isdir(category_path):
            num_images = len([f for f in os.listdir(category_path) if os.path.isfile(os.path.join(category_path, f))])
            class_counts[category] = num_images

    return class_counts

def test_class_distribution(class_dist):
    """Prints a warning if some expected classes are missing or imbalanced."""
    expected_classes = {"Tumor", "Cyst", "Stone", "Normal"}
    missing_classes = expected_classes - class_dist.keys()

    if missing_classes:
        print(f"⚠️ Warning: Missing classes detected: {missing_classes}")

    if any(count == 0 for count in class_dist.values()):
        print(f"⚠️ Warning: Some classes have zero images: {class_dist}")

def plot_class_distribution(class_dist):
    """Generates a simple bar chart for visualization."""
    plt.bar(class_dist.keys(), class_dist.values(), color=['red', 'blue', 'green', 'purple'])
    plt.xlabel("Class Labels")
    plt.ylabel("Number of Images")
    plt.title("Dataset Class Distribution")
    plt.show()

# =========================
# PYTEST FIXTURES & TESTS
# =========================
@pytest.fixture
def mock_data_directory(tmp_path):
    """Creates a temporary dataset directory for testing."""
    categories = {"Tumor": 10, "Cyst": 15, "Stone": 5, "Normal": 12}
    
    for category, num_images in categories.items():
        category_path = tmp_path / category
        category_path.mkdir()

        for i in range(num_images):
            (category_path / f"image_{i}.jpg").touch()

    return tmp_path

@pytest.fixture
def class_dist(mock_data_directory):
    """Fixture to return class distribution for testing."""
    return class_distribution(str(mock_data_directory))

def test_class_distribution_counts(class_dist):
    """Test if the function correctly counts the images in each category."""
    expected_counts = {"Tumor": 10, "Cyst": 15, "Stone": 5, "Normal": 12}
    assert class_dist == expected_counts, f"Expected {expected_counts}, but got {class_dist}"

def test_class_distribution_warnings(mock_data_directory, capsys):
    """Test if missing classes trigger warnings."""
    incomplete_dir = mock_data_directory / "MissingClass"
    incomplete_dir.mkdir()

    class_dist = class_distribution(str(mock_data_directory))
    test_class_distribution(class_dist)

    captured = capsys.readouterr()
    
    # ✅ Ensure the general warning is printed
    assert "⚠️ Warning: Some classes have zero images:" in captured.out, "Expected warning missing."
    
    # ✅ Check that the specific class ('MissingClass': 0) is in the message, but allow for other keys
    assert "'MissingClass': 0" in captured.out, "Unexpected missing class warning."
    
def test_plot_class_distribution(class_dist):
    """Ensure plot function runs without error."""
    try:
        plot_class_distribution(class_dist)
    except Exception as e:
        pytest.fail(f"plot_class_distribution raised an error: {e}")
