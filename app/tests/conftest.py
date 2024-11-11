import pytest
import os
import shutil

# Define the directories and files to clean up
dirs_to_delete = [
    "dbscan_clusters",
    "hierarchical_clusters",
    "k_means_clusters",
    "mlruns",
]

files_to_delete = [
    "dbscan_model.pkl",
    "dendrogram.png",
    "hierarchical_model.pkl",
    "k_distance_plot.png",
]


@pytest.fixture(scope="session", autouse=True)
def cleanup_generated_files():
    """
    A pytest fixture that automatically deletes specified directories and files
    from the root of the project after the test session completes.
    """
    yield  # Run tests first

    # Get the absolute path to the root of the project
    root_dir = os.path.dirname(os.path.abspath(__file__))  # Start from tests folder
    while not os.path.exists(
        os.path.join(root_dir, "README.md")
    ):  # Look for a marker file (e.g., README.md)
        root_dir = os.path.abspath(os.path.join(root_dir, ".."))  # Go up one level

    # Delete specified directories
    for dir_name in dirs_to_delete:
        dir_path = os.path.join(root_dir, dir_name)
        if os.path.exists(dir_path):
            print(f"Deleting directory: {dir_path}")
            shutil.rmtree(dir_path)

    # Delete specified files
    for file_name in files_to_delete:
        file_path = os.path.join(root_dir, file_name)
        if os.path.exists(file_path):
            print(f"Deleting file: {file_path}")
            os.remove(file_path)
