"""
Configuration of subsampling.
"""
from pathlib import Path

circle_names_with_diameter = {
    "Getaberget_20m_4_3_area": 50,
    "Getaberget_20m_9_2_area": 50,
    "Getaberget_20m_8_3_area": 50,
    "Getaberget_20m_7_1_area": 50,
    "Getaberget_20m_7_2_area": 20,  # 20 m
    "Getaberget_20m_5_1_area": 50,
    "Getaberget_20m_2_1_area": 40,  # 40 m
    "Getaberget_20m_2_2_area": 50,
    "Getaberget_20m_1_1_area": 50,
    "Getaberget_20m_1_2_area": 40,  # 40 m
    "Getaberget_20m_1_3_area": 20,  # 10 m
    "Getaberget_20m_1_4_area": 50,
    "Havsvidden_20m_1_area": 50,
}

results_path = Path("../results")

analysis_points_path = Path("../results/Ahvenanmaa_analysis_points.gpkg")

shoreline_geojson_url = (
    "https://raw.githubusercontent.com/nialov/"
    "fractopo-subsampling/master/misc/shoreline.geojson"
)
base_circle_reference_value_csv_path = results_path / "base_reference_values.csv"

subsampling_path = results_path / "subsampling"


base_circle_ids_csv_path = results_path / "base_circle_ids.csv"

filtered_analysis_points = results_path / "filtered_analysis_points.gpkg"
