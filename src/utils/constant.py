from pathlib import Path


curr_path = Path(__file__).parents[1]
data_directory = curr_path / 'data'

flipkart_dataset_path = data_directory / "review_flipkart.csv"
