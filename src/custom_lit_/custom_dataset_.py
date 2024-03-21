import pandas
from lit_nlp.api.dataset import Dataset
from src.utils.constant import flipkart_dataset_path
from lit_nlp.api import types as lit_types


class CustomDataset(Dataset):
    LABELS = ["positive", "negative"]

    def __init__(self):
        """Dataset constructor, loads the data into memory."""
        df = pandas.read_csv(flipkart_dataset_path)
        df = df[:10000]
        self._examples = [{
            'sentence': row['Summary'],
            'label': row['Sentiment'],
        } for _, row in df.iterrows()]

    def spec(self) -> lit_types.Spec:
        """Dataset spec, which should match the model"s input_spec()."""
        return {
            "sentence": lit_types.String(),
            "label": lit_types.CategoryLabel(vocab=self.LABELS),
        }
