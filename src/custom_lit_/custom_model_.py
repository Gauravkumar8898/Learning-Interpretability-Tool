# import tensorflow as tf
from lit_nlp.api.model import Model
from lit_nlp.api import types as lit_types
from lit_nlp.api import types
import joblib
from collections.abc import Iterable

Input = types.Input

class NLIModel(Model):
    """Wrapper for a Natural Language Inference model."""

    NLI_LABELS = ['positive', 'negative']

    def __init__(self, model_path):
        # Load the model into memory so, we're ready for interactive use.
        self._model = joblib.load(model_path)

    # LIT API implementations
    def predict(self, inputs: Iterable[Input]):
        import pandas
        df = pandas.DataFrame(inputs)
        # for d in inputs:
        #     print(d['sentence'])
        #     break
        print(df)
        """Predict on a stream of examples."""

        # examples = [self._model.convert_dict_input(d) for d in inputs]  # any custom preprocessing
        # print(examples)
        # return self._model.predict(examples)  # returns a dict for each input
        return self._model.predict(df.sentence)


    def input_spec(self) -> types.Spec:
        """Describe the inputs to the model."""
        return {
            'sentence': lit_types.String(),

        }

    def output_spec(self) -> types.Spec:
        """Describe the model outputs."""
        return {
            # The 'parent' keyword tells LIT where to look for gold labels when computing metrics.
            'probas': lit_types.MulticlassPreds(vocab=self.NLI_LABELS, parent='label'),
        }
