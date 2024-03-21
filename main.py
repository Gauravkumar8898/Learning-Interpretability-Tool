from src.custom_lit_.custom_dataset_ import CustomDataset
from src.custom_lit_.custom_model_ import NLIModel
from src.utils.constant import model_path
from lit_nlp.api.components import Metrics
from lit_nlp import dev_server
from absl import app
from collections.abc import Sequence
import sys
import logging
from typing import Optional
from absl import flags
from lit_nlp import server_flags


FLAGS = flags.FLAGS

FLAGS.set_default("development_demo", True)

_MAX_EXAMPLES = flags.DEFINE_integer(
    "max_examples", 1000, "Maximum number of examples to load into LIT. ")


def get_wsgi_app() -> Optional[dev_server.LitServerType]:
    """Returns a LitApp instance for consumption by gunicorn."""
    FLAGS.set_default("server_type", "external")
    FLAGS.set_default("demo_mode", True)
    # Parse flags without calling app.run(main), to avoid conflict with
    # gunicorn command line flags.
    unused = flags.FLAGS(sys.argv, known_only=True)
    if unused:
        logging.info(
            "toxicity_demo:get_wsgi_app() called with unused args: %s", unused
        )
    return main([])


def main(argv: Sequence[str]) -> Optional[dev_server.LitServerType]:
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")
    # Multi NLIData implements the Dataset API
    datasets = {
        'flipkart_review_dataset': CustomDataset(),
    }

    # NLIModel implements the Model API
    models = {
        'flipkart_review_model': NLIModel(model_path),
    }

    metrics = {
        # 'accuracy': MyAccuracyMetric(),
    }

    lit_demo = dev_server.Server(models, datasets, **server_flags.get_flags())
    return lit_demo.serve()


if __name__ == '__main__':
    app.run(main)
