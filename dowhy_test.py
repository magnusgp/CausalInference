

# Required libraries
import dowhy
from dowhy import CausalModel
import dowhy.datasets

# Avoiding unnecessary log messges and warnings
import logging
logging.getLogger("dowhy").setLevel(logging.WARNING)
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

# Load some sample data
data = dowhy.datasets.linear_dataset(
    beta=10,
    num_common_causes=5,
    num_instruments=2,
    num_samples=10000,
    treatment_is_binary=True,
    stddev_treatment_noise=10)

# I. Create a causal model from the data and domain knowledge.
model = CausalModel(
    data=data["df"],
    treatment=data["treatment_name"],
    outcome=data["outcome_name"],
    common_causes=data["common_causes_names"],
    instruments=data["instrument_names"])

model.view_model(layout="dot")
from IPython.display import Image, display
display(Image(filename="causal_model.png"))

# III. Estimate the target estimand using a statistical method.
propensity_strat_estimate = model.estimate_effect(identified_estimand,
                                 method_name="backdoor.dowhy.propensity_score_stratification")

print(propensity_strat_estimate)
"""
if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import math
    import dowhy
    from dowhy import CausalModel
    import dowhy.datasets, dowhy.plotter

    # Config dict to set the logging level
    import logging.config

    DEFAULT_LOGGING = {
        'version': 1,
        'disable_existing_loggers': False,
        'loggers': {
            '': {
                'level': 'INFO',
            },
        }
    }

    logging.config.dictConfig(DEFAULT_LOGGING)

    df = pd.read_csv('sample/data_569.csv', index_col = 0)

    dowhy.plotter.plot_treatment_outcome(df["A"], df["B"],
                                         df["C"])"""