import pandas as pd
import os
import numpy as np

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 200)


from table import Table
from blackbox import BlackBox


if __name__ == "__main__":
    # address = "raw_points/wood/rs-70_pcds/1717020290-942984619/3/wood_rs-70_pcds_1717020290-942984619_172_1.npz"
    
    address = os.path.join('C:\\', 'GitHub', 'ransac_tomato', 'Archive2', 'raw_points', 'synthetic', 'normal_noise', '0_0.2100.npz')
    # address = 'C:/GitHub/ransac_tomato/Archive2/raw_points/synthetic/normal_noise/0_0.2100.npz'

    # volume, semiaxis, confine
    scale_corrections = {
    "polygon": (1e-6, 1, 100),
    "wood": (1e-6, 1., 100.),
    "synthetic": (1, 1., 1.)
    }

    main_variables = {
        "sphere_resolution": 5000,
        "n_iterations": 10000
    }

    hyperparameters = {
        "threshold": 0.1,
        "confine_coeff": 5,
        "h": 3,
        "method": "count",
    }


    blackbox = BlackBox(address = address, main_variables=main_variables, scale_correction= scale_corrections["wood"])
    blackbox.fit()

    df = Table(blackbox=blackbox, hyperparameters=hyperparameters, main_variables=main_variables, debug=True).prepare()


    import polars as pl
    # df = df.dropna()
    df = df.filter(pl.col("rejected") == 0)
    df.head(20)


