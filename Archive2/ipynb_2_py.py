# %%
# !python -m pip install tables pyarrow --upgrade
# !pip install polars

# %%
import pandas as pd
import os
import numpy as np
import pyarrow
from my_plot import Plotter
import matplotlib.pyplot as plt
import numpy.linalg as la

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 200)

# %%
from table import Table
from blackbox import BlackBox

# %%

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




# %%
blackbox = BlackBox(address = address, main_variables=main_variables, scale_correction= scale_corrections["synthetic"])
blackbox.fit()

df = Table(blackbox=blackbox, hyperparameters=hyperparameters, main_variables=main_variables, debug=True).prepare()


import polars as pl
# df = df.dropna()
df = df.filter(pl.col("rejected") == 0)
df.head(20)

# %%
df.describe()

# %%
# 26 = 0-9 equation coefs
# 10-12 - center
# 13-15 - radii
# 16-24 - rotation
# 25 - rejection (>0 for rejected, 0 for accepted)
blackbox.get_predictions()[0]

# %%
# accepted
np.where(blackbox.get_predictions()[:, -1] == 0)

# %%


# %%
# True
T = blackbox.tomato.center
S = np.diag(blackbox.tomato.semiaxis)
R = blackbox.tomato.rotation

# %%
# 26 = 0-9 equation coefs
# 10-12 - center
# 13-15 - radii
# 16-24 - rotation
# 25 - rejection (>0 for rejected, 0 for accepted)

index = -1 + 9995
prediction = blackbox.get_predictions()[index]

T_pred = np.array(prediction[10:13])
S_pred = np.diag(prediction[13:16])
R_pred = np.array(prediction[16:25].reshape((3, 3)))

# %%
hypothesis = blackbox.get_predictions()[index][:10]
hypothesis

# %%
from my_plot import Plotter

plotter = Plotter()
plotter.plot_ellipsoid(S, R.T, T, name='True', alpha=0.6)
plotter.plot_ellipsoid(S_pred, R_pred.T, T_pred, name='predicted', alpha=0.5)

plotter.plot_points(blackbox.tomato.points.T, alpha=0.7, size=0.5)

plotter.show()

# %%
df.to_pandas().describe()

# %%
from ellipsoid import ELLIPSOID
import torch

synth_ellip = ELLIPSOID(
                                params = [T, np.diag(S), R], 
                                points = None, 
                                fill = True, 
                                resolution=10**4,
                                dev=torch.device('cpu')
                            )

from ellipsoid import ELLIPSOID
import torch

synth_ellip_pred = ELLIPSOID(
                                params = [T_pred, np.diag(S_pred), R_pred], 
                                points = None, 
                                fill = True, 
                                resolution=10**4,
                                dev=torch.device('cpu')
                            )


# %%
from my_plot import Plotter

plotter = Plotter()
plotter.plot_ellipsoid(S, R.T, T, name='True', alpha=0.6)
plotter.plot_ellipsoid(S_pred, R_pred.T, T_pred, name='predicted', alpha=0.5)

plotter.plot_points(synth_ellip.points.T, alpha=0.7, size=0.5)
plotter.plot_points(synth_ellip_pred.points.T, alpha=0.3, size=0.5)

plotter.show()

# %%
true_sur_points = plotter.gen_ellipse_points(S, R.T, T).reshape((3, -1)).T
true_sur_points

# %%
plotter = Plotter()
plotter.plot_ellipsoid(S, R.T, T, name='True', alpha=0.6)
# plotter.plot_ellipsoid(S_pred, R_pred.T, T_pred, name='predicted', alpha=0.5)

plotter.plot_points(true_sur_points.T, alpha=0.7, size=0.5)

plotter.show()

# %%
def subs_point(point, hypothesis):
    x, y, z = point
    pt = np.array([x**2, y**2, z**2, 2*x*y, 2*x*z, 2*y*z, x, y, z, -1])

    return pt @ np.array(hypothesis)

def subs_points(points, hypothesis):
    results = np.zeros(len(points))
    for i, point in enumerate(points):
        results[i] = subs_point(point, hypothesis)

    return results

# %%
def M_m(R, S, T):

    M = R.T @ la.inv(S)**2 @ R
    m = (-2) * T @ R.T @ la.inv(S)**2 @ R
    m_ = 1 - T @ R.T @ la.inv(S)**2 @ R @ T

    M /= m_
    m /= m_

    return M, m

def calc_hyp(R, S, T):
    M, m = M_m(R.T, S, T)

    return np.array([M[0, 0], M[1, 1], M[2, 2], M[0, 1], M[0, 2], M[1, 2], *m, 1])



# %%
true_hyp = calc_hyp(R, S, T)
true_hyp

# %%
buf = subs_points(true_sur_points, true_hyp)

plt.hist(buf)

# %%
buf = subs_points(synth_ellip.points, hypothesis)

plt.hist(buf)

# %%
buf = subs_points(synth_ellip_pred.points, true_hyp)

plt.hist(buf)

# %%
np.sum(buf < 0) / len(buf)

# # %%
# A U B = A + B - A^B

# # %%
# A^B / A U B

# %%
def count_in(points, hypothesis):
    buf = subs_points(points, hypothesis)

    return np.sum(buf < 0) / len(buf)

# %%
A = synth_ellip.points
B = synth_ellip_pred.points

A_hyp = true_hyp
B_hyp = hypothesis

count_in(B, B_hyp)

# %%
B_hyp

# %%
calc_hyp(R_pred, S_pred, T_pred)

# %%
buf = subs_points(synth_ellip_pred.points, hypothesis)

plt.hist(buf)

# %%

plotter = Plotter()
# plotter.plot_ellipsoid(S, R.T, T, name='True', alpha=0.6)
plotter.plot_ellipsoid(S_pred, R_pred.T, T_pred, name='predicted', alpha=0.5)

plotter.plot_points(B.T, alpha=0.7, size=0.5)
# plotter.plot_points(synth_ellip_pred.points.T, alpha=0.3, size=0.5)

plotter.show()


