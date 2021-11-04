# Core Library
import os
from pathlib import Path

# Third party
import numpy as np
import pandas as pd
from numpy.typing import NDArray

# First party
from metrics.ols import ols

cache = Path(os.getenv("XDG_CACHE_HOME")) / "py-did"  # type: ignore
cache.mkdir(exist_ok=True)


def cr_vce(X: NDArray, eps: NDArray, cluster_indicator: NDArray, correction: int = 1) -> NDArray:
    # Cluster-robust variance estimate
    # Correction: 1 - Liang and Zeger (1986)
    _, k = X.shape
    sorter = cluster_indicator.argsort()
    X_ordered, eps_ordered = X[sorter, :], eps[sorter]
    _, clu_starts = np.unique(cluster_indicator[sorter], return_index=True)
    clu_ends = np.append(clu_starts[1:], len(X))
    b_clu = np.zeros((k, k))
    for clu_start, clu_end in zip(clu_starts, clu_ends):
        X_g = X_ordered[clu_start:clu_end, :]
        eps_g = eps_ordered[clu_start:clu_end] * np.sqrt(correction)
        b_clu += X_g.T @ np.kron(eps_g[:, None], eps_g) @ X_g

    bread = np.linalg.inv(X_ordered.T @ X_ordered)
    return bread @ (b_clu) @ bread


def hew_vce(X: NDArray[np.float_], eps: NDArray[np.float_]) -> NDArray[np.float_]:
    # Huber-White Variance-Covariance Matrix
    # TODO: Consider implementing correction factors
    n, k = X.shape
    bread = np.linalg.inv(X.T @ X)
    vce_eps = np.zeros((n, n))
    np.fill_diagonal(vce_eps, eps ** 2)
    correction = n / (n - k)
    return correction * (bread @ (X.T @ vce_eps @ X) @ bread)


def hom_vce(X: NDArray, res: NDArray):
    n, k = X.shape
    inv = np.linalg.inv(X.T @ X)
    res_var = res @ res / (n - k)
    return inv * res_var


if __name__ == "__main__":
    # Card and Krueger
    data = (
        pd.read_stata("/Users/moritz.helm/Downloads/CK1994.dta")
        .assign(state=lambda df: df["state"].map({1: "NJ", 0: "PA"}).astype("category"))
        .assign(
            time=lambda df: df["time"]
            .map({0: "first_survey", 1: "second_survey"})
            .astype("category")
        )
    )
    data = data.dropna(subset=["empft", "emppt", "nmgrs"])
    data["emp_calc"] = data["nmgrs"] + data["empft"] + data["emppt"] * 0.5
    X = data[["time", "state"]]
    X["time"] = X.time.cat.codes
    X["state"] = X.state.cat.codes
    X["interaction"] = X["time"] * X["state"]
    X.insert(0, "constant", np.ones(len(X)))
    y = data["emp_calc"]

    to_save = pd.concat((X, y), axis=1).reset_index(drop=True)
    to_save.to_feather(cache / "data.feather")

    coeffs, res = ols(y.values, X.values)
    crve_ = cr_vce(X.values, res, data["state"].values)
    np.sqrt(np.diag(crve_))
    vce = hew_vce(X.values, res)
    np.sqrt(np.diag(vce))

    vce = hom_vce(X.values, res)
    np.sqrt(np.diag(vce))

    print(coeffs[0])
    gg = (
        data.groupby(["time", "state"])["emp_calc"]
        .mean()
        .unstack("state")
        .sort_index(axis=1, ascending=False)
        .assign(diff=lambda df: df["NJ"] - df["PA"])
    )

    gg.loc["second_survey", "diff"] - gg.loc["first_survey", "diff"]
    # Standard errors
    # Interpretation

    data2 = pd.read_feather(cache / "data2.feather")
    X = data2[["X1", "X2", "X3"]]
    X.insert(0, "constant", np.ones(len(X)))
    y = data2["y"]
    coeffs, res = ols(y.values, X.values)

    cr_vce(X.values, res, data2.cluster.values)
    np.sqrt(np.diag(cr_vce(X.values, res, data2.cluster)))
