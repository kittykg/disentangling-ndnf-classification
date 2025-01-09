import numpy as np
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo


def get_bcc_data(standardise: bool = False) -> tuple[np.ndarray, np.ndarray]:
    breast_cancer_coimbra = fetch_ucirepo(id=451)
    data = breast_cancer_coimbra.data  # data is pandas DataFrame

    X = data.features.to_numpy().astype(np.float32)  # type: ignore
    if standardise:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    y = data.targets.to_numpy().astype(np.float32).flatten() - 1  # type: ignore

    return X, y
