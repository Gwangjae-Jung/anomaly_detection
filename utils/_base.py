from    pathlib     import  Path
from    warnings    import  warn


__all__: list[str] = [
    'MVTEC_AD', 'MNIST', 'CIFAR10',
    'BASE_PATHS',
]


MVTEC_AD = 'mvtec_ad'
"""The keyword for the MVTec Anomaly Detection dataset.

-----
### Structure
Inside the MVTec-AD dataset, there are 15 categories:
>>> 'bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper'

Each category consists of 3 subcategories:
>>> 'ground_truth', 'train', 'test'

* Inside `'train'`, there is a unique subcategory `'good'`, as the training dataset for general anomaly detection task consists of normal data only.
* Inside `'test'`, there are subcategories (for example, as for the categiry `'pill'`, there are `'combined'`, `'crack'`, `'scratch'`, etc.), *including `'good'`* as in `'train'`.
* Inside `'ground_truth'`, there are subcategories in `'test'` *except for `'good'`. This subcategory saves masks for the anomalies in `'test'`.

-----
See https://www.mvtec.com/company/research/datasets/mvtec-ad for more information.
"""
MNIST   = 'mnist'
"""The keyword for the MNIST dataset.

-----
### Structure
The dataset is split into the training dataset (`train.npz`) and the test dataset (`test.npz`).
Each file saves the dictionary of `numpy.ndarray` objects, where the data is saved with the key `data`, and the targets are saved with the key `targets`.
"""
CIFAR10 = 'cifar10'
"""The keyword for the CIFAR-10 dataset.

-----
### Structure
The dataset is split into the training dataset (`train.npz`) and the test dataset (`test.npz`).
Each file saves the dictionary of `numpy.ndarray` objects, where the data is saved with the key `data`, and the targets are saved with the key `targets`.
"""

path__mvtec_ad: Path
path__mnist:    Path
path__cifar10:  Path
try:
    path__mvtec_ad  = Path(rf"/home/gwangjae_jung/anomaly_detection/datasets/mvtec_ad")
except FileNotFoundError:
    warn(f"The directory {path__mvtec_ad} is not found.")
try:
    path__mnist     = Path(rf"/home/gwangjae_jung/anomaly_detection/datasets/mnist")
except FileNotFoundError:
    warn(f"The directory {path__mnist} is not found.")
try:
    path__cifar10   = Path(rf"/home/gwangjae_jung/anomaly_detection/datasets/cifar10")
except FileNotFoundError:
    warn(f"The directory {path__cifar10} is not found.")

BASE_PATHS: dict[str, Path] = {
    MVTEC_AD:   path__mvtec_ad,
    MNIST:      path__mnist,
    CIFAR10:    path__cifar10,
}