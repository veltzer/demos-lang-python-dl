""" python deps for this project """

import config.shared


install_requires: list[str] = [
    # core numerics
    "numpy",
    "pandas",
    "pyarrow",
    "scikit-learn",
    "matplotlib",
    "seaborn",
    "psutil",
    # pytorch
    "torch",
    "torchvision",
    "torchaudio",
    # tensorflow / keras
    "tensorflow",
    "keras",
    # jax / flax
    "jax",
    "jaxlib",
    "flax",
    # huggingface
    "transformers",
    "datasets",
    "accelerate",
]
build_requires: list[str] = config.shared.BUILD
test_requires: list[str] = config.shared.TEST
types_requires: list[str] = [
    "types-termcolor",
    "types-PyYAML",
    "types-psutil",
    "types-seaborn",
    "pandas-stubs",
]
requires = install_requires + build_requires + test_requires + types_requires
