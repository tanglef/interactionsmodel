from setuptools import setup, find_packages
from interactionsmodel import __version__

name = "interactionsmodel"
description = "Penalized models with interactions"
author = "Florent Bascou"
author_email = "florent.bascou@umontpellier.fr"


setup(
    name="InteractionsModel",
    description="Making feature selection in penalized" + "models with interactions",
    author="Florent Bascou",
    author_email=author_email,
    packages=find_packages(),
    version=__version__,
)
