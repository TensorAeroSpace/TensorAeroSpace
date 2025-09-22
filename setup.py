"""
Конфигурация установки пакета TensorAeroSpace.

Этот файл содержит настройки для установки пакета TensorAeroSpace,
включая зависимости, метаданные и параметры сборки.
"""

from setuptools import find_packages, setup


with open('./requirements.txt') as f:
    required = f.read().splitlines()

setup(name='tensoraerospace',
        version='0.2.1',
        install_requires=required,
        packages=[package for package in find_packages() if package.startswith("tensoraerospace")],
        python_requires=">=3.7",
        author_email="mr8bit@yandex.ru",
        description="TensorAeroSpace! - RL for Aerospace.",
        url="https://github.com/tensoraerospace/tensoraerospace",
        keywords="reinforcement-learning machine-learning gym openai aerospace toolbox python data-science",
      )
