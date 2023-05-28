from setuptools import setup


with open('./requirements.txt') as f:
    required = f.read().splitlines()

setup(name='tensoraerospace',
        version='0.2.1',
        install_requires=required,
        packages=['tensoraerospace'],
        python_requires=">=3.7",
        author_email="mr8bit@yandex.ru",
        description="TensorAeroSpace! - RL for Aerospace.",
        url="https://github.com/tensoraerospace/tensoraerospace",
        keywords="reinforcement-learning machine-learning gym openai aerospace toolbox python data-science",
      )
