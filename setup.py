from setuptools import setup
from pip.req import parse_requirements

install_reqs = parse_requirements('./requirements.txt')
reqs = [str(ir.req) for ir in install_reqs]

setup(name='tensorairspace',
        version='0.1.1',
        install_requires=reqs,
        python_requires=">=3.7",
        description="TensorAirSpace! - RL for Aerospace.",
        url="https://github.com/TensorAirSpace/TensorAirSpace",
        keywords="reinforcement-learning machine-learning gym openai aerospace toolbox python data-science",
      )
