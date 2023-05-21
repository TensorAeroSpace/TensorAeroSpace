# TensorAirSpace 

[![Documentation Status](https://readthedocs.org/projects/tensoraerospace/badge/?version=latest)](https://tensoraerospace.readthedocs.io/en/latest/?badge=latest)
---


![](./img/logo-no-background.png)

**TensorAirSpace** - это набор объектов управления, сред моделирования OpenAI Gym и реализации алгоритмов Reinforcement Learning (RL)

## Запуск

Быстрая установка

```
git clone  https://github.com/TensorAirSpace/TensorAirSpace.git
cd TensorAirSpace
pip install -e .
```

Запуск Docker образа

```
docker build -t tensor_aero_space .
docker run -v example:/app/example -p 8888:8888 -it tensor_aero_space
```

## Примеры

Все примеры по запуску и работе с библиотекой TensorAirSpace находятся в папке `./example`

## Агенты

**TensorAirSpace** содержит такие RL алгоритмы как:

- IHDP (Incremental Heuristic Dynamic Programming)
- DQN (Deep Q Learning)
- SAC (Soft Actor Critic)

## Объекты управления

- General Dynamics F-16 Fighting Falcon
- Boeing-747
- ELV(Expendable launch vehicle)
- Модель ракеты
- БЛА в Unity среде


## Среды моделирования

### Unity Ml-Agents

![](./docs/example/env/img/img_demo_unity.gif)

**TensorAirSpace** умеет работать с системой Ml-Agents.

Пример среды для запуска можно найти в репозитории [UnityAirplaneEnvironment](https://github.com/TensorAirSpace/UnityAirplaneEnvironment)

В документации присутствуют примеры по настройке сети и работе с DQN агентом

### Matlab Simulink

**TensorAirSpace** содержит примеры по работе с Simulink моделями.

![](docs/example/simulink/img/model.png)

В документации приведены примеры по сборке и компиляции модели Simulink в работоспособный код который можно имплементировать в среду моделирования [OpenAI Gym](https://github.com/openai/gym)

### Матрицы пространств состояний

**TensorAirSpace** содержит объекты управления которые реализованы в виде матриц пространств состояний.
