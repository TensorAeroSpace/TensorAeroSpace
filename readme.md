# TensorAeroSpace 

[![Documentation Status](https://readthedocs.org/projects/tensoraerospace/badge/?version=latest)](https://tensoraerospace.readthedocs.io/en/latest/?badge=latest)
---


![](./img/logo-no-background.png)

**TensorAeroSpace** - это набор объектов управления, сред моделирования OpenAI Gym и реализации алгоритмов Reinforcement Learning (RL)

## Запуск

Быстрая установка

```
git clone  https://github.com/tensoraerospace/tensoraerospace.git
cd tensoraerospace
pip install -e .
```

Запуск Docker образа

```
docker build -t tensor_aero_space .
docker run -v example:/app/example -p 8888:8888 -it tensor_aero_space
```

## Примеры

Все примеры по запуску и работе с библиотекой TensorAeroSpace находятся в папке `./example`

## Агенты

**TensorAeroSpace** содержит такие RL алгоритмы как:

- IHDP (Incremental Heuristic Dynamic Programming)
- DQN (Deep Q Learning)
- SAC (Soft Actor Critic)
- A3C (Asynchronous Advantage Actor-Critic)

## Объекты управления

- General Dynamics F-16 Fighting Falcon
- Boeing-747
- ELV (Expendable launch vehicle)
- Модель ракеты
- McDonnell Douglas F-4C
- North American X-15
- Геостационарный спутник
- Спутник связи
- БЛА в State Space
- БЛА в Unity среде


## Среды моделирования

### Unity Ml-Agents

![](./docs/example/env/img/img_demo_unity.gif)

**tensoraerospace** умеет работать с системой Ml-Agents.

Пример среды для запуска можно найти в репозитории [UnityAirplaneEnvironment](https://github.com/TensorAeroSpace/UnityAirplaneEnvironment)

В документации присутствуют примеры по настройке сети и работе с DQN агентом

### Matlab Simulink

**tensoraerospace** содержит примеры по работе с Simulink моделями.

![](docs/example/simulink/img/model.png)

В документации приведены примеры по сборке и компиляции модели Simulink в работоспособный код который можно имплементировать в среду моделирования [OpenAI Gym](https://github.com/openai/gym)

### Матрицы пространств состояний

**tensoraerospace** содержит объекты управления которые реализованы в виде матриц пространств состояний.
