Установка
=========

Для установки склонируйте репозиторий 

.. code:: shell

    git clone https://github.com/TensorAirSpace/TensorAirSpace.git
    cd TensorAirSpace
    
Установите необходимые пакеты через ваш пакетный менеджер 

.. code:: shell

    pip install -r requirements.txt


Или же запустите проект через Docker образ

.. note::

    Мы рекомендуем работать через Docker так такой вариант минимизирует ошибки при запуске на разных платформах (x86/arm) и операционных системах (Linux/Windows/MacOS)

.. code:: shell

    docker build -t tensor_aero_space .
    docker run  -v ./tensorairspace:/app/tensorairspace -v ./example:/app/example -v ./docs:/app/docs -p 8888:8888 -it tensor_aero_space


.. note::

    Мы предлагаем монтировать папки что бы ваш результат сохранялся в исходном месте. 
    
