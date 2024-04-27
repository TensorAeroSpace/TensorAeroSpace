#!/bin/sh
poetry run jupyter notebook --notebook-dir=/app --ip=0.0.0.0 --no-browser --allow-root --port=8888 --NotebookApp.token='' --NotebookApp.password=''