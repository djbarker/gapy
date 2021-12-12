#!/usr/bin/env bash

export PYTHONPATH=../python/:$PYTHONPATH 

# python -m unittest discover ./ -v
pytest -v