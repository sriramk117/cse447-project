#!/usr/bin/env bash
set -e
set -v
python src/new_transformer_model.py test --work_dir work --test_data $1 --test_output $2 