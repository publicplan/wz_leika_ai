#!/bin/bash

echo "PULL data/nlp"
dvc pull -r readonly-upstream -R data/nlp
echo "PULL data/wz"
dvc pull -r readonly-upstream -R data/wz
echo "PULL weights/wz"
dvc pull -r readonly-upstream -R weights/wz

echo "Install"
pip install -e .

echo "Start wz-api"
wz-api --host 0.0.0.0 --port 80 --log-dir /api_log

