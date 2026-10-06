.PHONY: install data train live test lint serve demo docker
install: ; python -m pip install -r requirements-dev.txt
data:    ; python -m sentinel download
train:   ; python -m sentinel train
live:    ; python -m sentinel live
test:    ; python -m pytest -q
lint:    ; ruff check .
serve:   ; python -m sentinel serve
demo:    ; ./run.sh demo
docker:  ; docker build -t urlsentinel . && docker run --rm -p 8000:8000 -v "$$PWD/models:/app/models" urlsentinel
