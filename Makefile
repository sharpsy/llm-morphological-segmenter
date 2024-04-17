SHELL := bash
.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c
.DELETE_ON_ERROR:
MAKEFLAGS += --warn-undefined-variables
MAKEFLAGS += --no-builtin-rules
.RECIPEPREFIX = >

include config.mk


DOCKER := docker run
DOCKER_IMG := $(shell basename $(dir $(realpath $(lastword $(MAKEFILE_LIST)))))
DOCKER_ARGS := --rm \
               --env TRANSFORMERS_CACHE=/out/.cache \
               --env PYTHONUNBUFFERED=1 \
               --env PYTHONDONTWRITEBYTECODE=1 \
               --env SENTENCE_TRANSFORMERS_HOME='/out/.cache/sent-trans' \
               --env HF_DATASETS_CACHE="/out/.cache/datasets" \
               -v $$(realpath data):/data \
               -v $$(realpath src):/app \
               -v $$(realpath out):/out

ifeq ($(DEVICE), cuda)
DOCKER_ARGS += --gpus all
endif


METRIC := bpr

# Default - top level rule is what gets ran when you run just `make`
all: out/.build.sentinel out/.eval.sentinel out/.run.sentinel
.PHONY: all

build: out/.build.sentinel
.PHONY: build

run: out/.run.sentinel
.PHONY: run

prepare-test: out/.test.sentinel
.PHONY: prepare-test

prepare-gold: out/.gold.sentinel
.PHONY: prepare-gold

eval: out/.eval.sentinel
.PHONY: eval

clean:
> rm -rf out
.PHONY: clean


out/.split.sentinel: goldstd_trainset.segmentation.eng goldstd_trainset.segmentation.tur goldstd_trainset.segmentation.fin 
> mkdir -p out/llm-segm
> $(DOCKER) $(DOCKER_ARGS) $(DOCKER_IMG) python /app/split_data.py
> touch $@


out/.test.sentinel: out/.split.sentinel
> cat data/eng_test.segmentation.csv | sed -E 's/([^,]+),.+/\1/' > out/eng.test
> cat data/fin_test.segmentation.csv | sed -E 's/([^,]+),.+/\1/' > out/fin.test
> cat data/tur_test.segmentation.csv | sed -E 's/([^,]+),.+/\1/' > out/tur.test
> cat data/swati.clean.test.conll | sed -E 's/ \| .+//' > out/swati.test
> cat data/zulu.clean.test.conll | sed -E 's/ \| .+//' > out/zulu.test
> cat data/xhosa.clean.test.conll | sed -E 's/ \| .+//' > out/xhosa.test
# all of the above + handle ',' (replace with !)
> cat data/ndebele.clean.test.conll | sed -E 's/ \| .+//' | sed ' y/-,/ !/' > out/ndebele.test
> touch $@


out/.gold.sentinel: out/.split.sentinel
> cat data/eng_test.segmentation.csv | sed 'y/@/ /' | sed 's/,/\t/' | sed 's/,/, /g' > out/eng.gold
> cat data/fin_test.segmentation.csv | sed 'y/@/ /' | sed 's/,/\t/' | sed 's/,/, /g' > out/fin.gold
> cat data/tur_test.segmentation.csv | sed 'y/@/ /' | sed 's/,/\t/' | sed 's/,/, /g' > out/tur.gold
# Replace '-' with ' ' (space) to conform to the format of evaluation
> cat data/swati.clean.test.conll | cut -sd\| -f1-2 | sed 's/ | /\t/' | sed ' y/-/ /' > out/swati.gold
> cat data/zulu.clean.test.conll | cut -sd\| -f1-2 | sed 's/ | /\t/' | sed ' y/-/ /' > out/zulu.gold
> cat data/xhosa.clean.test.conll | cut -sd\| -f1-2 | sed 's/ | /\t/' | sed ' y/-/ /' > out/xhosa.gold
# Additionally, replace ',' as evaluation cannot work with this character (we replaced it with '!')
> cat data/ndebele.clean.test.conll | cut -sd\| -f1-2 | sed 's/ | /\t/' | sed ' y/-,/ !/' > out/ndebele.gold
> touch $@


out/.build.sentinel: Dockerfile
> DOCKER_BUILDKIT=1 docker build . --tag=$(DOCKER_IMG)
> DOCKER_BUILDKIT=1 docker build -f Dockerfile.eval --tag=$(DOCKER_IMG)-eval .
> mkdir -p out
> touch $@


out/.run.sentinel: out/.build.sentinel
> mkdir -p out/llm-segm
> $(DOCKER) $(DOCKER_ARGS) $(DOCKER_IMG) python /app/main.py
> touch $@


out/.eval.sentinel: out/.run.sentinel
> CMD="$(DOCKER) $(DOCKER_ARGS) $(DOCKER_IMG)-eval"
> $${CMD} morphoeval --metric ${METRIC} /out/eng.gold /out/llm-segm/eng.pred /out/llm-segm/eng-result.txt
> $${CMD} morphoeval --metric ${METRIC} /out/fin.gold /out/llm-segm/fin.pred /out/llm-segm/fin-result.txt
> $${CMD} morphoeval --metric ${METRIC} /out/tur.gold /out/llm-segm/tur.pred /out/llm-segm/tur-result.txt
> $${CMD} morphoeval --metric ${METRIC} /out/zulu.gold /out/llm-segm/zulu.pred /out/llm-segm/zulu-result.txt
> $${CMD} morphoeval --metric ${METRIC} /out/swati.gold /out/llm-segm/swati.pred /out/llm-segm/swati-result.txt
> $${CMD} morphoeval --metric ${METRIC} /out/xhosa.gold /out/llm-segm/xhosa.pred /out/llm-segm/xhosa-result.txt
> $${CMD} morphoeval --metric ${METRIC} /out/ndebele.gold /out/llm-segm/ndebele.pred /out/llm-segm/ndebele-result.txt
> $${CMD} python /app/calc_accuracy.py /out/eng.gold /out/llm-segm/eng.pred /out/llm-segm/eng-result-acc.txt
> $${CMD} python /app/calc_accuracy.py /out/fin.gold /out/llm-segm/fin.pred /out/llm-segm/fin-result-acc.txt
> $${CMD} python /app/calc_accuracy.py /out/tur.gold /out/llm-segm/tur.pred /out/llm-segm/tur-result-acc.txt
> $${CMD} python /app/calc_accuracy.py /out/zulu.gold /out/llm-segm/zulu.pred /out/llm-segm/zulu-result-acc.txt
> $${CMD} python /app/calc_accuracy.py /out/xhosa.gold /out/llm-segm/xhosa.pred /out/llm-segm/xhosa-result-acc.txt
> $${CMD} python /app/calc_accuracy.py /out/swati.gold /out/llm-segm/swati.pred /out/llm-segm/swati-result-acc.txt
> $${CMD} python /app/calc_accuracy.py /out/ndebele.gold /out/llm-segm/ndebele.pred /out/llm-segm/ndebele-result-acc.txt
> touch $@
