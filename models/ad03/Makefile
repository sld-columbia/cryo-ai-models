help:
	@echo "INFO: make <TAB> for targets"
.PHONY: help

run-console:
	hls4ml convert -c hls4ml_config.yml
.PHONY: run-console

run-python:
	python py/hls4ml_convert.py
.PHONY: run-python

clean:
	rm -rf anomaly_detector_prj
	rm -f anomaly_detector_prj.tar.gz
.PHONY: clean
