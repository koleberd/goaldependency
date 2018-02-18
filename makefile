run_basic: clean
	@rm -r -f simulation/2Dpath/*
	@rm -r -f simulation/trees/*
	@python src/main.py

t:
	@python src/test.py

clean:
	@rm -r -f src/__pycache__
	@rm -f trees/*.gv
	@rm -f trees/*.png

init:
	@mkdir simulation
	@mkdir simulation/2dpath
	@mkdir models
	@pip install imageio

	@mkdir json/simulation_stats

doc:
	@asciidoctor README.adoc -b html5

analysis:
	@python src/util/simAnalysis.py

loc:
	@find . -name '*.py' | xargs wc -l
