run: clean
	@rm -r -f simulation/2Dpath/*
	@rm -r -f simulation/trees/*
	@python src/main.py

test:
	@python src/test.py

clean:
	@rm -r -f src/__pycache__
	@rm -f simulation/trees/*.gv
	@rm -f simulation/trees/*.png
	@rm -f simulation/2Dpath/*.png

init:
	@mkdir -p simulation/2dpath
	@mkdir simulation/trees
	@mkdir trainedModels
	@mkdir -p json/simulation_configs
	@mkdir json/world_configs
	@mkdir json/benchmark_sets
	@mkdir json/simulation_stats
	@mkdir -p resources/2d
	@pip install imageio numpy graphviz PIL

doc:
	@asciidoctor README.adoc -b html5

loc:
	@find . -name '*.py' | xargs wc -l
