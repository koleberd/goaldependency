run_basic: clean
	
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
mv: clean
	@python src/controller.py


learn:
	@python src/tensorflow/blockDetector3.py



doc:
	@asciidoctor README.adoc -b html5
