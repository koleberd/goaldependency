run_basic: clean
	@python src/test.py
	@python src/main.py
t:
	@python src/test.py

clean:
	@rm -r -f src/__pycache__
	@rm -f trees/*.gv
	@rm -f trees/*.png
mv: clean
	@python src/controller.py

train:
	@python src/utils/trainingSetGenerator.py

learn:
	@python src/tensorflow/blockDetector2.py

convert:
	@python src/utils/trainingImageConverter.py
