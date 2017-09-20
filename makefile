run_basic:
	@rm -r -f src/__pycache__
	@python src/main.py
t:
	@python test/PlayerStateTest.py
