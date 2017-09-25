run_basic:
	@rm -r -f src/__pycache__
	@rm -f trees/*.gv
	@rm -f trees/*.png
	@python src/main.py
t:
	@python test/PlayerStateTest.py
