

docs:
	cd docs; make html


dist:
	python3 -m pip install --upgrade build
	python3 -m build

upload:
	python3 -m pip install --upgrade twine
	python3 -m twine upload --repository pypi dist/*
