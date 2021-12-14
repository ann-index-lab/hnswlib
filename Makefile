dataset: FORCE
	wget -O dataset/sift.tar.gz ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
	tar zxvf dataset/sift.tar.gz -C dataset/

FORCE:

pypi: dist
	twine upload dist/*

dist:
	-rm dist/*
	pip install build
	python3 -m build --sdist

test:
	python3 -m unittest discover --start-directory python_bindings/tests --pattern "*_test*.py"

clean:
	rm -rf *.egg-info build dist tmp var tests/__pycache__ hnswlib.cpython*.so

.PHONY: dist
