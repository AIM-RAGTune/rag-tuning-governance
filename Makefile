.PHONY: setup test reproduce-crag reproduce-multihop tables figures validate-publication clean

setup:
	pip install -r requirements.txt
	pip install -e .

test:
	pytest -q tests/publication

reproduce-crag:
	bash scripts/reproduce_crag_mock_api.sh

reproduce-multihop:
	bash scripts/reproduce_multihop_confirmatory.sh

reproduce-dataset-matrix:
	bash scripts/reproduce_dataset_matrix.sh

tables:
	python scripts/make_tables.py

figures:
	python scripts/make_figures.py

validate-publication:
	python scripts/validate_publication_bundle.py

package-for-approval:
	bash scripts/package_for_approval.sh

clean:
	rm -rf .pytest_cache __pycache__
