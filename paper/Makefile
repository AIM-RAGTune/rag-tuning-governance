PDF=main.pdf
TEX=main.tex

.PHONY: pdf abstract-check clean

pdf: abstract-check
	@if command -v latexmk >/dev/null 2>&1; then \
		latexmk -pdf -interaction=nonstopmode -halt-on-error $(TEX); \
	elif command -v tectonic >/dev/null 2>&1; then \
		tectonic --keep-logs $(TEX); \
	else \
		echo "No TeX engine found. Install latexmk/pdflatex or tectonic."; \
		exit 1; \
	fi

abstract-check:
	python3 ../scripts/check_arxiv_abstract.py

clean:
	latexmk -C $(TEX) || true
	rm -f main.bbl main.blg main.out main.fls main.fdb_latexmk
