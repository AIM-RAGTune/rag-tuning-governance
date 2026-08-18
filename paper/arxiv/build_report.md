# arXiv Source Build Report

- Abstract length: 1278/1920 characters.
- Abstract word count: 164.
- Abstract check: passed.
- Build command: `make -C paper pdf`.
- Clean temporary source-manifest build: passed_with_warnings.
- Page count: 20.
- Undefined references: 0.
- Missing citations: 0.
- Missing figures: 0.
- Raster figures: 0.
- TeX engine: Tectonic fallback; `latexmk`/`pdflatex` were not present on this Mac.
- Generated `paper/main.pdf`: yes during validation; not committed as a source artifact.
- Warning count: 11.

## Warning Summary

- `Package inputenc Warning: inputenc package ignored with utf8 based engines.`
- `Overfull \hbox (0.77803pt too wide) in paragraph at lines 29--32`
- `Overfull \hbox (0.21124pt too wide) in paragraph at lines 54--55`
- `Overfull \hbox (7.4497pt too wide) in paragraph at lines 97--98`
- `Overfull \hbox (4.99373pt too wide) in paragraph at lines 108--108`
- `Overfull \hbox (10.90942pt too wide) in paragraph at lines 118--119`
- `Overfull \hbox (3.13234pt too wide) in paragraph at lines 215--216`
- `Underfull \hbox (badness 1102) in paragraph at lines 26--32`
- `Underfull \hbox (badness 10000) in paragraph at lines 129--133`
- `Underfull \hbox (badness 10000) in paragraph at lines 129--133`
- `Underfull \hbox (badness 1424) in paragraph at lines 135--139`

## Hygiene

The arXiv source package includes LaTeX source, BibTeX, vector figure PDFs, source/data companions, and submission helper files. It does not include raw CRAG data, HotpotQA raw questions or contexts, prompts, generated answers, API responses, secrets, or private paths.
