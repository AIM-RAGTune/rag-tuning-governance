# History Sanitization Not Performed

Mode used: current-tree sanitization only.

The GitHub repository is private. This change removes CRAG raw query text from the current repository tree and from the files visible in the latest `main` branch, but it does not rewrite prior Git history.

Earlier private commits may still contain pre-sanitization artifact versions. Before any public release, use a fresh clean repository from the sanitized current tree or perform an explicit history rewrite after owner approval.

No destructive history rewrite was run in this task.
