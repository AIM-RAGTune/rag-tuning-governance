# History Rewrite Recommendation

Current-tree sanitization removes CRAG raw query text from the latest GitHub view. Prior private Git history may still contain earlier versions of the affected artifacts.

## Recommended Public-Release Route

Create a fresh clean repository from the sanitized current tree:

```bash
mkdir rag-tuning-governance-public-clean
rsync -a --exclude .git <sanitized-repository-bundle>/ rag-tuning-governance-public-clean/
cd rag-tuning-governance-public-clean
git init
git add .
git commit -m "Initial sanitized publication bundle"
git remote add origin <approved-public-or-internal-repository-url>
git push -u origin main
```

This gives public reviewers a single clean commit with no prior raw-text history.

## Alternative: Rewrite Current Private Repository

Use this only after explicit approval and after confirming collaborators will not be disrupted:

```bash
git branch backup/pre-crag-query-text-history-rewrite
# install git-filter-repo if needed
# rewrite or remove offending historical blobs
git push --force-with-lease origin main
```

Do not run this automatically. Set and document an explicit approval such as `RAGTUNE_ALLOW_HISTORY_REWRITE=true` before destructive history rewriting.
