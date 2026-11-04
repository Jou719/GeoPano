# Publishing checklist

Publish this directory, not the original experiment tree. Before announcing a
publicly reproducible release:

1. Host the checkpoint separately, specify its license, and add the actual URL
   and SHA-256 checksum to `checkpoints/README.md` and the root README.
2. Add verified paper title, authors, links, and BibTeX.
3. Review retained upstream licenses. Non-commercial restrictions apply.
4. Review the source and redistribution terms of the four author-provided real samples.
5. Reproduce setup in a fresh environment. GPU inference passed in the research environment.
   Current validation and its limits are recorded in `validation.md`.

After creating an empty GitHub repository, run:

```bash
git init
git add .
git status --short
git diff --cached --stat
git commit -m "Release GeoPano inference code"
git branch -M main
# Replace OWNER and REPOSITORY with your actual destination.
git remote add origin https://github.com/OWNER/REPOSITORY.git
git push -u origin main
```

Inspect the staged list before committing. Weights, point clouds, generated
caches, depth arrays, and inference results are ignored. Do not force-add them.
The included real panorama examples were supplied by the maintainer.
