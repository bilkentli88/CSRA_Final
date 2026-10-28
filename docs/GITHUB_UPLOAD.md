# Publishing this prepared repository

The ZIP contains a complete repository directory matched to the V10 Neurocomputing manuscript. No remote GitHub content has been changed by preparing it.

1. Extract the ZIP. Work inside the `CSRA_GitHub_R10` directory and run the README installation commands followed by `python reproduce.py`. Confirm `reproduced_results/paper/validation.json` says PASSED.
2. If updating an existing repository, preserve the old submitted version as an existing commit/tag before replacing its current tree. Use a new branch. Copy the complete new tree, including `.gitattributes`, `.gitignore` and `.github`, while retaining your existing `.git` directory. Remove obsolete tracked files only on that new branch after comparing the trees. Do not overwrite an old release/tag.
3. If creating a new GitHub repository, create it empty (without an automatically generated README/license), then run the following inside the extracted directory:

```bash
git init -b main
git add .
git commit -m "Add reproducible CSRA manuscript package"
git remote add origin YOUR_ACTUAL_GITHUB_REPOSITORY_URL
git push -u origin main
```

Replace the URL token with the actual destination. Authentication is handled by your Git/GitHub setup. The package includes no token or credentials. All supplied files are below GitHub's normal individual-file size limit; no model weights are included. Git is preferable to manually uploading hundreds of files in separate browser batches.

The workflow `.github/workflows/reproduce.yml` runs the CPU paper command on pushes and pull requests. It is supplied for automation; it has not run on GitHub until you publish the repository. Check its first result.

After reviewing that commit, create a versioned release, record its actual commit SHA and insert the real immutable URL into the manuscript's Data and code availability statement. Update CITATION.cff if you add that real URL or a genuine archive DOI. Version 0.11.4 in this prepared tree is metadata, not proof that a remote tag or DOI exists.

Keep `reproduced_results/`, environments and credentials out of Git. The supplied `.gitignore` does this. Keep `.gitattributes`: it prevents line-ending conversion from invalidating historical checksums on Windows. Do not run automatic formatters over frozen source/data directories.

This is an identified research archive. A separate anonymous-review copy, if needed for a venue, must address identifying metadata/links and document any redactions. Preserve the identified scientific archive and its authentic locks.
