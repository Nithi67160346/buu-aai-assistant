# Packaging notes

Prepared on 2026-10-30 from the existing local export. SHA256 values for the four
runtime Python modules, both fact-card JSON files, and the HTML template were
compared with the Raspberry Pi through its existing Connect shell and matched.
The checksums are recorded in `snapshot-sha256.json`.

The runtime modules, UI, fact-card content, source documents, collection/update
utilities and evaluation questions were copied without changing application logic.
The corrected dependency list includes Flask and NumPy, which the old list omitted.

Excluded from this repository folder:

- Old source versions, backups, patches and one-off data migration scripts.
- Runtime logs, user-session dumps, screenshots and benchmark result files.
- Python environments, bytecode and OCR caches.
- Generated SQLite database; recreate it using the README instructions.
- Model weights, training dataset and the obsolete 0.5B Modelfile.
- Nested ZIP archives, including the archive formerly inside `static/`.

The template has inline CSS/JavaScript and no dependency on files in `static/`.
An empty static directory is therefore not required.

This is a local publication-ready folder, not a published GitHub repository.
The owner can create an empty repository and run these commands from the folder:

```bash
git init -b main
git add .
git commit -m "Prepare AAI assistant source snapshot"
git remote add origin <YOUR_GITHUB_REPOSITORY_URL>
git push -u origin main
```
