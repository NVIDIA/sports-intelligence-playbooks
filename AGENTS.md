# Agents

## Updating a Single Dependency

To update a single Python dependency:

1. Add or update the pinned version using the repo uv wrapper:
   - Linux: `./repo.sh uv -- add pydata-sphinx-theme==0.17.0`
   - Windows: `repo.bat uv -- add pydata-sphinx-theme==0.17.0`

2. Remove the new entry that uv adds to `[project.dependencies]` in `pyproject.toml`. The dependency is transitive (pulled in by `nvidia-sphinx-theme`), so it should not be listed as a direct dependency.

3. Bump the version in `pyproject.toml`, `VERSION`, and add a changelog entry in `CHANGELOG.md`.

4. Regenerate the lockfile so it picks up the new version:
   - Linux: `./repo.sh uv -- lock`
   - Windows: `repo.bat uv -- lock`
