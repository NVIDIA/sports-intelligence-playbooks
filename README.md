# Sports Intelligence Cookbooks — documentation

Source for **Sports Intelligence Cookbooks** documentation (Sphinx + **repo_docs** via `doc_build_toolchain/`).

**Published documentation:** [https://nvidia.github.io/sports-intelligence-cookbooks/](https://nvidia.github.io/sports-intelligence-cookbooks/) (GitHub authentication required)

The published site title and left-hand navigation are defined in **`docs_src/index.rst`**. The sidebar is grouped into:

- **Multimodal Language Models** — sources under **`docs_src/multimodal/`**: Data Curation, Training (Setup, SFT, LoRA), Inference, Evaluation, Learnings.

A hidden **Reference** toctree links the **Licenses** page (`docs_src/licenses.rst`).

The visible product name is set in **`repo.toml`** under **`[repo_docs.projects.sports_intelligence_stack]`** (**`name`**) and **`metadata_global`** **`project-name`** (theme metadata).

## Python environment (uv)

Install **uv** (see [astral.sh/uv](https://docs.astral.sh/uv/getting-started/)), then from the repo root:

```bash
uv sync
```

Optional: `uv sync --python 3.12`

Activate when needed:

```bash
source .venv/bin/activate          # bash/zsh
source .venv/bin/activate.fish    # fish
```

## Build HTML documentation

From the repository root (with the Omniverse **repo** suite bootstrapped via `./repo.sh`):

```bash
./repo.sh docs
```

Build output:

`docs/latest/`

The editable Sphinx sources are under **`docs_src/`**. The generated **`docs/`** directory contains only the GitHub Pages deployment payload: **`latest/`**, **`versions1.json`**, **`.nojekyll`**, and a root redirect to **`latest/`**. Temporary build data and warning logs remain under **`_build/docs/`**.

Use **`./repo.sh docs --clean`** for a clean rebuild of the generated project output.

Faster iteration after a full build (RST/Markdown only):

```bash
./repo.sh docs --stage sphinx
```

Version comes from **`VERSION`**. The homepage title in **`docs_src/index.rst`** should stay aligned with **`repo.toml`** **`name`** / **`metadata_global`** **`project-name`** (currently **Sports Intelligence Cookbooks**).

### Preview in a browser (recommended)

Opening **`latest/index.html`** directly (`file://`) often breaks theme features that load extra files (for example the **version switcher**, which reads **`../versions1.json`**). Serve the built docs over HTTP instead.

**Foreground** (from the repository root; stop with **Ctrl+C**):

```bash
python -m http.server 8876 --bind 127.0.0.1 --directory docs
```

**Background** (from the repository root):

```bash
nohup python -m http.server 8876 --bind 127.0.0.1 --directory docs > _build/docs/server.log 2>&1 &
```

If you are already in **`docs/`**, run the server directly (do not `cd` into that path again):

```bash
nohup python -m http.server 8876 --bind 127.0.0.1 > ../_build/docs/server.log 2>&1 &
```

Then open **[http://127.0.0.1:8876/](http://127.0.0.1:8876/)** in your browser.

Use another port if **8876** is taken. To stop a background server, find its PID with `pgrep -f 'http.server 8876'` and run `kill <PID>`.

## Layout

| Path | Purpose |
|------|---------|
| `docs_src/` | Editable Sphinx sources: `index.rst`, `multimodal/`, `licenses.rst`, … |
| `docs/` | Generated, GitHub Pages-ready HTML output |
| `doc_build_toolchain/repo/docs/` | Vendored **repo_docs** implementation |
| `repo.toml` | `[repo_docs]` configuration |
| `repo_tools.toml` | Tool defaults |

## License

See `LICENSE` and the **Licenses** page in the HTML output. Third-party notices also appear under `licenses/`.
