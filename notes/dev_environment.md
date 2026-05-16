# Development Environments for Python / Deep Learning

You can do all of these exercises in nothing more than a text editor and a terminal —
but a real IDE saves time on debugging, refactoring, and managing virtual environments.
This note surveys the common choices.

## PyCharm (JetBrains)

The reference Python IDE. Two editions:

- **Community Edition** — free, open-source, sufficient for everything in this repo.
- **Professional Edition** — paid, adds web frameworks, database tools, scientific
  notebooks, and remote interpreters.

What you get:
- World-class refactoring (rename across files, extract method, etc.).
- A real visual debugger with breakpoints, watch expressions, conditional stops.
- Built-in venv management (creates one per project, switches automatically).
- A graphical Git client.
- Plotting that opens in a panel inside the IDE.

Setup for ML work:
1. `File → New Project`. Pick a folder. Under "Interpreter" choose "New environment" →
   "Virtualenv". PyCharm will create `.venv/` for you.
2. Inside the IDE, open the bottom-bar Terminal — it's pre-activated to your venv.
3. `File → Settings → Project → Python Interpreter`, click `+`, install `numpy`,
   `torch`, `torchvision`, `tensorflow`, `keras`, `jax`, `flax`, `transformers`,
  `datasets`, `numpy`, `pandas`, `matplotlib`.
4. Run any script with `Shift+F10` (run last config) or right-click → Run.

## Visual Studio Code

The most popular IDE-grade editor today. Free, highly extensible.

What you need to install:
- VS Code itself: [code.visualstudio.com](https://code.visualstudio.com/).
- The **Python extension** by Microsoft (auto-detects venvs, debugger, IntelliSense).
- The **Pylance** extension (faster type checking and autocompletion — usually
  installed automatically alongside Python).
- Optional but useful: **Jupyter** extension (run notebooks natively), **Ruff** or
  **Pylint** extensions (lint-on-save), **GitLens** (richer Git history).

Setup:
1. Open the project folder (`File → Open Folder`).
2. `Ctrl+Shift+P` → `Python: Create Environment` → `.venv` → pick your interpreter.
3. Open the integrated terminal (`Ctrl+\``); it's auto-activated.
4. `pip install -r requirements.txt` (or install per-package as in the venv note).
5. Open any `.py` file. Hit the green "Run" arrow at the top-right to execute.

## Jupyter / JupyterLab

Notebook-style execution: cells of Python whose outputs (text, plots, tables) appear
inline. The de facto standard for exploratory data science.

Two ways to run it:
- **Standalone**: `pip install jupyterlab` then `jupyter lab` opens a browser-based UI.
- **Inside an IDE**: VS Code (via the Jupyter extension) and PyCharm Professional both
  have first-class notebook UIs. PyCharm Community can open `.ipynb` files but with
  reduced interactivity.

Notebooks are great for *exploration*; they are bad for *long-lived code* (cells run in
non-obvious order, version control of the JSON file is painful). Use them to discover
what your code should do, then extract the working code into `.py` files for the real
project.

## On macOS specifically

The Mac comes with a few editors out of the box:
- **TextEdit** — useless for code.
- **vim** and **nano** — fine for quick edits in a terminal but no Python integration.
- **Xcode** — Apple's IDE, supports many languages, but Python support is third-party
  and limited. Don't use it for Python.

For real Python work on macOS install one of the cross-platform options above:
- **PyCharm CE** (`brew install --cask pycharm-ce`) — same experience as on Linux/Win.
- **VS Code** (`brew install --cask visual-studio-code`) — same experience as on
  Linux/Win.

The other macOS-specific option worth knowing about:
- **Cursor** — a VS Code fork with built-in AI assistance. Same Python extensions as
  VS Code but with deeper AI features. Free tier available.

## Lightweight terminal-only setup

If you don't want a full IDE:
- A modern terminal (Apple Terminal / iTerm2 / Windows Terminal / GNOME Terminal).
- An editor with good Python support: **vim** with `python-mode` and `coc.nvim`,
  **neovim** with `nvim-lspconfig` and `pyright`, or **emacs** with `lsp-mode`.
- `ipython` for an interactive REPL with autocomplete and history (`pip install
  ipython`).
- `pdb` or `ipdb` for command-line debugging.

This setup is fast and lightweight but doesn't give you the visual debugger or the
notebook integration. Fine for small scripts, tedious for serious data exploration.

## Recommendation

- **Just learning ML?** PyCharm Community on any OS. Best out-of-the-box experience.
- **Already comfortable in VS Code?** Stick with it — the Python + Jupyter extensions
  are excellent.
- **Heavy notebook work / experimenting?** JupyterLab in the browser, with code
  factored out to `.py` files once it stabilizes.
- **Already a vim/emacs power user?** Stay there with an LSP-based Python plugin.
