# Data

This directory is part of the project structure, but large data files are not
tracked by Git.

Suggested local layout:

```text
data/
├── raw/         # Original vendor/source files
├── interim/     # Intermediate cleaning outputs
├── processed/   # Model-ready datasets
├── external/    # Auxiliary datasets
└── sample/      # Small files that may be committed for examples or tests
```

The current raw Chinese A-share quote files should be placed under:

```text
data/raw/chinese/
```

