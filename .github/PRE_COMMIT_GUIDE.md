# Pre-commit Setup Guide

Complete guide for setting up and using pre-commit hooks with ruff, black, isort, and more.

## 🚀 Quick Setup

```bash
# 1. Install pre-commit
pip install pre-commit

# 2. Install the git hooks
pre-commit install

# 3. (Optional) Install commit-msg hook for conventional commits
pre-commit install --hook-type commit-msg

# Done! Hooks will now run automatically on git commit
```

## 📋 What's Included

Our pre-commit configuration includes:

### Code Formatting

- **Ruff** - Fast Python linter and formatter
- **Black** - The uncompromising Python code formatter
- **isort** - Sort Python imports
- **Prettier** - Format YAML, JSON, Markdown

### Code Quality

- **Ruff linter** - Comprehensive linting (replaces flake8, pylint)
- **Pyupgrade** - Automatically upgrade syntax for Python 3.8+
- **MyPy** - Static type checking
- **Pydocstyle** - Docstring style checking

### Security

- **Bandit** - Security issue detection
- **detect-private-key** - Prevent committing private keys

### General

- **trailing-whitespace** - Remove trailing whitespace
- **end-of-file-fixer** - Ensure files end with newline
- **check-yaml** - Validate YAML files
- **check-merge-conflict** - Detect merge conflict markers
- **markdownlint** - Markdown linting and fixing
- **shellcheck** - Shell script linting

## 🔧 Configuration Files

### `.pre-commit-config.yaml`

Main pre-commit configuration with all hooks.

### `pyproject.toml`

Tool-specific configurations:

- `[tool.ruff]` - Ruff linter settings
- `[tool.black]` - Black formatter settings
- `[tool.isort]` - Import sorting settings
- `[tool.mypy]` - Type checking settings
- `[tool.bandit]` - Security scanning settings
- `[tool.pydocstyle]` - Docstring style settings

## 📖 Usage

### Automatic (Recommended)

Once installed, hooks run automatically before each commit:

```bash
git add myfile.py
git commit -m "feat: add new feature"

# Pre-commit hooks run automatically:
# ✓ Check yaml..........................................Passed
# ✓ Fix trailing whitespace.............................Passed
# ✓ Ruff.................................................Passed
# ✓ Black................................................Passed
# ✓ isort................................................Passed
# ... (all hooks)

# If any hook fails, the commit is blocked
```

### Manual Execution

Run hooks manually on files:

```bash
# Run on all files
pre-commit run --all-files

# Run on staged files only
pre-commit run

# Run specific hook
pre-commit run black --all-files
pre-commit run ruff --all-files

# Run on specific files
pre-commit run --files visdrone_toolkit/dataset.py
```

### Update Hooks

Update hook versions:

```bash
# Update to latest versions
pre-commit autoupdate

# Update specific hook
pre-commit autoupdate --repo https://github.com/astral-sh/ruff-pre-commit
```

## 🎯 Common Workflows

### Before First Commit

Run on entire codebase to fix existing issues:

```bash
# Format everything
pre-commit run --all-files

# Review and commit the changes
git add -A
git commit -m "chore: apply pre-commit formatting"
```

### Daily Development

Just work normally - hooks run automatically:

```bash
# Make changes
vim visdrone_toolkit/dataset.py

# Commit (hooks run automatically)
git add visdrone_toolkit/dataset.py
git commit -m "feat: add new dataset loader"
```

### Skip Hooks (Emergency Only)

Sometimes you need to commit without running hooks:

```bash
# Skip all hooks (use sparingly!)
git commit --no-verify -m "wip: emergency commit"

# Or skip specific hooks with SKIP environment variable
SKIP=black,ruff git commit -m "temp: skip formatting"
```

### CI/CD Integration

Run in CI to catch issues:

```bash
# In your CI pipeline
pip install pre-commit
pre-commit run --all-files
```

## 🔨 Fixing Issues

### Auto-fix Available

Many hooks auto-fix issues:

```bash
# These hooks automatically fix problems:
# - trailing-whitespace
# - end-of-file-fixer
# - black
# - isort
# - ruff (with --fix)
# - prettier

# Just re-add and commit after auto-fix
git add .
git commit -m "feat: my feature"
```

### Manual Fix Required

Some hooks require manual fixes:

```bash
# MyPy type errors
# → Fix type hints in code

# Bandit security issues
# → Fix security vulnerabilities

# pydocstyle docstring errors
# → Add/fix docstrings
```

## ⚙️ Customization

### Disable Specific Hooks

Edit `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.12.1
    hooks:
      - id: black
        # Add this to disable:
        # exclude: ^(scripts/legacy/)
```

### Adjust Tool Settings

Edit `pyproject.toml`:

```toml
[tool.ruff]
line-length = 120  # Change from 100 to 120

[tool.black]
line-length = 120  # Must match ruff

[tool.isort]
line_length = 120  # Must match ruff
```

### Add Custom Hooks

Add to `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: local
    hooks:
      - id: custom-check
        name: Custom Check
        entry: python scripts/custom_check.py
        language: system
        pass_filenames: false
```

## 🚨 Troubleshooting

### Hook Installation Failed

```bash
# Clear cache and reinstall
pre-commit clean
pre-commit install --install-hooks
```

### Hook Times Out

```bash
# Increase timeout (default is 60 seconds)
# Add to .pre-commit-config.yaml under the hook:
hooks:
  - id: mypy
    args: [--timeout=300]
```

### Conflicts Between Tools

Our configuration avoids conflicts:

- Ruff and Black use same line length (100)
- isort uses black profile
- Ruff runs before Black
- All formatters use consistent settings

### Performance Issues

```bash
# Run hooks in parallel (faster)
# Add to .pre-commit-config.yaml:
default_stages: [commit]
fail_fast: false

# Or skip slow hooks during development
SKIP=mypy,bandit git commit -m "wip"
```

## 📊 Hook Execution Order

Hooks run in this order (by design):

1. **File checks** (trailing whitespace, end of file, YAML/JSON validation)
2. **Ruff** (linting and formatting)
3. **Black** (formatting - runs after ruff-format)
4. **isort** (import sorting - compatible with black)
5. **Prettier** (YAML, JSON, Markdown)
6. **Pyupgrade** (syntax upgrades)
7. **Pydocstyle** (docstring checks)
8. **MyPy** (type checking)
9. **Bandit** (security)
10. **Markdown/YAML linters**

## 🎓 Best Practices

### ✅ Do

- Install hooks immediately after cloning
- Run `pre-commit run --all-files` before first commit
- Update hooks regularly (`pre-commit autoupdate`)
- Fix issues immediately (don't skip hooks)
- Use conventional commit messages

### ❌ Don't

- Skip hooks regularly (only in emergencies)
- Commit `--no-verify` to bypass issues
- Ignore hook failures in CI
- Mix different line lengths across tools
- Commit without reviewing auto-fixes

## 📚 Tool Documentation

- **Ruff**: <https://docs.astral.sh/ruff/>
- **Black**: <https://black.readthedocs.io/>
- **isort**: <https://pycqa.github.io/isort/>
- **pre-commit**: <https://pre-commit.com/>
- **MyPy**: <https://mypy.readthedocs.io/>
- **Bandit**: <https://bandit.readthedocs.io/>

## 🔄 Migration from Old Tools

If migrating from older tools:

```bash
# Old: flake8
# New: ruff (includes flake8 rules)

# Old: pylint
# New: ruff (includes pylint rules)

# Old: autoflake, autopep8
# New: ruff --fix

# Ruff is much faster and replaces many tools!
```

## 💡 Tips

1. **Ruff is fast**: It can check/fix entire codebase in seconds
2. **Black is opinionated**: Don't fight it, embrace consistent formatting
3. **MyPy catches bugs**: Type hints help find issues before runtime
4. **Bandit finds security issues**: Critical for production code
5. **Pre-commit saves time**: Catch issues before CI/CD

## 🆘 Getting Help

```bash
# Show all hooks
pre-commit run --help

# Show specific hook info
pre-commit run black --help

# Verbose output for debugging
pre-commit run --all-files --verbose

# Try specific hook with verbose output
pre-commit run ruff --all-files --verbose
```

## 📝 Example Commit Flow

```bash
# 1. Make changes
vim visdrone_toolkit/dataset.py

# 2. Stage changes
git add visdrone_toolkit/dataset.py

# 3. Commit (hooks run automatically)
git commit -m "feat(dataset): add video support"

# Output:
# trim trailing whitespace.................................Passed
# fix end of files.........................................Passed
# check yaml...............................................Passed
# check toml...............................................Passed
# Ruff.....................................................Passed
# Ruff format..............................................Passed
# black....................................................Passed
# isort....................................................Passed
# mypy.....................................................Passed
# bandit...................................................Passed

# 4. Push
git push origin main
```

---

## 🎉 Ready to Use

Your code will now be automatically formatted and checked on every commit!

```bash
# Just install and forget
pre-commit install

# Everything else is automatic! ✨
```
