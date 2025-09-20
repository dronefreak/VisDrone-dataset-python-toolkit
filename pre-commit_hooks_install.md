# Pre-commit hooks installation

Some pre-commit hooks can be installed using the following steps.
Currently, only a flake8 hook is provided.

Install pre-commit (local install):

`curl https://pre-commit.com/install-local.py | python -`

Add the line

`export PATH="$PATH:$HOME/bin"`

to the file `~/.bashrc`.
Inside the git repository, run:

`pre-commit install`
