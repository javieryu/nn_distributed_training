# nn_distributed_training
Collective training of neural networks on distributed datasets.

## Setting up the venv

- Install ``pip`` and ``virtualenv`` if you don't have them already.

- Clone the repository and navigate to the folder in a terminal.

- Run ``virtualenv venv`` to make a new virtual environment for the project in the repo folder (the ``./venv/`` folder is git ignored so it won't push those files to the repo).

- Activate the virtual environment with: ``source venv/bin/activate``.

- Install the requirements with: ``pip install -r requirements.txt``.

- If you've installed new packages and want to add them to the ``requirements.txt`` just do: ``pip freeze > requirements.txt``.
