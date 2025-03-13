Installation
============

This guide covers different ways to install Elote for both users and developers.

Requirements
-----------

Elote requires:

- Python 3.8 or higher
- NumPy (automatically installed as a dependency)

Basic Installation
----------------

For most users, the simplest way to install Elote is via pip:

.. code-block:: bash

    pip install elote

This will install the latest stable release from PyPI along with all required dependencies.

If you prefer using Conda, you can install Elote via pip within your Conda environment:

.. code-block:: bash

    conda create -n elote-env python=3.9
    conda activate elote-env
    pip install elote

Development Installation
----------------------

If you want to contribute to Elote or need the latest development version, you can install directly from the GitHub repository:

.. code-block:: bash

    # Using Make (recommended)
    git clone https://github.com/yourusername/elote.git
    cd elote
    make install-dev

    # Or using pip
    git clone https://github.com/yourusername/elote.git
    cd elote
    pip install -e ".[dev]"

    # Or using uv
    git clone https://github.com/yourusername/elote.git
    cd elote
    uv pip install -e ".[dev]"

The development installation includes additional dependencies needed for testing, linting, and documentation.

Verifying Installation
--------------------

To verify that Elote is installed correctly, you can run a simple test in Python:

.. code-block:: python

    from elote import EloCompetitor
    
    # Create two competitors
    player1 = EloCompetitor(initial_rating=1500)
    player2 = EloCompetitor(initial_rating=1600)
    
    # Calculate expected score
    expected = player2.expected_score(player1)
    print(f"Installation successful! Expected score: {expected:.2%}")

If this runs without errors, Elote is installed correctly.

Installing Optional Dependencies
------------------------------

Elote has several optional dependency groups that can be installed based on your needs:

.. code-block:: bash

    # Install with visualization dependencies
    pip install "elote[viz]"
    
    # Install with all optional dependencies
    pip install "elote[all]"
    
    # Install development dependencies
    pip install "elote[dev]"

Troubleshooting
--------------

Common installation issues and their solutions:

NumPy Installation Errors
^^^^^^^^^^^^^^^^^^^^^^^^

If you encounter errors related to NumPy installation:

.. code-block:: bash

    # Install NumPy separately first
    pip install numpy
    pip install elote

Version Conflicts
^^^^^^^^^^^^^^^

If you have version conflicts with other packages:

.. code-block:: bash

    # Create a virtual environment
    python -m venv elote-env
    source elote-env/bin/activate  # On Windows: elote-env\Scripts\activate
    pip install elote

Permission Errors
^^^^^^^^^^^^^^^

If you encounter permission errors during installation:

.. code-block:: bash

    # Install for the current user only
    pip install --user elote

    # Or use a virtual environment (recommended)
    python -m venv elote-env
    source elote-env/bin/activate
    pip install elote

Getting Help
-----------

If you continue to experience installation issues:

1. Check the `GitHub Issues <https://github.com/yourusername/elote/issues>`_ to see if others have encountered the same problem
2. Open a new issue with details about your environment and the error messages
3. Reach out to the community for help 