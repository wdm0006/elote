Contributing to Elote
==================

Thank you for your interest in contributing to Elote! This guide will help you get started with contributing to the project.

Setting Up Your Development Environment
-------------------------------------

1. **Fork the Repository**

   Start by forking the Elote repository on GitHub.

2. **Clone Your Fork**

   .. code-block:: bash

       git clone https://github.com/your-username/elote.git
       cd elote

3. **Set Up Development Environment**

   .. code-block:: bash

       # Using Make (recommended)
       make install-dev

       # Or using pip
       pip install -e ".[dev]"

       # Or using uv
       uv pip install -e ".[dev]"

4. **Set Up Pre-commit Hooks**

   .. code-block:: bash

       pre-commit install

Development Workflow
------------------

1. **Create a Branch**

   Create a branch for your feature or bugfix:

   .. code-block:: bash

       git checkout -b feature/your-feature-name
       # or
       git checkout -b fix/your-bugfix-name

2. **Make Your Changes**

   Implement your changes, following the code style guidelines.

3. **Run Tests**

   Make sure your changes pass all tests:

   .. code-block:: bash

       # Run all tests
       make test

       # Run tests with coverage
       make test-cov

4. **Lint Your Code**

   Ensure your code follows the project's style guidelines:

   .. code-block:: bash

       # Check code style
       make lint

       # Auto-fix some linting issues
       make lint-fix

       # Format code
       make format

5. **Commit Your Changes**

   Follow the conventional commits format for your commit messages:

   .. code-block:: bash

       git commit -m "feat: add new feature"
       # or
       git commit -m "fix: resolve issue with X"

6. **Push Your Changes**

   Push your changes to your fork:

   .. code-block:: bash

       git push origin feature/your-feature-name

7. **Create a Pull Request**

   Open a pull request from your fork to the main Elote repository.

Code Style Guidelines
-------------------

Elote follows these code style guidelines:

- Use PEP 8 for Python code style
- Use docstrings for all public functions, classes, and methods
- Write clear, descriptive variable and function names
- Include type hints where appropriate
- Keep functions focused on a single responsibility
- Write unit tests for new functionality

Adding a New Rating System
------------------------

To add a new rating system to Elote:

1. Create a new file in the `elote/competitors` directory
2. Implement a new class that inherits from `BaseCompetitor`
3. Implement the required methods:
   - `expected_score(competitor)`
   - `update_rating(competitor, score)`
4. Add tests for your rating system in the `tests` directory
5. Update the documentation to include your new rating system

Here's a template for a new rating system:

.. code-block:: python

    from elote.competitors.base import BaseCompetitor

    class NewRatingCompetitor(BaseCompetitor):
        def __init__(self, initial_rating=1500, **kwargs):
            self.rating = initial_rating
            # Initialize other parameters

        def expected_score(self, competitor):
            """
            Calculate the expected score (probability of winning) against another competitor.
            
            Args:
                competitor: The opponent NewRatingCompetitor
                
            Returns:
                float: The probability of winning (between 0 and 1)
            """
            # Implement the expected score calculation
            pass

        def update_rating(self, competitor, score):
            """
            Update the rating based on a match result.
            
            Args:
                competitor: The opponent NewRatingCompetitor
                score: The actual score (1 for win, 0.5 for draw, 0 for loss)
            """
            # Implement the rating update logic
            pass

Documentation
------------

When adding new features or making significant changes, please update the documentation:

1. Add or update docstrings for all public functions, classes, and methods
2. Update the relevant RST files in the `docs/source` directory
3. If adding a new rating system, create a new RST file in `docs/source/rating_systems`
4. Build and check the documentation locally:

   .. code-block:: bash

       make docs
       # Open docs/build/html/index.html in your browser

Testing
------

Elote uses pytest for testing. When adding new features:

1. Write unit tests for your new code
2. Ensure all existing tests pass
3. Aim for high test coverage

.. code-block:: bash

    # Run tests
    make test

    # Run tests with coverage
    make test-cov

Reporting Issues
--------------

If you find a bug or have a feature request:

1. Check if the issue already exists in the GitHub issues
2. If not, create a new issue with:
   - A clear title and description
   - Steps to reproduce (for bugs)
   - Expected and actual behavior (for bugs)
   - Any relevant code snippets or error messages

Pull Request Process
------------------

1. Ensure your code passes all tests and linting checks
2. Update the documentation if needed
3. Add an entry to the CHANGELOG.md file
4. Submit your pull request with a clear description of the changes
5. Wait for review and address any feedback

Code of Conduct
-------------

Please note that Elote has a Code of Conduct. By participating in this project, you agree to abide by its terms.

Thank You!
---------

Your contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contribution you make is greatly appreciated! 