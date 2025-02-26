Changelog
=========

All notable changes to Elote will be documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_,
and this project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

[Unreleased]
-----------

Added
^^^^^

- Comprehensive Sphinx documentation
- Detailed pages for each rating system
- Real-world use case examples
- Improved README with badges and better examples

Changed
^^^^^^^

- Updated project structure to use modern Python packaging with pyproject.toml
- Improved code formatting and style consistency
- Enhanced type hints throughout the codebase

Fixed
^^^^^

- Various minor bugs and edge cases
- Documentation typos and inconsistencies

[1.0.0] - 2023-07-28
-------------------

Added
^^^^^

- Modern Python packaging with pyproject.toml
- Support for Python 3.8+
- Comprehensive test suite with pytest
- Makefile for common development tasks
- Pre-commit hooks for code quality

Changed
^^^^^^^

- Refactored code structure for better organization
- Updated dependencies to latest versions
- Improved error handling and edge cases

Removed
^^^^^^^

- Support for Python 3.7 and below
- Legacy setup.py-based packaging

[0.5.0] - 2022-03-21
-------------------

Added
^^^^^

- Ensemble rating system for combining multiple rating methods
- Basic Sphinx documentation structure
- Examples directory with sample use cases

Changed
^^^^^^^

- Improved performance of rating calculations
- Better handling of edge cases in all rating systems

Fixed
^^^^^

- Bug in Glicko rating deviation updates
- Edge case in ECF rating system for extreme rating differences

[0.4.0] - 2021-09-15
-------------------

Added
^^^^^

- DWZ (Deutsche Wertungszahl) rating system
- Support for ties/draws in all rating systems
- More comprehensive examples

Changed
^^^^^^^

- Refactored base competitor class for better extensibility
- Improved documentation and docstrings

[0.3.0] - 2021-02-10
-------------------

Added
^^^^^

- ECF (English Chess Federation) rating system
- Custom arena implementation for more flexibility
- Additional utility functions

Fixed
^^^^^

- Bug in Glicko implementation for inactive players
- Performance issues with large tournaments

[0.2.0] - 2020-08-05
-------------------

Added
^^^^^

- Glicko-1 rating system implementation
- LambdaArena for easier tournament management
- Basic documentation

Changed
^^^^^^^

- Improved Elo implementation with configurable K-factor
- Better handling of new competitors

[0.1.0] - 2020-03-21
-------------------

Initial release

Added
^^^^^

- Basic Elo rating system implementation
- Simple Arena class for managing competitions
- Basic examples and tests 