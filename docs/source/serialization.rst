Serialization
=============

Overview
--------

Elote provides a standardized serialization format for all competitor types, allowing for consistent saving and loading of competitor states. This is particularly useful for:

- Persisting ratings between application runs
- Sharing ratings between different systems
- Creating backups of rating data
- Analyzing rating changes over time

Standardized Format
------------------

As of version 1.0.0, all competitors use a standardized serialization format that includes the following fields:

.. code-block:: json

    {
        "type": "EloCompetitor",
        "version": 1,
        "created_at": 1625097600,
        "id": "550e8400-e29b-41d4-a716-446655440000",
        "parameters": {
            "initial_rating": 1500
        },
        "state": {
            "rating": 1550
        },
        "class_vars": {
            "k_factor": 32
        }
    }

Field Descriptions
^^^^^^^^^^^^^^^^^

- **type**: The class name of the competitor (e.g., "EloCompetitor", "GlickoCompetitor")
- **version**: The version of the serialization format (currently 1)
- **created_at**: Unix timestamp when the state was exported
- **id**: A unique identifier for this state export (UUID)
- **parameters**: The parameters used to initialize the competitor
  - **initial_rating**: The initial rating value
  - *Other parameters specific to the competitor type*
- **state**: The current state variables of the competitor
  - **rating**: The current rating value
  - *Other state variables specific to the competitor type*
- **class_vars**: Class variables for backward compatibility
  - *Class variables specific to the competitor type*

Backward Compatibility
---------------------

For backward compatibility, the serialized format also includes flattened parameters and state variables at the top level of the dictionary. This allows older code that expects these fields to continue working.

For example, in addition to the structured format above, the serialized JSON also includes:

.. code-block:: json

    {
        "type": "EloCompetitor",
        "version": 1,
        "created_at": 1625097600,
        "id": "550e8400-e29b-41d4-a716-446655440000",
        "parameters": { ... },
        "state": { ... },
        "class_vars": { ... },
        
        "initial_rating": 1500,
        "current_rating": 1550
    }

Serialization Methods
--------------------

To JSON
^^^^^^^

To serialize a competitor to JSON:

.. code-block:: python

    from elote import EloCompetitor

    # Create a competitor
    competitor = EloCompetitor(initial_rating=1500)

    # Serialize to JSON
    json_str = competitor.to_json()

From JSON
^^^^^^^^

To deserialize a competitor from JSON:

.. code-block:: python

    from elote import EloCompetitor

    # Deserialize from JSON
    competitor = EloCompetitor.from_json(json_str)

Cross-Competitor Compatibility
-----------------------------

Competitors can only be deserialized to the same type they were serialized from. Attempting to deserialize a competitor to a different type will raise an ``InvalidStateException``.

For example, this will fail:

.. code-block:: python

    from elote import EloCompetitor, GlickoCompetitor

    # Create and serialize an Elo competitor
    elo_competitor = EloCompetitor(initial_rating=1500)
    json_str = elo_competitor.to_json()

    # Attempt to deserialize as a Glicko competitor (will raise an exception)
    try:
        glicko_competitor = GlickoCompetitor.from_json(json_str)
    except Exception as e:
        print(f"Error: {e}")

Complete Example
--------------

Here's a complete example of serializing and deserializing a competitor:

.. code-block:: python

    from elote import EloCompetitor
    import json

    # Create a competitor
    competitor = EloCompetitor(initial_rating=1500)

    # Simulate some matches
    competitor.beat(EloCompetitor(initial_rating=1400))
    competitor.beat(EloCompetitor(initial_rating=1450))
    competitor.lost_to(EloCompetitor(initial_rating=1600))

    # Serialize to JSON
    json_str = competitor.to_json()

    # Print the serialized JSON (formatted for readability)
    print("Serialized competitor:")
    print(json.dumps(json.loads(json_str), indent=4))

    # Deserialize from JSON
    new_competitor = EloCompetitor.from_json(json_str)

    # Verify the ratings match
    print(f"Original rating: {competitor.rating}")
    print(f"Deserialized rating: {new_competitor.rating}")

    # Continue using the deserialized competitor
    new_competitor.beat(EloCompetitor(initial_rating=1450))
    print(f"Updated rating after another match: {new_competitor.rating}")

Implementing Serialization in Custom Competitors
----------------------------------------------

If you're implementing a custom competitor, you need to implement the following methods to support the standardized serialization format:

.. code-block:: python

    def _export_parameters(self) -> Dict[str, Any]:
        """Export the parameters used to initialize this competitor."""
        return {
            "initial_rating": self._initial_rating,
            # Include any other parameters used during initialization
        }

    def _export_current_state(self) -> Dict[str, Any]:
        """Export the current state variables of this competitor."""
        return {
            "rating": self._rating,
            # Include any other state variables that change during usage
        }

    def _import_parameters(self, parameters: Dict[str, Any]) -> None:
        """Import parameters from a state dictionary."""
        if "initial_rating" in parameters:
            initial_rating = parameters["initial_rating"]
            if initial_rating < self._minimum_rating:
                raise InvalidParameterException(f"Initial rating cannot be below the minimum rating of {self._minimum_rating}")
            self._initial_rating = initial_rating
        
        # Import any other parameters

    def _import_current_state(self, state: Dict[str, Any]) -> None:
        """Import current state variables from a state dictionary."""
        if "rating" in state:
            rating = state["rating"]
            if rating < self._minimum_rating:
                raise InvalidStateException(f"Rating cannot be below the minimum rating of {self._minimum_rating}")
            self._rating = rating
        
        # Import any other state variables

    @classmethod
    def _create_from_parameters(cls: Type[T], parameters: Dict[str, Any]) -> T:
        """Create a new competitor instance from parameters."""
        initial_rating = parameters.get("initial_rating", 1500)
        
        # Create a new instance with the parameters
        return cls(initial_rating=initial_rating)

For more details on implementing custom competitors, see the :doc:`implementing_competitors` guide. 