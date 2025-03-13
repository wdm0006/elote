Competitors
===========

Elo Competitor
--------------

.. autoclass:: elote.competitors.elo.EloCompetitor
    :members: export_state,expected_score,beat,tied,rating,to_json,from_json

Glicko Competitor
-----------------

.. autoclass:: elote.competitors.glicko.GlickoCompetitor
    :members: export_state,expected_score,beat,tied,rating,to_json,from_json

DWZ Competitor
--------------

.. autoclass:: elote.competitors.dwz.DWZCompetitor
    :members: export_state,expected_score,beat,tied,rating,to_json,from_json

ECF Competitor
--------------

.. autoclass:: elote.competitors.ecf.ECFCompetitor
    :members: export_state,expected_score,beat,tied,rating,to_json,from_json

BlendedCompetitor
-----------------

.. autoclass:: elote.competitors.ensemble.BlendedCompetitor
    :members: export_state,expected_score,beat,tied,rating,to_json,from_json

Serialization
------------

All competitor types in Elote support a standardized serialization format that allows for saving and loading competitor states.
The serialization format includes the following fields:

- **type**: The class name of the competitor
- **version**: The version of the serialization format
- **created_at**: Timestamp when the state was exported
- **id**: A unique identifier for this state export
- **parameters**: The parameters used to initialize the competitor
- **state**: The current state variables of the competitor
- **class_vars**: Class variables for backward compatibility

To serialize a competitor to JSON:

.. code-block:: python

    # Create a competitor
    competitor = EloCompetitor(initial_rating=1500)
    
    # Serialize to JSON
    json_str = competitor.to_json()

To deserialize a competitor from JSON:

.. code-block:: python

    # Deserialize from JSON
    competitor = EloCompetitor.from_json(json_str)

For backward compatibility, the serialized format also includes flattened parameters and state variables at the top level of the dictionary.
