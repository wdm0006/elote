Comparing Rating Systems
=====================

This page provides a side-by-side comparison of the rating systems implemented in Elote, helping you choose the right system for your specific use case.

Overview Comparison
------------------

.. list-table::
   :header-rows: 1
   :widths: 20 16 16 16 16 16

   * - Feature
     - Elo
     - Glicko
     - ECF
     - DWZ
     - Ensemble
   * - **Origin**
     - Chess (1960s)
     - Chess (1995)
     - England (1950s)
     - Germany (1990s)
     - Meta-system
   * - **Complexity**
     - Low
     - Medium
     - Low
     - Medium
     - High
   * - **Uncertainty Tracking**
     - No
     - Yes (RD)
     - No
     - Partial
     - Depends on components
   * - **Expected Score Formula**
     - Logistic
     - Modified Logistic
     - Linear
     - Logistic
     - Weighted Average
   * - **Inactivity Handling**
     - No
     - Yes
     - No
     - Partial
     - Depends on components
   * - **Implementation Difficulty**
     - Easy
     - Moderate
     - Easy
     - Moderate
     - Complex
   * - **Computational Cost**
     - Low
     - Medium
     - Low
     - Medium
     - High
   * - **Typical Use Cases**
     - General purpose
     - Sparse competitions
     - English chess
     - Youth development
     - Complex domains

Mathematical Formulation
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - System
     - Expected Outcome Formula
   * - **Elo**
     - :math:`E_A = \frac{1}{1 + 10^{(R_B - R_A) / 400}}`
   * - **Glicko**
     - :math:`E(A, B) = \frac{1}{1 + 10^{-g(RD_B) \times (r_A - r_B) / 400}}` where :math:`g(RD) = \frac{1}{\sqrt{1 + 3 \times RD^2 / \pi^2}}`
   * - **ECF**
     - :math:`E_A = 0.5 + \frac{R_A - R_B}{F}` where F is typically 120
   * - **DWZ**
     - :math:`W_e = \frac{1}{1 + 10^{-(R_A - R_B) / 400}}`
   * - **Ensemble**
     - :math:`E_{ensemble} = \sum_{i=1}^{n} w_i \times E_i` where :math:`w_i` are weights

Key Parameters
-------------

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - System
     - Key Parameters
   * - **Elo**
     - K-factor (determines rating change magnitude)
   * - **Glicko**
     - Initial rating, Initial RD, Volatility, Tau
   * - **ECF**
     - K-factor, F-factor (conversion factor)
   * - **DWZ**
     - Initial rating, Development coefficient
   * - **Ensemble**
     - Component systems, Weights

Strengths and Weaknesses
-----------------------

Elo
^^^

**Strengths:**
- Simple to understand and implement
- Widely recognized and used
- Works well with sufficient data
- Zero-sum in two-player games

**Weaknesses:**
- No uncertainty measurement
- Requires many matches for accuracy
- Fixed K-factor can be problematic
- Doesn't handle inactivity well

Glicko
^^^^^^

**Strengths:**
- Tracks rating reliability
- Handles inactivity appropriately
- More accurate for sparse competitions
- Better for matchmaking

**Weaknesses:**
- More complex to implement
- Higher computational requirements
- More parameters to tune
- Less intuitive interpretation

ECF
^^^

**Strengths:**
- Linear relationship is easy to calculate
- Designed for English chess ecosystem
- Simple to understand
- Long history of use

**Weaknesses:**
- Limited range of effectiveness
- Regional focus
- Less theoretical justification
- No uncertainty tracking

DWZ
^^^

**Strengths:**
- Handles youth development well
- Age and experience factors
- Good for tournament play
- National standardization

**Weaknesses:**
- Complex calculation
- Regional focus
- Parameter sensitivity
- Less international recognition

Ensemble
^^^^^^^^

**Strengths:**
- Combines strengths of multiple systems
- More robust predictions
- Adaptable to different domains
- Graceful degradation

**Weaknesses:**
- Most complex to implement
- Highest computational cost
- Requires weight tuning
- Less interpretable

Choosing the Right System
------------------------

Consider the following factors when choosing a rating system:

1. **Data Density**: How frequently do competitors face each other?
   - Sparse data: Consider Glicko
   - Dense data: Elo may be sufficient

2. **Domain Specifics**:
   - Chess in England: ECF
   - Chess in Germany: DWZ
   - Youth development: DWZ
   - General purpose: Elo or Glicko

3. **Computational Resources**:
   - Limited resources: Elo or ECF
   - Sufficient resources: Glicko, DWZ, or Ensemble

4. **Uncertainty Importance**:
   - Critical to track uncertainty: Glicko
   - Uncertainty less important: Elo or ECF

5. **Complexity Tolerance**:
   - Need simple explanation: Elo or ECF
   - Can handle complexity: Glicko, DWZ, or Ensemble

6. **Prediction Accuracy**:
   - Highest accuracy needed: Consider Ensemble
   - Reasonable accuracy sufficient: Any individual system

Code Comparison
--------------

Here's a quick comparison of how to use each system in Elote:

.. code-block:: python

    from elote import EloCompetitor, GlickoCompetitor, ECFCompetitor, DWZCompetitor, EnsembleCompetitor

    # Elo
    elo_player = EloCompetitor(initial_rating=1500, k_factor=32)

    # Glicko
    glicko_player = GlickoCompetitor(initial_rating=1500, initial_rd=350, volatility=0.06)

    # ECF
    ecf_player = ECFCompetitor(initial_rating=120, k_factor=16, f_factor=120)

    # DWZ
    dwz_player = DWZCompetitor(initial_rating=1600, initial_development_coeff=30)

    # Ensemble
    ensemble_player = EnsembleCompetitor(
        rating_systems=[
            (EloCompetitor(initial_rating=1500), 0.5),
            (GlickoCompetitor(initial_rating=1500), 0.5)
        ]
    )

    # Usage is the same for all systems
    opponent = EloCompetitor(initial_rating=1400)
    
    # Get expected scores
    print(f"Elo expected: {elo_player.expected_score(opponent):.2%}")
    print(f"Glicko expected: {glicko_player.expected_score(opponent):.2%}")
    print(f"ECF expected: {ecf_player.expected_score(opponent):.2%}")
    print(f"DWZ expected: {dwz_player.expected_score(opponent):.2%}")
    print(f"Ensemble expected: {ensemble_player.expected_score(opponent):.2%}")

    # Record a win
    elo_player.beat(opponent)
    glicko_player.beat(opponent)
    ecf_player.beat(opponent)
    dwz_player.beat(opponent)
    ensemble_player.beat(opponent)

Empirical Comparison
-------------------

While theoretical comparisons are useful, the best way to choose a rating system is through empirical testing on your specific domain. Elote makes it easy to experiment with different systems and compare their predictive accuracy.

Here's a simple approach to compare systems:

1. Split your historical match data into training and testing sets
2. Train each rating system on the training data
3. Evaluate prediction accuracy on the test data
4. Choose the system with the best performance for your specific use case

Remember that no rating system is universally best - the right choice depends on your specific requirements, data characteristics, and domain constraints. 