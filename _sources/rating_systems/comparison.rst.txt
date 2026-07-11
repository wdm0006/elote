Comparing Rating Systems
=====================

This page provides a side-by-side comparison of the rating systems implemented in Elote, helping you choose the right system for your specific use case.

Overview Comparison
------------------

.. list-table::
   :header-rows: 1
   :widths: 16 18 14 20 14 18

   * - System
     - Origin
     - Complexity
     - Uncertainty Tracking
     - Order Independent
     - Typical Use Cases
   * - **Elo**
     - Chess (1960s)
     - Low
     - No
     - No
     - General purpose
   * - **Glicko**
     - Chess (1995)
     - Medium
     - Yes (RD)
     - No
     - Sparse competitions
   * - **Glicko-2**
     - Chess (2000s)
     - Medium-High
     - Yes (RD + volatility)
     - No
     - Online chess, volatile skill
   * - **TrueSkill**
     - Microsoft (2007)
     - High
     - Yes (sigma)
     - No
     - Teams and multiplayer
   * - **ECF**
     - England (1950s)
     - Low
     - No
     - No
     - English chess
   * - **DWZ**
     - Germany (1990s)
     - Medium
     - Partial
     - No
     - Youth development
   * - **Colley Matrix**
     - Sports (2002)
     - Medium
     - No
     - Yes
     - Bias-free sports ranking
   * - **Bradley-Terry**
     - Statistics (1952)
     - Medium
     - No
     - Yes
     - Maximum-likelihood ranking
   * - **Ensemble**
     - Meta-system
     - High
     - Depends on components
     - Depends on components
     - Complex domains

Online systems (Elo, Glicko, Glicko-2, TrueSkill, ECF, DWZ) update ratings incrementally after each
result. Global-fit systems (Colley Matrix and Bradley-Terry) re-solve the whole connected group of
competitors from the full set of results, which makes them order independent.

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
   * - **Glicko-2**
     - Same logistic form as Glicko, evaluated on an internal transformed scale with an added volatility term
   * - **TrueSkill**
     - :math:`E_A = \Phi\!\left(\frac{\mu_A - \mu_B}{\sqrt{2\beta^2 + \sigma_A^2 + \sigma_B^2}}\right)` where :math:`\Phi` is the normal CDF
   * - **ECF**
     - :math:`E_A = 0.5 + \frac{R_A - R_B}{F}` where F is typically 120
   * - **DWZ**
     - :math:`W_e = \frac{1}{1 + 10^{-(R_A - R_B) / 400}}`
   * - **Colley Matrix**
     - Ratings solve :math:`C r = b`; :math:`E_A = \frac{1}{1 + e^{-4 (r_A - r_B)}}`
   * - **Bradley-Terry**
     - :math:`P(A \text{ beats } B) = \frac{p_A}{p_A + p_B} = \frac{1}{1 + e^{-(\beta_A - \beta_B)}}`
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
   * - **Glicko-2**
     - Initial rating, Initial RD, Initial volatility, Tau
   * - **TrueSkill**
     - Initial mu, Initial sigma, Beta, Tau, Draw probability
   * - **ECF**
     - Delta (max rating difference), Number of periods
   * - **DWZ**
     - Initial rating, Development coefficient (age-dependent)
   * - **Colley Matrix**
     - Initial rating (default 0.5)
   * - **Bradley-Terry**
     - Initial rating, Scale, Regularization, Max iterations
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

Glicko-2
^^^^^^^^

**Strengths:**
- Adds volatility to capture how consistent a competitor is
- Retains Glicko's uncertainty and inactivity handling
- Fast, stable convergence for both erratic and steady players
- Used by major online chess platforms

**Weaknesses:**
- The most involved of the Glicko family to implement
- Sensitive to the tau system constant
- Iterative volatility update adds computational cost
- Designed around rating periods rather than single games

TrueSkill
^^^^^^^^^

**Strengths:**
- Models both skill and uncertainty as a Gaussian
- Naturally extends to teams and multiplayer games
- Fast convergence as uncertainty shrinks
- Principled Bayesian foundation

**Weaknesses:**
- Most complex inference of the individual systems
- Rating is derived from mu and sigma (no direct setter)
- Works on a mu/sigma scale rather than the chess scale
- Sensitive to beta, tau, and draw-probability settings

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

Colley Matrix
^^^^^^^^^^^^^

**Strengths:**
- Order independent: depends only on the set of results
- Bias free: no margin-of-victory influence
- Clean least-squares interpretation with a unique solution
- Self-normalizing ratings in [0, 1] that sum to n/2

**Weaknesses:**
- Re-solves the whole connected group after each result
- Uses only win/loss/tie outcomes
- Ratings are on a [0, 1] scale, not the chess scale
- Requires a connected schedule to compare competitors

Bradley-Terry
^^^^^^^^^^^^^

**Strengths:**
- Order independent maximum-likelihood estimate
- Statistically principled paired-comparison model
- Reported on an Elo-compatible scale for easy interpretation
- Regularized so undefeated or winless competitors stay finite

**Weaknesses:**
- Re-fits the whole connected group after each result
- Uses only win/loss/tie outcomes
- The match graph cannot be serialized (only aggregate counts)
- More expensive than Elo for very large populations

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
   - Sparse data: Consider Glicko or Glicko-2
   - Dense data: Elo may be sufficient

2. **Domain Specifics**:
   - Chess in England: ECF
   - Chess in Germany: DWZ
   - Youth development: DWZ
   - Team and multiplayer games: TrueSkill
   - General purpose: Elo or Glicko

3. **Order Independence**: Do you need a ranking that ignores schedule order?
   - Yes: Colley Matrix or Bradley-Terry
   - No: any online system (Elo, Glicko, etc.)

4. **Computational Resources**:
   - Limited resources: Elo or ECF
   - Sufficient resources: Glicko, Glicko-2, TrueSkill, or Ensemble

5. **Uncertainty Importance**:
   - Critical to track uncertainty: Glicko, Glicko-2, or TrueSkill
   - Uncertainty less important: Elo or ECF

6. **Prediction Accuracy**:
   - Highest accuracy needed: Consider Ensemble
   - Reasonable accuracy sufficient: Any individual system

Code Comparison
--------------

Here's a quick comparison of how to use each system in Elote:

.. code-block:: python

    from elote import (
        EloCompetitor,
        GlickoCompetitor,
        Glicko2Competitor,
        TrueSkillCompetitor,
        ECFCompetitor,
        DWZCompetitor,
        ColleyMatrixCompetitor,
        BradleyTerryCompetitor,
        BlendedCompetitor,
    )

    # Elo
    elo_player = EloCompetitor(initial_rating=1500, k_factor=32)

    # Glicko
    glicko_player = GlickoCompetitor(initial_rating=1500, initial_rd=350)

    # Glicko-2
    glicko2_player = Glicko2Competitor(initial_rating=1500, initial_rd=350)

    # TrueSkill (mu/sigma scale)
    trueskill_player = TrueSkillCompetitor(initial_mu=25.0, initial_sigma=8.333)

    # ECF
    ecf_player = ECFCompetitor(initial_rating=100)

    # DWZ
    dwz_player = DWZCompetitor(initial_rating=1600)

    # Colley Matrix ([0, 1] scale)
    colley_player = ColleyMatrixCompetitor()

    # Bradley-Terry (Elo-compatible scale)
    bt_player = BradleyTerryCompetitor(initial_rating=1500)

    # Ensemble (blend of sub-competitors)
    ensemble_player = BlendedCompetitor(
        competitors=[
            {"type": "EloCompetitor", "competitor_kwargs": {"initial_rating": 1500}},
            {"type": "GlickoCompetitor", "competitor_kwargs": {"initial_rating": 1500}},
        ]
    )

    # Usage is the same for all systems: expected_score, beat, tied, lost_to
    player = EloCompetitor(initial_rating=1500)
    opponent = EloCompetitor(initial_rating=1400)
    print(f"Expected score: {player.expected_score(opponent):.2%}")
    player.beat(opponent)  # record a win

Empirical Comparison
-------------------

While theoretical comparisons are useful, the best way to choose a rating system is through empirical testing on your specific domain. Elote makes it easy to experiment with different systems and compare their predictive accuracy.

Here's a simple approach to compare systems:

1. Split your historical match data into training and testing sets
2. Train each rating system on the training data
3. Evaluate prediction accuracy on the test data
4. Choose the system with the best performance for your specific use case

Remember that no rating system is universally best - the right choice depends on your specific requirements, data characteristics, and domain constraints.
