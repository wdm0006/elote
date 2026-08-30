.. meta::
   :description: A sourced, measured comparison of Elo, Glicko-1, Glicko-2 and TrueSkill: the formulas, what each tracks, results on one identical split, and when a different system is the right answer.

Elo vs Glicko vs TrueSkill
============================

.. note::

   The measured tables below come from a snapshot verified on 2026-08-09.
   The three-line formula check in `Check the formulas against the library`_
   was re-run against the current published release and matches the papers
   exactly, including the TrueSkill line — the prediction defect described in
   the original version-caveat note below has since shipped a fix. The
   three-scenario result tables were not fully re-measured against the
   current release; treat them as illustrative of the shape of the
   differences between the systems rather than as exact current numbers.

**Use Elo when results are plentiful, two-sided, and you want a number people
already understand. Use Glicko-1 or Glicko-2 when competitors play
irregularly and you need to know how much to trust a rating. Use TrueSkill
when a result involves teams or more than two sides, or when you want the
same Bayesian treatment on a mu/sigma scale.** All three are implemented
behind the same four calls here, so the choice is reversible: it is a
constructor argument, not an architecture.

None of the three wins outright. On three generated scenarios run through
one identical split, TrueSkill has the best Brier score on all three and the
best accuracy outright on none of them, while Glicko-1 and Glicko-2 lead on
accuracy where draws are common — so the ordering by accuracy is not the
ordering by Brier score. The measured tables are below, and so is the code
that produced them.

The short answer
-------------------

.. list-table::
   :header-rows: 1
   :widths: 30 15 15 15 15

   * - Question
     - Elo
     - Glicko-1
     - Glicko-2
     - TrueSkill
   * - Publishes an uncertainty value
     - no
     - ``rd``
     - ``rd``, ``volatility``
     - ``sigma``
   * - Models more than two sides
     - no
     - no
     - no
     - designed for it
   * - Handles irregular participation
     - poorly
     - yes, by design
     - yes, plus volatility
     - yes, via ``sigma``
   * - Scale a reader recognises
     - yes
     - yes
     - yes
     - no, ``mu``/``sigma``
   * - Parameters you will actually tune
     - K-factor
     - initial RD
     - initial RD, tau
     - beta, tau, draw probability
   * - Cost per result
     - lowest
     - low
     - low
     - low
   * - Reported rating is
     - the rating
     - the rating
     - the rating
     - ``mu - 3 * sigma``

That last row catches people out; see `What TrueSkill reports`_.

What each one actually computes
-----------------------------------

The three formulas are short enough to state, and stating them is more
useful than adjectives, because the differences between these systems are
visible in them.

Elo
~~~~

Elo's expected score is the logistic function of the rating difference, with
base 10 and a 400-point scale:

.. code-block:: text

    E_A = 1 / (1 + 10 ** ((R_B - R_A) / 400))

A 200-point favourite is expected to score about 0.76. After the game, each
competitor moves by ``K * (actual - expected)``; Elote's default ``K`` is 32
and its default base is 400. Note that FIDE's own published regulations
implement the expected score as a lookup table of rating difference against
scoring probability rather than as this closed form, so "the Elo formula" is
a family with a common shape rather than one universal constant set.
[source: `FIDE Rating Regulations effective from 1 March 2024, section
8.1.2 <https://handbook.fide.com/chapter/B022024>`_] [source: Elo, A. E.
(1978), *The Rating of Chessplayers, Past and Present*]

Glicko-1 and Glicko-2
~~~~~~~~~~~~~~~~~~~~~~~~

Glicko adds a rating deviation, RD, which is the standard deviation of the
rating: a measure of how uncertain it is. An unrated player starts at
rating 1500 with RD 350, RD falls with every game played, and RD rises with
time spent not playing. The expected score discounts the rating difference
by the *opponent's* uncertainty:

.. code-block:: text

    g(RD) = 1 / sqrt(1 + 3 * q**2 * RD**2 / pi**2),  q = ln(10) / 400 = 0.0057565
    E     = 1 / (1 + 10 ** (-g(RD_j) * (r - r_j) / 400))

[source: `Mark E. Glickman, The Glicko system <https://glicko.net/glicko/glicko.pdf>`_]

That ``g`` term is the whole idea in one factor. Against an opponent whose
rating is well established, ``g`` is near 1 and Glicko behaves almost
exactly like Elo. Against an opponent whose rating is a guess, ``g`` shrinks
the difference toward 0.5, because a large rating gap you are not sure
about is not much evidence.

Glicko-2 adds a third quantity, volatility, which "indicates the degree of
expected fluctuation in a player's rating" — high for a competitor with
erratic results, low for a consistent one. It works on an internally
transformed scale (``mu = (r - 1500) / 173.7178``), and a system constant
tau constrains how fast volatility may move; Glickman recommends values
between 0.3 and 1.2, or as low as 0.2 where extremely improbable runs of
results are expected. An unrated player starts at 1500 / 350 / 0.06.
[source: `Mark E. Glickman, Example of the Glicko-2 system
<https://glicko.net/glicko/glicko2.pdf>`_]

Elote's defaults are those recommended values.

TrueSkill
~~~~~~~~~~~

TrueSkill models each competitor as a Gaussian belief with mean ``mu`` and
standard deviation ``sigma``, and was built for the multiplayer and team
case rather than the two-player one. [source: `Herbrich, Minka and Graepel,
TrueSkill(tm): A Bayesian Skill Rating System, NIPS 2006
<https://papers.nips.cc/paper_files/paper/2006/hash/f44ee263952e65b3610b8ba51229d1f9-Abstract.html>`_]
For two competitors, Elote's expected score decomposes the outcome into a
win, a draw, and a loss against a draw margin, then scores a draw as half a
win:

.. code-block:: text

    c    = sqrt(2 * beta**2 + sigma_A**2 + sigma_B**2)
    eps  = sqrt(2) * beta * Phi_inverse((p_draw + 1) / 2)
    win  = Phi((mu_A - mu_B - eps) / c)
    draw = Phi((mu_A - mu_B + eps) / c) - win
    E_A  = win + draw / 2

``beta`` is the performance noise: how much a single result is chance
rather than skill. It is counted twice because both sides have it. Elote's
class defaults are ``mu`` 25.0, ``sigma`` 8.333, ``beta`` 4.166, ``tau``
0.083, and an assumed draw probability of 0.10.

Check the formulas against the library
------------------------------------------

Nothing above needs to be taken on trust. This reproduces all three from
their published form and compares them to what the library returns.
``beta`` and the assumed draw probability are class variables rather than
constructor arguments, which is why they are read off the class:

.. code-block:: python

    import math
    from statistics import NormalDist

    from elote import EloCompetitor, GlickoCompetitor, TrueSkillCompetitor

    phi = NormalDist().cdf

    # Elo: E_A = 1 / (1 + 10 ** ((R_B - R_A) / 400))
    a, b = EloCompetitor(initial_rating=1600), EloCompetitor(initial_rating=1400)
    print(f"Elo       library {a.expected_score(b):.10f}  paper {1 / (1 + 10 ** ((1400 - 1600) / 400)):.10f}")

    # Glicko: E = 1 / (1 + 10 ** (-g(RD_j) (r - r_j) / 400)),  g(RD) = 1 / sqrt(1 + 3 q^2 RD^2 / pi^2)
    g, h = GlickoCompetitor(initial_rating=1600), GlickoCompetitor(initial_rating=1400)
    q = 0.0057565  # ln(10) / 400, rounded exactly as the paper prints it
    g_term = 1 / math.sqrt(1 + 3 * q**2 * h.rd**2 / math.pi**2)
    print(f"Glicko-1  library {g.expected_score(h):.10f}  paper {1 / (1 + 10 ** (-g_term * (1600 - 1400) / 400)):.10f}")

    # TrueSkill: win + half the draw mass, with c^2 = 2 beta^2 + sigma_A^2 + sigma_B^2
    # and the draw margin eps = sqrt(2) beta Phi^-1((p_draw + 1) / 2).
    t, u = TrueSkillCompetitor(initial_mu=30.0), TrueSkillCompetitor(initial_mu=25.0)
    beta, p_draw = TrueSkillCompetitor._beta, TrueSkillCompetitor._draw_probability
    c = math.sqrt(2 * beta**2 + t.sigma**2 + u.sigma**2)
    eps = math.sqrt(2) * beta * NormalDist().inv_cdf((p_draw + 1) / 2)
    win = phi((t.mu - u.mu - eps) / c)
    draw = phi((t.mu - u.mu + eps) / c) - win
    print(f"TrueSkill library {t.expected_score(u):.10f}  paper {win + draw / 2:.10f}")

.. code-block:: text

    Elo       library 0.7597469266  paper 0.7597469266
    Glicko-1  library 0.6835840246  paper 0.6835840246
    TrueSkill library 0.6476185605  paper 0.6476185605

Two details in that output are worth reading rather than skimming. The
Glicko line uses ``q`` rounded to the seven digits the paper prints; at full
precision the two sides differ in the seventh decimal, which is the library
following its source rather than a defect. And the two 1600/1400
competitors give different answers under Elo and Glicko — 0.7597 against
0.6836 — purely because Glicko discounts a rating whose RD is still the
unrated 350.

What TrueSkill reports
--------------------------

``TrueSkillCompetitor.rating`` is the conservative estimate ``mu - 3 *
sigma``, not ``mu``. At construction that is close to zero, and a single
drawn game against an identical opponent raises it noticeably while the
expected score stays at exactly 0.5000 — the belief did not move, the
uncertainty around it shrank.

That is the right number for a leaderboard, because it will not promote a
competitor who has simply not played enough to be caught out. It is the
wrong number to feed anywhere expecting a skill estimate; use ``mu`` for
that, and ``sigma`` for the uncertainty.

Uncertainty, measured
------------------------

Elo has one number and no notion of how much evidence backs it. The other
three publish a spread that shrinks as results accumulate:

.. code-block:: python

    from elote import EloCompetitor, GlickoCompetitor, TrueSkillCompetitor


    def after(system, games, **kwargs):
        a, b = system(**kwargs), system(**kwargs)
        for _ in range(games):
            a.beat(b)
        return a


    print(f"{'games':>6}{'elo rating':>12}{'glicko rd':>12}{'trueskill sigma':>18}")
    for games in (0, 1, 5, 20):
        elo = after(EloCompetitor, games, initial_rating=1500)
        glicko = after(GlickoCompetitor, games, initial_rating=1500)
        trueskill = after(TrueSkillCompetitor, games)
        print(f"{games:>6}{elo.rating:>12.1f}{glicko.rd:>12.2f}{trueskill.sigma:>18.3f}")

.. code-block:: text

     games  elo rating   glicko rd   trueskill sigma
         0      1500.0      350.00             8.333
         1      1516.0      290.23             7.171
         5      1566.8      223.26             5.585
        20      1665.4      177.98             4.413

Glicko-2's ``rd`` follows Glicko-1's closely over the same 20 games, and its
volatility barely moves, because a run of identical results is exactly the
non-erratic case volatility exists to detect.

One limitation specific to this library: Glicko's RD also grows during
inactivity, but Elote's arena rejects a match timestamp earlier than the
competitor's creation time, so that decay cannot be driven through an arena
on historical data. On a back-test it is inert, and the results below were
produced with it inert for both Glicko systems.

Measured on the same split
------------------------------

Three generated scenarios, one seed, one time-ordered 70/30 split, 420
training rows and 180 held-out bouts each, every system starting from 1500
where it accepts a starting rating. Lower is better for Brier score and log
loss. These figures are from the 2026-08-09 snapshot described in the note
at the top of this page.

``balanced`` — 30 competitors, 600 matchups, 2 drawn bouts in the evaluation set:

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 15

   * - System
     - Accuracy
     - Brier
     - Log loss
   * - Elo
     - 0.8778
     - 0.0900
     - 0.3310
   * - Glicko-1
     - 0.9167
     - 0.0605
     - 0.2206
   * - Glicko-2
     - 0.9167
     - 0.0600
     - 0.2191
   * - TrueSkill
     - 0.9111
     - **0.0570**
     - **0.1996**

``draw_heavy`` — same size, higher draw probability and higher generator
noise, 26 drawn bouts in the evaluation set:

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 15

   * - System
     - Accuracy
     - Brier
     - Log loss
   * - Elo
     - 0.6556
     - 0.1219
     - 0.4886
   * - Glicko-1
     - **0.7000**
     - 0.1057
     - 0.4327
   * - Glicko-2
     - **0.7000**
     - 0.1062
     - 0.4338
   * - TrueSkill
     - 0.6944
     - **0.1050**
     - **0.4282**

``sparse`` — 120 competitors sharing the same 600 matchups, so most pairs
have never met:

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 15

   * - System
     - Accuracy
     - Brier
     - Log loss
   * - Elo
     - 0.6389
     - 0.1807
     - 0.5553
   * - Glicko-1
     - 0.7611
     - 0.1255
     - 0.3944
   * - Glicko-2
     - 0.7611
     - 0.1255
     - 0.3943
   * - TrueSkill
     - 0.7611
     - **0.1218**
     - **0.3783**

Read it this way, and no further:

- Elo is last on every scenario **in this configuration**, and the gap is
  widest where data is thin. That is the case Glicko was designed for, so
  it is the expected result rather than a surprising one.
- Glicko-1 and Glicko-2 are effectively tied here. Volatility earns its
  keep on competitors whose form actually changes, which a generator
  sampling from a fixed latent strength does not produce.
- TrueSkill has the best Brier score and log loss on all three, and the
  best accuracy outright on none of them — it ties for best on ``sparse``
  and is second on the other two. If you take one thing from this page,
  take that: two reasonable metrics disagree about the winner even between
  these three closely related systems.

Answering the actual selection questions
--------------------------------------------

**"My competitors play irregularly, and some are new."** Glicko-1 or
Glicko-2. The RD term is exactly the mechanism for it, and the sparse table
above is the measured version of that claim. If you also need to withhold
a competitor from a leaderboard until their rating is trustworthy, RD is
the threshold to use.

**"Results involve teams, or more than two sides."** TrueSkill. Elo and
Glicko are two-player systems; anything else is a workaround. TrueSkill was
designed for the multiplayer case.

**"Draws are a large share of my results."** Any of the three will accept
them, and TrueSkill's expected score has a draw margin built into it.
Compare on a proper scoring rule rather than accuracy, because a
threshold-based accuracy number on draw-heavy data mostly measures where
you put the thresholds.

**"I need people to understand the number."** Elo, or Glicko reported as a
rating. TrueSkill's ``mu``/``sigma`` scale is unfamiliar, and its reported
rating is a conservative estimate rather than the belief itself.

**"I need the ranking not to depend on the order results arrived."** None
of these three. All are incremental. See the next section.

**"I want to know which is best on my data."** Run all three. The
comparison is a loop over a dictionary and the whole harness is on the
:doc:`decision guide <../choose_a_rating_system>`.

When none of these three is the answer
-------------------------------------------

Elote exports a dozen concrete rating systems. These three are the ones
people name; they are not the ones that suit every problem. A subset, by the
axis most likely to rule them in or out:

.. list-table::
   :header-rows: 1
   :widths: 18 15 12 18 37

   * - System
     - Update model
     - Uncertainty
     - Reads score margins
     - Choose it when
   * - Elo
     - Incremental
     - no
     - no
     - Results are plentiful and the number must be recognisable.
   * - Glicko-1
     - Incremental
     - ``rd``
     - no
     - Participation is irregular and confidence matters.
   * - Glicko-2
     - Incremental
     - ``rd``, ``volatility``
     - no
     - The same, and form genuinely changes over time.
   * - TrueSkill
     - Incremental
     - ``sigma``
     - no
     - Teams, multiplayer, or a Bayesian belief you will use directly.
   * - ECF
     - Incremental
     - no
     - no
     - You must match the English chess federation's published grading.
   * - DWZ
     - Incremental
     - no
     - no
     - You must match the German federation's published scheme.
   * - Colley Matrix
     - Global fit
     - no
     - no
     - A finished season, ranked once, outcomes only, no margin influence.
   * - Massey
     - Global fit
     - no
     - **yes**
     - You want predicted point spreads on the scale of the scores.
   * - Keener
     - Global fit
     - no
     - **yes**
     - Scores should inform the ranking without one blowout dominating it.
   * - Bradley-Terry
     - Global fit
     - no
     - no
     - You want the maximum-likelihood paired-comparison fit on an Elo-like scale.

``BlendedCompetitor`` combines any of these with weights, and inherits
whatever its components do on every axis. ``PythagoreanCompetitor`` and
``WholeHistoryRatingCompetitor`` are two further systems not covered by this
head-to-head; see :doc:`Pythagorean <pythagorean>` and their own pages.

The division that matters most is the middle column. Incremental systems
update two competitors per result and depend on the order results arrived;
global-fit systems re-solve the whole connected group and do not. Reordering
the same five results changes Elo's final ratings and leaves Bradley-Terry's
untouched. Order independence costs real time — on 420 rows, Bradley-Terry
took roughly ten thousand times as long as Elo in the 2026-08-09 snapshot —
and it needs a connected schedule, because competitors who have never met
even transitively are fitted in separate groups. The
:doc:`decision guide <../choose_a_rating_system>` works through that choice.

What this comparison does not show
---------------------------------------

- The data is **generated**. It samples outcomes from a latent
  per-competitor strength, which is the model these systems assume, so
  their agreement with it is an upper bound rather than evidence about
  messy real data.
- It measures prediction on held-out results only — not recovery of a known
  true ordering, not convergence speed, not calibration after binning, not
  behaviour under adversarial scheduling.
- It compares Elote's systems to each other. It says nothing about any
  other library.
- Threshold optimization is not used; accuracy is computed at fixed
  0.45 / 0.55 thresholds.
- Glicko's inactivity-driven RD growth is inert in these runs, so Glicko is
  measured without one of its own mechanisms.

Reproducing it
------------------

Every code block on this page runs as written on Python 3.10 through 3.12
with Elote installed and nothing else. The three-scenario table comes from
the harness on the :doc:`decision guide <../choose_a_rating_system>`, with
``SHORTLIST`` cut to these four systems and the scenario constructed as
``SyntheticDataset(num_competitors=30, num_matchups=600, seed=20260809)``,
``draw_probability=0.5, noise_std=200.0`` added for ``draw_heavy``, and
``num_competitors=120`` for ``sparse``.

Related
----------

- :doc:`How to choose a rating system <../choose_a_rating_system>` — the
  decision framework across all of Elote's rating systems.
- :doc:`Comparing rating systems <comparison>` — per-system origins,
  parameters, strengths and weaknesses.
- :doc:`Elo <elo>`, :doc:`Glicko <glicko>`, :doc:`Glicko-2 <glicko2>`,
  :doc:`TrueSkill <trueskill>` — the individual system pages.
