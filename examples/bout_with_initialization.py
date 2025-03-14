from elote import EloCompetitor

good = EloCompetitor(initial_rating=500)
better = EloCompetitor(initial_rating=450)
best = EloCompetitor(initial_rating=400)

print("Starting ratings:")
print(
    "%7.2f, %7.2f, %7.2f"
    % (
        good.rating,
        better.rating,
        best.rating,
    )
)

print("\nAfter matches")

for _ in range(20):
    better.beat(good)
    best.beat(better)
    print(
        "%7.2f, %7.2f, %7.2f"
        % (
            good.rating,
            better.rating,
            best.rating,
        )
    )
