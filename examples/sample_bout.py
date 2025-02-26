from elote import EloCompetitor

good = EloCompetitor()
better = EloCompetitor()
best = EloCompetitor()

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

for _ in range(10):
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
