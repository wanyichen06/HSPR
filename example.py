"""Small synthetic example of task selection and score recovery on CPU."""
import numpy as np

from hspr import SealedResponses, predict_decisions, signature_bank


def main():
    # Eight historical agents on four task groups. Each group has three tasks
    # with identical historical responses; target outcomes can differ within it.
    signatures = np.array([
        [0, 0, 1, 1], [0, 1, 1, 1], [0, 0, 0, 1], [1, 1, 1, 1],
        [0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 1, 1], [1, 1, 0, 1],
    ], dtype=float)
    source = np.repeat(signatures, 3, axis=1)
    target = np.array([0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1], dtype=float)
    bank = signature_bank(source)
    oracle = SealedResponses(target)
    config = dict(rank=1, precision=4.0, acquisition="score")
    result = predict_decisions(
        bank, oracle, budget=4, config=config, precision=1.0,
        policy="posterior_score", seed=0,
    )
    estimate = result["scores"]["median"][0.0]
    print("Synthetic example")
    print("Queried task indices:", result["selected_items"])
    print(f"Estimated full score: {100 * estimate:.1f}%")


if __name__ == "__main__":
    main()
