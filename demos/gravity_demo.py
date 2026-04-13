"""
Gravitational Scorer Demo.
Shows attraction and repulsion between entities.
"""

import sys, torch
sys.path.insert(0, '.')
from solar_ring.gravitational_scorer import GravitationalScorer
from solar_ring.contextual_embedder import ContextualEmbedder

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def demo():
    print('=' * 60)
    print('Gravitational Scorer — Attraction & Repulsion')
    print('=' * 60)

    embedder = ContextualEmbedder(DEVICE)
    scorer   = GravitationalScorer(d=384).to(DEVICE)

    # Test sentences
    tests = [
        {
            'sentence':   'The trophy did not fit the suitcase because it was too big.',
            'pronoun':    'it',
            'candidates': [('trophy', 1), ('suitcase', 4)],
            'expected':   'trophy',
        },
        {
            'sentence':   'The hawk chased the rabbit because it was hungry.',
            'pronoun':    'it',
            'candidates': [('hawk', 1), ('rabbit', 4)],
            'expected':   'hawk',
        },
        {
            'sentence':   'The water filled the bucket until it overflowed.',
            'pronoun':    'it',
            'candidates': [('bucket', 3), ('water', 1)],
            'expected':   'bucket',
        },
        {
            'sentence':   'Sarah helped Beth because she was tired.',
            'pronoun':    'she',
            'candidates': [('sarah', 0), ('beth', 2)],
            'expected':   'sarah',
        },
        {
            'sentence':   'The workers obeyed the managers because they gave clear orders.',
            'pronoun':    'they',
            'candidates': [('managers', 3), ('workers', 1)],
            'expected':   'managers',
        },
    ]

    correct = 0
    for test in tests:
        sent     = test['sentence']
        expected = test['expected']

        result = scorer.resolve_pronoun(
            sent, test['candidates'], embedder, DEVICE
        )

        winner = result['winner']
        ok = winner == expected
        if ok:
            correct += 1

        print(f'\n{"✓" if ok else "✗"} {sent[:55]}')
        print(f'  pronoun "{test["pronoun"]}" → {winner}')
        print(f'  expected: {expected}')
        print()

        # Show physics for each candidate
        for r in result['all']:
            print(f'  {r["word"]:12} '
                  f'Φ={r["total_phi"]:+.4f}  '
                  f'attract={r["attraction"]:.4f}  '
                  f'repel={r["repulsion"]:.4f}  '
                  f'mass={r["mass"]:.2f}')
        print(f'  margin: {result["margin"]:.4f}')

    print()
    print('=' * 60)
    print(f'Score: {correct}/{len(tests)} = {correct / len(tests) * 100:.0f}%')
    print()
    print('Physics interpretation:')
    print('  Φ > 0 → attraction (entity fits context)')
    print('  Φ < 0 → repulsion (entity conflicts context)')
    print('  Higher Φ → stronger gravitational bond → correct referent')
    print('=' * 60)


if __name__ == '__main__':
    demo()
