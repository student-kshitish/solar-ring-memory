"""
Demo: unified light field memory.
One formula covers all relationships,
reasoning, facts, and identity.
"""

import sys, torch
sys.path.insert(0,'.')
from solar_ring.unified_memory import UnifiedMemory

def demo():
    print('='*60)
    print('Unified Light Field Memory Demo')
    print('One formula: Phi = lambda * G * C * R * (1-BH)')
    print('='*60)

    # Initialize with Kshitish's identity
    mem = UnifiedMemory('Kshitish', d=300)

    # Learn facts
    print('\nLearning facts:')
    facts = [
        ('kshitish', 'lives in', 'hostel'),
        ('kshitish', 'studies at', 'SUIIT'),
        ('kshitish', 'studies', 'BTech'),
        ('kshitish', 'lives in', 'Burla'),
    ]
    for subj, pred, obj in facts:
        mem.learn_fact(subj, pred, obj)
        print(f'  Learned: {subj} {pred} {obj}')

    # Learn relationships
    print('\nLearning relationships:')
    relationships = [
        ('Ram', 'parent'),
        ('Sita', 'parent'),
        ('John', 'sibling'),
        ('Priya', 'classmate'),
        ('Rahul', 'best_friend'),
        ('Dr Kumar', 'professor'),
        ('Arjun', 'cousin'),
        ('Stranger', 'stranger'),
    ]
    for name, rel in relationships:
        key, d = mem.learn_relationship(name, rel)
        c = mem.field.c('relationship')
        d_light = d / c
        lam = mem.field.redshift(d_light)
        mass = mem.entities[key]['mass']
        phi = lam * mass**2 / max(d**2, 1)
        print(f'  {name:12} {rel:15} '
              f'd={d} d_light={d_light:.3f} '
              f'lambda={lam:.3f} Phi={phi:.3f}')

    # Light intensity spectrum
    print('\nLight intensity spectrum from Kshitish:')
    print('(How brightly each entity illuminates memory)')
    print()

    self_entity = mem.entities[mem.identity]
    self_entity['pos'] = 0

    receivers = []
    for name, info in mem.entities.items():
        if name == mem.identity:
            continue
        conns = info.get('connections', {})
        d = list(conns.values())[0]['hops'] if conns else 3
        recv = dict(info)
        recv['pos'] = d
        receivers.append(recv)

    spectrum = mem.field.intensity_spectrum(
        self_entity, receivers, 'relationship'
    )

    print(f'  {"Name":15} {"Intensity":12} {"Phi":10} {"d_light":10} {"Visible?"}')
    print('-'*65)
    for r in spectrum:
        visible = 'BRIGHT' if r['intensity'] > 0.5 else (
                  'DIM' if r['intensity'] > 0.1 else 'DARK')
        print(f'  {r["name"].title():15} '
              f'{r["intensity"]:12.4f} '
              f'{r["phi"]:10.4f} '
              f'{r["d_light"]:10.4f} '
              f'{visible}')

    # Answer questions
    print('\nAnswering questions:')
    questions = [
        'Where do I live?',
        'What is my college?',
        'Who is Ram?',
        'Who is Rahul?',
        'Who is closest to me?',
        'Who is Stranger?',
        'List all relationships',
    ]
    for q in questions:
        ans = mem.query(q)
        print(f'\n  Q: {q}')
        print(f'  A: {ans}')

    # Show phi matrix for close entities
    print('\nPhi matrix (attraction/repulsion):')
    names = ['ram','john','rahul','arjun','stranger']
    available = [n for n in names if n in mem.entities]

    entities_list = [
        {**mem.entities[n], 'pos': i}
        for i, n in enumerate(available)
    ]

    if len(entities_list) > 1:
        matrix = mem.field.phi_matrix(
            entities_list, 'relationship'
        )
        print(f'  {"":12}', end='')
        for n in available:
            print(f'{n:10}', end='')
        print()
        for i, ni in enumerate(available):
            print(f'  {ni:12}', end='')
            for j, nj in enumerate(available):
                val = matrix[i,j].item()
                print(f'{val:10.3f}', end='')
            print()

    print()
    mem.summary()
    print()
    print('GPT: loses facts after context window')
    print('Solar Ring: Phi never forgets — O(N) memory')
    print('='*60)

if __name__ == '__main__':
    demo()
