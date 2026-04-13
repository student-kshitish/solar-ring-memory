"""
Probability and statistics solver for Solar Ring Memory.
Handles probability, combinations, permutations,
mean/median/mode, variance, z-score, Bayes theorem.
"""

import sys, math
sys.path.insert(0, '.')
from benchmarks.math_reasoning import parse_number, clean
from collections import Counter


def prob_stats_solve(problem: str, question: str) -> str:
    """
    Solve probability and statistics problems.
    Returns string answer.
    """
    tl = (problem + ' ' + question).lower()
    nums = []
    for w in problem.split():
        n = parse_number(w.rstrip('.,;%'))
        if n is not None:
            nums.append(n)
    q = question.lower()

    def _fmt(v):
        if isinstance(v, float):
            # Only collapse to int for whole numbers >= 1 (e.g. 3.0, 54.0)
            # Never collapse small fractions like 0.6 → 0 or 6
            if v.is_integer() and v >= 1.0:
                return str(int(v))
            return str(round(v, 4))
        if isinstance(v, list):
            return str(v[0]) if len(v) == 1 else str(v)
        return str(v)

    if 'probability' in tl or 'chance' in tl or 'likelihood' in tl:
        if 'die' in tl or 'dice' in tl:
            if 'even' in tl: return '0.5'
            if 'odd' in tl: return '0.5'
            if 'not' in q and len(nums) >= 1:
                return _fmt(round(1 - nums[0] / 6, 4))
            if len(nums) >= 1: return _fmt(round(nums[0] / 6, 4))
            return _fmt(round(1 / 6, 4))

        if 'coin' in tl:
            if 'both head' in tl or 'two head' in tl: return '0.25'
            return '0.5'

        if 'card' in tl or 'deck' in tl:
            if 'ace' in tl:   return _fmt(round(4 / 52, 4))
            if 'king' in tl:  return _fmt(round(4 / 52, 4))
            if 'queen' in tl: return _fmt(round(4 / 52, 4))
            if 'heart' in tl: return _fmt(round(13 / 52, 4))
            if 'spade' in tl: return _fmt(round(13 / 52, 4))
            if 'red' in tl:   return _fmt(round(26 / 52, 4))
            if 'black' in tl: return _fmt(round(26 / 52, 4))
            if 'face' in tl:  return _fmt(round(12 / 52, 4))
            if len(nums) >= 1: return _fmt(round(nums[0] / 52, 4))

        if ('and' in q or 'both' in q) and len(nums) >= 2:
            return _fmt(round(nums[0] * nums[1], 4))
        if 'or' in q and len(nums) >= 2:
            return _fmt(round(nums[0] + nums[1] - nums[0] * nums[1], 4))

        # Bag/ball/marble/box problems — total = sum of all counts
        # "Bag has 6 red 4 blue" → favorable/total not nums[0] alone
        if any(w in tl for w in ('bag', 'balls', 'marbles',
                                  'tokens', 'chips', 'box')):
            if len(nums) >= 2:
                total = sum(nums)
                all_colors = ['red', 'blue', 'green', 'yellow',
                              'black', 'white', 'orange', 'purple']
                # Order colors by position in problem text (not fixed list)
                present = sorted(
                    [(c, tl.index(c)) for c in all_colors if c in tl],
                    key=lambda x: x[1]
                )
                problem_colors = [c for c, _ in present]
                for i, color in enumerate(problem_colors):
                    if color in q and i < len(nums):
                        favorable = nums[i]
                        p = favorable / total
                        if 'not' in q:
                            p = 1 - p
                        return _fmt(round(p, 4))
                # fallback: first number / total
                p = nums[0] / total
                if 'not' in q:
                    p = 1 - p
                return _fmt(round(p, 4))

        # "X out of Y" or "X red and Y blue" → X/(X+Y)
        if len(nums) >= 2:
            # check for "X out of Y" pattern
            if 'out of' in tl:
                p = nums[0] / nums[1]
            else:
                # treat first num as favourable, sum as total
                p = nums[0] / sum(nums)
            if 'not' in q: p = 1 - p
            return _fmt(round(p, 4))

    if 'permutation' in tl or 'arrange' in tl or 'arrangement' in tl:
        if len(nums) >= 2:
            n, r = int(nums[0]), int(nums[1])
            return str(math.factorial(n) // math.factorial(n - r))

    if 'combination' in tl or 'choose' in tl or 'select' in tl or 'pick' in tl:
        if len(nums) >= 2:
            n, r = int(nums[0]), int(nums[1])
            return str(math.comb(n, r))

    if 'mean' in tl or 'average' in tl:
        if len(nums) >= 2:
            return _fmt(round(sum(nums) / len(nums), 2))

    if 'median' in tl:
        if len(nums) >= 3:
            s = sorted(nums)
            mid = len(s) // 2
            if len(s) % 2 == 0:
                return _fmt((s[mid - 1] + s[mid]) / 2)
            return _fmt(s[mid])

    if 'mode' in tl:
        if len(nums) >= 3:
            c = Counter(nums)
            return _fmt(c.most_common(1)[0][0])

    if 'range' in tl and 'standard' not in tl and len(nums) >= 2:
        return _fmt(max(nums) - min(nums))

    if 'variance' in tl and len(nums) >= 3:
        mean = sum(nums) / len(nums)
        return _fmt(round(sum((x - mean) ** 2 for x in nums) / len(nums), 2))

    if 'standard deviation' in tl and len(nums) >= 3:
        mean = sum(nums) / len(nums)
        var = sum((x - mean) ** 2 for x in nums) / len(nums)
        return _fmt(round(var ** 0.5, 2))

    if 'factorial' in tl and len(nums) >= 1:
        return str(math.factorial(int(nums[0])))

    if ('z-score' in tl or 'zscore' in tl or 'z score' in tl) and len(nums) >= 3:
        x, mean, sd = nums[0], nums[1], nums[2]
        return _fmt(round((x - mean) / sd, 4)) if sd else '0'

    if ('bayes' in tl or 'bayes theorem' in tl) and len(nums) >= 3:
        p_b_given_a, p_a, p_b = nums[0], nums[1], nums[2]
        return _fmt(round(p_b_given_a * p_a / p_b, 4)) if p_b else '0'

    return 'unknown'
