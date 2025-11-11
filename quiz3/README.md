 Quiz 3 – CSCE 580 (Fall 2025)

Student: David Dinh  
Date: Nov 11, 2025  
Folder: `Quiz3/`

KEY: (E) = CONTAINS IN. i dont know how to write contains in. 
-----------

 Q1: Search and Heuristics
---------------------------------------------------------------------------------------
a) What is an admissible heuristic?  
- A heuristic \(h(n)\) is admissible if it never overestimates the true cost to the goal.
- Formally: \(h(n) \leq h^*(n)\) for every node \(n\), where \(h^*(n)\) is the actual optimal cost from \(n\) to a goal.
- Intuition: it is optimistic but idk.
---------------------------------------------------------------------------------------
b) Suppose you are given `h = 0` as the heuristic. Is it admissible?  
- Yes.
- Because 0 can never be greater than the true cost.
- This is basically Uniform-Cost Search / Dijkstra lmfao.
---------------------------------------------------------------------------------------
c) Suppose you are given `h = k` where `k` is any fixed number (e.g. 1). Is it admissible?  
- Not always.
- If `k = 1` but the real remaining cost is 0 (we are at the goal), then `1 > 0` -> overestimation -> not admissible.
- So we cannot say “`h = k` is admissible” in general unless `k = 0`.
---------------------------------------------------------------------------------------
d) Given 3 heuristics `h1, h2, h3`, one of which is admissible:
- Is `h = min(h1, h2, h3)` admissible?  
  - Yes. The min will always be ≤ the admissible one, so it also never overestimates.
- Is `h = max(h1, h2, h3)` admissible?  
  - Not guaranteed.
  - If one is inadmissible (overestimates), the max can also overestimate.
---------------------------------------------------------------------------------------
---

 Q2: Missionaries and Cannibals CHECK CODE 

The problem: move M missionaries and C cannibals from left bank to right bank with a 1–2 person boat, never letting cannibals outnumber missionaries on a nonzero side.

 Q2.1 State representation and search strategy

State representation used (in `MCAgent.py`):

We represent a state as:

- `left_missionaries`
- `left_cannibals`
- `right_missionaries`
- `right_cannibals`
- `boat_position` (E) {`"left"`, `"right"`}

So a state looks like:

```text
(left_m, left_c, right_m, right_c, boat_side)
---------------------------------------------------------------------------------------



Q3: Cryptarithm CSP Formulation

 Problem
```
  T W 0
+ T W 0
--------
  F O U R
```
--------------------------------------------------------------------------------------------------------------
 a) Variables, Domains, and Constraints

Variables:  
`{T, W, F, O, U, R}`

Domains:  
- `T (E) {1–9}` (cannot be 0 because it's a leading digit)  
- `F (E) {1–9}` (leading digit of result)  
- `W, O, U, R (E) {0–9}`  
- All values must be distinct (All Different constraint)

Constraints:  
1. Column 1 (units): `0 + 0 = R` -> R = 0  
2. Column 2 (tens): `W + W = U + 10*c1` -> 2W = U + 10*c1, `c1 (E) {0,1}`  
3. Column 3 (hundreds): `T + T + c1 = O + 10*c2` -> 2T + c1 = O + 10*c2, `c2 (E) {0,1}`  
4. Column 4 (thousands): `F = c2`  
5. AllDifferent(T, W, F, O, U, R)
--------------------------------------------------------------------------------------------------------------

 b) Using Node/Arc Consistency (Simplification)

Goal: Simplify domains before search using constraint propagation.

Pseudo-code:

```
1. Initialize domains for all variables.
2. R = 0 -> domain(R) = {0}
3. Enforce AllDifferent -> remove 0 from other domains.
4. For constraint 2W = U + 10*c1:
    for each W in domain(W):
        for c1 in {0,1}:
            U = 2W - 10*c1
            keep (W,U) pairs where U (E) [0,9]
5. For constraint 2T + c1 = O + 10*c2:
    for each T in domain(T):
        for c1,c2 (E) {0,1}:
            O = 2T + c1 - 10*c2
            keep values where O (E) [0,9]
6. F = c2 -> domain(F) = {1} since F is a leading digit.
7. Repeat arc consistency until no further reductions.
```

This reduces domain size significantly and makes later backtracking easier.  
Even if a full solution is not derived, consistency pruning simplifies the CSP substantially.

---


