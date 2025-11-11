from collections import deque


class MCAgent:
    """
    Missionaries and Cannibals Agent
    Q2.1: original BFS version
    Q2.2: added DFS version to show a different search strategy
    """

    def __init__(self):
        pass

    class State:
        def __init__(self, left_m, left_c, right_m, right_c, boat_pos, parent=None):
            self.left_m = left_m
            self.left_c = left_c
            self.right_m = right_m
            self.right_c = right_c
            self.boat_pos = boat_pos  # "left" or "right"
            self.parent = parent

        def is_goal(self, init_m, init_c):
            return (
                self.left_m == 0
                and self.left_c == 0
                and self.right_m == init_m
                and self.right_c == init_c
                and self.boat_pos == "right"
            )

        def is_valid(self):
            # no side can have negative people
            if (
                self.left_m < 0 or self.left_c < 0 or
                self.right_m < 0 or self.right_c < 0
            ):
                return False
            # missionaries cannot be outnumbered on a nonzero side
            if self.left_m > 0 and self.left_c > self.left_m:
                return False
            if self.right_m > 0 and self.right_c > self.right_m:
                return False
            return True

        def __eq__(self, other):
            return (
                self.left_m == other.left_m and
                self.left_c == other.left_c and
                self.right_m == other.right_m and
                self.right_c == other.right_c and
                self.boat_pos == other.boat_pos
            )

        def __hash__(self):
            return hash((
                self.left_m, self.left_c,
                self.right_m, self.right_c,
                self.boat_pos
            ))

    def _successors(self, state: "MCAgent.State"):
        moves = [(2, 0), (0, 2), (1, 1), (1, 0), (0, 1)]
        succs = []
        if state.boat_pos == "left":
            # move from left to right
            for m, c in moves:
                new_state = self.State(
                    state.left_m - m,
                    state.left_c - c,
                    state.right_m + m,
                    state.right_c + c,
                    "right",
                    parent=state
                )
                if new_state.is_valid():
                    succs.append(new_state)
        else:
            # move from right to left
            for m, c in moves:
                new_state = self.State(
                    state.left_m + m,
                    state.left_c + c,
                    state.right_m - m,
                    state.right_c - c,
                    "left",
                    parent=state
                )
                if new_state.is_valid():
                    succs.append(new_state)
        return succs

    # ---------- Q2.1: BFS (original approach) ----------
    def _bfs(self, init_m, init_c):
        start = self.State(init_m, init_c, 0, 0, "left")
        if start.is_goal(init_m, init_c):
            return start

        queue = deque([start])
        visited = set([start])

        while queue:
            node = queue.popleft()
            if node.is_goal(init_m, init_c):
                return node
            for child in self._successors(node):
                if child not in visited:
                    visited.add(child)
                    queue.append(child)
        return None

    # ---------- Q2.2: alternative strategy (DFS) ----------
    def _dfs(self, init_m, init_c):
        start = self.State(init_m, init_c, 0, 0, "left")
        stack = [start]
        visited = set([start])

        while stack:
            node = stack.pop()
            if node.is_goal(init_m, init_c):
                return node
            for child in self._successors(node):
                if child not in visited:
                    visited.add(child)
                    stack.append(child)
        return None

    def _reconstruct_moves(self, goal_state):
        path = []
        current = goal_state
        while current.parent is not None:
            prev = current.parent
            move = (
                abs(current.left_m - prev.left_m),
                abs(current.left_c - prev.left_c)
            )
            path.append(move)
            current = prev
        path.reverse()
        return path

    def solve(self, initial_missionaries, initial_cannibals):
        """
        Default: BFS to match the provided tester notebook.
        """
        result = self._bfs(initial_missionaries, initial_cannibals)
        if result is None:
            return []
        return self._reconstruct_moves(result)

    def solve_with_dfs(self, initial_missionaries, initial_cannibals):
        """
        Extra method to show a different search strategy.
        """
        result = self._dfs(initial_missionaries, initial_cannibals)
        if result is None:
            return []
        return self._reconstruct_moves(result)
