from collections.abc import Callable, Iterable
from collections import deque
from typing import Union, List

import numpy as np

from game import *


class PlayerController:
    """
    v39: Numpy-enhanced agent with Voronoi-aware BFS.
    Uses numpy distance maps to prefer directions that maximize
    our reachable territory (cells closer to us than opponent).
    Keeps erase-step from v23.
    """

    def __init__(self, player_parity: int, time_left: Callable):
        self.parity = player_parity
        self.opp = -player_parity

    def bid(self, board: Board, player_parity: int, time_left: Callable) -> int:
        return 0

    def _bfs_dist(self, board, start_r, start_c, rows, cols):
        """BFS distance map from a position, respecting walls."""
        dist = np.full((rows, cols), 9999, dtype=np.int32)
        dist[start_r][start_c] = 0
        q = deque([(start_r, start_c)])
        while q:
            r, c = q.popleft()
            d = dist[r][c]
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < rows and 0 <= nc < cols and not board.cells[nr][nc].is_wall and dist[nr][nc] > d+1:
                    dist[nr][nc] = d + 1
                    q.append((nr, nc))
        return dist

    def play(
        self,
        board: Board,
        player_parity: int,
        time_left: Callable,
    ) -> Union[Action.Move, Action.Paint, Iterable[Action.Move | Action.Paint]]:
        me = board.get_player(player_parity)
        opp = board.get_opponent(player_parity)
        rows, cols = board.board_size.r, board.board_size.c
        my_r, my_c = me.loc.r, me.loc.c
        opp_r, opp_c = opp.loc.r, opp.loc.c
        stamina = me.stamina
        opp_stamina = opp.stamina

        DR = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        DIR_MAP = {(-1, 0): Direction.UP, (1, 0): Direction.DOWN,
                   (0, -1): Direction.LEFT, (0, 1): Direction.RIGHT}
        INV_DIR = {Direction.UP: (-1, 0), Direction.DOWN: (1, 0),
                   Direction.LEFT: (0, -1), Direction.RIGHT: (0, 1)}

        def valid(r, c):
            return 0 <= r < rows and 0 <= c < cols and not board.cells[r][c].is_wall

        def mdist(r1, c1, r2, c2):
            return abs(r1 - r2) + abs(c1 - c2)

        def cell_owner(r, c):
            return board.cells[r][c].owner_parity

        stamina_diff = stamina - opp_stamina
        if stamina_diff > 30:
            SAFE_DIST = 3
        elif stamina_diff < -30:
            SAFE_DIST = 6
        else:
            SAFE_DIST = 5

        def count_paintable(r, c):
            count = 0
            for dr, dc in DR:
                nr, nc = r + dr, c + dc
                if not valid(nr, nc):
                    continue
                o = cell_owner(nr, nc)
                if o == 0:
                    count += 2
                elif o == player_parity and abs(board.cells[nr][nc].paint_value) < GameConstants.MAX_PAINT_VALUE:
                    count += 1
            return count

        # --- Erase step for hill cells with opponent paint ---
        if stamina >= 55:
            for dr, dc in DR:
                nr, nc = my_r + dr, my_c + dc
                if not valid(nr, nc):
                    continue
                ecell = board.cells[nr][nc]
                if (ecell.hill_id and ecell.hill_id != 0 and
                    ecell.owner_parity == self.opp):
                    if nr == opp_r and nc == opp_c:
                        continue
                    return [Action.Move(DIR_MAP[(dr, dc)], move_type=MoveType.ERASE)]

        # --- Compute distance maps for Voronoi scoring ---
        my_dist = self._bfs_dist(board, my_r, my_c, rows, cols)
        opp_dist = self._bfs_dist(board, opp_r, opp_c, rows, cols)

        # --- BFS with Voronoi-enhanced scoring ---
        best_first_dir = None
        best_priority = -999999

        visited = set()
        visited.add((my_r, my_c))
        queue = deque()

        for dr, dc in DR:
            nr, nc = my_r + dr, my_c + dc
            if not valid(nr, nc):
                continue
            d_opp = mdist(nr, nc, opp_r, opp_c)

            if nr == opp_r and nc == opp_c:
                if cell_owner(nr, nc) == self.opp:
                    continue
                return Action.Move(DIR_MAP[(dr, dc)])

            if cell_owner(nr, nc) == self.opp and d_opp <= SAFE_DIST:
                continue

            visited.add((nr, nc))
            queue.append((nr, nc, DIR_MAP[(dr, dc)], 1))

        while queue:
            r, c, first_dir, depth = queue.popleft()
            if depth > 20:
                break

            cell = board.cells[r][c]
            priority = -999999

            if cell.hill_id and cell.hill_id != 0:
                hill = board.hills[cell.hill_id]
                if hill.controller_parity != player_parity:
                    if cell.owner_parity != player_parity:
                        priority = 2000 - depth * 20
                    else:
                        priority = 1000 - depth * 20
                elif cell.owner_parity == 0:
                    priority = 800 - depth * 15

            if cell.powerup:
                pup_val = 1500 - depth * 25
                if stamina < 60:
                    pup_val += 300
                priority = max(priority, pup_val)

            if cell.owner_parity == 0 and priority < -900:
                priority = 900 - depth * 20
                # Voronoi bonus: prefer neutral cells we can reach before opponent
                if my_dist[r][c] < opp_dist[r][c]:
                    priority += 50  # we'll reach this first
                elif my_dist[r][c] > opp_dist[r][c] + 2:
                    priority -= 100  # opponent reaches well before us

            if cell.owner_parity == self.opp and priority < -900:
                d_opp = mdist(r, c, opp_r, opp_c)
                if d_opp > SAFE_DIST:
                    priority = 300 - depth * 15

            if priority > -900:
                priority += count_paintable(r, c) * 2

            if priority > best_priority:
                best_priority = priority
                best_first_dir = first_dir

            if depth < 20:
                for dr, dc in DR:
                    nr, nc = r + dr, c + dc
                    if (nr, nc) in visited or not valid(nr, nc):
                        continue
                    d_opp = mdist(nr, nc, opp_r, opp_c)
                    if cell_owner(nr, nc) == self.opp and d_opp <= SAFE_DIST:
                        continue
                    visited.add((nr, nc))
                    queue.append((nr, nc, first_dir, depth + 1))

        if best_first_dir is None:
            for dr, dc in DR:
                nr, nc = my_r + dr, my_c + dc
                if valid(nr, nc):
                    best_first_dir = DIR_MAP[(dr, dc)]
                    break
            if best_first_dir is None:
                return Action.Move(Direction.UP)

        actions: List = []
        actions.append(Action.Move(best_first_dir))

        ddr, ddc = INV_DIR[best_first_dir]
        new_r, new_c = my_r + ddr, my_c + ddc

        if not valid(new_r, new_c):
            return actions

        reserve = 25 if board.turn_count > 1400 else 10
        paint_budget = stamina - reserve
        paint_spent = 0

        paint_candidates = []
        for dr, dc in DR:
            pr, pc = new_r + dr, new_c + dc
            if not (0 <= pr < rows and 0 <= pc < cols):
                continue
            pcell = board.cells[pr][pc]
            if pcell.is_wall or pcell.beacon_parity == player_parity:
                continue
            if pcell.owner_parity != player_parity and pcell.owner_parity != 0:
                continue
            if pcell.owner_parity == player_parity and abs(pcell.paint_value) >= GameConstants.MAX_PAINT_VALUE:
                continue

            pscore = 0
            if pcell.hill_id and pcell.hill_id != 0:
                pscore += 200
            if pr == my_r and pc == my_c:
                pscore += 150
            if pcell.owner_parity == 0:
                pscore += 100
            else:
                pscore += 10
            paint_candidates.append((pscore, pr, pc))

        paint_candidates.sort(key=lambda x: -x[0])
        for _, pr, pc in paint_candidates:
            if paint_spent + GameConstants.PAINT_STAMINA_COST > paint_budget:
                break
            actions.append(Action.Paint(Location(pr, pc)))
            paint_spent += GameConstants.PAINT_STAMINA_COST

        return actions

    def commentate(self, board: Board, player_parity: int, time_left: Callable) -> str:
        return ""
