from collections.abc import Callable, Iterable
from collections import deque
from typing import Union, List

from game import *


class PlayerController:
    """
    BFS expansion agent v3: aggressive expansion with paint-behind strategy.

    Key improvements over v2:
    - Paint cell we just left (behind us) for efficient territory filling
    - More aggressive paint budget (reserve only 10 stamina)
    - Better hill targeting with weighted BFS
    - Smarter direction tiebreaking (prefer directions with more paintable neighbors)
    """

    def __init__(self, player_parity: int, time_left: Callable):
        self.parity = player_parity
        self.opp = -player_parity

    def bid(self, board: Board, player_parity: int, time_left: Callable) -> int:
        return 0

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

        SAFE_DIST = 5
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

        # Count paintable neighbors from a position (for direction tiebreaking)
        def count_paintable(r, c):
            count = 0
            for dr, dc in DR:
                nr, nc = r + dr, c + dc
                if not valid(nr, nc):
                    continue
                o = cell_owner(nr, nc)
                if o == 0:
                    count += 2  # neutral is best
                elif o == player_parity and abs(board.cells[nr][nc].paint_value) < GameConstants.MAX_PAINT_VALUE:
                    count += 1  # can reinforce
            return count

        # BFS to find best direction
        best_first_dir = None
        best_priority = -999999

        visited = set()
        visited.add((my_r, my_c))
        queue = deque()

        # Seed with immediate neighbors
        candidates = []
        for dr, dc in DR:
            nr, nc = my_r + dr, my_c + dc
            if not valid(nr, nc):
                continue
            d_opp = mdist(nr, nc, opp_r, opp_c)
            # Hard safety: don't step on enemy cell near opponent
            if cell_owner(nr, nc) == self.opp and d_opp <= SAFE_DIST:
                continue
            # Don't walk into opponent
            if d_opp == 0:
                continue
            d_enum = DIR_MAP[(dr, dc)]
            visited.add((nr, nc))
            queue.append((nr, nc, d_enum, 1))
            candidates.append((nr, nc, d_enum))

        while queue:
            r, c, first_dir, depth = queue.popleft()
            if depth > 15:
                break

            cell = board.cells[r][c]
            priority = -999999

            # Score this cell as a target
            if cell.hill_id and cell.hill_id != 0:
                hill = board.hills[cell.hill_id]
                if hill.controller_parity != player_parity:
                    if cell.owner_parity != player_parity:
                        priority = 2000 - depth * 20  # uncaptured hill, unpainted
                    else:
                        priority = 1000 - depth * 20  # uncaptured hill, already painted
                elif cell.owner_parity == 0:
                    priority = 800 - depth * 15  # captured hill, neutral cell (reinforce)

            if cell.owner_parity == 0 and priority < -900:
                priority = 900 - depth * 20  # neutral expansion

            if cell.owner_parity == self.opp and priority < -900:
                d_opp = mdist(r, c, opp_r, opp_c)
                if d_opp > SAFE_DIST:
                    priority = 300 - depth * 15

            # Tiebreak: prefer directions with more paintable neighbors
            if priority > -900:
                priority += count_paintable(r, c) * 2

            if priority > best_priority:
                best_priority = priority
                best_first_dir = first_dir

            if depth < 15:
                for dr, dc in DR:
                    nr, nc = r + dr, c + dc
                    if (nr, nc) in visited:
                        continue
                    if not valid(nr, nc):
                        continue
                    d_opp = mdist(nr, nc, opp_r, opp_c)
                    if cell_owner(nr, nc) == self.opp and d_opp <= SAFE_DIST:
                        continue
                    visited.add((nr, nc))
                    queue.append((nr, nc, first_dir, depth + 1))

        # Fallback
        if best_first_dir is None:
            for dr, dc in DR:
                nr, nc = my_r + dr, my_c + dc
                if valid(nr, nc):
                    best_first_dir = DIR_MAP[(dr, dc)]
                    break
            if best_first_dir is None:
                return Action.Move(Direction.UP)

        # Build action list
        actions: List = []
        ddr, ddc = INV_DIR[best_first_dir]
        new_r, new_c = my_r + ddr, my_c + ddc

        if not valid(new_r, new_c):
            return [Action.Move(best_first_dir)]

        # Check for double-move opportunity: 2nd cell in same direction
        stop2_r, stop2_c = new_r + ddr, new_c + ddc
        # Double move when: high stamina, 2nd cell valid, 2nd cell not enemy near opp
        use_double = False
        if stamina >= 70 and valid(stop2_r, stop2_c):
            d_opp2 = mdist(stop2_r, stop2_c, opp_r, opp_c)
            if not (cell_owner(stop2_r, stop2_c) == self.opp and d_opp2 <= SAFE_DIST):
                if d_opp2 > 0:  # don't collide
                    use_double = True

        reserve = 10
        # Account for extra move cost in budget
        extra_move_cost = 10 if use_double else 0
        paint_budget = stamina - reserve - extra_move_cost
        paint_spent = 0
        painted_cells = set()

        def collect_paint_candidates(cr, cc, old_r, old_c):
            """Collect paintable cells adjacent to (cr, cc)."""
            cands = []
            for dr, dc in DR:
                pr, pc = cr + dr, cc + dc
                if (pr, pc) in painted_cells:
                    continue
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
                if pr == old_r and pc == old_c:
                    pscore += 150
                if pcell.owner_parity == 0:
                    pscore += 100
                else:
                    pscore += 10
                cands.append((pscore, pr, pc))
            cands.sort(key=lambda x: -x[0])
            return cands

        # 1. First move
        actions.append(Action.Move(best_first_dir))

        # 2. Paint at first stop
        for _, pr, pc in collect_paint_candidates(new_r, new_c, my_r, my_c):
            if paint_spent + GameConstants.PAINT_STAMINA_COST > paint_budget:
                break
            actions.append(Action.Paint(Location(pr, pc)))
            paint_spent += GameConstants.PAINT_STAMINA_COST
            painted_cells.add((pr, pc))

        # 3. Optional second move + paint
        if use_double:
            actions.append(Action.Move(best_first_dir))
            for _, pr, pc in collect_paint_candidates(stop2_r, stop2_c, new_r, new_c):
                if paint_spent + GameConstants.PAINT_STAMINA_COST > paint_budget:
                    break
                actions.append(Action.Paint(Location(pr, pc)))
                paint_spent += GameConstants.PAINT_STAMINA_COST
                painted_cells.add((pr, pc))

        return actions

    def commentate(self, board: Board, player_parity: int, time_left: Callable) -> str:
        return ""
