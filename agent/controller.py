from collections.abc import Callable, Iterable
from collections import deque
from typing import Union, List

from game import *


class PlayerController:
    """
    gemini-1-v5: BFS expansion with Strategic Bidding & Refined Hill Defense.

    Improvements over v4:
    - Strategic Bidding: bid 1 stamina to gain initiative when near hills or powerups.
    - Refined Hill Scoring: even higher priority for contesting hill cells.
    - Better stamina thresholding for powerups and bidding.
    """

    def __init__(self, player_parity: int, time_left: Callable):
        self.parity = player_parity
        self.opp = -player_parity

    def bid(self, board: Board, player_parity: int, time_left: Callable) -> int:
        me = board.get_player(player_parity)
        if me.stamina < 20: return 0
        
        # Bid 1 if near critical targets
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                nr, nc = me.loc.r + dr, me.loc.c + dc
                if 0 <= nr < board.board_size.r and 0 <= nc < board.board_size.c:
                    cell = board.cells[nr][nc]
                    if (cell.hill_id and cell.hill_id != 0) or cell.powerup:
                        return 1
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

        def count_paintable(r, c):
            count = 0
            for dr, dc in DR:
                nr, nc = r + dr, c + dc
                if not valid(nr, nc): continue
                o = cell_owner(nr, nc)
                if o == 0: count += 2
                elif o == player_parity and abs(board.cells[nr][nc].paint_value) < GameConstants.MAX_PAINT_VALUE:
                    count += 1
            return count

        # BFS for best targets
        best_first_dir = None
        best_priority = -999999
        best_depth = 99

        visited = set()
        visited.add((my_r, my_c))
        queue = deque()

        for dr, dc in DR:
            nr, nc = my_r + dr, my_c + dc
            if not valid(nr, nc): continue
            d_opp = mdist(nr, nc, opp_r, opp_c)
            if cell_owner(nr, nc) == self.opp and d_opp <= SAFE_DIST: continue
            if d_opp == 0: continue
            
            d_enum = DIR_MAP[(dr, dc)]
            visited.add((nr, nc))
            queue.append((nr, nc, d_enum, 1))

        while queue:
            r, c, first_dir, depth = queue.popleft()
            if depth > 15: break

            cell = board.cells[r][c]
            priority = -999999

            # 1. Hill Cells
            if cell.hill_id and cell.hill_id != 0:
                hill = board.hills[cell.hill_id]
                if hill.controller_parity != player_parity:
                    if cell.owner_parity == self.opp:
                        priority = 3000 - depth * 20
                    elif cell.owner_parity == 0:
                        priority = 2500 - depth * 20
                    else:
                        priority = 1000 - depth * 20
                else:
                    # Defend hill
                    if cell.owner_parity == self.opp:
                        priority = 2200 - depth * 20
                    elif cell.owner_parity == 0:
                        priority = 1200 - depth * 20
                    elif abs(cell.paint_value) < GameConstants.MAX_PAINT_VALUE:
                        priority = 600 - depth * 15

            # 2. Powerups
            if cell.powerup and stamina < 75:
                p_val = 1800 - depth * 30
                if p_val > priority: priority = p_val

            # 3. Neutral Expansion
            if cell.owner_parity == 0 and priority < -900:
                priority = 900 - depth * 20

            # 4. Take opponent territory
            if cell.owner_parity == self.opp and priority < -900:
                d_opp = mdist(r, c, opp_r, opp_c)
                if d_opp > SAFE_DIST:
                    priority = 400 - depth * 15

            if priority > -900:
                priority += count_paintable(r, c) * 2

            if priority > best_priority or (priority == best_priority and depth < best_depth):
                best_priority = priority
                best_first_dir = first_dir
                best_depth = depth

            if depth < 15:
                for dr, dc in DR:
                    nr, nc = r + dr, c + dc
                    if (nr, nc) in visited: continue
                    if not valid(nr, nc): continue
                    d_opp = mdist(nr, nc, opp_r, opp_c)
                    if cell_owner(nr, nc) == self.opp and d_opp <= SAFE_DIST: continue
                    visited.add((nr, nc))
                    queue.append((nr, nc, first_dir, depth + 1))

        if best_first_dir is None:
            for dr, dc in DR:
                nr, nc = my_r + dr, my_c + dc
                if valid(nr, nc):
                    best_first_dir = DIR_MAP[(dr, dc)]
                    break
            if best_first_dir is None: return Action.Move(Direction.UP)

        actions: List = []
        ddr, ddc = INV_DIR[best_first_dir]
        new_r, new_c = my_r + ddr, my_c + ddc

        if not valid(new_r, new_c):
            return [Action.Move(best_first_dir)]

        reserve = 10
        paint_budget = stamina - reserve
        paint_spent = 0
        painted_cells = set()

        def can_paint(pr, pc):
            if not (0 <= pr < rows and 0 <= pc < cols): return False
            pcell = board.cells[pr][pc]
            if pcell.is_wall or pcell.beacon_parity == player_parity: return False
            if pcell.owner_parity != player_parity and pcell.owner_parity != 0: return False
            if pcell.owner_parity == player_parity and abs(pcell.paint_value) >= GameConstants.MAX_PAINT_VALUE:
                return False
            return True

        # 1. Pre-paint
        pre_candidates = []
        new_neighbors = set()
        for dr, dc in DR: new_neighbors.add((new_r + dr, new_c + dc))
        for dr, dc in DR:
            pr, pc = my_r + dr, my_c + dc
            if (pr, pc) in new_neighbors or (pr, pc) == (new_r, new_c): continue
            if not can_paint(pr, pc): continue
            pscore = 0
            pcell = board.cells[pr][pc]
            if pcell.hill_id and pcell.hill_id != 0: pscore += 200
            if pcell.owner_parity == 0: pscore += 100
            else: pscore += 10
            pre_candidates.append((pscore, pr, pc))

        pre_candidates.sort(key=lambda x: -x[0])
        for _, pr, pc in pre_candidates:
            if paint_spent + GameConstants.PAINT_STAMINA_COST > paint_budget: break
            actions.append(Action.Paint(Location(pr, pc)))
            paint_spent += GameConstants.PAINT_STAMINA_COST
            painted_cells.add((pr, pc))

        # 2. Move
        actions.append(Action.Move(best_first_dir))

        # 3. Post-paint
        post_candidates = []
        for dr, dc in DR:
            pr, pc = new_r + dr, new_c + dc
            if (pr, pc) in painted_cells: continue
            if not can_paint(pr, pc): continue
            pscore = 0
            pcell = board.cells[pr][pc]
            if pcell.hill_id and pcell.hill_id != 0: pscore += 200
            if pr == my_r and pc == my_c: pscore += 150
            if pcell.owner_parity == 0: pscore += 100
            else: pscore += 10
            post_candidates.append((pscore, pr, pc))

        post_candidates.sort(key=lambda x: -x[0])
        for _, pr, pc in post_candidates:
            if paint_spent + GameConstants.PAINT_STAMINA_COST > paint_budget: break
            actions.append(Action.Paint(Location(pr, pc)))
            paint_spent += GameConstants.PAINT_STAMINA_COST

        return actions

    def commentate(self, board: Board, player_parity: int, time_left: Callable) -> str:
        return ""
