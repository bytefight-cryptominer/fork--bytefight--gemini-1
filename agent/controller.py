from collections.abc import Callable, Iterable
from collections import deque
from typing import Union, List

from game import *


class PlayerController:
    """
    v45: Forecast-based direction selection + BFS fallback.
    Simulates each direction using board.forecast_turn to pick the
    move with the best actual outcome. Falls back to BFS for
    long-range targeting. Keeps erase-step from v23.
    """

    def __init__(self, player_parity: int, time_left: Callable):
        self.parity = player_parity
        self.opp = -player_parity

    def bid(self, board: Board, player_parity: int, time_left: Callable) -> int:
        return 0

    def _build_actions(self, board, player_parity, direction, my_r, my_c, stamina):
        """Build a full action set (move + paint) for a given direction."""
        rows, cols = board.board_size.r, board.board_size.c
        INV_DIR = {Direction.UP: (-1, 0), Direction.DOWN: (1, 0),
                   Direction.LEFT: (0, -1), Direction.RIGHT: (0, 1)}
        DR = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        ddr, ddc = INV_DIR[direction]
        new_r, new_c = my_r + ddr, my_c + ddc
        if not (0 <= new_r < rows and 0 <= new_c < cols) or board.cells[new_r][new_c].is_wall:
            return None

        actions = [Action.Move(direction)]
        reserve = 10
        budget = stamina - reserve
        spent = 0

        candidates = []
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
            candidates.append((pscore, pr, pc))

        candidates.sort(key=lambda x: -x[0])
        for _, pr, pc in candidates:
            if spent + GameConstants.PAINT_STAMINA_COST > budget:
                break
            actions.append(Action.Paint(Location(pr, pc)))
            spent += GameConstants.PAINT_STAMINA_COST

        return actions

    def _eval_board(self, board, player_parity):
        """Evaluate board state: territory + hills + local control."""
        me = board.get_player(player_parity)
        rows, cols = board.board_size.r, board.board_size.c
        territory = 0
        for row in board.cells:
            for cell in row:
                if cell.owner_parity == player_parity:
                    territory += 1
        hills = len(me.controlled_hills) * 200
        local = 0
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                nr, nc = me.loc.r + dr, me.loc.c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    if board.cells[nr][nc].owner_parity == player_parity:
                        local += 2
        return territory + hills + local + me.stamina * 0.1

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

        # --- Collision pursuit ---
        for dr, dc in DR:
            nr, nc = my_r + dr, my_c + dc
            if valid(nr, nc) and nr == opp_r and nc == opp_c:
                if cell_owner(nr, nc) != self.opp:
                    return [Action.Move(DIR_MAP[(dr, dc)])]

        # --- BFS for long-range hill targeting ---
        bfs_dir = None
        bfs_priority = -999999
        visited = set()
        visited.add((my_r, my_c))
        queue = deque()

        for dr, dc in DR:
            nr, nc = my_r + dr, my_c + dc
            if not valid(nr, nc):
                continue
            d_opp = mdist(nr, nc, opp_r, opp_c)
            if cell_owner(nr, nc) == self.opp and d_opp <= SAFE_DIST:
                continue
            if d_opp == 0:
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

            if cell.powerup:
                pup_val = 1500 - depth * 25
                if stamina < 60:
                    pup_val += 300
                priority = max(priority, pup_val)

            if priority > bfs_priority:
                bfs_priority = priority
                bfs_dir = first_dir

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

        # --- Forecast-based direction selection ---
        # If BFS found a high-priority target (hill/powerup), use BFS direction
        if bfs_priority >= 1500 and bfs_dir is not None:
            actions = self._build_actions(board, player_parity, bfs_dir, my_r, my_c, stamina)
            if actions:
                return actions

        # Otherwise, simulate all 4 directions and pick best outcome
        best_actions = None
        best_eval = -999999
        for direction in Direction.cardinals():
            actions = self._build_actions(board, player_parity, direction, my_r, my_c, stamina)
            if actions is None:
                continue
            # Check safety
            ddr, ddc = INV_DIR[direction]
            nr, nc = my_r + ddr, my_c + ddc
            d_opp = mdist(nr, nc, opp_r, opp_c)
            if cell_owner(nr, nc) == self.opp and d_opp <= SAFE_DIST:
                continue
            if d_opp == 0:
                continue
            # Simulate
            try:
                sim_board, ok = board.forecast_turn(player_parity, actions)
                if ok:
                    score = self._eval_board(sim_board, player_parity)
                    if score > best_eval:
                        best_eval = score
                        best_actions = actions
            except Exception:
                pass

        if best_actions:
            return best_actions

        # Fallback to BFS direction or any valid move
        if bfs_dir is not None:
            actions = self._build_actions(board, player_parity, bfs_dir, my_r, my_c, stamina)
            if actions:
                return actions

        for dr, dc in DR:
            nr, nc = my_r + dr, my_c + dc
            if valid(nr, nc):
                return [Action.Move(DIR_MAP[(dr, dc)])]
        return [Action.Move(Direction.UP)]

    def commentate(self, board: Board, player_parity: int, time_left: Callable) -> str:
        return "v45c"
