from collections.abc import Callable, Iterable
from collections import deque
from typing import Union, List

from game import *


class PlayerController:
    """
    v61: Forecast-based direction selection + BFS depth 35 fallback.
    Simulates each direction using board.forecast_turn to pick the
    move with the best actual outcome. Falls back to BFS for
    long-range targeting (hills/powerups). Includes enemy traversal
    at depth > 1 from v52/v58.
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
        reserve = 25 if board.turn_count > 1400 else 10
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
        paint_strength = 0
        for row in board.cells:
            for cell in row:
                if cell.owner_parity == player_parity:
                    territory += 1
                    paint_strength += abs(cell.paint_value)
        hills = len(me.controlled_hills) * 200
        local = 0
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                nr, nc = me.loc.r + dr, me.loc.c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    if board.cells[nr][nc].owner_parity == player_parity:
                        local += 2
        return territory + paint_strength * 0.1 + hills + local + me.stamina * 0.1

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

        avg_dim = (rows + cols) / 2
        base_safe = max(2, min(5, int(avg_dim / 4)))
        
        stamina_diff = stamina - opp_stamina
        if stamina_diff > 30:
            SAFE_DIST = max(1, base_safe - 2)
        elif stamina_diff < -30:
            SAFE_DIST = base_safe + 1
        else:
            SAFE_DIST = base_safe

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
                    if stamina >= 65:
                        for dr2, dc2 in DR:
                            off_r, off_c = nr + dr2, nc + dc2
                            if valid(off_r, off_c) and cell_owner(off_r, off_c) != self.opp:
                                if mdist(off_r, off_c, opp_r, opp_c) > SAFE_DIST:
                                    return [
                                        Action.Move(DIR_MAP[(dr, dc)], move_type=MoveType.ERASE),
                                        Action.Move(DIR_MAP[(dr2, dc2)]),
                                        Action.Paint(Location(nr, nc))
                                    ]
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
            visited.add((nr, nc))
            queue.append((nr, nc, DIR_MAP[(dr, dc)], 1))

        while queue:
            r, c, first_dir, depth = queue.popleft()
            if depth > 35:
                break
            cell = board.cells[r][c]
            priority = -999999

            if cell.hill_id and cell.hill_id != 0:
                hill = board.hills[cell.hill_id]
                if hill.controller_parity == self.opp:
                    if cell.owner_parity != player_parity:
                        priority = 2500 - depth * 20
                    else:
                        priority = 1200 - depth * 20
                elif hill.controller_parity == 0:
                    if cell.owner_parity != player_parity:
                        priority = 2000 - depth * 20
                    else:
                        priority = 1000 - depth * 20
                elif cell.owner_parity == 0:
                    priority = 800 - depth * 15

            if cell.powerup:
                pup_val = 1500 - depth * 25
                stamina_ratio = max(0, min(1, stamina / 100))
                pup_val += int(1000 * (1 - stamina_ratio))
                priority = max(priority, pup_val)
                
            if cell.owner_parity == 0 and priority < -900:
                priority = 900 - depth * 20

            if cell.owner_parity == self.opp and priority < -900:
                d_opp = mdist(r, c, opp_r, opp_c)
                if d_opp > SAFE_DIST:
                    priority = 1100 - depth * 20

            if priority > -900:
                priority += count_paintable(r, c) * 2

            if priority > bfs_priority:
                bfs_priority = priority
                bfs_dir = first_dir

            if depth < 35:
                for dr, dc in DR:
                    nr, nc = r + dr, c + dc
                    if (nr, nc) in visited or not valid(nr, nc):
                        continue
                    # Enemy traversal allowed at depth > 1 (we don't check SAFE_DIST here)
                    visited.add((nr, nc))
                    queue.append((nr, nc, first_dir, depth + 1))

        # --- Forecast-based direction selection ---
        # If BFS found a high-priority target (hill/powerup), use BFS direction
        # Modified threshold to account for depth 35: (2000 - 35*20 = 1300)
        if bfs_priority >= 1200 and bfs_dir is not None:
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
            # Simulate 1st ply
            try:
                sim_board_1, ok_1 = board.forecast_turn(player_parity, actions)
                if not ok_1:
                    continue
                
                # Evaluate 2nd ply
                max_ply2_eval = -999999
                sim_me = sim_board_1.get_player(player_parity)
                sim_my_r, sim_my_c = sim_me.loc.r, sim_me.loc.c
                sim_stamina = sim_me.stamina
                
                sim_opp = sim_board_1.get_opponent(player_parity)
                sim_opp_r, sim_opp_c = sim_opp.loc.r, sim_opp.loc.c
                
                for dir2 in Direction.cardinals():
                    actions2 = self._build_actions(sim_board_1, player_parity, dir2, sim_my_r, sim_my_c, sim_stamina)
                    if actions2 is None:
                        continue
                        
                    ddr2, ddc2 = INV_DIR[dir2]
                    nr2, nc2 = sim_my_r + ddr2, sim_my_c + ddc2
                    d_opp2 = mdist(nr2, nc2, sim_opp_r, sim_opp_c)
                    
                    if sim_board_1.cells[nr2][nc2].owner_parity == self.opp and d_opp2 <= SAFE_DIST:
                        continue
                    if d_opp2 == 0:
                        continue
                        
                    try:
                        sim_board_2, ok_2 = sim_board_1.forecast_turn(player_parity, actions2)
                        if ok_2:
                            score = self._eval_board(sim_board_2, player_parity)
                            if score > max_ply2_eval:
                                max_ply2_eval = score
                    except Exception:
                        pass
                
                if max_ply2_eval == -999999:
                    max_ply2_eval = self._eval_board(sim_board_1, player_parity)
                    
                if max_ply2_eval > best_eval:
                    best_eval = max_ply2_eval
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
        return ""
