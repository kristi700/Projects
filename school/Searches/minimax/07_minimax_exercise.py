from itertools import groupby
from typing import Callable, Optional

import numpy as np
import FreeSimpleGUI as sg # type: ignore

from framework.gui import BoardGUI

BOARD_SIZE = 3
NEEDED_TO_WIN = 3
BLANK_IMAGE_PATH = 'tiles/gomoku_blank_scaled.png'
X_IMAGE_PATH = 'tiles/gomoku_X_scaled.png'
O_IMAGE_PATH = 'tiles/gomoku_O_scaled.png'


# Gomoku (5 in a row): O is the maximizing player (the player controlled by
# Artificial Intelligence)

# The BOARD will be represented as a numpy array
# The fields are:
# 0: empty
# 1: X
# 2: O

class GomokuBoard:
    def __init__(self, n: int = BOARD_SIZE, board: Optional[np.ndarray] = None):
        if board is None:
            self.n = n
            self.board = np.zeros((self.n, self.n), dtype=int)
        else:
            self.board = board
            self.n = board.shape[0]

    def winner(self) -> Optional[int]:
        """Return the winner of the BOARD from the point of view of the X player.

        Return winners:
        1: X wins
        2: O wins
        0: draw
        None: game is still ongoing
        """

        # check the rows
        for player, length in ((key, len(list(group))) for row in self.board
                               for key, group in groupby(row) if key != 0):
            if length >= NEEDED_TO_WIN:
                return player
        # check the columns
        for player, length in ((key, len(list(group))) for row in self.board.T
                               for key, group in groupby(row) if key != 0):
            if length >= NEEDED_TO_WIN:
                return player
        # check the diagonals (use self.board.diagonal())
        # upper left to lower right
        diagonals = (self.board.diagonal(i) for i in range(
            NEEDED_TO_WIN-self.board.shape[0], self.board.shape[0] - NEEDED_TO_WIN + 1))
        for player, length in ((key, len(list(group))) for diag in diagonals
                               for key, group in groupby(diag) if key != 0):
            if length >= NEEDED_TO_WIN:
                return player
        # lower left to upper right
        diagonals = (np.flipud(self.board).diagonal(i) for i in range(
            NEEDED_TO_WIN-self.board.shape[0], self.board.shape[0] - NEEDED_TO_WIN + 1))
        for player, length in ((key, len(list(group))) for diag in diagonals
                               for key, group in groupby(diag) if key != 0):
            if length >= NEEDED_TO_WIN:
                return player
        # check the draw
        if len(np.where(self.board == 0)[0]) == 0:
            return 0
        # otherwise, return None
        return None

    def copy(self):
        return GomokuBoard(board=self.board.copy())

    def possible_steps(self) -> np.ndarray:
        return np.argwhere(self.board == 0)

    def next_boards(self, player: int):
        """Produces a list of boards for all the possible next steps of
        player."""
        boards = []
        for row_ind, col_ind in self.possible_steps():
            next_board = self.copy()
            next_board[row_ind, col_ind] = player
            boards.append(next_board)
        return boards

    def __getitem__(self, index: tuple[int, int] | int) -> int:
        if isinstance(index, tuple):
            i, j = index
            return self.board[i][j]
        else:
            return self.board[index]

    def __setitem__(self, index: tuple[int, int] | int, item: int) -> None:
        if isinstance(index, tuple):
            i, j = index
            self.board[i][j] = item
        else:
            self.board[index] = item

    def reset(self) -> None:
        self.board = np.zeros((self.n, self.n), dtype=int)


def switch_player(player: int) -> int:
    return 3 - player


def random_move(board: GomokuBoard) -> None:
    """This strategy just makes a random allowed move on the board."""
    zero_ind = np.where(board.board == 0)
    index = np.random.randint(len(zero_ind[0]))
    row, col = zero_ind[0][index], zero_ind[1][index]
    board[row,col] = 2

# YOUR CODE HERE

MiniMaxReturnType = tuple[Optional[int], Optional[list[GomokuBoard]]]
# A tuple of an integer (or None) and a list if boards (or None)
# The list of boards contains the solution and
#   the integer describes the evaluation value of that solution

#TODO: implement the minimax algorithm
def minimax(board: GomokuBoard) -> None:
    """Do the next step computed by the minimax algorithm."""
    def value(board: GomokuBoard, player: int) -> MiniMaxReturnType:
        if player == 2:
            return maximize(board, player)
        return minimize(board, player)

    def minimize(board: GomokuBoard, player: int) -> MiniMaxReturnType:
        winner = board.winner()
        if winner is not None:
            if winner==2:
                return (float('inf'), [board])
            elif winner==1:
                return (float('-inf'), [board])
            else:
                return (0, [board])
        
        min_value = float('inf')
        min_solution = None

        for possible_step in board.possible_steps():
            current_board = board.copy()
            current_board[possible_step[0], possible_step[1]] = player
            value, solution = maximize(current_board, switch_player(player))

            if min_value >= value and solution is not None:
                min_value = value
                min_solution = [current_board] + solution
        return min_value, min_solution

        # Idea: Check MiniMaxReturnType as a hint for what to return!
        #       Is the game over? Do we have a winner? Is it a draw?
        #       If it is over, the minimax value should be a positive or 
        #           a negative number, depending on the winner; or with 0 if draw
        #       If it is not over, check the next possible states (boards)!
        #       Finally, return both the min value and the corresponding solution

    def maximize(board: GomokuBoard, player: int) -> MiniMaxReturnType:
        winner = board.winner()
        if winner is not None:
            if winner==2:
                return (float('inf'), [board])
            elif winner==1:
                return (float('-inf'), [board])
            else:
                return (0, [board])
        
        max_value = float('-inf')
        max_solution = None

        for possible_step in board.possible_steps():
            current_board = board.copy()
            current_board[possible_step[0], possible_step[1]] = player
            value, solution = minimize(current_board, switch_player(player))
            if max_value <= value and solution is not None:
                max_value = value
                max_solution = [current_board] + solution
        return max_value, max_solution
        # Idea: very similar approach

    _, solution = value(board, 2)
    if solution is not None:
        board.board = solution[0].board


def alpha_beta(board: GomokuBoard) -> None:
    """Do the next step computed by the alpha-beta search."""
    def value(board: GomokuBoard, player: int, alpha, beta) -> MiniMaxReturnType:
        if player == 2:
            return maximize(board, player, alpha, beta)
        return minimize(board, player, alpha, beta)

    def minimize(board: GomokuBoard, player: int, alpha, beta) -> MiniMaxReturnType:
        winner = board.winner()
        if winner is not None:
            if winner==2:
                return (float('inf'), [board])
            elif winner==1:
                return (float('-inf'), [board])
            else:
                return (0, [board])
        
        min_value = float('inf')
        min_solution = None

        for possible_step in board.possible_steps():
            current_board = board.copy()
            current_board[possible_step[0], possible_step[1]] = player
            value, solution = maximize(current_board, switch_player(player), alpha, beta)

            if min_value >= value and solution is not None:
                min_value = value
                min_solution = [current_board] + solution

            beta = min(beta, min_value)
            if beta <= alpha:
                break             

        return min_value, min_solution
    
    def maximize(board: GomokuBoard, player: int, alpha, beta) -> MiniMaxReturnType:
        winner = board.winner()
        if winner is not None:
            if winner==2:
                return (float('inf'), [board])
            elif winner==1:
                return (float('-inf'), [board])
            else:
                return (0, [board])
        
        max_value = float('-inf')
        max_solution = None

        for possible_step in board.possible_steps():
            current_board = board.copy()
            current_board[possible_step[0], possible_step[1]] = player
            value, solution = minimize(current_board, switch_player(player), alpha, beta)
            if max_value <= value and solution is not None:
                max_value = value
                max_solution = [current_board] + solution

            alpha = max(alpha, max_value)
            if alpha >= beta:
                break
        return max_value, max_solution

    alpha = float('-inf')
    beta = float('inf')
    _, solution = value(board, 2, alpha, beta)
    if solution is not None:
        board.board = solution[0].board
    # Idea: Implement the alpha-beta pruning in the minimax algorithm
    #       You can start by just copying your minimax algorithm here
    #       Introduce alpha and beta as function parameters in each 
    #       implemented function


# END OF YOUR CODE


class GameLogic:

    def __init__(self, board: GomokuBoard, move_ai: Callable):
        self.board = board
        self.move_ai = move_ai
        self.current_player = 1

    def play(self, row_ind, col_ind):
        if self.board[row_ind][col_ind] != 0:
            return None
        self.board[row_ind][col_ind] = self.current_player
        winner = self.board.winner()
        if winner is not None:
            return self.board.winner()
        self.switch_player()
        self.move_ai(self.board)
        self.switch_player()
        winner = self.board.winner()
        if winner is not None:
            return self.board.winner()

    def switch_player(self):
        self.current_player = switch_player(self.current_player)

    def reset(self):
        self.board.reset()
        self.current_player = 1


sg.ChangeLookAndFeel('SystemDefault')

algorithms = {
    'Random move': random_move,
    'Minimax': minimax,
    'Alpha-beta search': alpha_beta
}

GOMOKU_DRAW_DICT = {
    0: ('', ('black', 'lightgrey'), BLANK_IMAGE_PATH),
    1: ('', ('black', 'lightgrey'), X_IMAGE_PATH),
    2: ('', ('black', 'lightgrey'), O_IMAGE_PATH),
}

BOARD = GomokuBoard(BOARD_SIZE)
GAME_LOGIC = GameLogic(BOARD, random_move)
BOARD_GUI = BoardGUI(BOARD, GOMOKU_DRAW_DICT) #type:ignore


def create_window(board_gui):
    layout = [[sg.Column(board_gui.board_layout)],
              [
                  sg.Frame('Algorithm settings',
                           [[
                               sg.T('Algorithm: '),
                               sg.Combo([algo for algo in algorithms],
                                        key='algorithm',
                                        readonly=True,
                                        default_value=[algo for algo in algorithms][0])
                           ]])],
              [
                  sg.Button('Restart'),
                  sg.Button('Exit')
            ]]

    window = sg.Window('Gomoku',
                       layout,
                       default_button_element_size=(10, 1),
                       auto_size_buttons=False,
                       location=(0,0))
    return window


window = create_window(BOARD_GUI)

while True:  # Event Loop
    event, values = window.Read()
    if event is None or event == 'Exit' or event == sg.WIN_CLOSED:
        break
    if event == 'Restart':
        GAME_LOGIC.move_ai = algorithms[values['algorithm']]
        BOARD.reset()
        BOARD_GUI.update()
        # change current player
    if isinstance(event, tuple):
        row, col = event
        winner = GAME_LOGIC.play(row, col)
        BOARD_GUI.update()
        if winner is not None:
            if winner == 1:
                sg.Popup('You have won, congrats!')
            elif winner == 2:
                sg.Popup('You have lost, you must be a great AI programmer! :)')
            else:
                sg.Popup("It's a draw!")
            GAME_LOGIC.reset()


window.Close()
