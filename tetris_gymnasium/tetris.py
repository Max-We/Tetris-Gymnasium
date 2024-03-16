"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import numpy as np
import random


class TetrisBag:
    def __init__(self, size):
        self.bag = np.arange(size, dtype=np.int8)
        self.index = 0
        self.shuffle_bag()
        self.x = None
        self.y = None

    def shuffle_bag(self):
        """Shuffles the pieces in the bag."""
        random.shuffle(self.bag)
        self.index = 0  # Reset index to the start

    def get_next_piece(self):
        """Returns the next piece from the bag and refills it if necessary."""
        piece = self.bag[self.index]
        self.index += 1
        # If we've reached the end of the bag, shuffle and start over
        if self.index >= len(self.bag):
            self.shuffle_bag()
        return piece

class Tetris:
    piece_colors = [
        (0, 0, 0),
        (255, 255, 0),
        (147, 88, 254),
        (54, 175, 144),
        (255, 0, 0),
        (102, 217, 238),
        (254, 151, 32),
        (0, 0, 255)
    ]

    tetrominoes = [
        np.array([[0, 0, 0, 0],
                  [1, 1, 1, 1],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0]]),
        np.array([[2, 2],
                  [2, 2]]),
        np.array([[0, 3, 0],
                  [3, 3, 3],
                  [0, 0, 0]]),
        np.array([[0, 4, 4],
                  [4, 4, 0],
                  [0, 0, 0]]),
        np.array([[5, 5, 0],
                  [0, 5, 5],
                  [0, 0, 0]]),
        np.array([[6, 0, 0],
                  [6, 6, 6],
                  [0, 0, 0]]),
        np.array([[0, 0, 7],
                  [7, 7, 7],
                  [0, 0, 0]])
    ]

    def __init__(self, height=20, width=10):
        self.active_tetromino = None
        self.height = height
        self.width = width
        self.reset()

    def reset(self):
        self.board = np.zeros((self.height, self.width), dtype=np.int8)

        # Create bag
        self.bag = TetrisBag(len(self.tetrominoes))

        # Get first piece from bag
        self.current_piece = self.tetrominoes[self.bag.get_next_piece()]

        self.x, self.y = (self.width // 2 - len(self.current_piece[0]) // 2, 0)
        self.gameover = False

    def rotate(self, piece, clockwise=True):
        return np.rot90(piece, k=(1 if clockwise else -1))

    def spawn_tetromino(self):
        self.active_tetromino = self.tetrominoes[self.bag.get_next_piece()]
        pff = self.active_tetromino.shape
        self.x, self.y = self.width // 2 - self.active_tetromino.shape[0] // 2, 0
        self.gameover = self.check_collision(self.active_tetromino, self.x, self.y)

    def check_collision(self, tetromino, x, y):
        """
        Check if there is a collision when placing a tetromino onto the board using NumPy's
        efficient array operations.

        :param board: The game board, a 2D numpy array.
        :param tetromino: The tetromino piece, a 2D numpy array.
        :param new_position: A tuple (row, col) representing the top-left position where
                         the tetromino is trying to be placed on the board.
        :return: True if there is a collision, False otherwise.
        """
        tetromino_height, tetromino_width = tetromino.shape

        # Boundary check
        if (x < 0 or x + tetromino_width > self.width) or (y < 0 or y + tetromino_height > self.height):
            return True

        # Extract the subarray of the board where the tetromino will be placed
        board_subarray = self.board[y:y + tetromino_height, x:x + tetromino_width]

        # Check collision using numpy's element-wise operations.
        # This checks if any corresponding cells (both non-zero) of the subarray and the
        # tetromino overlap, indicating a collision.
        if np.any(board_subarray[tetromino > 0] > 0):
            return True

        # No collision detected
        return False

    def place_active_tetromino(self):
        # # Create a mask for non-zero elements in the active Tetromino
        # mask = self.active_tetromino > 0
        #
        # # Calculate the shape to identify range for height and width
        # tetromino_height, tetromino_width = self.active_tetromino.shape
        #
        # # Create coordinate grids for the destination based on the mask
        # destination_y, destination_x = np.ogrid[self.y:self.y + tetromino_height, self.x:self.x + tetromino_width]
        #
        # # Use the mask to place the Tetromino's non-zero values onto the board
        # self.board[destination_y[mask], destination_x[mask]] = self.active_tetromino[mask]
        tetromino_height, tetromino_width = self.active_tetromino.shape
        idk = self.board[self.y:self.y + tetromino_height, self.x:self.x + tetromino_width]
        self.board[self.y:self.y + tetromino_height, self.x:self.x + tetromino_width] += self.active_tetromino
        self.active_tetromino = None

    def clear_filled_rows(self):
        """
        Clear filled rows in the Tetris board and shift the rows above down.
        This function alters the board in-place.

        :param board: The game board, a 2D numpy array.
        :return: The number of cleared rows.
        """
        # Check for filled rows. A row is filled if it does not contain any zeros.
        filled_rows = ~(self.board == 0).any(axis=1)
        num_filled = np.sum(filled_rows)

        if num_filled > 0:
            # Identify the rows that are not filled.
            unfilled_rows = self.board[~filled_rows]

            # Create a new top part of the board with zeros to compensate for the cleared rows.
            free_space = np.zeros((num_filled, self.width), dtype=np.int8)

            # Concatenate the new top with the unfilled rows to form the updated board.
            self.board[:] = np.concatenate((free_space, unfilled_rows), axis=0)

        return num_filled

    def drop_active_tetromino(self):
        while not self.check_collision(self.active_tetromino, self.x, self.y+1):
            self.y += 1


    def step(self, action_id):
        if self.active_tetromino is None:
            self.spawn_tetromino()
        else:
            if action_id == 0: # move left
                if not self.check_collision(self.active_tetromino, self.x-1, self.y):
                    self.x -= 1
            elif action_id == 1: # move right
                if not self.check_collision(self.active_tetromino, self.x+1, self.y):
                    self.x += 1
            elif action_id == 2: # rotate clockwise
                if not self.check_collision(self.rotate(self.active_tetromino, True), self.x, self.y):
                    self.active_tetromino = self.rotate(self.active_tetromino, True)
            elif action_id == 3: # rotate counterclockwise
                if not self.check_collision(self.rotate(self.active_tetromino, False), self.x, self.y):
                    self.active_tetromino = self.rotate(self.active_tetromino, False)
            elif action_id == 4: # hard drop
                self.drop_active_tetromino()
                self.place_active_tetromino()
            elif action_id == 5: # fall down
                if not self.check_collision(self.active_tetromino, self.x, self.y+1):
                    self.y += 1
                else:
                    self.place_active_tetromino()

    def render(self):
        if self.active_tetromino is not None:
            tetromino_height, tetromino_width = self.active_tetromino.shape
            view = self.board.copy()
            view[self.y:self.y + tetromino_height, self.x:self.x + tetromino_width] += self.active_tetromino
        else:
            view = self.board

        char_field = np.where(view == 0, '.', view.astype(str))
        field_str = '\n'.join(''.join(row) for row in char_field)
        print(field_str)
        print("==========")

# Create an instance of Tetris
tetris_game = Tetris()

# Main game loop
while not tetris_game.gameover:
    # Render the current state of the game as text
    tetris_game.render()

    # Take a random action (for demonstration purposes)
    action = random.randint(0, 5)

    # Perform the action
    tetris_game.step(action)

# Game over
print("Game Over!")
