# TODO refactor, make it faster and more efficient
# TODO check length of game state np array if it has correct length (len_triangle + (N_areas * 2) + N_players)

import numpy as np
from dicewars.client.game.board import Board


class Serializer:
    numAreas = 34  # Number of areas on board (names from 1 to 34 (including)

    def __init__(self, board: Board, players):
        """
        At start it determines the matrix of neighbouring areas, because that doesn't change
        """
        self.numPlayers = players
        neighbours_matrix = self.get_neighbours_matrix(board)
        self.neighbours_triangle = get_matrix_upper_triangle(neighbours_matrix)

    def serialize_board_state(self, board: Board):
        """
        TODO refactor
        """
        # each round we need new info about areas and biggest regions
        areas_info = self.get_array_of_area_info(board)
        biggest_regions = self.get_array_of_biggest_regions(board)
        # concatenate arrays into one
        game_state = np.concatenate((self.neighbours_triangle, areas_info, biggest_regions), dtype=int)

        return game_state

    def get_neighbours_matrix(self, board: Board):
        # TODO refactor
        matrix = []

        for i in range(1, self.numAreas + 1):
            row = []
            area = board.get_area(i)
            neighbours = area.get_adjacent_areas_names()
            for j in range(1, self.numAreas + 1):
                if j in neighbours:
                    row.append(1)
                else:
                    row.append(0)
            matrix.append(row)
        return matrix

    def get_array_of_area_info(self, board: Board):
        # TODO refactor
        areas_info = []
        for i in range(1, self.numAreas + 1):
            area = board.get_area(i)
            dice = area.get_dice()
            owner = area.get_owner_name()
            areas_info.append(dice)
            areas_info.append(owner)
        return areas_info

    def get_array_of_biggest_regions(self, board: Board):
        # TODO refactor
        biggest_regions = []
        for i in range(1, self.numPlayers + 1):
            player_regions = board.get_players_regions(i)
            biggest_regions.append(max(len(elem) for elem in player_regions))
        return biggest_regions


def get_matrix_upper_triangle(matrix):
    """
    N = 3
    matrix:
    [[00,01,02],
     [10,11,12],
     [20,21,22]]
    upper_triangle:
     --,01,02
     --,--,12
     --,--,--
    as single array: [01,02,12]

    TODO - refactor
    """
    triangle = []
    N = len(matrix)
    for row in range(0, N):
        for col in range(row + 1, N):
            triangle.append(matrix[row][col])
    return triangle


def attack_simulation(board, attack):
    """
    Simulate attack on board and do changes

    return board with changes
    """
    if attack is None:
        return board

    source, target = attack
    source_area = board.get_area(source.get_name())
    target_area = board.get_area(target.get_name())
    target_area.set_owner(source.get_owner_name())
    target_area.set_dice(source.get_dice() - 1)
    source_area.set_dice(1)
    return board


def evaluate_board(board, player):
    """
    Evaluate board somehow
    TODO moze byt ine ??? evaluation vynasobit s pravdepodobnostou utoku
    """
    # Count number of dices on all fields of player
    dices_number = board.get_player_dice(player)

    # Get size of biggest region of player
    biggest_region_size = 0
    player_regions = board.get_players_regions(player)
    for region in player_regions:
        if len(region) > biggest_region_size:
            biggest_region_size = len(region)

    # Return some evaluation of the board
    evaluation = dices_number + biggest_region_size

    return evaluation
