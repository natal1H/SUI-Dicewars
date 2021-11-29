# TODO refactor, make it faster and more efficient
# TODO check length of game state np array if it has correct length (len_triangle + (N_areas * 2) + N_players)

import numpy as np
import torch

from dicewars.client.game.board import Board
from ..utils import probability_of_holding_area
from dicewars.ai.xholko02.ml.nn import *


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
        return torch.from_numpy(game_state).type(torch.float32)

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
    Simulate attack on board and do changes.

    Return:
        board -> board with changes
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


def get_player_win_prob(probs, player):
    """
    Get probabylity for player [1/2/3/4] winning
    """
    return probs.numpy()[0][player - 1]


def evaluate_board_NN(model, serializer, board, player):
    board_state = serializer.serialize_board_state(board)
    with torch.no_grad():
        probabilities = model(board_state[None, ...])  # Has to be this way for some reason
        probabilities_normalized = torch.softmax(probabilities, dim=1)
    evaluation = get_player_win_prob(probabilities_normalized, player)
    return evaluation


def evaluate_board(board, player):
    """
    Evaluate board somehow
    TODO Machine-learning
    TODO tuto sa da volat NN tiez a je tu na vstupe player aj board cize data + label
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


def get_transfer_to_border(board, player_name):
    """
    TODO NEviem ci to ponechat a len odzdrojovat alebo to treba prpisat nejako
    """
    border_names = [a.name for a in board.get_player_border(player_name)]
    all_areas = board.get_player_areas(player_name)
    inner = [a for a in all_areas if a.name not in border_names]

    for area in inner:
        if area.get_dice() < 2:
            continue

        for neigh in area.get_adjacent_areas_names():
            if neigh in border_names and board.get_area(neigh).get_dice() < 8:
                return area.get_name(), neigh

    return None


def areas_expected_loss(board, player_name, areas):
    """
    TODO NEviem ci to ponechat a len odzdrojovat alebo to treba prpisat nejako
    """
    hold_ps = [probability_of_holding_area(board, a.get_name(), a.get_dice(), player_name) for a in areas]
    return sum((1-p) * a.get_dice() for p, a in zip(hold_ps, areas))


def get_transfer_from_endangered(board, player_name):
    """
    TODO NEviem ci to ponechat a len odzdrojovat alebo to treba prpisat nejako
    """
    border_names = [a.name for a in board.get_player_border(player_name)]
    all_areas_names = [a.name for a in board.get_player_areas(player_name)]

    retreats = []

    for area in border_names:
        area = board.get_area(area)
        if area.get_dice() < 2:
            continue

        for neigh in area.get_adjacent_areas_names():
            if neigh not in all_areas_names:
                continue
            neigh_area = board.get_area(neigh)

            expected_loss_no_evac = areas_expected_loss(board, player_name, [area, neigh_area])

            src_dice = area.get_dice()
            dst_dice = neigh_area.get_dice()

            dice_moved = min(8-dst_dice, src_dice - 1)

            area.dice -= dice_moved
            neigh_area.dice += dice_moved

            expected_loss_evac = areas_expected_loss(board, player_name, [area, neigh_area])

            area.set_dice(src_dice)
            neigh_area.set_dice(dst_dice)

            retreats.append(((area, neigh_area), expected_loss_no_evac - expected_loss_evac))

    retreats = sorted(retreats, key=lambda x: x[1], reverse=True)

    if retreats:
        retreat = retreats[0]
        if retreat[1] > 0.0:
            return retreat[0][0].get_name(), retreat[0][1].get_name()

    return None