import logging

from dicewars.client.ai_driver import BattleCommand, EndTurnCommand
from dicewars.ai.xholko02.utils import Serializer


class AI:
    def __init__(self, player_name, board, players_order, max_transfers):
        self.player_name = player_name
        self.logger = logging.getLogger('AI')
        self.serializer = Serializer(board, len(players_order))

    def ai_turn(self, board, nb_moves_this_turn, nb_transfers_this_turn, nb_turns_this_game, time_left):
        game_state_tmp = self.serializer.serialize_board_state(board)

        return EndTurnCommand()
