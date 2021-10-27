import random
import logging

from dicewars.ai.utils import possible_attacks
from dicewars.ai.utils import probability_of_successful_attack
from dicewars.ai.utils import attack_succcess_probability

from dicewars.client.ai_driver import BattleCommand, EndTurnCommand
from dicewars.ai.xholko02.utils import Serializer


class AI:
    def __init__(self, player_name, board, players_order, max_transfers):
        self.player_name = player_name
        self.logger = logging.getLogger('AI')
        self.serializer = Serializer(board, len(players_order))

    def minmax(self, attacks, player, depth):
        """
        TODO MinMax-N
        """

        chosen_attack = None

        if depth <= 0:
            return chosen_attack

        chosen_attack = self.minmax(attacks, player, depth - 1)

        ######################################################
        X = 0

        # Prejdenie utokov a zistenie ich ohodnotenia
        for attack in attacks:
            source, target = attack
            attack_probability = attack_succcess_probability(source.get_dice(), target.get_dice())

            if attack_probability > X:
                X = attack_probability
                chosen_attack = attack

        return chosen_attack

    def depth_one_best(self, attacks):
        """
        Return best possible attack from attacks, based on attack success probability.

        TODO: Prehladavanie len do hlbky 1 , na toto naviazat strojove ucenie a NOT BAD riesenie.

        :param attacks:
        :return:
        """
        chosen_attack = None
        comparable_probability = 0

        for attack in attacks:
            source, target = attack
            attack_probability = attack_succcess_probability(source.get_dice(), target.get_dice())

            if attack_probability > comparable_probability:
                comparable_probability = attack_probability
                chosen_attack = attack

        return chosen_attack

    def ai_turn(self, board, nb_moves_this_turn, nb_transfers_this_turn, nb_turns_this_game, time_left):
        """
        AI turn.

        :param board:
        :param nb_moves_this_turn:
        :param nb_transfers_this_turn:
        :param nb_turns_this_game:
        :param time_left:
        :return:
        """
        game_state_tmp = self.serializer.serialize_board_state(board)
        attacks = list(possible_attacks(board, self.player_name))

        if attacks:
            #best_attack = self.minmax(attacks, self.player_name, 3)
            best_attack = self.depth_one_best(attacks)
            best_attack_source, best_attack_target = best_attack
            return BattleCommand(best_attack_source.get_name(), best_attack_target.get_name())
            #return EndTurnCommand()
        else:
            self.logger.debug("No more possible turns.")
            return EndTurnCommand()
