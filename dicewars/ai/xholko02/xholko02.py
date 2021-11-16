import copy
import logging

from dicewars.ai.utils import attack_succcess_probability
from dicewars.ai.utils import possible_attacks
from dicewars.client.ai_driver import BattleCommand, EndTurnCommand
from dicewars.ai.xholko02.utils import evaluate_board
from dicewars.ai.xholko02.utils import attack_simulation


# SPUSTENIE python3 ./scripts/dicewars-human.py --ai dt.sdc dt.rand xholko02
# SPUSTENIE LEN NASE AI python3 ./scripts/dicewars-human.py --ai xholko02 xholko02 xholko02
class FinalAI:
    """
    ExpectiMiniMax player agent
    """

    def __init__(self, player_name, board, players_order, max_transfers):
        self.board = board
        self.player_name = player_name
        self.logger = logging.getLogger('AI')
        self.player_order = players_order
        self.max_transfers = max_transfers

    def evaluate_attack(self, attack):
        # 1 Ohodnot board
        current_board_evaluation = evaluate_board(self.board, self.player_name)

        # 2 Urob kopiu boardu C_board
        board_simulation = copy.deepcopy(self.board)

        # 3 Simuluj moj attack na C_board
        board_simulation = attack_simulation(board_simulation, attack)

        # 4 Pre kadzdeho enemy simuluj ich najlepsi utok
        for enemy in self.player_order:
            if enemy != self.player_name:
                enemy_attacks = list(possible_attacks(board_simulation, enemy))
                if enemy_attacks:
                    best_enemy_attack = None
                    best_enemy_attack_possibility = 0
                    for enemy_attack in enemy_attacks:
                        source, target = enemy_attack
                        attack_probability = attack_succcess_probability(source.get_dice(), target.get_dice())
                        if attack_probability > best_enemy_attack_possibility:
                            best_enemy_attack = enemy_attack
                            best_enemy_attack_possibility = attack_probability

                    board_simulation = attack_simulation(board_simulation, best_enemy_attack)

        # TODO 5 a 6 to by bolo do hlbky 3 ale je to dobry napad ?, bolo by to len dve iteracie ako hore
        # 5 Simuluj zasa moj utok z novej mnoziny
        # 6 Simuluj zasa utoky nepratelov

        # 7 Ohodnot C_board
        board_simulation_evaluation = evaluate_board(board_simulation, self.player_name)

        # TODO mozno doplnit situaciu kedy sa nevykona ziaden utok, lebo stav by bol lepsi, ale to chce znova taku istu iteraciu bez simulacie mojho stavu
        # 8 Porovnaj ohodnotenia board a C_board

        return [attack, board_simulation_evaluation]

    def choose_best_attack(self, attacks):
        """
        From all possible attacks choose one with best evaluation of final state of board.
        """
        # Count evaluation for every possible attack.
        evaluated_attacks = []
        for attack in attacks:
            eval_attack = self.evaluate_attack(attack)
            evaluated_attacks.append(eval_attack)

        # Choose attack with best evaluation value.
        best_attack = None
        best_attack_eval = 0
        for eval_attack in evaluated_attacks:
            if eval_attack[1] > best_attack_eval:
                best_attack = eval_attack[0]  # attack
                best_attack_eval = eval_attack[1]  # evaluation value


        return best_attack

    def ai_turn(self, board, nb_moves_this_turn, nb_transfers_this_turn, nb_turns_this_game, time_left):
        """
        Agent turn. Choose best possible attack or does nothing.
        """

        attacks = list(possible_attacks(board, self.player_name))
        best_attack = self.choose_best_attack(attacks)

        if best_attack:
            source, target = best_attack
            return BattleCommand(source.get_name(), target.get_name())
        else:
            return EndTurnCommand()
