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
        board_simulation = copy.deepcopy(self.board)
        board_simulation = attack_simulation(board_simulation, attack)

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

        source, target = attack
        player_attack_poss = attack_succcess_probability(source.get_dice(), target.get_dice())
        board_simulation_evaluation = evaluate_board(board_simulation, self.player_name)

        return [attack, board_simulation_evaluation * player_attack_poss]

    def choose_best_attack(self, attacks):
        """
        From all possible attacks choose one with best evaluation of final state of board.
        """
        # Count evaluation for every possible attack.
        evaluated_attacks = []
        for attack in attacks:
            source, target = attack
            if source.get_dice() >= target.get_dice():
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

    # for every area in my border with enemy vybrat policko kam presuvame (min , daco, ked maju nepriatelia oproti nasemu policku vela dices) a vybereme policko
    # TODO ktore je najmenej ohodnotene a tam presuvame. Druha heuristika je ako a z kade presuvat.
    # TODO presuvanie bude tak ze kukame policka ktore niesu na hrane a su susedmi daneho policka a ktore ma najviac
    # TODO dices tak z neho presuvame a toto volame iterativne (2 krat, teda hlbka 2) ze na ten co ma najviac tak ten presune dopredu.
    def move_dices(self, board, player, nb_transfers_this_turn):
        # Choose weakest area --------------------------------------------------------------
        weak_area = None
        player_border = board.get_player_border(player)
        for border_area in player_border:
            if weak_area is None:
                weak_area = border_area
            else:
                if weak_area.get_dice() > border_area.get_dice():
                    weak_area = border_area

        # Choose mover ---------------------------------------------------------------------
        mover = None
        weak_area_neighbours = weak_area.get_adjacent_areas_names()
        for name in weak_area_neighbours:
            area = board.get_area(name)
            if area.get_owner_name() == player and area not in player_border:
                if mover is None:
                    mover = area
                else:
                    if mover.get_dice() < area.get_dice():
                        mover = area

        # Move dices from mover to weak_area ---------------------------------------------- DONE
        if weak_area is not None and mover is not None:
            if weak_area.get_dice() != 8 and mover.get_dice() != 1:
                if weak_area.get_dice() == 1:
                    weak_area.set_dice(mover.get_dice())
                    mover.set_dice(1)
                else:
                    need = 8 - weak_area.get_dice()
                    if mover.get_dice() > need:
                        weak_area.set_dice(8)
                        mover.set_dice(mover.get_dice() - need)
                    else:
                        weak_area.set_dice(weak_area.get_dice() + mover.get_dice() - 1)
                        mover.set_dice(1)

                nb_transfers_this_turn += 1

    def ai_turn(self, board, nb_moves_this_turn, nb_transfers_this_turn, nb_turns_this_game, time_left):
        """
        Agent turn. Choose best possible attack or does nothing.
        """
        attacks = list(possible_attacks(board, self.player_name))
        if nb_transfers_this_turn < 3:
            self.move_dices(board, self.player_name, nb_transfers_this_turn)
        best_attack = self.choose_best_attack(attacks)

        if best_attack:
            if nb_transfers_this_turn < 6:
                self.move_dices(board, self.player_name, nb_transfers_this_turn)
            source, target = best_attack
            return BattleCommand(source.get_name(), target.get_name())
        else:
            return EndTurnCommand()
