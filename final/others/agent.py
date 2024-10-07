import json, os
from game.players import BasePokerPlayer
from game.engine.card import Card

class CallPlayer(BasePokerPlayer):
    def __init__(self, **config):
        self.config = dict()
        self.config.update(config)
        with open(os.path.join(os.path.dirname(__file__), "robbi_2.json"), "r") as f:
            self.card_combinations = json.load(f)
    
    def get_my_seats(self, seats):
        return [i for i, j in enumerate(seats) if j['uuid'] == self.uuid][0]

    def fold_win(self, round_state):
        seat = self.get_my_seats(round_state['seats'])
        stack = round_state["seats"][seat]["stack"]
        round = round_state["round_count"]
        blinds = (21 - round) // 2 * 3 * round_state["small_blind_amount"]
        if round % 2 == 0:
            if seat == round_state["big_blind_pos"]:
                blinds += 2 * round_state["small_blind_amount"]
            elif seat == round_state["small_blind_pos"]:
                blinds += round_state["small_blind_amount"]

    def declare_action(self, valid_actions, hole_card, round_state):
        if self.fold_win(round_state):
            return "fold", 0

        same_suit = 0
        ranks = tuple(Card.from_str(i).rank for i in hole_card)
        suits = tuple(Card.from_str(i).suit for i in hole_card)
        if suits[0] == suits[1]:
            same_suit = 1
        if card1 == 14:
            card1 = 1
        if card2 == 14:
            card2 = 1
        card1, card2 = sorted(ranks)
        
        status = self.find_status(card1, card2, same_suit)
        print(f"Card1: {card1}, Card2: {card2}, Same Suit: {same_suit}, Status: {status}")
        
        # Implement your action logic based on the status here
        # This is just an example
        if status == 1:
            return "raise", valid_actions[2]["amount"]["max"]
        else:
            return "fold", 0

    def find_status(self, card1, card2, same_suit):
        for combination in self.card_combinations:
            if (combination['card1'] == card1 and
                combination['card2'] == card2 and
                combination['same_suit'] == same_suit):
                return combination['status']
        return None

    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass

def setup_ai():
    return CallPlayer()
