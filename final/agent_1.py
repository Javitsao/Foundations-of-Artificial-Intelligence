import json, os, random
from game.players import BasePokerPlayer
from game.engine.card import Card
from collections import Counter
from itertools import combinations
BB = 10
bluff = 0.1
def prob(value):
        if value == 1:
            return 1.0
        if value == 2:
            return 1.0
        if value == 3:
            return 1.0
        if value == 4:
            return 1.0
        if value == 5:
            return 1.0
        if value == 6:
            return 0.9
        if value == 7:
            return 0.7
        if value == 8:
            return 0.5
        if value == 9:
            return 0.1
        return 0.0
def evaluate_hand(hand, community):
    # Combine hand and community cards
    all_cards = hand + community
    ranks = '23456789TJQKA'
    suits = {'D': [], 'C': [], 'H': [], 'S': []}
    rank_counts = {rank: 0 for rank in ranks}

    # Categorize cards by suit and rank
    for card in all_cards:
        rank, suit = card[1], card[0]
        suits[suit].append(rank)
        rank_counts[rank] += 1

    # Helper functions to check for hand types
    def is_straight_flush():
        for suit, cards in suits.items():
            if len(cards) >= 5:
                sorted_cards = sorted(cards, key=lambda x: ranks.index(x))
                for i in range(len(sorted_cards) - 4):
                    if ranks.index(sorted_cards[i+4]) - ranks.index(sorted_cards[i]) == 4:
                        return True
        return False

    def is_four_of_a_kind():
        return 4 in rank_counts.values()

    def is_full_house():
        return sorted(rank_counts.values(), reverse=True)[:2] == [3, 2]

    def is_flush():
        for suit, cards in suits.items():
            if len(cards) >= 5:
                return True
        return False

    def is_straight():
        sorted_ranks = sorted(set(all_cards), key=lambda x: ranks.index(x[1]))
        for i in range(len(sorted_ranks) - 4):
            if ranks.index(sorted_ranks[i+4][1]) - ranks.index(sorted_ranks[i+3][1]) == 1 and ranks.index(sorted_ranks[i+3][1]) - ranks.index(sorted_ranks[i+2][1]) == 1 and ranks.index(sorted_ranks[i+2][1]) - ranks.index(sorted_ranks[i+1][1]) == 1 and ranks.index(sorted_ranks[i+1][1]) - ranks.index(sorted_ranks[i][1]) == 1:
                return True
        return False

    def is_three_of_a_kind():
        return 3 in rank_counts.values()

    def is_two_pair():
        return list(rank_counts.values()).count(2) >= 2

    def is_one_pair():
        return 2 in rank_counts.values()

    # Check for each hand type
    if is_straight_flush():
        return 1#"Straight Flush"
    elif is_four_of_a_kind():
        return 2#"Four of a Kind"
    elif is_full_house():
        return 3#"Full House"
    elif is_flush():
        return 4#"Flush"
    elif is_straight():
        return 5#"Straight"
    elif is_three_of_a_kind():
        return 6#"Three of a Kind"
    elif is_two_pair():
        return 7#"Two Pair"
    elif is_one_pair():
        return 8#"One Pair"
    else:
        return 9#"High Card"
def high_pair(hand, community):
    all_cards = hand + community
    ranks = '23456789TJQKA'
    rank_counts = {rank: 0 for rank in ranks}
    for card in all_cards:
        rank = card[1]
        rank_counts[rank] += 1
    for rank in ranks[::-1]:
        if rank_counts[rank] >= 2:
            if rank >= max(community, key=lambda x: ranks.index(x[1]))[1]:
                return True
            return False
class CallPlayer(BasePokerPlayer):
    def __init__(self, **config):
        self.config = dict()
        self.config.update(config)
        with open(os.path.join(os.path.dirname(__file__), "sb_bet.json"), "r") as f:
            self.card_combinations_sb_bet = json.load(f)
        with open(os.path.join(os.path.dirname(__file__), "sb_4bet.json"), "r") as f:
            self.card_combinations_sb_4bet = json.load(f)
        with open(os.path.join(os.path.dirname(__file__), "bb_3bet.json"), "r") as f:
            self.card_combinations_bb_3bet = json.load(f)
    def size(self, round_state):
        seat = self.get_my_seats(round_state['seats'])
        round = round_state["round_count"]
        blinds = (21 - round) // 2 * 3 * round_state["small_blind_amount"]
        if round % 2 == 0:
            if seat == round_state["big_blind_pos"]:
                blinds += 2 * round_state["small_blind_amount"]
            elif seat == round_state["small_blind_pos"]:
                blinds += round_state["small_blind_amount"]
        lead = (self.mystack - self.botstack - 2 * blinds) / BB
        if lead > -5:
            return 0.6
        elif lead > -10:
            return 0.7
        elif lead > -15:
            return 0.8
        elif lead > -20:
            return 0.9
        else:
            return 1
    def bet(self, amount, valid_actions, round_state):
        mul = self.size(round_state)
        if valid_actions[2]["amount"]["min"] == -1:
            return "call", valid_actions[1]["amount"]
        return "raise", max(min(amount, valid_actions[2]["amount"]["max"]), valid_actions[2]["amount"]["min"])
    def get_my_seats(self, seats):
        return [i for i, j in enumerate(seats) if j['uuid'] == self.uuid][0]

    def fold_win(self, valid_actions, round_state):
        seat = self.get_my_seats(round_state['seats'])
        #stack = round_state["seats"][seat]["stack"]
        round = round_state["round_count"]
        blinds = (21 - round) // 2 * 3 * round_state["small_blind_amount"]
        if round % 2 == 0:
            if seat == round_state["big_blind_pos"]:
                blinds += 2 * round_state["small_blind_amount"]
            elif seat == round_state["small_blind_pos"]:
                blinds += round_state["small_blind_amount"]
        if self.mystack - self.botstack > 2 * blinds:
            return True
        return False
    def declare_action(self, valid_actions, hole_card, round_state):
        # rank, best_combination = best_hand(hole_card, round_state["community_card"])
        # hand_names = ["Royal Flush", "Straight Flush", "Four of a Kind", "Full House", "Flush", 
        #             "Straight", "Three of a Kind", "Two Pair", "One Pair", "High Card"]
        # print(f"Best hand: {hand_names[rank-1]} with cards {best_combination}")
        value = evaluate_hand(hole_card, round_state["community_card"])
        print(value)
        if round_state["street"] == "preflop":
            if self.fold_win(valid_actions, round_state):
                print("fold to win")
                return "fold", 0
        print("111")
        same_suit = 0
        ranks = tuple(Card.from_str(i).rank for i in hole_card)
        suits = tuple(Card.from_str(i).suit for i in hole_card)
        print("222")
        if suits[0] == suits[1]:
            same_suit = 1
        if ranks[0] == 14:
            ranks[0] = 1
        if ranks[1] == 14:
            ranks[1] = 1
        print(ranks)
        card1 = min(ranks)
        card2 = max(ranks)
        print("333")
        status_sb_bet = self.find_status(card1, card2, same_suit, self.card_combinations_sb_bet)
        status_sb_4bet = self.find_status(card1, card2, same_suit, self.card_combinations_sb_4bet)
        status_bb_3bet = self.find_status(card1, card2, same_suit, self.card_combinations_bb_3bet)
        print(f"Card1: {card1}, Card2: {card2}, Same Suit: {same_suit}, Status_sb_bet: {status_sb_bet}, Status_sb_4bet: {status_sb_4bet}, Status_bb_3bet: {status_bb_3bet}")
        
        # Implement your action logic based on the status here
        # This is just an example
        
        mystack = round_state["seats"][self.get_my_seats(round_state['seats'])]["stack"]
        botstack = round_state["seats"][1 - self.get_my_seats(round_state['seats'])]["stack"]
        blinds = (21 - round_state["round_count"]) // 2 * 3 * round_state["small_blind_amount"]
        if round_state["round_count"] % 2 == 0:
            if self.get_my_seats(round_state['seats']) == round_state["big_blind_pos"]:
                blinds += 2 * round_state["small_blind_amount"]
            elif self.get_my_seats(round_state['seats']) == round_state["small_blind_pos"]:
                blinds += round_state["small_blind_amount"]
        print (blinds)
        # if mystack + round_state["pot"]["main"]["amount"] - botstack > blinds * 2:
        #     return bet(valid_actions[2]["amount"]["max"], valid_actions)
        if botstack + round_state["pot"]["main"]["amount"] - mystack > blinds * 2:
            return self.bet(valid_actions[2]["amount"]["max"], valid_actions, round_state)
        mul = self.size(round_state)
        mul = 1
        if round_state["community_card"] == []:
            print("bot_raise", self.bot_raise)
            if self.get_my_seats(round_state['seats']) == round_state["small_blind_pos"]:
                if self.bot_raise == 0:
                    if status_sb_bet == 1:
                        return self.bet(mul * BB * 2.5, valid_actions, round_state)
                    else:
                        return "fold", 0
                elif self.bot_raise == 1:
                    if status_sb_4bet == 1:
                        if self.bot_raise_amount <= 13:
                            return self.bet(mul * self.bot_raise_amount * BB * 2.1, valid_actions, round_state)
                        else:
                            return self.bet(mul * valid_actions[2]["amount"]["max"], valid_actions, round_state)
                    elif status_sb_4bet == 0:
                        return "call", valid_actions[1]["amount"]
                    else:
                        if self.bot_raise_amount <= 6.5:
                            return "call", valid_actions[1]["amount"]
                        return "fold", 0
                else:
                    return self.bet(mul * valid_actions[2]["amount"]["max"], valid_actions, round_state)
            else:
                if self.bot_raise == 0:
                    #return self.bet(mul * BB * 4, valid_actions, round_state)
                    if status_bb_3bet == 1:
                        return self.bet(mul * BB * 4, valid_actions, round_state)
                    else:
                        return "call", valid_actions[1]["amount"]
                elif self.bot_raise == 1:
                    if status_bb_3bet == 1:
                        return self.bet(mul * self.bot_raise_amount * BB * 4, valid_actions, round_state)
                    elif status_bb_3bet == 0:
                        return "call", valid_actions[1]["amount"]
                    else:
                        return "fold", 0
                elif self.bot_raise == 2:
                    return self.bet(mul * valid_actions[2]["amount"]["max"], valid_actions, round_state)
                
                    if (card1,card2) == (1,1) or (card1,card2) == (1,13) or (card1,card2) == (1,12) or (card1,card2) == (13,13) or (card1,card2) == (12,12) or (card1,card2) == (11,11) or (card1,card2) == (10,10) or (card1,card2) == (9,9):
                        return self.bet(mul * valid_actions[2]["amount"]["max"], valid_actions, round_state)
                    else:
                        return "call", valid_actions[1]["amount"]
                    
        else:
            if value <= 6:
                if mystack + round_state["pot"]["main"]["amount"] - botstack > blinds * 2:
                    return self.bet(valid_actions[2]["amount"]["max"], valid_actions, round_state)
                else:
                    return self.bet((blinds * 2 + botstack - mystack - round_state["pot"]["main"]["amount"]) / 2 + 1, valid_actions, round_state)
            elif value == 7:
                if evaluate_hand([], round_state["community_card"]) == 7:
                    if valid_actions[1]["amount"] == 0:
                        return "call", 0
                    return "fold", 0
                elif evaluate_hand([], round_state["community_card"]) == 8:
                    return "call", valid_actions[1]["amount"]
                else:
                    if mystack + round_state["pot"]["main"]["amount"] - botstack > blinds * 2:
                        return self.bet(valid_actions[2]["amount"]["max"], valid_actions, round_state)
                    else:
                        return self.bet((blinds * 2 + botstack - mystack - round_state["pot"]["main"]["amount"]) / 2 + 1, valid_actions, round_state)
                    
            elif value == 8 and evaluate_hand([], round_state["community_card"]) != 8:
                if round_state["street"] == "flop":
                    if mystack + round_state["pot"]["main"]["amount"] - botstack > blinds * 2:
                        return self.bet(valid_actions[2]["amount"]["max"], valid_actions, round_state)
                    else:
                        #return self.bet(round_state["pot"]["main"]["amount"], valid_actions, round_state)
                        return self.bet((blinds * 2 + botstack - mystack - round_state["pot"]["main"]["amount"]) / 2 + 1, valid_actions, round_state)
                elif round_state["street"] == "turn":
                    return "call", valid_actions[1]["amount"]
                else:
                    # if high_pair(hole_card, round_state["community_card"]):
                    #     return "call", valid_actions[1]["amount"]
                    if valid_actions[1]["amount"] < round_state["pot"]["main"]["amount"] / 2:
                        return "call", valid_actions[1]["amount"]
                    return "fold", 0
            else:
                def high_in_community():
                    highest_card = max(hole_card + round_state["community_card"], key=lambda card: Card.from_str(card).rank)
                    if highest_card in round_state["community_card"]:
                        return True
                    else:
                        return False
                # if botstack == 0:
                #     return "fold", 0
                # if mystack + round_state["pot"]["main"]["amount"] - botstack > blinds * 2:
                #     return self.bet(valid_actions[2]["amount"]["max"], valid_actions, round_state)
                if valid_actions[1]["amount"] <= BB + 1:
                    if mystack + round_state["pot"]["main"]["amount"] - botstack > blinds * 2 and not high_in_community():
                        return self.bet(valid_actions[2]["amount"]["max"], valid_actions, round_state)
                    #if self.get_my_seats(round_state['seats']) == round_state["big_blind_pos"]:
                    if self.bot_last_amount == 0:
                        if random.random() < 11 / 26:
                            return self.bet(round_state["pot"]["main"]["amount"] - 1, valid_actions, round_state ) #BB
                    return "call", valid_actions[1]["amount"]
                return "fold", 0
                if random.random() < bluff:
                    return self.bet(valid_actions[2]["amount"]["max"], round_state)
                # if(random.random() < prob(value) and round_state["pot"]["main"]["amount"] < blinds * 2 - botstack + mystack):
                #     return "call", valid_actions[1]["amount"]
                else:
                    if round_state["round_count"] > 17 and mystack < botstack:
                        return self.bet(valid_actions[2]["amount"]["max"], valid_actions, round_state)
                    return "fold", 0

    def find_status(self, card1, card2, same_suit, card_combinations):
        for combination in card_combinations:
            if (combination['card1'] == card1 and
                combination['card2'] == card2 and
                combination['same_suit'] == same_suit):
                return combination['status']
        return None

    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        self.mode = None
        self.bot_raise = 0
        self.bot_raise_amount = 0
        self.mystack = seats[self.get_my_seats(seats)]["stack"]
        self.botstack = seats[1 - self.get_my_seats(seats)]["stack"]
        self.round = round_count

    def receive_street_start_message(self, street, round_state):
        self.bot_raise = 0
        self.bot_raise_amount = 0

    def receive_game_update_message(self, action, round_state):
        if action["action"] == "raise" and action["player_uuid"] != self.uuid:
            self.bot_raise += 1
            print("bot_raise", self.bot_raise)
            self.bot_raise_amount = action["amount"] / (round_state["small_blind_amount"] * 2)
        if action["player_uuid"] != self.uuid:
            self.bot_last_amount = action["amount"]

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass

def setup_ai():
    return CallPlayer()
