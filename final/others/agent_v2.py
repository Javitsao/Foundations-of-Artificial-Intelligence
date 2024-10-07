import json, os, random
from game.players import BasePokerPlayer
from game.engine.card import Card
from collections import Counter
from itertools import combinations
BB = 10
bluff = 0.1
# RANKS = "23456789TJQKA"
# SUITS = "CDHS"
# def card_value(card):
#     return RANKS.index(card[1])

# def is_royal_flush(cards):
#     suits = [card[0] for card in cards]
#     ranks = [card[1] for card in cards]
#     return len(set(suits)) == 1 and all(rank in ranks for rank in 'TJQKA')

# def is_straight_flush(cards):
#     return is_flush(cards) and is_straight(cards)

# def is_four_of_a_kind(cards):
#     counts = Counter(card[1] for card in cards)
#     return 4 in counts.values()

# def is_full_house(cards):
#     counts = Counter(card[1] for card in cards)
#     return 3 in counts.values() and 2 in counts.values()

# def is_flush(cards):
#     suits = [card[0] for card in cards]
#     return len(set(suits)) == 1

# def is_straight(cards):
#     values = sorted(set(card_value(card) for card in cards))
#     return len(values) == 5 and values[-1] - values[0] == 4

# def is_three_of_a_kind(cards):
#     counts = Counter(card[1] for card in cards)
#     return 3 in counts.values()

# def is_two_pair(cards):
#     counts = Counter(card[1] for card in cards)
#     return list(counts.values()).count(2) == 2

# def is_one_pair(cards):
#     counts = Counter(card[1] for card in cards)
#     return 2 in counts.values()

# def high_card(cards):
#     values = [card_value(card) for card in cards]
#     return max(values)

# def best_hand(hand, community):
#     all_cards = hand + community
#     best_rank = (11, [])

#     for comb in combinations(all_cards, 5):
#         print(comb)
#         if is_royal_flush(comb):
#             rank = 1
#         elif is_straight_flush(comb):
#             rank = 2
#         elif is_four_of_a_kind(comb):
#             rank = 3
#         elif is_full_house(comb):
#             rank = 4
#         elif is_flush(comb):
#             rank = 5
#         elif is_straight(comb):
#             rank = 6
#         elif is_three_of_a_kind(comb):
#             rank = 7
#         elif is_two_pair(comb):
#             rank = 8
#         elif is_one_pair(comb):
#             rank = 9
#         else:
#             rank = 10

#         if rank < best_rank[0] or (rank == best_rank[0] and high_card(comb) > high_card(best_rank[1])):
#             best_rank = (rank, comb)

#     return best_rank
def bet(amount, valid_actions):
    if valid_actions[2]["amount"]["min"] == -1:
        return "call", valid_actions[1]["amount"]
    return "raise", max(min(amount, valid_actions[2]["amount"]["max"]), valid_actions[2]["amount"]["min"])
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
class CallPlayer(BasePokerPlayer):
    def __init__(self, **config):
        self.config = dict()
        self.config.update(config)
        with open(os.path.join(os.path.dirname(__file__), "robbi_2.json"), "r") as f:
            self.card_combinations = json.load(f)
    
    def get_my_seats(self, seats):
        return [i for i, j in enumerate(seats) if j['uuid'] == self.uuid][0]

    def fold_win(self, valid_actions, round_state):
        seat = self.get_my_seats(round_state['seats'])
        stack = round_state["seats"][seat]["stack"]
        round = round_state["round_count"]
        blinds = (21 - round) // 2 * 3 * round_state["small_blind_amount"]
        if round % 2 == 0:
            if seat == round_state["big_blind_pos"]:
                blinds += 2 * round_state["small_blind_amount"]
            elif seat == round_state["small_blind_pos"]:
                blinds += round_state["small_blind_amount"]
        if stack - (2000 - stack) > 2 * blinds:
            return True
        return False
    def declare_action(self, valid_actions, hole_card, round_state):
        # rank, best_combination = best_hand(hole_card, round_state["community_card"])
        # hand_names = ["Royal Flush", "Straight Flush", "Four of a Kind", "Full House", "Flush", 
        #             "Straight", "Three of a Kind", "Two Pair", "One Pair", "High Card"]
        # print(f"Best hand: {hand_names[rank-1]} with cards {best_combination}")
        value = evaluate_hand(hole_card, round_state["community_card"])
        print(value)
        if self.fold_win(valid_actions, round_state):
            print("fold to win")
            return "fold", 0

        same_suit = 0
        ranks = tuple(Card.from_str(i).rank for i in hole_card)
        suits = tuple(Card.from_str(i).suit for i in hole_card)
        if suits[0] == suits[1]:
            same_suit = 1
        if ranks[0] == 14:
            ranks[0] = 1
        if ranks[1] == 14:
            ranks[1] = 1
        card1, card2 = sorted(ranks)
        
        status = self.find_status(card1, card2, same_suit)
        print(f"Card1: {card1}, Card2: {card2}, Same Suit: {same_suit}, Status: {status}")
        
        # Implement your action logic based on the status here
        # This is just an example
        mystack = round_state["seats"][self.get_my_seats(round_state['seats'])]["stack"]
        botstack = 2000 - mystack
        if round_state["community_card"] == []:
            if self.bot_raise == 1 and round_state["pot"]["main"]["amount"] >= 4*BB and round_state["round_count"] < 18 and self.bot_high_raise <= 3:
                self.bot_high_raise += 1
                return "fold", 0
            else:
                if status == 1:
                    if self.bot_raise >= 1:
                        return bet(BB * self.bot_raise_amount * 3.5, valid_actions)
                    else:
                        return bet(BB * 3.5, valid_actions)
                else:
                    if self.bot_raise >= 1:
                        if self.bot_raise_amount > round_state["pot"]["main"]["amount"] / 1.5:
                            self.bot_high_raise += 1
                            if self.bot_high_raise <= 3:
                                return "fold", 0
                        random = random.random()
                        if random >= 0 and random < 0.2:
                            return bet(BB * 3.5, valid_actions)
                        elif random >= 0.2 and random < 0.9:
                            return "call", valid_actions[1]["amount"]
                        else:
                            return "fold", 0
                
                    return "call", valid_actions[1]["amount"]
        else:
            if value <= 3:
                if mystack + round_state["pot"]["main"]["amount"] - botstack > 300:
                    return "call", valid_actions[1]["amount"]
                else:
                    return bet(round_state["pot"]["main"]["amount"] * 3, valid_actions)
            if value <= 7:
                if mystack + round_state["pot"]["main"]["amount"] - botstack > 300:
                    return "call", valid_actions[1]["amount"]
                else:
                    return bet(round_state["pot"]["main"]["amount"] * 2, valid_actions)
            if value <= 8:
                if mystack + round_state["pot"]["main"]["amount"] - botstack > 300:
                    return "call", valid_actions[1]["amount"]
                else:
                    return bet(round_state["pot"]["main"]["amount"], valid_actions)
            else:
                if valid_actions[1]["amount"] == 0:
                    return "call", 0
                if random.random() < bluff:
                    valid_actions[2]["amount"]["max"]
                if(random.random() < prob(value) and valid_actions[1]["amount"] <= 10*BB):
                    return "call", valid_actions[1]["amount"]
                else:
                    if round_state["round_count"] > 17 and mystack < botstack:
                        return "call", valid_actions[1]["amount"]
                    return "fold", 0

    def find_status(self, card1, card2, same_suit):
        for combination in self.card_combinations:
            if (combination['card1'] == card1 and
                combination['card2'] == card2 and
                combination['same_suit'] == same_suit):
                return combination['status']
        return None

    def receive_game_start_message(self, game_info):
        self.bot_high_raise = 0

    def receive_round_start_message(self, round_count, hole_card, seats):
        self.mode = None
        self.bot_raise = 0
        self.bot_raise_amount = 0
        
    def receive_street_start_message(self, street, round_state):
        self.bot_raise = 0
        self.bot_raise_amount = 0

    def receive_game_update_message(self, action, round_state):
        if action["action"] == "raise":
            self.bot_raise += 1
            self.bot_raise_amount = action["amount"] / (round_state["small_blind_amount"] * 2)

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass

def setup_ai():
    return CallPlayer()
