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
print(high_pair(['D5', 'C6'], ['H2', 'D3', 'C6']))