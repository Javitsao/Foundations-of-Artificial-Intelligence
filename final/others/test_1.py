import subprocess
import re

def run_game_and_check_winner(game_number):
    # Run the start_game.py script and capture its output
    result = subprocess.run(['python', f'start_game_{game_number}.py'], capture_output=True, text=True)
    output = result.stdout

    # Print the output of start_game.py to the terminal
    print(output)

    # Use regular expression to find all occurrences of "stack" and their values
    matches = re.findall(r'"stack": (\d+)', output)
    if matches and len(matches) >= 2:
        # The second occurrence of "stack" corresponds to p2
        p2_stack = int(matches[1])
        # Check if p2's stack is greater than 1000, indicating a win
        if p2_stack > 1000:
            return True
    return False
def calculate_score(wins):
    if wins >= 3:
        return 5
    elif wins == 2:
        return 3
    elif wins == 1:
        return 1.5
    else:
        return 0
def main(k):
    total_runs = k * 5
    p2_wins = 0

    results = [0, 0, 0, 0, 0, 0, 0]
    iter_results = []
    scores = []
    for iteration in range(k):
        iter_results = []
        score = 0
        for game_number in range(1, 8):
            for _ in range(5):
                if run_game_and_check_winner(game_number):
                    p2_wins += 1
                for i, result in enumerate(iter_results, 1):
                    print(f"\nPlayer 2 won {result} out of {5} games in start_game_{i}.py.  {calculate_score(result)} points")
                print(f"Player 2 won {p2_wins} out of {_ + 1} games in start_game_{game_number}.py.  {calculate_score(p2_wins)} points")
                for iter in range(iteration):
                    print(f"Score for iteration {iter + 1}: {scores[iter]}")
                print(f"Score for iteration {iteration + 1}: {score + calculate_score(p2_wins)} (Current)")
            iter_results.append(p2_wins)
            results[game_number - 1] += p2_wins
            score += calculate_score(p2_wins)
            p2_wins = 0
        scores.append(score)
        
    for i, result in enumerate(results, 1):
        print(f"\nPlayer 2 won {result} out of {total_runs} games in start_game_{i}.py.  win rate: {result/total_runs*100}%")
    for iter in range(k):
        print(f"Score for iteration {iter + 1}: {scores[iter]}")
    print("Average Score: ", sum(scores)/k)

if __name__ == "__main__":
    k = int(input("Enter the number of runs: "))
    main(k)