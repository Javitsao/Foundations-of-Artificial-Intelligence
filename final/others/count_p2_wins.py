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


def main(total_runs):
    p2_wins = 0

    results = []
    for game_number in range(1, 8):
        for _ in range(total_runs):
            if run_game_and_check_winner(game_number):
                p2_wins += 1
            for i, result in enumerate(results, 1):
                print(f"\nPlayer 2 won {result} out of {total_runs} games in start_game_{i}.py.")
            print(f"Player 2 won {p2_wins} out of {_ + 1} games in start_game_{game_number}.py.")
        results.append(p2_wins)
        p2_wins = 0

    for i, result in enumerate(results, 1):
        print(f"\nPlayer 2 won {result} out of {total_runs} games in start_game_{i}.py.")

if __name__ == "__main__":
    total_runs = int(input("Enter the number of runs: "))
    main(total_runs)
