def check_coordinates_diff_by_one(coords):
    # Iterate through the list of coordinates starting from the second coordinate
    for i in range(1, len(coords)):
        # Get the current and previous coordinates
        current_coord = coords[i]
        previous_coord = coords[i - 1]
        
        # Check if the absolute difference between x and y coordinates is exactly 1
        if abs(current_coord[0] - previous_coord[0]) != 1 and abs(current_coord[1] - previous_coord[1]) != 1:
            # If the condition is not satisfied, return False
            print("false")
    
    # If the loop completes without returning False, return True
    print("true")

# Given list of coordinates

