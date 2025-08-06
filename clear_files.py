import json

def clear():
# Overwrite with an empty dictionary
    with open("embeddings.json", "w") as file:
        pass  # Use [] if it's meant to be a list
    # Open the text file in write mode, which clears it
    with open("policy.txt", "w") as file:
        pass  # No need to write anything — this clears the file
