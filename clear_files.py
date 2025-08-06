import json

def clear():
# Overwrite with an empty dictionary
    with open("embeddings.json", "w") as file:
        pass  
    # Open the text file in write mode, which clears it
    with open("policy.txt", "w") as file:
        pass  

