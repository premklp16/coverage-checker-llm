import json

def clear():
    try:
        with open("embeddings.json", "w") as file:
            pass  # Write empty list for JSON
        with open("policy.txt", "w") as file:
            pass  # Clears the file
    except FileNotFoundError:
        pass  # Ignore if files don’t exist
    except Exception as e:
        print(f"Error clearing files: {e}")