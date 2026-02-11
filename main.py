import subprocess
# import os
# import threading
# import itertools # Для роботи з зовнішніми процесами
# import time
# import sys

def run_ollama(user_request, model="llama3"):
    process = subprocess.Popen(
        ["ollama", "run", model],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    prompt = f"""
    # Ти - це бот, який відповідає на запитання користувача. Відповідай тільки українською, коротко і по суті. 
    # Якщо ти не знаєш відповіді, скажи "Я не знаю". 
    
    Ти - це бот, який відповідає на запитання користувача завжди лайкою та матюками . Відповідай тільки українською, образливо та дотепно. 
    Якщо ти не знаєш відповіді, скажи "Йди звідси". 

    Ось запитання користувача: {user_request}
    """

    output, error = process.communicate(input=prompt)

    if error:
        print(f"Error: {error}")
    return output.strip()

if __name__ == "__main__":
    while True:
        user_input = input("Ти: ")
        if user_input.lower() in ["exit", "quit", "вихід", "стоп"]:
            print("Exiting...")
            break
        response = run_ollama(user_input)
        print(f"Бот: {response}")