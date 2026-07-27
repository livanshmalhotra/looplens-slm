import sys
import time
import requests
import colorama
from colorama import Fore, Style

colorama.init(autoreset=True)

SERVER_URL = "http://localhost:8000/api/ask"

def main():
    print(Fore.CYAN + Style.BRIGHT + "=" * 65)
    print(Fore.CYAN + Style.BRIGHT + "  🤖 LoopLens SLM Terminal Interactive Chat Assistant")
    print(Fore.CYAN + Style.BRIGHT + "=" * 65)
    print(Fore.WHITE + "  Type any question about procurement spend, suppliers, contracts,")
    print(Fore.WHITE + "  or RFQs. Type 'exit' or 'quit' to terminate.\n")

    while True:
        try:
            user_input = input(Fore.GREEN + Style.BRIGHT + "User > " + Fore.RESET).strip()
            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit", "q"]:
                print(Fore.YELLOW + "\nGoodbye!")
                break

            t0 = time.time()
            print(Fore.MAGENTA + "LoopLens AI is thinking..." + Fore.RESET)

            try:
                response = requests.post(SERVER_URL, json={"question": user_input}, timeout=60)
                if response.status_code == 200:
                    data = response.json()
                    ans = data.get("answer", "")
                    sec = data.get("response_time_sec", round(time.time() - t0, 3))
                    model_str = data.get("model_name", "LoopLens SLM")

                    print("\n" + Fore.CYAN + Style.BRIGHT + f"LoopLens AI ({model_str}) [{sec}s]:")
                    print(Fore.WHITE + f"{ans}\n")
                else:
                    print(Fore.RED + f"Error from server (status {response.status_code}): {response.text}\n")
            except requests.exceptions.ConnectionError:
                print(Fore.YELLOW + f"[!] Could not connect to FastAPI server at {SERVER_URL}.")
                print(Fore.YELLOW + "    Please start the server first using: uvicorn src.serve:app --port 8000\n")

        except KeyboardInterrupt:
            print(Fore.YELLOW + "\nExiting chat session...")
            break

if __name__ == "__main__":
    main()
