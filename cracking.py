import hashlib
import json

# List of popular dog, girls, boys names
names = [
    "bella", "luna", "max", "charlie", "daisy", "bailey", "lucy", "leo", "milo", "rocky",
    "olivia", "emma", "ava", "sophia", "isabella", "mia", "amelia", "keren", "evelyn", "abigail",
    "Liam", "Noah", "Oliver", "john", "James", "William", "Benjamin", "Lucas", "Henry", "Alexander",
    "spring", "summer", "autumn", "winter"
]

def hash_password_sha1(password: str) -> str:
    sha1 = hashlib.sha1()
    sha1.update(password.encode())
    hashed = sha1.hexdigest().upper()
    return hashed

def count_hash_in_file(hash_to_check: str, filename: str) -> dict:
    counts = {}
    try:
        with open(filename, 'r') as file:
            for line in file:
                if hash_to_check in line:
                    parts = line.strip().split(':')
                    if len(parts) > 1:
                        count = int(parts[-1])
                        counts[line.strip()] = count
    except FileNotFoundError:
        print(f"The file {filename} does not exist.")
    return counts

def main():
    base_passwords = ["123456", "12345", "1234", "123", "12", "1"]
    filename = r"cracking_text_files/first_10000_lines.txt"
    # filename = r"cracking_text_files/first_1000_lines.txt"
    json_output_file = r"cracking_text_files/combined_results.json"

    all_results = []

    for base_password in base_passwords:
        for name in names:
            new_password = name + base_password
            hashed_password = hash_password_sha1(new_password)
            print(f"Checking hash for password: {new_password} -> {hashed_password}")

            counts = count_hash_in_file(hashed_password, filename)

            if counts:
                for line, count in counts.items():
                    if count > 0:
                        result_message = {
                            "hash": hashed_password,
                            "count": count,
                            "line": line,
                            "new_password": new_password,
                            "base_password": base_password
                        }
                        print(f"Found: {result_message}")
                        all_results.append(result_message)
    if all_results:
        with open(json_output_file, 'w') as json_file:
            json.dump(all_results, json_file, indent=4)
        print(f"\nAll results have been written to {json_output_file}")
    else:
        print("No hashes found.")

if __name__ == "__main__":
    main()

