import hashlib
import json
from tqdm import tqdm
from collections import defaultdict

# List of popular dog, girls, boys names
# capital_names = ["",
#     "Amelia", "Abigail", "Ava", "Alexander", "Bailey", "Bella", "Benjamin", "Luna", "Charlie", "Daisy",
#     "Dana", "Daniel", "Emma", "Evelyn", "Henry", "Isabella", "John", "James", "Keren", "Liam",
#     "Lucas","Lucy", "Leo", "Milo", "Max", "Mia", "Michael", "Noah", "Oliver", "Olivia", "Rocky",
#     "Sophia", "William", "Spring", "Summer", "Autumn", "Winter", "A", "B", "C", "D", "E", "F", "G",
#     "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"
# ]

directory_path = "password_strength_text_files/"
#
with open(directory_path + "word.txt", 'r') as f:
    word_list = f.readlines()
word_set = set()
for w in word_list:
    word_set.add(w.replace('\n', ''))
word_list = sorted(list(word_set), key=len)
word_list.append("")

with open(directory_path + "cracking_number_list_with_dates.txt", 'r') as f:
    number_list = f.readlines()
num_set = set()
for w in number_list:
    num_set.add(w.replace('\n', ''))
number_list = sorted(list(num_set), key=len)
number_list.append("")

# with open(directory_path + "cracking_name_list.txt", 'r') as f:
#     name_list = f.readlines()
# name_set = set()
# for w in name_list:
#     name_set.add(w.replace('\n', ''))
# name_list = sorted(list(name_set))
# name_list.append("")


# with open(directory_path + "rockyou_popular_words.txt", 'r') as f:
#     rockyou_popular_words = f.readlines()
# word_set = set()
# for w in rockyou_popular_words:
#     word_set.add(w.replace('\n', ''))
# rockyou_popular_words = sorted(list(word_set), key=len)
# rockyou_popular_words.append("")
#
# with open(directory_path + "rockyou_popular_numbers.txt", 'r') as f:
#     rockyou_popular_numbers = f.readlines()
# num_set = set()
# for w in rockyou_popular_numbers:
#     num_set.add(w.replace('\n', ''))
# rockyou_popular_numbers = sorted(list(num_set), key=len)
# rockyou_popular_numbers.append("")


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
    # base_numbers = ["123456", "12345", "1234", "123", "12", "1"]
    filename = r"cracking_text_files/first_10000_lines.txt"
    # filename = r"cracking_text_files/first_1000_lines.txt"
    json_file_name = "cracking_text_files/lower_wordlist_plus_cracking_number_list_without_dates_"
    found_passwords = list()
    name_count = dict()
    number_count = defaultdict(int)
    names_not_found = list()
    all_results = []
    old_len = 0
    for base_word in tqdm(word_list):
        for base_number in number_list:

            new_password = base_word.lower() + base_number

            hashed_password = hash_password_sha1(new_password)

            counts = count_hash_in_file(hashed_password, filename)

            if counts:
                for line, count in counts.items():
                    if count > 0:
                        result_message = {
                            "hash": hashed_password,
                            "count": count,
                            "line": line,
                            "detected_password": new_password,
                            "base_number": base_number,
                            "base_word": base_word
                        }
                        number_count[base_number] += 1
                        all_results.append(result_message)
                        found_passwords.append(new_password)

        if len(all_results) == old_len:
            names_not_found.append(base_word)
        else:
            name_count[base_word] = len(all_results) - old_len
        old_len = len(all_results)

    if all_results:
        json_output_file = json_file_name + str(len(all_results)) + ".json"
        with open(json_output_file, 'w') as json_file:
            json.dump(all_results, json_file, indent=4)
        print(f"\nfound {len(all_results)} passwords")
        print(f"All results have been written to {json_output_file}")
    else:
        print("No hashes found.")

    with open("password_strength_text_files/cracked_passwords.txt", 'w', encoding='utf-8') as file:
        for password in found_passwords:
            file.write(password + '\n')

    # sorted_name_dict = dict(sorted(name_count.items(), key=lambda item: item[1], reverse=True))
    # print(f"NAMES BY POPULARITY: {len(sorted_name_dict)}")
    # for k,v in sorted_name_dict.items():
    #     print(f"{k}\t:\t{v}")
    #     # print(f"{k}")
    #
    # sorted_number_dict = dict(sorted(number_count.items(), key=lambda item: item[1], reverse=True))
    # print(f"NUMBERS BY POPULARITY: {len(sorted_number_dict)}")
    # for k, v in sorted_number_dict.items():
    #     print(f"{k}\t:\t{v}")
    #     # print(f"{k}")
    #
    # print(f"NAMES NOT FOUND: {len(names_not_found)}")
    # for k in sorted(names_not_found):
    #     print(f"XX \t {k}")


if __name__ == "__main__":
    main()
