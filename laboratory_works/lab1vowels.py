
def count_vowels(string: str) -> int:
    count:int = 0
    for letter in string:
        if letter in 'aeiouAEIOU':
            count += 1
    return count


def main():
    hint:str = "Enter text: "
    for i in range(5):
        count:int = count_vowels(input(hint))
        print(f"Vowels count: {count}\n")

main()