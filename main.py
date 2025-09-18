from laboratory_works.lab1vowels import count_vowels


def main():
    hint:str = "Enter text: "
    for i in range(5):
        count:int = count_vowels(input(hint))
        print(f"Vowels count: {count}\n")

main()

