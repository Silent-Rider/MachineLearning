def count_vowels(string: str):
    count: int = 0
    for letter in string:
        if letter in 'aeiouAEIOU':
            count += 1
    return count
