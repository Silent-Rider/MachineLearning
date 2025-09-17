def count_vowels(string: str) -> int:
    count: int = 0
    for letter in string:
        if letter in 'aeiouAEIOU':
            count += 1
    return count
