def count_vowels(string: str):
    count: int = 0
    for letter in string:
        if letter in 'aeiouAEIOU':
            count += 1
    return count

a:int
b:str
c:float
d:bool
e = None

class Student:
    def __init__(self, name: str, age: int):
        self.name = name
        self.age = age

    def __str__(self) -> str:
        return "Student {name: " + self.name + " age: " + str(self.age) + "}"

    def __eq__(self, other):
        return self.name == other.name and self.age == other.age

    def hash(self):
        return hash(self.name) * 30 + hash(self.age)

student:Student = Student("John", 22)

def check_age():
    age:int = int(input("Please enter your age: "))

    if age < student.age:
        print("Age is less than available student age")
    elif age > student.age:
        print("Age is greater than available student age")
    else:
        print("Student age is equal to available student age")

