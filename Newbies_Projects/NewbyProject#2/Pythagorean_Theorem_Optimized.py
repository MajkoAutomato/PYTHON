# Importing only the sqrt function from the math module
from math import sqrt  

def pythag(a, b):  # Function definition for calculating the length of the hypotenuse of a right-angled triangle
    return sqrt(a * a + b * b)  # Returning the square root of the sum of squares of the two given numbers (Pythagorean theorem)


# Prompting the user to input the lengths of the legs
a = float(input("Enter the length of side 'a' (leg of the right-angled triangle): "))
b = float(input("Enter the length of side 'b' (leg of the right-angled triangle): "))

# Calculating the length of the hypotenuse using the pythag function
solution = pythag(a, b)

print("The length of the hypotenuse (side 'c') is:", solution)  # Printing the value of the hypotenuse
