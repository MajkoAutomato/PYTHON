# Importing the math module to use the sqrt function
import math  

def pythag(a, b):
    sum_of_squares = a * a + b * b  # Calculating the sum of squares of the two given numbers
    c = math.sqrt(sum_of_squares)  # Taking the square root of the sum to find the length of the hypotenuse

    return c  # Returning the length of the hypotenuse

# Prompting the user to input the lengths of the legs
a = float(input("Enter the length of side 'a' (leg of the right-angled triangle): "))
b = float(input("Enter the length of side 'b' (leg of the right-angled triangle): "))

# Calculating the length of the hypotenuse using the pythag function
solution = pythag(a, b)

print("The length of the hypotenuse (side 'c') is:", solution)  # Printing the value of the hypotenuse
