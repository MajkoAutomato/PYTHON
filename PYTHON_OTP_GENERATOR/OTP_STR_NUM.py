# Importing library
import random
import string

# Declaring variables
def generate_random_password(size):
    password = ''.join([random.choice(string.ascii_letters + string.digits) for _ in range(size)])
    return password

# Driver code
if __name__ == "__main__":
    password_length = 13
    generated_password = generate_random_password(password_length)
    print(generated_password)
