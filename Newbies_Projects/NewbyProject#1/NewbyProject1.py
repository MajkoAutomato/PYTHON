# Importing library
import random

# Shows print text in console
print('Your password: ')

# Defining variables
chars = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ,.-;:!$§%&/()=?@'

# Store input
password = ''

# Driver code
for x in range(16):
    password += random.choice(chars)

print(password)

