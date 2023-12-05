
# Import library
import random

# Declaring variable and set range of length
def generate_otp(length=6):
    characters = '0123456789'
    return ''.join(random.choice(characters) for _ in range(length))

# Driver Code
if __name__ == "__main__":
    otp_length = 6
    generated_otp = generate_otp(otp_length)
    print(f"OTP of length {otp_length}: {generated_otp}")

