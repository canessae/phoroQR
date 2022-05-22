import base64

import Crypto
from Crypto.PublicKey import RSA
from Crypto import Random
from Crypto.Cipher import AES, PKCS1_OAEP

def generate_keys():
    # Generate keys
    random_generator = Random.new().read
    key = RSA.generate(1024, random_generator)  # generate public and private keys

    return key

if __name__ =="__main__":
    key = generate_keys()
    msg = "vediamo quanto lo allunga"
    cipher = PKCS1_OAEP.new(key.public_key())
    encrypted = cipher.encrypt(msg.encode())
    print(len(encrypted))
    print(encrypted)

    cipher = PKCS1_OAEP.new(key)
    decrypted = cipher.decrypt(encrypted)
    print(decrypted)
    print(type(encrypted))