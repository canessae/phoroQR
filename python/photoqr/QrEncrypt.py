import os

from Crypto.Cipher import PKCS1_OAEP
from Crypto.PublicKey import RSA

class QrEncrypt:

    privkey = None
    pubkey = None

    __enc_params = {
        "PRIVATE_KEY": "private",
        "PUBLIC_KEY": "public"
    }

    def __init__(self, params):
        self.__enc_params = params
        self.privkey, self.pubkey = self.__read_keys()
        self.cipher = PKCS1_OAEP.new(self.privkey)

    def __generate_keys(self):
        # Generate keys
        random_generator = Random.new().read
        key = RSA.generate(1024, random_generator)  # generate public and private keys

        # Save public key
        # print("public:  " + str(key.publickey().exportKey()))
        fid = open(self.PUBLIC_KEY, "wb")
        fid.write(key.publickey().export_key())
        fid.close()

        # Save private key
        # print("private: " + str(key.export_key()))
        fid = open(self.PRIVATE_KEY, "wb")
        fid.write(key.export_key())
        fid.close()
        return key

    def __read_keys(self):
        privatekey = publickey = None

        try:
            with open(self.__enc_params["PRIVATE_KEY"], "r") as fid:
                privatekey = RSA.import_key(fid.read())
        except IOError:
            pass

        try:
            with  open(self.__enc_params["PUBLIC_KEY"], "r") as fid:
                publickey = RSA.import_key(fid.read())
        except IOError:
            pass

        return privatekey, publickey

    def decryptMessage(self, msg):
        default_length = 128
        encrypt_byte = bytes(msg.decode(), 'iso_8859_1')
        length = len(encrypt_byte)
        # print(type(encrypt_byte))
        # print(length)
        # print(encrypt_byte)

        if length <= default_length:
            decrypt_byte = self.cipher.decrypt(encrypt_byte)
        else:
            offset = 0
            res = []
            while length - offset > 0:
                if length - offset > default_length:
                    res.append(self.cipher.decrypt(encrypt_byte[offset:offset + default_length]))
                else:
                    res.append(self.cipher.decrypt(encrypt_byte[offset:]))
                offset += default_length
            decrypt_byte = b''.join(res)
        decrypted = decrypt_byte.decode()
        return decrypted

    def encryptMessage(self, msg):
        rsa = PKCS1_OAEP.new(self.privkey)
        return rsa.encrypt(str(msg).encode())