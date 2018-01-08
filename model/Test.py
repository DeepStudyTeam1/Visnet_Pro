import os

print (os.getcwd())
path = os.path.split(os.getcwd())[0]
print(path + "/data/street2shop")