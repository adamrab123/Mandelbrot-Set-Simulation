import os
temp = []
chars = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"]
for char in chars:
	temp.append(char + "a")
	temp.append(char + "b")
	temp.append(char + "c")
	temp.append(char + "d")

for i in range(1,101):
	os.system("mv output/{}.bmp output/{}.bmp".format(i,temp[i]))