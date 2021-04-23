from git import Repo
import os
from shutil import copy as copyFile

def main():
	###########################################################################################################################################################################
	if not os.path.isdir("build"):
		os.system('mkdir build')
	os.system('cd build/ && cmake .. && make')

	print("------------------------")
	print("Setup faimGraph done")
	print("------------------------")

if __name__ == "__main__":
	main()