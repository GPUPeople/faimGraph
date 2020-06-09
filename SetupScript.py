from git import Repo
import os
from shutil import copy as copyFile

def main():
	###########################################################################################################################################################################
	print("------------------------")
	print("Setup CUB")
	print("------------------------")
	if not os.path.isdir("include/cub"):
		Repo.clone_from("https://github.com/NVlabs/cub.git", "include/cub")
		print("CUB cloning done!")
	else:
		print("CUB already cloned")

	if not os.path.isdir("build"):
		os.system('mkdir build')
	os.system('cd build/ && cmake .. && make -j4')

	print("------------------------")
	print("Setup faimGraph done")
	print("------------------------")

if __name__ == "__main__":
	main()