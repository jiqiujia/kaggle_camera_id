import os
import sys
from PIL import Image

def compressMe(subdir, file, verbose=False):
	filepath = os.path.join(subdir, file)
	oldsize = os.stat(filepath).st_size
	picture = Image.open(filepath)
	dim = picture.size
	
	#set quality= to the preferred quality. 
	#I found that 85 has no difference in my 6-10mb files and that 65 is the lowest reasonable number
	picture.save(os.path.join(subdir, "Compressed70_"+file),"JPEG",quality=70) #,optimize=True
	picture.save(os.path.join(subdir, "Compressed90_"+file),"JPEG",quality=90)

	print(filepath)
	#newsize = os.stat(os.path.join(os.getcwd(),"Compressed_"+file)).st_size
	#percent = (oldsize-newsize)/float(oldsize)*100
	#if (verbose):
		#print "File compressed from {0} to {1} or {2}%".format(oldsize,newsize,percent)
	#return percent

def main():
	verbose = False
	#checks for verbose flag
	if (len(sys.argv)>1):
		if (sys.argv[1].lower()=="-v"):
			verbose = True

	#finds present working dir
	pwd = os.getcwd()

	tot = 0
	num = 0
	folder = '/media/dl/data1/datasets/kaggle_camera/train2/'
	for subdir, dirs, files in os.walk(folder):
		for file in files:
			compressMe(subdir, file, verbose)
	print "Average Compression: %d" % (float(tot)/num)
	print "Done"

if __name__ == "__main__":
	main()