#!/usr/bin/env Python
# -*- coding: utf-8 -*-

### Names: Abby Bechtel, Nic Cordova, Nate Everett, Suleiman Karkoutli, Matthew Parnham
# IDs: 2312284, 2302109, 2296318, 2275013, 2287511
# Emails: abechtel@chapman.edu, cordova@chapman.edu, everett@chapman.edu, karko101@chapman.edu, parnham@chapman.edu
# Course: CPSC393 Interterm 2020
# Assignment: Final
###

"""
This program module uses the face_recognition package to identify visitors to the International Space Station in photos.

The classifier is first trained on a set of labeled (known) faces and can then predict the person
in an unknown image by finding the similar faces (images with closet face-features under eucledian distance)
in its training set, and performing a majority vote.

Usage:

1. Properly run the encode_faces.py program and encode the desired training data photos

2. Run the command: python recognize_faces_image.py --encodings encodings.pickle --image examples/test_image.jpg
"""

import face_recognition
import argparse
import pickle
import cv2
import os
import operator
import math
import sys

#convert a distance to a confidence through normalization, not exact percent
def face_distance_to_conf(face_distance, face_match_threshold=0.6):
	if face_distance > face_match_threshold:
		range = (1.0 - face_match_threshold)
		linear_val = (1.0 - face_distance) / (range * 2.0)
		return linear_val
	else:
		range = face_match_threshold
		linear_val = 1.0 - (face_distance / (range * 2.0))
		return linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))

#reads an image in from given command line arg
def readImage(im):
	# load the input image and convert it from BGR to RGB
	image = cv2.imread(im)

	if image is None: #the image file could not be loaded
		raise TypeError

	r = 1000.0 / image.shape[1]
	dim = (1000, int(image.shape[0] * r))

	# perform the actual resizing of the image
	image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
	return image

def printToFile(im, fNames, fDist):
	# Get the name of the image to place in output folder
	imgName = im
	imgName = imgName.replace(".png","")
	imgName = imgName.replace(".jpg", "")
	imgName = imgName.replace(".jpeg", "")
	imgName = imgName.replace('examples/', '')

	currPath = os.path.dirname(os.path.realpath(__file__))
	subdir = "output"
	filename = imgName
	newPath = os.path.join(currPath, subdir, filename)

	try:
		f = open(newPath + ".txt", "w")
		f.write("name, confidence\n")
		for x,y in zip(fNames, fDist):
			confidence = face_distance_to_conf(y, 0.5)
			f.write(x + ", " + str(confidence) + "\n")
		f.close()

	except IOError:
		print("Wrong path provided")

#makes the box over the face in the image
def makeBox(fname, fnames, im, fBoxes):
	# loop over the recognized faces
	for ((top, right, bottom, left), fname) in zip(fBoxes, fnames):
		# draw the predicted face name on the image
		cv2.rectangle(im, (left, top), (right, bottom), (0, 255, 0), 2)
		y = top - 15 if top - 15 > 15 else top + 15
		cv2.putText(im, fname, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
			0.75, (0, 255, 0), 2)

def main():
	#parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-e", "--encodings", required=True,
		help="path to serialized db of facial encodings")
	ap.add_argument("-i", "--image", required=True,
		help="path to input image")
	ap.add_argument("-d", "--detection-method", type=str, default="cnn",
		help="face detection model to use: either `hog` or `cnn`")
	args = vars(ap.parse_args())

	# load the known faces and embeddings
	print("[INFO] loading encodings...")
	data = pickle.loads(open(args["encodings"], "rb").read())

	try:
		image = readImage(args["image"])
	except TypeError:
		print("The file image file was not properly loaded")
		sys.exit(1)

	rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	# detect the (x, y)-coordinates of the bounding boxes corresponding
	# to each face in the input image, then compute the facial embeddings for each face
	print("[INFO] recognizing faces...")
	boxes = face_recognition.face_locations(rgb,
		model=args["detection_method"])
	encodings = face_recognition.face_encodings(rgb, boxes)

	# initialize the list of names and distances for each face detected
	names = []
	distances = []

	# loop over the facial embeddings
	for encoding in encodings:
		# attempt to match each face in the input image to our known
		# encodings
		matches = face_recognition.compare_faces(data["encodings"],
			encoding, tolerance = 0.5)
		#see how far apart the test image is from the known faces
		faceDistances = face_recognition.face_distance(data["encodings"], encoding)
		name = "Unknown"

		# check to see if we have found a match
		if True in matches:
			# find the indexes of all matched faces then initialize a
			# dictionary to count the total number of times each face was matched
			matchedIdxs = [i for (i, b) in enumerate(matches) if b]
			counts = {}
			distDict = {}

			# loop over the matched indexes and maintain a count for each recognized face
			for i in matchedIdxs:
				name = data["names"][i]
				counts[name] = counts.get(name, 0) + 1

			for i, faceDistance in enumerate(faceDistances):
				distDict[data["names"][i]] = faceDistance

			# determine the recognized face with the largest number of votes
			if name in names:
				secondVote = list(sorted(counts.values()))[-2]
				name = list(counts.keys())[list(counts.values()).index(secondVote)]

			else:
				name = max(counts, key=counts.get)

			distance = distDict.get(name)

		# update the list of names and distances
		names.append(name)
		distances.append(distance)

	printToFile(args['image'], names, distances)
	makeBox(name, names, image, boxes)

	# show the output image
	cv2.imshow("Image", image)
	cv2.waitKey(0)

if __name__ == "__main__":
	main()
