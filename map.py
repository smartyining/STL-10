#!/usr/bin/python

import sys

table1 = []
table2 = []
course_file = sys.argv[1]
professor_file = sys.argv[2]

lines = open(course_file).readlines() +  open(professor_file).readlines()

for line in lines :
	[t, key] = line.strip().split(',', 1)
	# print key

	if int(t) == 1:
		table1.append(key)
	elif int(t) == 2:
		table2.append(key)



for i in table1:
	print '%s\t%s' %(i, table2)




