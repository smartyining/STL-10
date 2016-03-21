#!/usr/bin/python

import sys

table1 = []
table2 = []

for line in sys.stdin:
	[t, key] = line.strip().split(',', 1)
	# print key

	if int(t) == 1:
		table1.append(key)
	elif int(t) == 2:
		table2.append(key)



for i in table1:
	print '%s\t%s' %(i, table2)




