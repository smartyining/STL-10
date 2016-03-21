#!/usr/bin/python

import sys
import re

delimiters = "['", "', '", "']"
regexPattern = '|'.join(map(re.escape, delimiters))
output_file = open(sys.argv[1],'w')
#key = None

#input comes from STDIN (stream data that goes to the program)
for line in sys.stdin:
    line =  line.strip()
    key, value = line.split('\t',1)
    value_lst = re.split(regexPattern, value)

    for i in value_lst:
        if len(i) > 0:
        	output_file.write('%s,%s\n' %(key, i))
            #print '%s,%s' %(key, i)
