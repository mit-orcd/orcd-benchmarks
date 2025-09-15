#!/bin/bash

max=6
numlist=(1)
i=1
while [ $((i * 4)) -lt $max ]; do
	i=$((2*i))
	numlist+=($i)
done
numlist+=($((max / 2)) $max $((3 * max / 2)) $((2 * max)))

echo "Here's the list ${numlist[*]}"

for i in ${!numlist[@]}; do
	echo "${numlist[i]}"
done

		
