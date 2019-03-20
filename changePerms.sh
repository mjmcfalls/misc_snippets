#!/bin/sh

filename="$1"
group="$2"

while read -r line; do
        name="$line"
        echo "chmod -R +a group $group $cmd $line"
done < "$filename"

