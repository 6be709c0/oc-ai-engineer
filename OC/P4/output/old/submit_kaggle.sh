#!/bin/bash  
  
# Iterate over each CSV file in the ./submissions directory  
for file in $(find ./submissions/v2 -name "*.csv"); do  
    # Extract the file name without the directory path  
    filename=$(basename "$file")  
      
    # Submit the file using the kaggle command  
    kaggle competitions submit -c home-credit-default-risk -f "./submissions/v2/$filename" -m "V2"  
done  