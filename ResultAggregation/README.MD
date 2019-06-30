# Result Aggregation
This is the final step of our journey:

 1. We need the tags for each tweet by both LR (Datasets\LRPredictions) and SVM  (Datasets\SVMPredictions)
 2. We also need the true tags of Sentiment140, preferably transformed (Datasets\[SA]Sentiment140_CleanShave)
 3. We need the topics of OLDA (Datasets\OLDA_CLEAN) and MABED (Datasets\MABED_CLEAN)
 4. Now we try to match the tweets with the topics based on the number of common words and then give a label based on the majoritarian one
 5. After that we can aggregate the results and draw some conclusions

This was written in C# as it runs way faster in a compiled program rather than on python.
