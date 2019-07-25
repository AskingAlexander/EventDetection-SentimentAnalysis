# EventDetection-SentimentAnalysis
The main module will run the ED and SA Tasks.

For the configuration, call:
```python main.py --help```

Sample run for SA, it will run both models against the sample files:
```python main.py --taskType SA```

Sample run for ED, it will run both methods against the sample files:
```python main.py --taskType ED```

## SA over ED
In order to run Sentiment Analysis over Event Detection you need to run the main from ```ResultAggregation/Program.cs``` and follow the instructions from the README. (It will need to have both the ED and SA results before running)

## Prerequisites:
Read the SA and ED READMEs.