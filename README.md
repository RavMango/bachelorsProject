# Exploring Architectural Knowledge in Open Source System Mailing Lists

This repository contains the dataset (*datasetExported.xlsx* & *dataset.atlproj23*) as well as the parsing tool (*akConceptParser.py*).

## akConceptParser.py

The `akConceptParser.py` script parses and analyzes the AK concepts within the `datasetExported.xlsx`. 

### Features

- **Data Loading and Cleaning**: Load the dataset, sort the rows accordingly to the order they were manually parsed, and clean the data by removing empty rows.
- **Combination Generation**: Generate all possible combinations of AK concepts and tags.
- **Counting Co-occurrences**: Count the occurrences of individual and combined AK concepts and tags within emails.
- **Statistical Analysis**: Perform statistical tests to analyze the significance of the co-occurrences.
- **Visualization**: Create visual representations of the data distribution and analysis results.

## Requirements
The script requires the following libraries to be installed:

- `pandas`
- `numpy`
- `itertools`
- `matplotlib`
- `scipy`
- `seaborn`

## Usage 

You can change the `codes_to_count`, `tags`, and `ignore_tags` variables to include the AK concepts and tags you want to analyze. The script will then count the occurrences of these concepts and tags within the dataset.

```python
# Here is where the codes that are to be counted are defined
# (make sure they're in alphabetical order)
codes_to_count = [
    "Assumption", "Component Behavior", "Constraints", "Decision Rule", "Design Configuration", 
    "Quality Issue", "Requirements", "Solution Benefits and Drawbacks", "Solution Comparison", 
    "Solution Evaluation", "Solution Risks", "Solution Trade-off"
]

# Tags to count (make sure they're in alphabetical order)
tags = ["tag:existence", "tag:property", "tag:technology"]

# Tags to ignore (also alphabetical order)
ignore_tags = ["tag:notag", "tag:other", "tag:process"]
```

The `filename` can also be altered. As long as it is an exported .xlsx file from an Atlas.ti project, the script will be able to parse it.
``` python
if __name__ == "__main__":
    filename = 'datasetExported.xlsx'
    counter(filename)
```
