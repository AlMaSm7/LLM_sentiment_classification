import datasets

# Example: Loading the raw English annotations directly from the GitHub URL
url = "https://raw.githubusercontent.com/Helsinki-NLP/XED/master/"

languages = {

    "english": "https://raw.githubusercontent.com/Helsinki-NLP/XED/master/AnnotatedData/en-annotated.tsv",
    "swedish": "https://raw.githubusercontent.com/Helsinki-NLP/XED/master/Projections/sv-projected.txt",
    "german": "https://raw.githubusercontent.com/Helsinki-NLP/XED/master/Projections/de-projected.txt"
}

# The files are tab-separated (sentence \t labels)
df = pd.read_csv(url, sep='\t', header=None, names=['sentence', 'labels'])
print(df.head())