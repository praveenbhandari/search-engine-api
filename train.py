import os
import pickle
import suggestion
import nltk

# Example reading from a sample text file
""" base_file = open('combined_output.txt', 'rt')
raw_text = base_file.read()
base_file.close()
print("Text read from file : ",raw_text[:200]) """

# nltk.download('gutenberg')
# 
# print(nltk.corpus.gutenberg.fileids())
base_file = open('combined_output.txt', 'rt')
raw_text = base_file.read()
# raw_text = nltk.corpus.gutenberg.raw('/Users/praveenlawyantra/Desktop/Search-Engine-API/train.py')
base_file.close()
tokens = suggestion.tokenize(raw_text, pad_start=True, pad_end=True)
model = suggestion.train(tokens, num=2)

print("\nSample token list : ", tokens[:10])
print("\nTotal Tokens : ",len(tokens))

with open('final11.pickle', 'wb') as handle:
    pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)


