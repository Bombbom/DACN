import re
import os
import warnings
import numpy as np
from gensim.models import Word2Vec
import pandas

def parse_file(filename):
    df = pandas.read_csv(filename, index_col = None)  
    for i in range(len(df)):
        fragment = []
        
        row = list(df.loc[i]) 
        fragment_val = int(row[8]) 
        fragment = row[4]


        
        yield fragment, fragment_val   
        
def split_line_opcode(opcode: str) -> list[list]:
    opcode = opcode.split()
    res = []
    tmp = []
    for i in opcode:
        if i == "JUMPDEST":
            res.append(tmp)
            tmp = []
            tmp.append(i)        
        else:
            tmp.append(i)
    if len(tmp)>0:
        res.append(tmp)   
    return res       
        

 
if __name__ == "__main__":
    
    filename = r"10000_reentrancy.csv"
    sentences = []
    fragments = []
    vector_length = int(100)
    for fragment, val in parse_file(filename):
        opcode = split_line_opcode(fragment)
        sentences.extend(opcode)
        row = {"fragment": fragment.split(), "val": val}
        fragments.append(row)
    model = Word2Vec(sentences, min_count=1, vector_size=vector_length, sg=0)
    # model.save("word2vec.model")
    # print(model.wv)
    vectors = []
    for fragment in fragments:
        # print(type(fragment))
        # print(fragment["fragment"] )
        # break
        vector = np.zeros(shape=(300, vector_length))
        for i in range(min(len(fragment["fragment"]), 300)):
            vector[i] = model.wv[fragment['fragment'][i]]
        row = {"vector": vector, "val": fragment["val"]}
        vectors.append(row)
    df = pandas.DataFrame(vectors)
    df.to_pickle("pre_train_5000_opcode.pkl")