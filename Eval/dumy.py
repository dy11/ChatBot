import  pickle

example = []
example.append([[[1,2,3],["a","b","c"]],[[1,3,4,],["a","c","d"]]])
example.append([[[4,5,6],["d","e","f"]],[[2,3,4,],["b","c","d"]]])

pickle.dump(example,open('db','wb'))
