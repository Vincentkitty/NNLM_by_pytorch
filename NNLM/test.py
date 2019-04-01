import NNLM
import codecs
with codecs.open("doc") as f:
    pro=NNLM.Processing(f)
    pro.run()
    print(pro.target_batch)
    print(pro.input_batch)

