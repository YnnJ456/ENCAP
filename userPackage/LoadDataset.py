import os
import re
import sys


class LoadDataset:
    def __init__(self):
        pass

    def readFasta(self, File, minSeqLength=5):
        if not os.path.exists(File):
            print('Error: "' + File + '" does not exist.')
            sys.exit(1)
        with open(File) as f:
            records = f.read()
        if re.search('>', records) is None:
            print('The input file seems not in fasta format.')
            sys.exit(1)
        records = records.split('>')[1:]
        SeqDict = {}
        for fasta in records:
            array = fasta.split('\n')
            name = array[0].split()[0]
            sequence = re.sub('[^ARNDCQEGHILKMFPSTWYV-]', '-', ''.join(array[1:]).upper())
            if len(sequence) >= minSeqLength:
                SeqDict[name] = sequence
        return SeqDict

