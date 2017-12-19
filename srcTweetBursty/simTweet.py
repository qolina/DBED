from argparse import ArgumentParser

def get_args():
    parser = ArgumentParser(description='Event related tweets(by openie).')
    parser.add_argument('-input_file', type=str, help="input file path")
    parser.add_argument('-input_dir', type=str, help="input file dir path")
    parser.add_argument('-in_format', type=str, default="ollie", help="format of input file(s): ollie or reverb")
    args = parser.parse_args()
    return args

# in eg: 0.549: (she; had down; her safety drills)
# out: (score, [sub, rel, obj])
def read_triple(tripl_str):
    score = float(tripl_str[:tripl_str.find(": ")])
    tripl = tripl_str[tripl_str.find("("):-1].split("; ")
    return (score, tripl)
    

# return: sentence: [triples]
def read_openie(filename, in_format="ollie"):
    content = open(filename, "r").read()
    if in_format == "ollie":
        output_openie = read_ollies(content)
    elif in_format == "reverb":
        output_openie = read_reverbs(content)
    else:
        print "##Wrong format!"

def read_reverbs(content):
    output_openie = []
    return output_openie

def read_ollies(content):
    output_openie = []
    sent_structs = content.split("\n\n")
    for sent_struct in sent_structs:
        struct_arr = sent_struct.split("\n")
        sent = struct_arr[0]
        triples = struct_arr[1:]
        triples = [read_triple(item) for item in triples]

        output_openie.append((sent, triples))
        #print output_openie[-1]
        #break
    return output_openie
    
if __name__ == "__main__":
    
    args = get_args()

    read_openie(args.input_file, in_format=args.in_format)
