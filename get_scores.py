import sys




def process(filename):
       
       i = 0
       sent = 0
       score = 0
       with open(filename, 'r') as f:
            for line in f:
                 if i%4==3:
                       line = line.strip().split('\t')[-1]
                       line = line.split()
                       print(line)
                       line = [-float(l) for l in line]
                       score += sum(line)
                       sent +=1
                                
                 i+=1
       print(score/sent)



if __name__=="__main__":
       process(sys.argv[1])
                       
                       
