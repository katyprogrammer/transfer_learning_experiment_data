import sys
import subprocess
import multiprocessing as mp

def A():
    subprocess.call('python 20newsgroup.py -r A -d A.pkl -e 10000 > A.txt', shell=True)
def B():
    subprocess.call('python 20newsgroup.py -r B -d B.pkl -e 10000 > B.txt', shell=True)

# runA, runB = mp.Process(target=A), mp.Process(target=B)
# runA.start()
# runA.join()
# runB.start()
# runB.join()

def runR(r, i):
    subprocess.call('python 20newsgroup.py -r B -l A.pkl -d B_{0}.pkl -e 10000 -R {0} > B_{0}_{1}.txt'.format(r, i), shell=True)
def runRevR(r):
    subprocess.call('python 20newsgroup.py -r A -l B.pkl -d A_{0}.pkl -e 10000 -R {0} > A_{0}.txt'.format(r), shell=True)

runs = []
tn = 0
# all layers
R = [1,5,10,50,100,200]
tn = 0
for i in range(5):
    for r in R:
        AB = mp.Process(target=runR, args=(r,i))
        AB.start()
        AB.join()
        # BA = mp.Process(target=runRevR, args=(r,))
        # BA.start()
        # BA.join()
