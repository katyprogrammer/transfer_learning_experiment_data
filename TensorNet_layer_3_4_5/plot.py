import optparse
import matplotlib.pyplot as plt
import pandas as pd
import re
import glob

EPOCH = 'Epoch (\d*).*\n.*training loss:\t\t.*\n.*validation loss:\t\t.*\n.*validation accuracy:\t\t.* %'
TLOSS = 'Epoch \d*.*\n.*training loss:\t\t(.*)\n.*validation loss:\t\t.*\n.*validation accuracy:\t\t.* %'
VLOSS = 'Epoch \d*.*\n.*training loss:\t\t.*\n.*validation loss:\t\t(.*)\n.*validation accuracy:\t\t.* %'
VACC = 'Epoch \d*.*\n.*training loss:\t\t.*\n.*validation loss:\t\t.*\n.*validation accuracy:\t\t(.*) %'
FINAL_LOSS = 'Final results:\n.*test loss:\t\t\t(.*)\n.*test accuracy:\t\t.* %'
FINAL_ACC = 'Final results:\n.*test loss:\t\t\t.*\n.*test accuracy:\t\t(.*) %'
BASE = None
REV = None

def parse_arg():
    parser = optparse.OptionParser('usage%prog [-i infile] [-o outfile] [-A all] [-r A->B or B->A]')
    parser.add_option('-i', dest='fin')
    parser.add_option('-o', dest='fout')
    parser.add_option('-A', dest='All', action='store_true', default=False)
    parser.add_option('-r', dest='rev', action='store_true', default=False)
    (options, args) = parser.parse_args()
    return options

def plot(fin,fout,title,isB=False):
    global Base
    with open(fin, 'r') as f:
        data = f.read()
    epoch = [float(x) for x in re.findall(EPOCH, data)]
    tloss = [float(x) for x in re.findall(TLOSS, data)]
    vloss = [float(x) for x in re.findall(VLOSS, data)]
    vacc = [float(x) for x in re.findall(VACC, data)]
    floss = float(re.findall(FINAL_LOSS, data)[0])
    facc = float(re.findall(FINAL_ACC, data)[0])
    if isB:
        Base = [vacc,facc]
    df = pd.DataFrame({'training loss':tloss, 'validation loss':vloss}, index=epoch)
    df.plot()
    plt.title(title)
    plt.plot([len(epoch)+1],[floss],marker='o')
    plt.annotate('test loss = {0}'.format(floss), xy=(len(epoch)+1,floss), xytext=(len(epoch)*0.9,floss*1.1))
    plt.xlabel('epoch')
    plt.savefig('loss_{0}.png'.format(fout))
    plt.clf()
    if isB:
        df = pd.DataFrame({'validation accuracy':vacc}, index=epoch)
    else:
        if not REV:
            df = pd.DataFrame({'validation accuracy':vacc, 'B':Base[0]}, index=epoch)
        else:
            df = pd.DataFrame({'validation accuracy':vacc, 'A':Base[0]}, index=epoch)
    df.plot()
    plt.title(title)
    plt.plot([len(epoch)+1],[facc],marker='o',markersize=10)
    plt.annotate('test accuracy = {0}'.format(facc), xy=(len(epoch)+1,facc), xytext=(len(epoch)*0.9,facc*1.02))
    if not isB:
        plt.plot([len(epoch)+1],[Base[1]],marker='o',markersize=10)
        plt.annotate('B = {0}'.format(Base[1]), xy=(len(epoch)+1,Base[1]), xytext=(len(epoch)*0.9,Base[1]*1.02))
    plt.xlabel('epoch')
    plt.savefig('val_acc_{0}.png'.format(fout))
    return floss, facc, tloss[0], vloss[0], vacc[0]

def plotData(df, title, a, b):
    df.plot()
    if REV:
        plt.axhline(y=a, label='A', linestyle='-.', color='r')
    else:
        plt.axhline(y=b, label='B', linestyle='-.', color='r')
    plt.xlabel('# Rank1 tensor transferred')
    plt.legend()
    plt.title(title)
    plt.savefig('{0}.png'.format(title))
    plt.clf()
def plotAll():
    files = glob.glob('*.txt')
    A, B = None, None
    if not REV:
        A = plot('A.txt', 'A', 'A', isB=True)
        B = plot('B.txt', 'B', 'B', isB=True)
    else:
        B = plot('B.txt', 'B', 'B', isB=True)
        A = plot('A.txt', 'A', 'A', isB=True)
    FL, FA, TL, VL, VA, R = [], [], [], [], [], []
    for f in files:
        fname = f.split('.')[0]
        ABR = fname.split('_')
        if len(ABR) == 1:
            continue
        else:
            if not REV:
                fl, fa, tl, vl, va = plot(f, fname, 'B with {0} Rank1 from A '.format(ABR[1]))
            else:
                fl, fa, tl, vl, va = plot(f, fname, 'A with {0} Rank1 from B '.format(ABR[1]))
            FL += [fl]
            FA += [fa]
            TL += [tl]
            VL += [vl]
            VA += [va]
            R += [int(ABR[1])]
    plotData(pd.DataFrame({'test loss':FL}, index=R).sort_index(), 'test loss', A[0], B[0])
    TA = pd.DataFrame({'test accuracy':FA}, index=R).sort_index()
    plotData(TA, 'test accuracy', A[1], B[1])
    plotData(pd.DataFrame({'training loss at epoch=1':TL}, index=R).sort_index(), 'training loss at epoch=1', A[2], B[2])
    plotData(pd.DataFrame({'validation loss at epoch=1':VL}, index=R).sort_index(), 'validation loss at epoch=1', A[3], B[3])
    plotData(pd.DataFrame({'validation accuracy at epoch=1':VA}, index=R).sort_index(), 'validation accuracy at epoch=1', A[4], B[4])
    TA['index']=R
    TA.to_csv('test_acc.csv')
opts = parse_arg()
if opts.All is False:
    plot(opts.fin, opts.fout)
else:
    REV = opts.rev
    plotAll()