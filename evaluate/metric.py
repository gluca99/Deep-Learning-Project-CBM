# reference: https://github.com/mabirck/CatastrophicForgetting-EWC/blob/master/logShow.py
import csv
import numpy as np
import glob
import json
import pandas as pd
import argparse

def parseargs():
    parser = argparse.ArgumentParser()
    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)
    aa("--file_dir",type=str,help='result dir from the experiment')
    aa("--task_num",type=int,default=5,help='number of tasks in the experiment')
    aa("--strategy",type=str,default="SRT",help='continual learning algorithm')
    args = parser.parse_args()
    return args

def plotData(allData, legend,task_num):

    print("allData",len(allData))

    for mode, data in enumerate(allData):
        print("MODE:",legend[mode])
        pdlist=[]
        pdlist.append(legend[mode])
        for i,d in enumerate(data):
            np.set_printoptions(precision=4,suppress=True)
            strd=[str('%.4f' % j) if j!=0 else str(0.0) for j in d]
            pdlist=pdlist+strd

        first_list=[1]
        for i in range(task_num,1,-1):
            first_list.append(first_list[-1]+i)
        average_acc=forget=0
        for t in range(2,task_num+1):
            cur_avg=cur_for=0
            for c_t in range(t):
                first=first_list[c_t]
                final=(t-c_t-1)+first
                cur_avg+=float(pdlist[final])
                cur_for+=(float(pdlist[first])-float(pdlist[final]))
            average_acc+=cur_avg/(t)
            forget+=cur_for/(t-1)

        average_acc/=(task_num-1)
        forget/=(task_num-1)
        print("average incremental accuracy:",average_acc)
        print("average incremental forgetting:",forget)

        pdlist=[[] for _ in range(task_num)]
        for i,d in enumerate(data):
            print(f"task{i+1}",end=' ')
            np.set_printoptions(precision=4,suppress=True)
            print(d,end=" ")
            strd=[str('%.4f' % j) if j!=0 else str(0.0) for j in d]
            pdlist[i]=strd
        print("\n")

        average_acc=forget=learn=0
        for t in pdlist:
            average_acc+=float(t[-1])
            forget+=float(t[0])-float(t[-1])
            learn+=float(t[0])
        average_acc/=len(pdlist)
        forget/=len(pdlist)-1
        learn/=len(pdlist)
        print("average accuracy: ",average_acc, "learning accuracy:",learn, "average forgetting:", forget)



def main():
    args = parseargs()

    task_num=args.task_num
    allData = []

    accuracy_task = list()
    #file_name=args.file_dir+args.strategy+'_acc.txt'
    file_name=args.file_dir + '/acc2.txt'
    with open(file_name, 'r') as j:
        contents = json.loads(j.read())
    for i in range(task_num):
        accuracy_task.append(np.asarray(contents[str(i)]))

    allData.append(accuracy_task)

    plotData(allData, args.strategy,task_num)

if __name__ == "__main__":
    main()
