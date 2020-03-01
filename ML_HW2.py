import numpy as np
import matplotlib.pyplot as plt
import argparse 


def GetData(size = 20, ratio = 0.2):
    #noise_idx = np.arange(size)
    #np.random.shuffle(noise_idx)
    #noise_idx = noise_idx[:int(size*ratio)]
    data = np.random.uniform(-1, 1, size)
    sign_label = np.sign(data)
    prob = np.random.uniform( 0, 1, size)
    sign_label[prob<ratio] *= -1
    #np.negative.at(sign_label, noise_idx)
    return data, sign_label

def Theta(data,size=20):
    theta = []
    for i in range(size+1):
     if i == 0:
        theta.append([data[0] - 1])
     elif i == size:
        theta.append([data[-1] + 1])
     else:
        theta.append([(data[i] + data[i-1])/2])
    theta = np.array(theta)
    print("theta",theta)
    return theta

def Decision_Stump(data,label,size):
    data = np.sort(data)
    theta = Theta(data,size)
    data_shape = data.shape[0]
    theta_shape = theta.shape[0]
    data = np.tile(data, (theta_shape, 1))
    #for s = 1
    y1 = np.sign( data - theta )
    y2 = np.sign( data - theta ) * (-1)
    #caculate error 
    err1 = np.sum(y1 != label, axis = 1)
    err2 = np.sum(y2 != label, axis = 1)
    #min_index
    min_err1 = np.argmin(err1)
    min_err2 = np.argmin(err2)
    best_s = 1 
    best_theta = 1
    error = 0
    if err1[min_err1] < err2[min_err2]:
        best_s = 1
        best_theta = theta[min_err1][0]
        error = err1[min_err1]/data_shape
    else:
        best_s = -1
        best_theta = theta[min_err2][0]
        error = err2[min_err2]/data_shape
    return best_s, best_theta, error
   
def calculate(times,figname,size):
    avg_Ein = []
    avg_Eout = []
    for _ in range(times):
        data,label = GetData(size)
        print("data",data)
        print("label",label)
        best_s, best_theta, error = Decision_Stump(data,label,size)
        Eout = 0.5 + 0.3*best_s*(np.abs(best_theta)-1)
        avg_Ein.append(error)
        avg_Eout.append(Eout)
    avg_Ein = np.array(avg_Ein)
    avg_Eout = np.array(avg_Eout)
    Ein_Eout = avg_Ein - avg_Eout
    print('avg_Ein',np.mean(avg_Ein))
    print('avg_Eout',np.mean(avg_Eout))                                                                   
    plt.hist(Ein_Eout,bins=int((max(Ein_Eout)-min(Ein_Eout))/0.01))
    plt.title(figname)
    plt.xlabel('Ein-Eout')
    plt.ylabel('Frequency')
    plt.savefig(f"{figname}.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("n", help="repeat time", type=int)
    parser.add_argument("-u", "--fig_name", dest="figname")
    parser.add_argument("size",help='size',type=int)
    args = parser.parse_args()
    print("args",args)
    calculate(args.n,args.figname,args.size)

    
