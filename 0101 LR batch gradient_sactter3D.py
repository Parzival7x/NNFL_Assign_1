import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


# from sklearn.preprocessing import MinMaxScaler
import numpy as np
# import scipy
import matplotlib.pyplot as plt
# from sklearn.neighbors import LocalOutlierFactor
# df = pd.read_csv('german_data-numeric', delim_whitespace=True, header=None)


flag = 0
u = float(0)
lu = [float(0)]
v = float(0)
lv = [float(0)]
w = float(0)
lw = [float(0)]
a = float(0.00001)
ls2 = []
lscost = []
# a=float(0.1)

val=[0,0,0]

df = pd.read_csv('data.csv', nrows=349, delim_whitespace=False, sep=',', header=None)
df = df.astype('float64')
scaler = MinMaxScaler().fit(df)
ls = scaler.transform(df)

print(ls)

# ls = df2.values.tolist()


def predict(x,y,a,b,c):
    req = a + b*float(x) + c*float(y)
    return req

def cost():
    s = 0
    for i in range(348):
        s += pow((hypo(ls[i]) - ls[i][2]), 2)
    s/=2
    return s

def test():
    temp = np.prod(val)
    return temp


def hypo(vec):
    x = vec[0]
    y = vec[1]
    z = vec[2]
    h = u + v*x + w*y
    return h

def derivative(j):
    s = int(0)
    for i in range(348):
        if j==0:
            s += (hypo(ls[i]) - ls[i][2])
        else:
            s += (hypo(ls[i]) - ls[i][2]) * ls[i][j]


    return s


def cal():
    global u
    global v
    global w
    global lu
    global lv
    global lw

    for j in range(3):
        # print(j)
        w_old = 0
        if val[j] == 1:
            continue
        if j == 0 :
            # ut=global()u
            w_old =  u
        elif j == 1:
            # vt=v
            w_old = v
        else:
            # wt=w
            w_old = w

        # print(derivative(j))

        w_new = w_old - float(a) * float(derivative(j))
        if j == 0 :
            u = w_new
            lu.append(w_new)
        elif j == 1:
            v= w_new
            lv.append(w_new)
        else:
            w = w_new
            lw.append(w_new)
        # print(w_new)
        if w_new == w_old:
            val[j]=1



# while not(test()):
#     print("----------------------------------")
#     cal()

min_cost = 999
iteration = 0
for i in range(5000):
    # global ls2
    ls2.append(i)
    # if i != 0:
    #     temp = (lscost[-1])
    temp1 = (cost())
    if temp1<min_cost:
        min_cost = temp1
        iteration = i
    # if i>1000 and flag==0 and temp1 > temp:
    #     flag = i
    lscost.append(temp1)
    cal()
ls2.append(5000)
lscost.append(cost())
# print(ls2)
# print(lu)
# print(lv)
# print(lw)
# print(lscost)

# print("the required iteration: ",flag)
# print(lscost[flag-2])
# print(lscost[flag-1])
# print(lscost[flag])
# print(lscost[flag+1])
# print(lscost[flag+2])
print("\n*-----------------------------------------------------------------*")
print("The Cost Function is minimised at the following values :",min_cost,iteration)
print("w0: ",lu[iteration-1])
print("w1: ",lv[iteration-1])
print("w2: ", lw[iteration-1])
print("*-----------------------------------------------------------------*")
# print("Sample Prediction")
# print("Input the value of x\n")
# p = input()
# print("input the value of y\n")
# q = input()
# print("The predicted values: \n",predict(p,q,lu[iteration-1],lv[iteration-1],lw[iteration-1]))


plt.plot(ls2,lscost)
plt.show()
# plt.plot(lv, lw, lscost)
# # plt.show()
# print(len(lu))
# print(len(lv))
# print(len(lw))
# print(len(lscost))

plt.plot(ls2,lu)
plt.show()
plt.plot(ls2,lv)
plt.show()
plt.plot(ls2,lw)
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')

ax.scatter(lv, lw, lscost, c='r',marker = 'o')
ax.set_xlabel('W1')
ax.set_ylabel('W2')
ax.set_zlabel('J(w) [Cost Function]')

plt.show()


fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')

ax.scatter(lv, lw, lu, c='r',marker = 'o')
ax.set_xlabel('W1')
ax.set_ylabel('W2')
ax.set_zlabel('W0')

plt.show()





# print(df)



























