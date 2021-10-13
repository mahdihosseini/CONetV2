import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import scipy 




# #PRINTING ALPHA SWEEPS
fig, ax = plt.subplots()


ALPHA1 = 5
ALPHA2 = 35
ALPHA3 = 1
Temp = 1
x = np.arange(0.0, 1, 0.01)
y1 = np.exp((-1/(ALPHA1*x*Temp)))
y2 = np.exp((-1/(ALPHA2*x*Temp)))
y3 = np.exp((-1/(ALPHA3*x*Temp)))
z = [1]*100
ax.plot(x, [1]*100, "k--", label="Random Number Bounding Box")
ax.plot([1]*100, x, "k--", label="Random Number Bounding Box")
ax.plot(x, [0]*100, "k--", label="Random Number Bounding Box")
ax.plot([0]*100, x, "k--", label="Random Number Bounding Box")

ax.plot(x, y2, "b", linewidth=3)
ax.plot(x, y1, "r", linewidth=3)
ax.plot(x, y3, "b", linewidth=3)

ax.fill_between(x, y3, y2, facecolor='blue', alpha=0.5)

ax.fill_between(x, [0]*100, y3, facecolor='green', alpha=0.2)
ax.grid()

ax.set(title = "Accepting Function vs Temperature",
       xlabel = "Temp",
       ylabel = "Accepting Function (" + (r'$\zeta$') + ")")


plt.show()




N = [r'$\gamma_{0}$',r'$\gamma_{0.5}$',r'$\gamma_{0.6}$', r'$\gamma_{0.7}$', 
    r'$\gamma_{0.8}$', r'$\gamma_{0.9}$']


ResnetGreedy_ACC = [63.47,68.58,71.13,74.46,77.73,78.19]
ResnetGreedy_Param = [0.17,0.394,0.854,1.812,7.166,17.43]

ResnetSA_ACC = [68.32,70.4,73.35,75.7,78.23,78.6]
ResnetSA_Param = [0.43,0.682,1.288,2.502,8.85,18.37]

ResnetCS_ACC = [76.2,77.54,78.33]
ResnetCS_Param = [5.02,10.1,15.19]

ResnetBaseline_ACC = [77.88]
ResnetBaseline_Param = [21.28]


logTrend = np.polyfit(np.log(ResnetGreedy_Param), ResnetGreedy_ACC, 2)


# plot the 3 sets
plt.plot(ResnetGreedy_Param,logTrend[0]*np.log(ResnetGreedy_Param)+logTrend[1],'o--',label='Resnet Greedy Search')
plt.plot(ResnetSA_Param,ResnetSA_ACC, 'o--',label='Resnet SA')
plt.plot(ResnetCS_Param,ResnetCS_ACC, 'o--',label='Resnet CS')

plt.plot(ResnetBaseline_Param,ResnetBaseline_ACC, 'o--',label='Resnet Baseline')


for i,(loc) in enumerate(zip(ResnetGreedy_Param,ResnetGreedy_ACC)):
    plt.annotate(N[i], (loc),textcoords="offset points",
                 xytext=(0,10),
                 ha='center')

for i,(loc) in enumerate(zip(ResnetSA_Param,ResnetSA_ACC)):
    plt.annotate(N[i], (loc),textcoords="offset points",
                 xytext=(0,10),
                 ha='center')

# call with no parameters
plt.legend(loc=4)

plt.title('CIFAR100 Accuracy vs Parameters')
plt.xlabel('Parameters (M)')
plt.ylabel('Accruacy (%)')
 

plt.show()