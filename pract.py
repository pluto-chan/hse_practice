###### command line arguments parsing

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("imname", help = "path to the graph file")
parser.add_argument("xax", help = "x axis overall length", type=int)
parser.add_argument("yax", help = "y axis overall length", type=int)
parser.add_argument("func", help = "function type")

args = parser.parse_args()

imname = args.imname
xdim = args.xax
ydim = args.yax
func = args.func

###### 

from scipy.ndimage import imread
import numpy as np
import random

#symbols lib for derivatives all math stuff should be used according to this lib
from sympy import * 

###### reading image containing graph 

def open_img(name):
    try:
        graph = imread(imname, flatten = True)
        print(graph.shape)
        #plt.imshow(graph)
        #plt.show()

        return(graph)
    except Exception as e:
        print ('image cannot be opened')
        return(-1)

###### getting points from the graph
###### 10% of points are used
###### threshold defines the threshold of a point

def get_points(graph, xdim, ydim, threshold = 200.0):
    xx = [] 
    yy = []
    
    for i in range(graph.shape[0]):
        for j in range(graph.shape[1]):
            if graph[i, j] < threshold:

                #scaling graph
                #reversing y axis

                yy.append((graph.shape[0]/2-i)*(ydim/graph.shape[0]))
                xx.append((j-graph.shape[1]/2)*(xdim/graph.shape[1]))

    freq = int(len(xx)/10)
    smxx = xx[::freq]
    smyy = yy[::freq]
    
    # plotting the points used
    #plt.scatter(smxx, smyy, marker='.')
    #plt.show()

    X = np.asarray(smxx)
    Y = np.asarray(smyy)

    return([X, Y])

###### function 
###### types currently supported
###### lin - linear, k * x + b
###### sqr - square, A * x^2 + B * x + C
###### sqrt - square root, (k * x + b)^(1/2)
###### hyp - hyperbole, A / (B * x + C) + D

def F(x, coeff, func):
    #Asin(B*x+C)+D
    As, Bs, Cs, Ds = symbols('As Bs Cs Ds')
    cfs = [As, Bs, Cs, Ds]
    
    func_types = {
    'lin': cfs[0]*x + cfs[1],
    'sqr': cfs[0]*x**2 + cfs[1]*x + cfs[2],
    'sqrt': (cfs[0]*x + cfs[1])**(1/2), 
    'hyp': cfs[0]/(cfs[1]*x+cfs[2]) + cfs[3]
    }
    
    try:
        Fs = func_types[func]
    except KeyError as e:
        raise ValueError('No such function type!')
        return (-1)

    F = Fs.subs([(cfs[0], coeff[0]), (cfs[1], coeff[1]), (cfs[2], coeff[2]), (cfs[3], coeff[3])])

    return F

####### gradient descend 

def full_grad_4(X, Y, func, EPS  = 0.01, MAX_ITERATION = 50000, lam = 0.001):
    
    grad=np.zeros(4)
    g0, g1, g2, g3 = symbols('g0 g1 g2 g3')
    sym_grad = [g0, g1, g2, g3]
    i = 0

    As, Bs, Cs, Ds = symbols('As Bs Cs Ds')
    cfs = [As, Bs, Cs, Ds]

    L1 = []
    L = 10
    k = 0
    c = [1.0, 1.0, 1.0, 1.0]

    flag = False  
    random.seed()
    if func == 'sqrt':
        while flag == False:
            flag = True        
            for i in range(X.shape[0]):
                if c[0]*X[i]+c[1] < 0:
                    flag = False
            if not flag:
              c[0] = random.uniform(-5.0, 5.0)
              c[1] = random.uniform(-5.0, 5.0)
            print(c)

    print(c)
    while L > EPS and k < MAX_ITERATION:  
        grad = np.zeros(4)
        for i in range(X.shape[0]):
        
            for j in range(len(cfs)):
            #derivatives
                loss = (F(X[i], cfs, func) - Y[i])**2
                sym_grad[j] = loss.diff(cfs[j])

            for j in range(len(cfs)):
                grad[j] += float(sym_grad[j].subs([(cfs[0], c[0]), (cfs[1], c[1]), (cfs[2], c[2]), (cfs[3], c[3])]))


        #updating
        for j in range(len(c)):
            c[j] = c[j] - lam * grad[j]/(2*X.shape[0])
    
        L = 0                                           
        for i in range(X.shape[0]): 
            loss = (F(X[i], cfs, func) - Y[i])**2
            L += loss.subs([(cfs[0], c[0]), (cfs[1], c[1]), (cfs[2], c[2]), (cfs[3], c[3])])
        L = L / (2*X.shape[0])                                      
        L1.append(L)
        #print loss every 100 steps
        if k%100 == 0:
            
            if func == 'lin' or func == 'sqrt':
                print(c[0:2])
            elif func == 'sqr': 
                print(c[0:4])
            else:
                print([c])
            print('loss = ', L)

        k += 1
    print(k, ' iterations needed, ', L)
    return c

####### so here it goes

funcs = [   'lin', 
            'sqr', 
            'sqrt', 
            'hyp'
        ] 

if xdim <= 5 or ydim <= 5:
    print('Incorrect axis length')
else:
    if func not in funcs:
        print('Incorrect function type')

    else:
        graph = open_img(imname)
        if type(graph) == np.ndarray: 
            X, Y = get_points(graph, xdim, ydim)
            fin_c = full_grad_4(X, Y, func)
            
            if func == 'lin' or func == 'sqrt':
                print(fin_c[0:2])
            elif func == 'sqr': 
                print(fin_c[0:4])
            else:
                print([fin_c])