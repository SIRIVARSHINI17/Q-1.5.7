import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots()
A = np.array([1, -1])
B = np.array([-4, 6])
C = np.array([-3, -5])

I=(1/(np.sqrt(37) + 4 + np.sqrt(61)))*np.array([[np.sqrt(61) - 16 - 3*np.sqrt(37)],[-np.sqrt(61) + 24 - 5*np.sqrt(37)]])    
n = np.array([[11], [1]])   #n is normal vector 
nT = n.T   #taking transpose of n
r = abs((nT @ I) - (nT @ B))/(np.linalg.norm(n))   #r is distance between I and BC (nT.I - nT.B=0, n and I are vectors)

def dir_vec(A, B):
    return B - A
    
def line_gen(A, B):
    len = 10
    dim = A.shape[0]
    x_AB = np.zeros((dim, len))
    lam_1 = np.linspace(0, 1, len)
    for i in range(len):
        temp1 = A + lam_1[i] * (B - A)
        x_AB[:, i] = temp1.T
    return x_AB
    
D = I + r * (n / np.linalg.norm(n))
x_inradius = line_gen(I, D)
x_AB = line_gen(A,B)
x_BC = line_gen(B,C)
x_CA = line_gen(C,A)



plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')
plt.plot(x_BC[0,:],x_BC[1,:],label='$BC$')
plt.plot(x_CA[0,:],x_CA[1,:],label='$CA$') 
plt.plot(x_inradius[0, :], x_inradius[1, :], label='Inradius ($r$)')
plt.plot(I[0], I[1], label='Incenter ($I$)')

circle = patches.Circle(I, r, fill=False, label='Incircle')
ax.add_patch(circle)

A = A.reshape(-1,1)
B = B.reshape(-1,1)
C = C.reshape(-1,1)    
tri_coords = np.block([[A,B,C,I]])
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['A','B','C','I']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(-10,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')


plt.show()
