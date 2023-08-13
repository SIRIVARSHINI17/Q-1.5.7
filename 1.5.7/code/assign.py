#code for showing out the parameters for drawing the circle
import numpy as np
B = np.array([[-4],[6]])  
I=(1/(np.sqrt(37) + 4 + np.sqrt(61)))*np.array([[np.sqrt(61) - 16 - 3*np.sqrt(37)],[-np.sqrt(61) + 24 - 5*np.sqrt(37)]])    #Here I is incentre (point of intersection of angle bisectors)
n = np.array([[11], [1]])   #n is normal vector 
nT = n.T   #taking transpose of n
r = abs((nT @ I) - (nT @ B))/(np.linalg.norm(n))   #r is distance between I and BC (nT.I - nT.B=0, n and I are vectors)
print("center of the circle(incenter):",I)
print("radius of the circle(inradius):",r)


