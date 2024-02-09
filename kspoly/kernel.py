from sklearn.gaussian_process.kernels import Matern
from scipy.special import gamma, factorial
import numpy as np
import scipy

__all__ = ['kernel', 'sigma']


#@title Default title text
selected_kernel = "Gaussian" #@param ["Gaussian", "Matern", "Constant"]
sigma = 700.5 #@param {type:"slider", min:0, max:200, step:0.5}
matern_rho = 800 #@param {type:"slider", min:0, max:200, step:1}
matern_nu = 1.5 #@param {type:"slider", min:0, max:4, step:0.1}


k_gauss = lambda t, sigma: np.exp(-0.5*(t/sigma**2))/(sigma*np.sqrt(2*np.pi))
k_matern = lambda d, sigma, nu, rho: (sigma**2)*((2**(1-nu))/gamma(nu))*((np.sqrt(2*nu)*d/rho)**nu)*scipy.special.kv(nu, np.sqrt(2*nu)*d/rho)

def kernel_gaussian(x,y, MX, MY, X_ALL, Y_ALL):   
   N = MX.shape[0]
   D2 = (x-MX)**2 + (y-MY)**2
   D2_ALL = (x-X_ALL)**2 + (y-Y_ALL)**2
   #return np.sum(np.exp(-0.5*(D2/sigma**2))/(N*sigma*np.sqrt(2*np.pi)))
   return np.sum(k_gauss(D2, sigma))/np.sum(k_gauss(D2_ALL, sigma))

def kernel_entry_gaussian(mx, my, X, Y):
   D2 = (mx - X)**2 + (my - Y)**2
   Kxy = k_gauss(mx, my)


def kernel_matern(x,y, MX, MY, X_ALL, Y_ALL):  
   N = MX.shape[0]
   D2 = (x-MX)**2 + (y-MY)**2
   D2_ALL = (x-X_ALL)**2 + (y-Y_ALL)**2
   #return np.sum(np.exp(-0.5*(D2/sigma**2))/(N*sigma*np.sqrt(2*np.pi)))
   return np.sum(k_matern(np.sqrt(D2), sigma=1, nu=matern_nu, rho=matern_rho))/np.sum(k_matern(np.sqrt(D2_ALL), sigma=1, nu=matern_nu, rho=matern_rho))
  
def kernel_constant(x,y, MX, MY, X_ALL, Y_ALL):
   sigma = 100
   N = MX.shape[0]
   D2 = (x-MX)**2 + (y-MY)**2
   if (np.sum((D2 < 140**2)) > 5): 
     return 1
   else: 
     return 0

def kernel_NNS(x,y, MX, MY):
   sigma = 100
   N = MX.shape[0]
   D2 = (x-MX)**2 + (y-MY)**2
   return (np.sum((D2 < 170**2)) / 3)



def kernel(X,Y,A, xnum=400, ynum=200):
    # vectorize the kernel function so it can perform well on tensors/matrices 
    if (selected_kernel == 'Gaussian'):
        kernelF = np.vectorize(kernel_gaussian, excluded=['MX', 'MY', 'X_ALL', 'Y_ALL'])
    elif (selected_kernel == 'Matern'):
        kernelF = np.vectorize(kernel_matern, excluded=['MX', 'MY', 'X_ALL', 'Y_ALL'])
    elif (selected_kernel == 'Constant'):
        kernelF = np.vectorize(kernel_constant, excluded=['MX', 'MY', 'X_ALL', 'Y_ALL'])


    
    x = np.linspace(X.min(), X.max(), xnum)
    y = np.linspace(Y.min(), Y.max(), ynum)
    xx , yy = np.meshgrid(x,y)
    zz = kernelF(xx,yy, MX=X[A==True], MY=Y[A==True], X_ALL = X, Y_ALL=Y)
    return x, y, zz


def kernel_gaussian_func(x,y, X, Y, F):
    D2 = (x-X)**2 + (y-Y)**2
    kg = k_gauss(D2, sigma)
    z = np.dot(kg,F)
    return z

def kernel_vectorfield_comp(X,Y,V1,V2):
    margin = .01*(X.max() - X.min())
    x = np.linspace(X.min()-margin, X.max()+margin, 30)
    y = np.linspace(Y.min()-margin, Y.max()+margin, 15)
    xx , yy = np.meshgrid(x,y)

    kernelF = np.vectorize(kernel_gaussian_func, excluded=['X', 'Y', 'F'])
    zz1 = kernelF(xx,yy, X=X, Y=Y, F=V1)
    zz2 = kernelF(xx,yy, X=X, Y=Y, F=V2)
    x = xx.reshape(-1)
    y = yy.reshape(-1)
    zz1 = zz1.reshape(-1)
    zz2 = zz2.reshape(-1)
    return x, y, zz1, zz2


def kernel_by_value_G(X,Y,Z, xnum= 100, ynum=50):
    margin = .01*(X.max() - X.min())
    x = np.linspace(X.min()-margin, X.max()+margin, xnum)
    y = np.linspace(Y.min()-margin, Y.max()+margin, ynum)
    xx , yy = np.meshgrid(x,y)

    kernelF = np.vectorize(kernel_gaussian_func, excluded=['X', 'Y', 'F'])
    zz = kernelF(xx,yy, X=X, Y=Y, F=Z)
    #x = xx.reshape(-1)
    #y = yy.reshape(-1)
    #zz = zz.reshape(-1)
    return x, y, zz

from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
from scipy import sparse


def create_kernel_matrix_nnbr(X,Y, xx ,yy):
    # Creates kernel matrix that can transform a function defined 
    # on monte Carlo points to a function defined on grid points. 
    # X: x coordinate of Monte Carlo points 
    # Y: y coordinate of Monte Carlo points 
    # xx: x coordinate of a mesh grid
    # yy: y coordinate of a mesh grid

    # normsq = lambda a: (a*a).sum()
    # fn_kernel = np.vectorize(lambda i,j, MP,GP:
    #      k_gauss(normsq(MP[i,:]-GP[j,:]), sigma)
    #      , excluded=['MP', 'GP'], otypes=[float])


    MP = np.stack([X,Y],axis=1)
    XX = xx.reshape(-1)
    YY = yy.reshape(-1)
    GP = np.stack([XX,YY], axis=1)
    n_MC_neighbors = 6
    print('  create_kernel_matrix_nnbr 1')
    nbrs = NearestNeighbors(n_neighbors=n_MC_neighbors, algorithm='ball_tree', leaf_size=50).fit(MP)
    print('  create_kernel_matrix_nnbr 2')
    distances, indices = nbrs.kneighbors(GP)
    print('  create_kernel_matrix_nnbr 3')


    print('multi... 0')

    #K= np.zeros((GP.shape[0], MP.shape[0]))
    #K= csr_matrix((GP.shape[0], MP.shape[0]))


    fn = lambda d: k_gauss(d**2, sigma)
    sh = distances.shape

    print('multi... 1')
    #with concurrent.futures.ThreadPoolExecutor(max_workers=30) as executor:
    #    print('multi... 2')
    #    g_dist_l = executor.map(fn, distances.reshape(-1))


    g_dist_l =  map(fn, distances.reshape(-1))
    g_dist = np.array(list(g_dist_l)).reshape(sh)

    print('  create_kernel_matrix_nnbr 4 ', GP.shape[0])

    I = np.repeat(range(GP.shape[0]),n_MC_neighbors).reshape(-1)
    J = indices.reshape(-1,1).reshape(-1)
    V = g_dist.reshape((-1,1)).reshape(-1)
    print('  create_kernel_matrix_nnbr 5')

    K = sparse.coo_matrix((V,(I,J)),shape=(GP.shape[0], MP.shape[0]))

    #for i in range(GP.shape[0]):    
    #    K[i,indices[i,:]] = g_dist[i,:] #k_gauss(D**2, sigma)

    # for i in range(GP.shape[0]):
    #     for jnbr in range(n_MC_neighbors):
    #         j = indices[i,jnbr]
    #         #D = distances[i,jnbr]
    #         K[i,j] = g_dist[i,jnbr] #k_gauss(D**2, sigma)
    K = normalize(K, axis=1, norm='l1')
    print('  create_kernel_matrix_nnbr 6')

    return K


from scipy.spatial.distance import cdist

from numba import jit, prange
from scipy.integrate import dblquad

@jit(nopython=True, parallel = True)
def compute_gaussian_marginals(GP, xlim, ylim, out_mar):    
    N = (sigma**2)*2*np.pi
    #print(xlim)
    a = xlim[0]
    b = xlim[1]
    c = ylim[0]
    d = ylim[1]

    # Define the number of subintervals in each direction
    n = 1000

    # Calculate the width and height of each subinterval
    dx = (b - a) / n
    dy = (d - c) / n

    for k in prange(GP.shape[0]):
        p1 = GP[k, 0]
        p2 = GP[k, 1]



        # Initialize the sum to zero
        integral = 0

        # Loop over the subintervals and add the contribution to the integral
        for i in range(n):
            for j in range(n):
                # Calculate the midpoint of the current subinterval
                x1 = a + (i + 0.5) * dx
                x2 = c + (j + 0.5) * dy
                # Add the contribution of the current subinterval to the integral
                integral += (np.exp(-((x1-p1)**2 + (x2 - p2)**2)  / (2 * sigma ** 2))/N) * dx * dy


        #a = xmin - p1
        #b = xmax - p2
        #v = ( np.exp(-(a-b)**2/(2*sigma**2)) -1)*(sigma*np.sqrt(2/np.pi)) + (a-b)*scipy.special.erf((a-b)/(sigma*np.sqrt(2)))
        #fn = lambda x1, x2: np.exp(-((x1-p1)**2 + (x2 - p2)**2)  / (2 * sigma ** 2))/N
        #res, err = dblquad(lambda x1, x2: np.exp(-((x1-p1)**2 + (x2 - p2)**2)  / (2 * sigma ** 2)), xmin, xmax, ymin, ymax)
        out_mar[k] = integral






@jit(nopython=True, parallel = True)
def create_kernel_matrix_jit(MP, GP, W, marginals, res):
    # MP = np.stack([X,Y],axis=1)
    # XX = xx.reshape(-1)
    # YY = yy.reshape(-1)
    # GP = np.stack([XX,YY], axis=1)
    n = GP.shape[0]
    m = MP.shape[0]
    
    #sigma = 1500
    N = (sigma**2)*2*np.pi

    for i in prange(n):
        mar = marginals[i]
        for j in range(m):
            w = W[j]            
            X = GP[i,0] - MP[j,0]
            Y = GP[i,1] - MP[j,1]
            D2 = X*X + Y*Y
            res[i,j] = w*np.exp(-D2 / (2 * sigma ** 2))/(N*mar)
    return res


def create_kernel_matrix_tr(X,Y, xx ,yy):
    MP = np.stack([X,Y],axis=1)
    XX = xx.reshape(-1)
    YY = yy.reshape(-1)
    GP = np.stack([XX,YY], axis=1)

    # print('shape---- ', MP.shape)
    # dist_matrix = cdist(MP, GP)
    # print(dist_matrix)
    # return np.zeros((GP.shape[0], MP.shape[0]))
    # Compute the Gaussian kernel matrix using the distance matrix
    K = np.exp(-dist_matrix ** 2 / (2 * sigma ** 2))



def create_kernel_matrix(X,Y, xx ,yy):
    # Creates kernel matrix that can transform a function defined 
    # on monte Carlo points to a function defined on grid points. 
    # X: x coordinate of Monte Carlo points 
    # Y: y coordinate of Monte Carlo points 
    # xx: x coordinate of a mesh grid
    # yy: y coordinate of a mesh grid

    normsq = lambda a: (a*a).sum()
    fn_kernel = np.vectorize(lambda i,j, MP,GP:
         k_gauss(normsq(MP[i,:]-GP[j,:]), sigma)
         , excluded=['MP', 'GP'], otypes=[float])

    MP = np.stack([X,Y],axis=1)

    XX = xx.reshape(-1)
    YY = yy.reshape(-1)
    GP = np.stack([XX,YY], axis=1)
    I = np.arange(MP.shape[0])
    J = np.arange(GP.shape[0])
    II, JJ = np.meshgrid(I,J)
    print(xx.shape, II.shape, JJ.shape, MP.shape, GP.shape)
    # K= np.zeros((GP.shape[0], MP.shape[0]))
    # for k in range(II.shape[0]):
    #     for l in range(II.shape[1]):
    #         i = II[k,l]
    #         j = JJ[k,l]
    #         K[k,l] = k_gauss(np.linalg.norm(MP[i,:]-GP[j,:])**2, sigma)
    

    K = fn_kernel(II,JJ, MP=MP, GP=GP)
    #K = normalize(K, axis=1, norm='l1')
    return K

def apply_kernel(K,F,out_shape):

    G = (K*F)





def save():
    return
    now = datetime.now() # current date and time
    filename = selected_kernel+  now.strftime("_%m_%d_%Y__%H_%M_%S")+ '.png'
    plt.savefig(filename)
    print('Figure saved as '+filename)
    plt.show()
