
import numpy as np 

import kernel
#importlib.reload(kernel) 

import triangulation as tri



class scene:
    def __init__(self, sigma, mesh_size, xlim, ylim):
        self.sigma = sigma
        self.mesh_size = mesh_size
        self.xlim = xlim
        self.ylim = ylim
        self.meshgrid_shape = (mesh_size[0],mesh_size[1])

        self.gaussian_marginals = None

        self.frame_x = None 
        self.frame_y = None 
        self.frame_xx = None 
        self.frame_yy = None 

        self.frame_MCX = None 
        self.frame_MCY = None
        self.frame_MCf = None  
        self.tr_weights = None  
        self.tr_list = None          
        self.frame_K = None
        self.data = None

    def set_land(self, polygons):
        pass
 
    def polygon_to_triangle_map(self, data):
        value = data[self.frame_MCf]
        return value
        
    def smoothit(self, frame_data): 

        # load frame data
        self.data = frame_data

        # setup the kernel weights given polygons 
        self.construct_kernel()

        # construct smooth fields 
        res = dict()

        #res['mass'] = self.frame_K.dot(mass)
        #res['velocity'] = self.frame_K.dot(vel)
        #A = (MCf > -1).astype(int)
        #f_gr = K.dot(A)
        #f_gr = f_gr.reshape(self.meshgrid_shape)

        for n in self.data.field_names:            
            value = self.polygon_to_triangle_map(self.data.fields[n])
            shape_old = list(value.shape)
            value = value.reshape((shape_old[0], -1))
            res[n] = self.frame_K.dot(value)
            shape_old[0] = self.frame_K.shape[0]
            res[n] = res[n].reshape(shape_old)
            shape_old[0] = self.meshgrid_shape[1]
            shape_old = [self.meshgrid_shape[0]] + shape_old
            res[n] = res[n].reshape(shape_old)


        return res
        

    def construct_kernel(self):
        
        traingle_area_p = 0.5 * np.float_power(self.sigma, 1.33)
        triangle_gen = tri.triangulation()
        self.frame_MCX, self.frame_MCY, self.frame_MCf, self.tr_weights, self.tr_list = triangle_gen.generate_triangle_points(self.xlim, self.ylim, self.data.polygons, traingle_area_p )

        # Create a meshgrid
        self.frame_x = np.linspace(self.xlim[0], self.xlim[1], self.mesh_size[0])
        self.frame_y = np.linspace(self.ylim[0], self.ylim[1], self.mesh_size[1])
        self.frame_xx ,self.frame_yy = np.meshgrid(self.frame_x,self.frame_y)


        # Compute the kernel function that transforms functions 
        
        kernel.sigma = self.sigma

        MP = np.stack([self.frame_MCX,self.frame_MCY],axis=1)
        XX = self.frame_xx.reshape(-1)
        YY = self.frame_yy.reshape(-1)
        GP = np.stack([XX,YY], axis=1)        
        res = np.zeros((GP.shape[0], MP.shape[0]))

        if (self.gaussian_marginals is None):
            marginal = np.zeros((GP.shape[0]))
            kernel.compute_gaussian_marginals( GP, self.xlim, self.ylim, marginal)
            self.gaussian_marginals = marginal

        self.frame_K = kernel.create_kernel_matrix_jit(MP,GP,self.tr_weights, self.gaussian_marginals,  res)      
        self.frame_K = res

