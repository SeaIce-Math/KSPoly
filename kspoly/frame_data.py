

import numpy as np 


class frame_data:
    def __init__(self, polygons, velocities = None):
        self.polygons = polygons
        self.fields = dict()
        self.field_names = []
        
        if velocities is not None:
            self.set_field(velocities, 'velocity')
    
    def set_field(self, value, name):
        self.fields[name] = np.array(value)
        if self.fields[name].shape[0] != len(self.polygons):
            raise Exception('The number of field values (first dimension of the tensor) should be equal to the number of polygons!')
        
        if self.fields[name].ndim == 1:
            self.fields[name] = np.expand_dims(self.fields[name], axis=1)
        self.field_names.append(name)
