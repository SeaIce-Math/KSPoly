import numpy as np
import triangle as tr


class triangulation:
    def __init__(self) -> None:
        pass

    def get_floe_area(self, polygon):
        X = polygon[0]
        Y = polygon[1]
        return 0.5*np.abs(np.dot(X,np.roll(Y,1))-np.dot(Y,np.roll(X,1)))
    
    def get_rough_center(self, polygon):
        X = polygon[0]
        Y = polygon[1]
        
        cx = np.sum(X)/len(X)
        cy = np.sum(Y)/len(Y)
        return cx, cy

# Generates a triangulation from a set of polygons with one point from each triangle. 
    def generate_triangle_points(self, xlim, ylim, polygons, area_p):
    
            lstX = []
            lstY = []
            lstf = []
            lstW = []
            triangleList = []

            ATTR = 1234.


            total_area = (xlim[1]-xlim[0])*(ylim[1]-ylim[0])
            #floe_idx = 0

            k=-1

            for p in polygons:

                k = k+1
                
                
                x,y = self.get_rough_center(p)

                if x < xlim[0] or x > xlim[1] or y < ylim[0] or y > ylim[1]:
                    continue

                area = self.get_floe_area(p)


                if False: #area < 1*self.area_threshold:
                    print('SKIPPED')
                    lstX.append(x)
                    lstY.append(y)
                    lstf.append(k)
                    lstW.append(area/1000000.0)
                else:
                    vertices = []
                    segments = []
                    regions = []
                    B0 = p 
                    
                    Bpts = B0.shape[1]
                    B = np.copy(B0)
                    j = 0
                    for i in range(Bpts):                    
                        vertices.extend([[B[0,i], B[1,i]]])
                        segments.extend([(j,(j+1)%Bpts)])
                        j = j+1
                    
                    regions.append([x,y,1234, 0.]) # attribute = 100

     
                    TR_IN = dict(vertices=vertices, segments=segments, regions=regions)
                    if area>2 * area_p:
                        TR = tr.triangulate(TR_IN, 'ipa'+str(area_p))
                    else:
                        TR = tr.triangulate(TR_IN, 'ip')
                    triangleList.append(TR)

                    points = TR['vertices']
                    for triang in TR['triangles']:
                        x1, y1 = points[triang[0]]
                        x2, y2 = points[triang[1]]
                        x3, y3 = points[triang[2]]
                        x,y = (x1+x2+x3)/3.0  , (y1+y2+y3)/3.0
                        tr_area = 0.5 * abs((x2-x1)*(y3-y1) - (x3-x1)*(y2-y1))
                        lstX.append(x)
                        lstY.append(y)
                        lstf.append(k)
                        #lstW.append(tr_area/total_area)
                        lstW.append(tr_area)                    
            X = np.array(lstX)
            Y = np.array(lstY)
            f = np.array(lstf)
            W = np.array(lstW)
            return X,Y,f,W, triangleList
