import numpy as np
import pandas as pd
import matplotlib.pyplot as plot

class ComplexRadar():
    def __init__(self, fig, variables, ranges,
                 n_ordinate_levels=6):
        angles = np.arange(0, 360, 360./len(variables))

        axes = [fig.add_axes([0.1,0.1,0.9,0.9],polar=True,
                label = "axes{}".format(i)) 
                for i in range(len(variables))]
        l, text = axes[0].set_thetagrids(angles, labels=variables)
        [txt.set_rotation(angle-90) for txt, angle 
             in zip(text, angles)]
        for ax in axes[1:]:
            ax.patch.set_visible(False)
            ax.grid("off")
            ax.xaxis.set_visible(False)
            #ax.set_rgrids(range(1, 6), angle=angle, labels=variables)
            
        labels = [ [0.0, 0.1, 0.2, 0.3, 0.4, 0.5] for i in range(10)]
        
        for ax, angle, label in zip(axes, angles, labels):
            ax.set_rgrids(range(1, 6), angle=angle, labels=label)
            ax.spines["polar"].set_visible(False)
            ax.set_ylim(0, 0.5)
            
        #for i, ax in enumerate(axes):
            #grid = np.linspace(*ranges[i], 
            #                   num=n_ordinate_levels)
            #gridlabel = ["{}".format(round(x,2)) for x in grid]
            #if ranges[i][0] > ranges[i][1]:
            #   grid = grid[::-1] # hack to invert grid
                          # gridlabels aren't reversed
            #gridlabel[0] = "" # clean up origin
            #ax.set_rgrids(grid, labels=gridlabel,angle=angles[i])
            #ax.set_rgrids(range(1, 6), angle=angles[i], labels=gridlabel)
            #ax.set_ylim(0, 0.5)
        # variables for plotting
        self.angle = np.deg2rad(np.r_[angles, angles[0]])
        self.ranges = ranges
        self.ax = axes[0]
    def plot(self, data, *args, **kw):
        sdata = _scale_data(data, self.ranges)
        self.ax.plot(self.angle, np.r_[sdata, sdata[0]], *args, **kw)
    def fill(self, data, *args, **kw):
        sdata = _scale_data(data, self.ranges)
        self.ax.fill(self.angle, np.r_[sdata, sdata[0]], *args, **kw)