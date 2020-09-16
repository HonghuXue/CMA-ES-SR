'''
Copyright (C) 2013 Travis DeWolf

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.



tau keeps the same trajectory shape, only with the effect of elogating/shortening evenly the whole trajectory
rollout(timesteps) 
cs.runtime makes sure that the rbf is evenly distributed in Z space ,   e.g. cs.run_time =1 , tau = 1, timesteps = 500, num_rbf = 50   v.s. cs.run_time =5. , tau = 1, timesteps = 500, num_rbf = 50
'''
import math as ms
import matplotlib.pyplot as plt
import numpy as np
from dmp import DMPs

    

class DMPs_discrete(DMPs):
    """An implementation of discrete DMPs"""

    def __init__(self, **kwargs):
        """
        """

        # call super class constructor
        super(DMPs_discrete, self).__init__(pattern='discrete', **kwargs)

        self.gen_centers()

        # set variance of Gaussian basis functions
        # trial and error to find this spacing
        self.h = np.ones(self.n_bfs) * self.n_bfs**1.5 / self.c / self.cs.ax

        self.check_offset()

    def gen_centers(self):
        """Set the centre of the Gaussian basis
        functions be spaced evenly throughout run time"""

        '''x_track = self.cs.discrete_rollout()
        t = np.arange(len(x_track))*self.dt
        # choose the points in time we'd like centers to be at
        c_des = np.linspace(0, self.cs.run_time, self.n_bfs)
        self.c = np.zeros(len(c_des))
        for ii, point in enumerate(c_des):
            diff = abs(t - point)
            self.c[ii] = x_track[np.where(diff == min(diff))[0][0]]'''

        # desired activations throughout time
        des_c = np.linspace(0, self.cs.run_time, self.n_bfs)

        self.c = np.ones(len(des_c))
        for n in range(len(des_c)):
            # finding x for desired times t
            self.c[n] = np.exp(-self.cs.ax * des_c[n])

    def gen_front_term(self, x, dmp_num):
        """Generates the diminishing front term on
        the forcing term.

        x float: the current value of the canonical system
        dmp_num int: the index of the current dmp
        """
        return x# * (self.goal[dmp_num] - self.y0[dmp_num])

    def gen_goal(self, y_des):
        """Generate the goal for path imitation.
        For rhythmic DMPs the goal is the average of the
        desired trajectory.

        y_des np.array: the desired trajectory to follow
        """

        return np.copy(y_des[:, -1])

    def gen_psi(self, x):
        """Generates the activity of the basis functions for a given
        canonical system rollout.

        x float, array: the canonical system state or path
        """

        if isinstance(x, np.ndarray):
            x = x[:, None]
        return np.exp(-self.h * (x - self.c)**2)

    def gen_weights(self, f_target):
        """Generate a set of weights over the basis functions such
        that the target forcing term trajectory is matched.

        f_target np.array: the desired forcing term trajectory
        """

        # calculate x and psi
        x_track = self.cs.rollout()
        psi_track = self.gen_psi(x_track)

        # efficiently calculate BF weights using weighted linear regression
        self.w = np.zeros((self.n_dmps, self.n_bfs))
        for d in range(self.n_dmps):
            # spatial scaling term
            k = (self.goal[d] - self.y0[d])
            for b in range(self.n_bfs):
                numer = np.sum(x_track * psi_track[:, b] * f_target[:, d])
                denom = np.sum(x_track**2 * psi_track[:, b])
                self.w[d, b] = numer / (k * denom)
        self.w = np.nan_to_num(self.w)

# ==============================
# Test code
# ==============================

def alpha_beta_rescaling(alpha, conf):
    slope = 0.1
    if isinstance(alpha, np.ndarray):
        alpha_rescaled= np.ones(len(alpha))
        for i in range(len(alpha)):
            if alpha[i] >= (conf['alpha'][1] - conf['alpha'][0])/(2*slope):
                alpha_rescaled[i] = conf['alpha'][1]
            elif alpha[i] <= (-conf['alpha'][1] + conf['alpha'][0])/(2*slope):
                alpha_rescaled[i] = conf['alpha'][0]
            else:
                alpha_rescaled[i] =  slope * alpha[i] + (conf['alpha'][1] + conf['alpha'][0])/2
    else:
        if alpha >= (conf['alpha'][1] - conf['alpha'][0])/(2*slope):
            alpha_rescaled = conf['alpha'][1]
        elif alpha <= (-conf['alpha'][1] + conf['alpha'][0])/(2*slope):
            alpha_rescaled = conf['alpha'][0]
        else:
            alpha_rescaled =  slope * alpha + (conf['alpha'][1] + conf['alpha'][0])/2
    return alpha_rescaled   



if __name__ == "__main__":
#2.5 * np.random.randn(2, 4) + 3

    np.random.seed(33)    
    # test normal run
     # weight+ alpha  as parameters, goal as known. total time (timesteps)/y_0 should be flexible 
   
    #-----------------------------------------------------------With accel and no goal pose------------------------------------------------------
    xbest = np.array([  438.41860402,  -401.82946442,   105.73570447,   503.73672949, -206.55342512,  -797.88760264,   413.88697689,   175.28364035,
                       -1834.60959196,    45.3757339 ,   998.5847565 ,  -211.26862071,
                         524.06012456,  -231.40094561,  -242.50339368,   835.91957963,
                        -112.50276038,  -393.0644123 ,  -380.75583821, -1057.92753059,
                        1036.84921659,   535.30897483,  -240.76826766,  -522.89368164,
                        -678.80816124, -1778.46259382,  1425.02726347,   565.15602852,
                          71.48070622,    38.11720633,   142.39484487,  -120.48270387,
                         696.65704011,  -573.96230792,    96.38220042,  -683.82984746,
                         259.67361763,   889.33995356,   269.61917472,  1148.37708049,
                          40.90049273,  1267.18259372,  -457.8270891 , -1063.87336529,
                        -211.92767698,   536.29377725, -1400.88325524,  -824.46343484,
                        -123.60297969,   -66.26311144,   406.24115893,   230.88779322,
                         236.45418989, -1086.98561864,  1491.89771415,   688.98206837,
                        -900.2167454 ,   423.7612044 , -1002.50755745,  1516.54109931,
                        1010.56914322,   597.98856711,   395.48647569,    89.58462917,
                         395.42038765, -2934.50704387,  -135.43716434,   184.31221322,
                        -563.40674566,  -652.8194862 ,  -169.66315445,   340.02033106,
                         121.67284144,  -811.46414173,  -289.51136018,   533.13036255,
                         901.81863879]).reshape(7,-1)
    
    conf = {'initial_joint_angles': np.array([0., 0., 0., -1./18.*ms.pi, 0., ms.pi* 10/18, 2./9.*ms.pi]),
            'angular_accel_constraint': np.array([15, 7.5, 10, 12.5, 15, 20, 20])/ms.pi * 180,
            'alpha':[1., 20.], 'beta':[1.,20.]}  
    
    n_dmps = 1 
    n_bfs = 10
    slow_down_multitude = 4
    tau = 1
    save = False
    
    
    dmp = DMPs_discrete(dt=.01, n_dmps=n_dmps, n_bfs=n_bfs, w=1000*np.random.normal(0, 1, size = (n_dmps, n_bfs)), goal=0, ay=5 * np.ones(n_dmps), y0=0)
    y_track, dy_track, ddy_track = dmp.rollout(timesteps = round(500/tau), tau = tau)  
    plt.figure(1, figsize=(6, 3))
    plt.plot(y_track, lw=2)
    plt.title('Angle profile')
    plt.xlabel('time (ms)')
    plt.ylabel('Angle [deg]')
    plt.legend(['goal', 'system state'], loc='lower right')
    plt.tight_layout()
    
    
#    ay=alpha_beta_rescaling(xbest[-i][-1]*np.ones(n_dmps), conf)
    
#    for i in range(7):
#        ay=alpha_beta_rescaling(xbest[-i][-1]*np.ones(n_dmps), conf)   
#        print(ay)
#        dmp = DMPs_discrete(dt=.01, n_dmps=1, n_bfs=n_bfs, w=np.expand_dims(xbest[i][:-1], axis=0), goal= conf['initial_joint_angles'][i]*180 /ms.pi, ay=alpha_beta_rescaling(xbest[i][-1]*np.ones(n_dmps), conf), by=alpha_beta_rescaling(xbest[i][-1]*np.ones(n_dmps), conf)/4., y0=conf['initial_joint_angles'][i]*180 /ms.pi)
#        y_track, dy_track, ddy_track = dmp.rollout(timesteps = round(500/tau), tau = tau)   
#    
#        plt.figure(i+1, figsize=(6, 3))
##        plt.plot(np.ones(len(ddy_track)) * conf['angular_accel_constraint'][i], 'r--', lw=2)
##        plt.plot(-np.ones(len(ddy_track)) * conf['angular_accel_constraint'][i], 'r--', lw=2)
#        plt.plot(y_track, lw=2)
#        plt.title('Angle profile')
#        plt.xlabel('time (ms)')
#        plt.ylabel('Angle [deg]')
#        plt.legend(['goal', 'system state'], loc='lower right')
#        plt.tight_layout()


    if save == True:
        dmp = DMPs_discrete(dt=.01, n_dmps=7, n_bfs=n_bfs, w=xbest[:,:-2], goal=xbest[:,-2]+ conf['initial_joint_angles']*180 /ms.pi, ay=alpha_beta_rescaling(xbest[:,-1]*np.ones(n_dmps), conf), by=alpha_beta_rescaling(xbest[:,-1]*np.ones(n_dmps), conf)/4., y0=conf['initial_joint_angles']*180 /ms.pi)# weight as parameters, goal as known
        y_track, dy_track, ddy_track = dmp.rollout(timesteps = round(500/tau), tau = tau)     
        np.savetxt('Joint_angle.txt', y_track, delimiter = ';' , fmt = '%10.5f')
    
#    plt.figure(2, figsize=(6, 3))
#    plt.plot(np.ones(len(y_track))*dmp.goal, 'r--', lw=2)
#    plt.plot(dy_track, lw=2)
#    plt.title('Angular Velocity Profile')
#    plt.xlabel('time (ms)')
#    plt.ylabel('\u03C9  [deg/s]')
#    plt.legend(['goal', 'system state'], loc='lower right')
#    plt.tight_layout()

#    # test imitation of path run
#    plt.figure(2, figsize=(6, 4))
#    n_bfs = [10]
#
#    # a straight line to target
#    path1 = np.sin(np.arange(0, 1, .01)*5)
#    # a strange path to target
#    path2 = np.zeros(path1.shape)
#    path2[int(len(path2) / 2.):] = .5
#
#    for ii, bfs in enumerate(n_bfs):
#        dmp = DMPs_discrete(n_dmps=2, n_bfs=bfs)
#
#        dmp.imitate_path(y_des=np.array([path1, path2]))
#        # change the scale of the movement
#        dmp.goal[0] = 3
#        dmp.goal[1] = 2
#
#        y_track, dy_track, ddy_track = dmp.rollout()
#
#        plt.figure(2)
#        plt.subplot(211)
#        plt.plot(y_track[:, 0], lw=2)
#        plt.subplot(212)
#        plt.plot(y_track[:, 1], lw=2)
#
#    plt.subplot(211)
#    a = plt.plot(path1 / path1[-1] * dmp.goal[0], 'r--', lw=2)
#    plt.title('DMP imitate path')
#    plt.xlabel('time (ms)')
#    plt.ylabel('system trajectory')
#    plt.legend([a[0]], ['desired path'], loc='lower right')
#    plt.subplot(212)
#    b = plt.plot(path2 / path2[-1] * dmp.goal[1], 'r--', lw=2)
#    plt.title('DMP imitate path')
#    plt.xlabel('time (ms)')
#    plt.ylabel('system trajectory')
#    plt.legend(['%i BFs' % i for i in n_bfs], loc='lower right')
#
#    plt.tight_layout()
#    plt.show()

    plt.figure()
    psi_track = dmp.gen_psi(dmp.cs.rollout())
    plt.plot(psi_track)
    