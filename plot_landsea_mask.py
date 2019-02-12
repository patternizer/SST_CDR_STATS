#!/usr/bin/env python 

# Code include segment in plot_stt.py
  
fig = plt.figure()
grid = plt.GridSpec(1, 10, wspace=0.3, hspace=0.2)
main_ax = plt.subplot(grid[0, 9])

cmap = 'tab20c' # https://matplotlib.org/users/colormaps.html#qualitative
#cmap = plt.get_cmap('viridis',100)
#cmap.set_under('white')
#cmap.set_over('grey')

ax1 = plt.subplot(grid[0,0:3])
plt.plot(lat_fraction,lat_vec,'b-')
plt.xticks(np.arange(0, 1, step=0.2), fontsize=12)
#plt.yticks(np.arange(-90, 90, step=30), fontsize=12)
plt.xlim([0,1])
#ax1.set_ylim([-85,85])
ticks = ax1.get_yticks()
ax1.set_yticks(np.linspace(-90, 90, 7))
plt.ylim([-90,90])
plt.xlabel(r'Ocean fraction', fontsize=12)
plt.ylabel(r'Latitude / $\mathrm{\degree N}$', fontsize=12)

ax2 = plt.subplot(grid[0, 3:], sharey=ax1)
plot_bitfield(ax2, x[::10].values, y[::10].values, z[0,::10,::10].values.astype('int'), {1: "water", 2: "land", 4: "lake", 8: "ice"}, cmap)
plt.xticks(np.arange(-180, 180, step=60), fontsize=12)
#ax1.set_ylim([-85,85])
ticks = ax1.get_yticks()
ax1.set_yticks(np.linspace(-90, 90, 7))
#plt.yticks(np.arange(-90, 90, step=30), fontsize=12)
plt.xlim([-180,180])
plt.ylim([-90,90])
plt.setp(ax2.get_yticklabels(), visible=False)
plt.xlabel(r'Longitude / $\mathrm{\degree E}$', fontsize=12)
plt.savefig('mask_map.png')

