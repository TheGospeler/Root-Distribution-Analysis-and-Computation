import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.interpolate import RBFInterpolator
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from matplotlib.colors import PowerNorm
# import functions  # imports the relevant functions used in processing and making the plots
#1: Functions that creates the cylindrical dataset in cartesian coordinate

# Creating a function that creates the X, Y coordinates from 
# a given radius and the number of columns or traces

def create_CardPoints(rad, trace_num):
    """Create 1 2D array with with shape (trace_num x 2).
    
    parameters:
    ----------
    rad: radius from the center of the trunk in centimeters.
    
    Returns:
    ρ, θ, X, and Y
    """
    dist = trace_num * 0.02  # The distance of the horizontal profile
    
    arr = np.ones((trace_num, 4))
    arr[:, 0] = rad/100  # converting the radius to meters since the distance is in meters
    arr[:, 1] = np.linspace(0.02, dist, trace_num) / (arr[:, 0])
    # θ = distance / radius
    arr[:, 2] = arr[:, 0] * np.cos(arr[:, 1])
    arr[:, 3] = arr[:, 0] * np.sin(arr[:, 1])
    
    return arr   

def createZ(sample_pts, depth = 'time'):
    """Convert the number of digitized points into depth profile.
    
    The vertical axis can be converted to depth in time or depth by
    spreading out the numbers of digitized points across the 45.45 ns
    the EM waves were collected.
    """
    
    if depth == 'time':
        calc_depth = np.linspace(0, 45.45, sample_pts)  # Depth in time
        return calc_depth
        
    # To convert to distance, distance = velocity * time/2
    # the velocity was discovered to vary between 0.1 - 0.15 ns
    
    return np.linspace(0, 45.45, sample_pts) * .5 * 0.1  
        

#2: the cylindrical dataset in cartesian coordinate
def create_cartes_init(tree_folder = 'Tree4_6x6_migrated', stored_rad='04840', start_radius = 48.4):
    radius = start_radius 

    transect1 = np.genfromtxt(f"{tree_folder}/Migrated_{stored_rad}.ASC", dtype=None)
    scaler = MinMaxScaler(feature_range=(-1, 1))  # initializing scalar
    amp_array = scaler.fit_transform(transect1)

    # creating an array that stores the X and Y coordinate for all the transects
    coord_array = create_CardPoints(radius, transect1.shape[1])
    depth_z = createZ(transect1.shape[0], depth='dist')

    return coord_array, amp_array, depth_z


#3: Function that compiles transects 2-end of the tree in cartesian coordinate into an array
def create_cartes_arr(coord_array, amp_array, tree_folder = 'Tree4_6x6_migrated', 
                      start_radius = 58.4):
    list_dir = os.listdir(tree_folder)  # a list of the tree's transects
    list_dir = sorted(list_dir)  # Added sorted to the list

    # starting radius of the second transect
    radius = start_radius

    for index, transect in enumerate(list_dir[1:]):
        selected_trans = np.genfromtxt(f"{tree_folder}/{transect}", dtype=None)

        # store the individual file amplitude in the dictionary
        # Normalized the amplitude to be between -1 and 1
        
        scaler = MinMaxScaler(feature_range=(-1, 1))  # initializing scalar
        amp_array = np.hstack((amp_array, scaler.fit_transform(selected_trans)))  # fixed column
        coord_array = np.vstack((coord_array, create_CardPoints(radius, selected_trans.shape[1])))

        radius += 10  # incrementing 10 cm increase for each transect.

    # Transpose the amp_arr
    amp_array = amp_array.T
    
    return coord_array, amp_array


#4: Function that Plots the cylindrical dataset in cartesian coordinate
def plot_init_transect(coord_array, depth, top_view=False):
    fig = plt.figure(figsize=(11,9), dpi=100)
    ax = plt.axes(projection='3d')

    # The variables are
    X = coord_array[:, 2]
    Y = coord_array[:, 3]
    Z = depth.reshape(-1, 1)
    ax.plot_surface(X, Y, Z)
    if top_view:
        ax.view_init(azim=180, elev= 90)  # azim=180 rotates the 0 to start from the East
    ax.set_xlabel('X-axis (m)')
    ax.set_ylabel('Y-axis (m)')
    ax.set_zlabel('Depth (m)')
    ax.invert_zaxis()


    plt.show()

def plot_cartes_coord(coord_array, depth, top_view=False):
    fig = plt.figure(figsize=(11,9), dpi=100)
    ax = plt.axes(projection='3d')

    # The variables are
    X = coord_array[:, 2]
    Y = coord_array[:, 3]
    Z = Z = np.linspace(0, depth[-1], coord_array[:, 2].shape[0])
    ax.scatter3D(X, Y, Z, s=2)
    if top_view:
        ax.view_init(azim=180, elev= 90)  # azim=180 rotates the 0 to start from the East
    ax.invert_zaxis()
    ax.set_xlabel('X-axis (m)')
    ax.set_ylabel('Y-axis (m)')
    ax.set_zlabel('Depth (m)')

    plt.show()

    
    
#5: Function that creates the Mesh that distributes the amplitude across the cartesian coordinates

def create_mesh(coord_array, amp_array, depth_z):
    xy_coord = coord_array[:, 2:]
    x_arr = []
    y_arr = []
    z_arr = []
    c_arr = []
    
    for i, (x, y) in enumerate(xy_coord):
        for j in range(depth_z.shape[0]):
            x_arr.append(x), y_arr.append(y)
            z_arr.append(depth_z[j])
            # creating a new index that can be modified
            c_arr.append(amp_array[i, j]) 
        
    # converting the list to array
    x_arr, y_arr, z_arr, c_arr = np.array(x_arr), np.array(y_arr), np.array(z_arr), np.array(c_arr) 
    
    return x_arr, y_arr, z_arr, c_arr



#6: Function that slices out the original array to a specific length
def arr_subsetting(arr, depth):
    '''
    Param
    -----
    arr: A list that contains x, y, z, and c arrays in this order
    depth: The prefered point to slice the array.
    '''
    mask = (arr[2] <=depth)
    x_main =arr[0][mask]
    y_main =arr[1][mask]
    z_main =arr[2][mask]
    c_main =arr[3][mask]
    
    return x_main, y_main, z_main, c_main



#7: Functions that Plots the amplitude in the cartesian coordinate
def matplot_amplitude(x_arr, y_arr, z_arr, c_arr, top_view=False):
    fig = plt.figure(figsize=(11,9), dpi=100)
    ax = plt.subplot(111, projection='3d')

    cmap = plt.cm.rainbow
    plot = ax.scatter3D(x_arr, y_arr, z_arr, c=c_arr, cmap=cmap)
    ax.set_xlabel('X-axis (m)')
    ax.set_ylabel('Y-axis (m)')
    ax.set_zlabel('Depth (m)')
    
    if top_view:
        ax.view_init(azim=0, elev= 90)
    ax.invert_zaxis()
    fig.colorbar(plot)
    plt.show()

def plotly_amplitude(x_arr, y_arr, z_arr, c_arr, top_view=False):
    fig = px.scatter_3d(x=x_arr, y=y_arr, z=z_arr,
                  color=c_arr, color_continuous_scale='rainbow')
    fig.update_layout(
        scene={
            'zaxis': {'autorange': 'reversed'}, # reverse automatically
        }
    )
    
    if top_view:
        name = 'eye = (x:0., y:0., z:2.5)'
        camera = dict(
            eye=dict(x=0., y=0., z=2.5)
        )

        fig.update_layout(scene_camera=camera, title=name)
    fig.show()



#8: Function that creates and plots the Depth Slices for the tree
def depth_slice(x, y, z, c, depth=0.05, plot_coarseness = 1):
    '''Plot the depth slice passed into the function.
    
    Parameters
    ----------
    depth: The particular depth to be plotted.
    plot_coarseness: How rough the surface will be.
    
    returns:
    A list containing the x, y, z, and c arrays which can be used to
    calculate the volume.
    '''
    mask = (z >= depth) & (z <depth+.01)
    x =x[mask]
    y =y[mask]
    z =z[mask]
    c =c[mask]


    cmap = plt.cm.rainbow
    fig = px.scatter_3d(x=x, y=y, z=z,
                  color=c, color_continuous_scale='rainbow', opacity=plot_coarseness)

    cmap = plt.cm.rainbow

    fig.update_layout(
        scene={
            'zaxis': {'autorange': 'reversed'}, # reverse automatically
            
        }
    )
    
    name = 'eye = (x:0., y:0., z:2.5)'
    camera = dict(
        eye=dict(x=0., y=0., z=2.5)
    )

    fig.update_layout(scene_camera=camera, title=name)

    fig.show()
    
    return [x, y, z, c]


#9: Function that creates and plots the Depth Slice 5 cm - 45 cm for the tree
def return_arr(x, y, z, c, depth=0.05):
    mask = (z >= depth) & (z <depth+.01)
    x_slice =x[mask]
    y_slice =y[mask]
    z_slice =z[mask]
    c_slice =c[mask]
    
    return [x_slice, y_slice, z_slice, c_slice]


def depth_slice_multiple(x, y, z, c, depth=0.05, plot_coarseness = 1, title='Tree'):
    '''Plot the depth slice passed into the function.
    
    Parameters
    ----------
    depth: The particular depth to be plotted.
    plot_coarseness: How rough the surface will be.
    
    returns:
    A list containing the x, y, z, and c arrays which can be used to
    calculate the volume.
    '''

    # Creating the 6x6 depth
    fig = make_subplots(
    rows=3, cols=3,
    shared_yaxes=True,
    horizontal_spacing=0.06,
    shared_xaxes=True,
    vertical_spacing=0.06,
    subplot_titles=("Depth = 5 cm", "Depth = 10 cm", "Depth = 15 cm", "Depth = 20 cm", 
                "Depth = 25 cm", "Depth = 30 cm", "Depth = 35 cm", "Depth = 40 cm", "Depth = 45 cm"),
    )

    xyz_5cm = return_arr(x=x, y=y, z=z, c=c, depth = .05)
    fig.add_trace(go.Scatter(x=xyz_5cm[0], y=xyz_5cm[1],
                               mode ='markers',
                               marker=dict(
                               color=xyz_5cm[3],
                               colorscale='Rainbow', coloraxis="coloraxis",
                               opacity=plot_coarseness)),
                  row=1, col=1)

    xyz_10cm = return_arr(x=x, y=y, z=z, c=c, depth = .10)
    fig.add_trace(go.Scatter(x=xyz_10cm[0], y=xyz_10cm[1],
                               mode ='markers',
                               marker=dict(
                               color=xyz_10cm[3],
                               colorscale='Rainbow', coloraxis="coloraxis",
                               opacity=plot_coarseness)),
                  row=1, col=2)

    xyz_15cm = return_arr(x=x, y=y, z=z, c=c, depth = .15)
    fig.add_trace(go.Scatter(x=xyz_15cm[0], y=xyz_15cm[1],
                               mode ='markers',
                               marker=dict(
                               color=xyz_15cm[3],
                               colorscale='Rainbow', coloraxis="coloraxis",
                               opacity=plot_coarseness)),
                  row=1, col=3)

    xyz_20cm = return_arr(x=x, y=y, z=z, c=c, depth = .20)
    fig.add_trace(go.Scatter(x=xyz_20cm[0], y=xyz_20cm[1],
                               mode ='markers',
                               marker=dict(
                               color=xyz_20cm[3],
                               colorscale='Rainbow', coloraxis="coloraxis",
                               opacity=plot_coarseness)),
                  row=2, col=1)

    xyz_25cm = return_arr(x=x, y=y, z=z, c=c, depth = .25)
    fig.add_trace(go.Scatter(x=xyz_25cm[0], y=xyz_25cm[1],
                               mode ='markers',
                               marker=dict(
                               color=xyz_25cm[3],
                               colorscale='Rainbow', coloraxis="coloraxis",
                               opacity=plot_coarseness)),
                  row=2, col=2)

    xyz_30cm = return_arr(x=x, y=y, z=z, c=c, depth = .30)
    fig.add_trace(go.Scatter(x=xyz_30cm[0], y=xyz_30cm[1],
                               mode ='markers',
                               marker=dict(
                               color=xyz_30cm[3],
                               colorscale='Rainbow', coloraxis="coloraxis",
                               opacity=plot_coarseness)),
                  row=2, col=3)

    xyz_35cm = return_arr(x=x, y=y, z=z, c=c, depth = .35)
    fig.add_trace(go.Scatter(x=xyz_35cm[0], y=xyz_35cm[1],
                               mode ='markers',
                               marker=dict(
                               color=xyz_35cm[3],
                               colorscale='Rainbow', coloraxis="coloraxis",
                               opacity=plot_coarseness)),
                  row=3, col=1)

    xyz_40cm = return_arr(x=x, y=y, z=z, c=c, depth = .40)
    fig.add_trace(go.Scatter(x=xyz_40cm[0], y=xyz_40cm[1],
                               mode ='markers',
                               marker=dict(
                               color=xyz_40cm[3],
                               colorscale='Rainbow', coloraxis="coloraxis",
                               opacity=plot_coarseness)),
                  row=3, col=2)

    xyz_45cm = return_arr(x=x, y=y, z=z, c=c, depth = .45)
    fig.add_trace(go.Scatter(x=xyz_45cm[0], y=xyz_45cm[1],
                               mode ='markers',
                               marker=dict(
                               color=xyz_45cm[3],
                               colorscale='Rainbow', coloraxis="coloraxis",
                               opacity=plot_coarseness)),
                  row=3, col=3)

    fig.update_layout(height=1000, width=1000,
                  title_text=f"{title}")


    # Update xaxis properties
    fig.update_xaxes(title_text="xaxis(meter)", showgrid=False, row=3, col=1)
    fig.update_xaxes(title_text="xaxis(meter)", row=3, col=2, showgrid=False)
    fig.update_xaxes(title_text="xaxis(meter)", row=3, col=3, showgrid=False)

    # Update yaxis properties
    fig.update_yaxes(title_text="yaxis(meter)", row=1, col=1, showgrid=False)
    fig.update_yaxes(title_text="yaxis(meter)", showgrid=False, row=2, col=1)
    fig.update_yaxes(title_text="yaxis(meter)", showgrid=False, row=3, col=1)

    fig.update_layout(coloraxis=dict(colorscale='Rainbow'), showlegend=False)
    # Update title and height
    #fig.update_layout(title_text="Customizing Subplot Axes", height=700)

    fig.show()



# 10: Plotting the Interpolation plot using matplotlib library
def multiplot_interpolated(int_mesh, ampl_list, norm_val = 1, inner_rad=0.54, outer_rad=2.54, title='Tree 7'):
    
    # For creating the small radius
    theta = np.linspace(0, 2 * np.pi, 150)
    x_inner = inner_rad * np.cos( theta )
    y_inner = inner_rad * np.sin( theta )
    
    # used for heightening the value of the roots.
    norm = PowerNorm(gamma=norm_val)
    
    # extracting the x_data and y_data
    x_data, y_data = int_mesh[0], int_mesh[1]
    
    # Unpacking the amplitude
    amp5, amp10, amp15, amp20, amp25, amp30, amp35, amp40, amp45 = ampl_list
    
    # Create a mask for points outside the circle
    # Circle center and radius
    center_x, center_y = 0, 0
    radius = outer_rad

    mask = (x_data - center_x)**2 + (y_data - center_y)**2 > radius**2
    amp5[mask] = np.nan
    amp10[mask] = np.nan
    amp15[mask] = np.nan
    amp20[mask] = np.nan
    amp25[mask] = np.nan
    amp30[mask] = np.nan
    amp35[mask] = np.nan
    amp40[mask] = np.nan
    amp45[mask] = np.nan


    # Create the figure and axis
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3, 3, figsize=(13, 12))

    cmap = plt.cm.rainbow
    
    # Title
    fig.suptitle(f'{title}')
    
    # Display the pcolormesh inside the circle
    ax1.pcolormesh(x_data, y_data, amp5, cmap=cmap, norm=norm)
    ax1.fill_between(x_inner, y_inner, color='white')
    ax1.set_title('Depth Slice = 5 cm')
    ax1.set_aspect('equal', adjustable='box')  # Set aspect ratio to equal
    
    ax2.pcolormesh(x_data, y_data, amp10, cmap=cmap, norm=norm)
    ax2.fill_between(x_inner, y_inner, color='white')
    ax2.set_title(f'Depth Slice = 10 cm')
    ax2.set_aspect('equal', adjustable='box')  # Set aspect ratio to equal
    
    ax3.pcolormesh(x_data, y_data, amp15, cmap=cmap, norm=norm)
    ax3.fill_between(x_inner, y_inner, color='white')
    ax3.set_title('Depth Slice = 15 cm')
    ax3.set_aspect('equal', adjustable='box')  # Set aspect ratio to equal
    
    ax4.pcolormesh(x_data, y_data, amp20, cmap=cmap, norm=norm)
    ax4.fill_between(x_inner, y_inner, color='white')
    ax4.set_title('Depth Slice = 20 cm')
    ax4.set_aspect('equal', adjustable='box')  # Set aspect ratio to equal
    
    ax5.pcolormesh(x_data, y_data, amp25, cmap=cmap, norm=norm)
    ax5.fill_between(x_inner, y_inner, color='white')
    ax5.set_title('Depth Slice = 25 cm')
    ax5.set_aspect('equal', adjustable='box')  # Set aspect ratio to equal
    
    ax6.pcolormesh(x_data, y_data, amp30, cmap=cmap, norm=norm)
    ax6.fill_between(x_inner, y_inner, color='white')
    ax6.set_title('Depth Slice = 30 cm')
    ax6.set_aspect('equal', adjustable='box')  # Set aspect ratio to equal
    
    ax7.pcolormesh(x_data, y_data, amp35, cmap=cmap, norm=norm)
    ax7.fill_between(x_inner, y_inner, color='white')
    ax7.set_title('Depth Slice = 35 cm')
    ax7.set_aspect('equal', adjustable='box')  # Set aspect ratio to equal
    
    ax8.pcolormesh(x_data, y_data, amp40, cmap=cmap, norm=norm)
    ax8.fill_between(x_inner, y_inner, color='white')
    ax8.set_title('Depth Slice = 40 cm')
    ax8.set_aspect('equal', adjustable='box')  # Set aspect ratio to equal
        
    cbar = ax9.pcolormesh(x_data, y_data, amp45, cmap=cmap, norm=norm)
    ax9.fill_between(x_inner, y_inner, color='white')
    ax9.set_title('Depth Slice = 45 cm')
    ax9.set_aspect('equal', adjustable='box')  # Set aspect ratio to equal



    for ax in fig.get_axes():
        ax.label_outer()
        
    # getting the colorbar for all the subplots
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(cbar, cbar_ax)
        
    # Display the plot
    plt.show()


# 11: Single Interpolated Plotting
def plot_interpolated(int_mesh, ampl, norm_val = 1, inner_rad=0.54, outer_rad=2.54, val=5):
    
    # For creating the small radius
    theta = np.linspace(0, 2 * np.pi, 150)
    x_inner = inner_rad * np.cos( theta )
    y_inner = inner_rad * np.sin( theta )
    
    # used for heightening the value of the roots.
    norm = PowerNorm(gamma=norm_val)
    
    # extracting the x_data and y_data
    x_data, y_data = int_mesh[0], int_mesh[1]
    
    # Create a mask for points outside the circle
    # Circle center and radius
    center_x, center_y = 0, 0
    radius = outer_rad

    mask = (x_data - center_x)**2 + (y_data - center_y)**2 > radius**2
    ampl[mask] = np.nan

    cmap = plt.cm.rainbow

    # Create the figure and axis
    fig, ax = plt.subplots()

    # Display the pcolormesh inside the circle
    c = ax.pcolormesh(x_data, y_data, ampl, cmap=cmap, norm=norm)
    ax.fill_between(x_inner, y_inner, color='white')
    ax.set_title(f'Depth Slice = {val} cm')

    # Set aspect ratio to equal
    ax.set_aspect('equal', adjustable='box')

    # Display the plot
    plt.show()
    
# 12: Creating the Interpolation points
# build a mesh to interpolate onto
def interpolator(data_3d, num_grid=200, grid_ext=3):
    '''
    Interpolate the Scatter Plots.
    
    Param
    -----
    data_3d: contains the x, y, z, and amplitude values.
    '''

    # Will change the dimension when necessary
    int_mesh = np.mgrid[-1*grid_ext:grid_ext:num_grid*1j, -1*grid_ext:grid_ext:num_grid*1j]
    mesh = int_mesh.reshape(2,-1).T

    # perform the interpolation
    points = np.column_stack((data_3d[0], data_3d[1]))
    # print(x, y, points)
    new_vals = RBFInterpolator(points, data_3d[3])(mesh)
    interp_values = new_vals.reshape(num_grid, num_grid)
    
    return int_mesh, interp_values


# 13:  Return array from 5 cm to 45 cm deep below the soil

def return_arr_depth_5_45cm(x_arr, y_arr, z_arr, c_arr):
    
    _5cm = return_arr(x_arr, y_arr, z_arr, c_arr, depth=0.05)  # Depth = 5 cm
    _10cm = return_arr(x_arr, y_arr, z_arr, c_arr, depth=0.10)  # Depth = 10 cm
    _15cm = return_arr(x_arr, y_arr, z_arr, c_arr, depth=0.15)  # Depth = 15 cm
    _20cm = return_arr(x_arr, y_arr, z_arr, c_arr, depth=0.20)  # Depth = 20 cm
    _25cm = return_arr(x_arr, y_arr, z_arr, c_arr, depth=0.25)  # Depth = 25 cm
    _30cm = return_arr(x_arr, y_arr, z_arr, c_arr, depth=0.30)  # Depth = 30 cm
    _35cm = return_arr(x_arr, y_arr, z_arr, c_arr, depth=0.35)  # Depth = 35 cm
    _40cm = return_arr(x_arr, y_arr, z_arr, c_arr, depth=0.40)  # Depth = 40 cm
    _45cm = return_arr(x_arr, y_arr, z_arr, c_arr, depth=0.45)  # Depth = 45 cm
    
    return _5cm ,_10cm, _15cm, _20cm, _25cm, _30cm, _35cm, _40cm, _45cm


# 14: Used in reducing the array size to correct for stationary matrices
def downsample_array(x_arr, y_arr, z_arr, c_arr, downsamp_step=2):
    
    # Get the shape of the array
    arr_size =x_arr.shape[0]
    downsized = np.arange(0, arr_size, downsamp_step)
    
    #x_arr, y_arr, z_arr, c_arr = array
    c_arr[0]
    # downsampling the array
    x_arr = x_arr[downsized]
    y_arr = y_arr[downsized]
    z_arr = z_arr[downsized]
    c_arr = c_arr[downsized]
    #print(x_arr)
    
    return x_arr, y_arr, z_arr, c_arr













