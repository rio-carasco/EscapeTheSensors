#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import numpy as np
from simanneal import Annealer
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap, Normalize
import random


# In[3]:


def Dist2Sens(Sens, Point):
    distances = np.linalg.norm(Sens - Point, axis=1)
    return distances


# In[4]:


def PropSens(dist, sig = 1):
    p = 1/(1+((dist/sig)**2))
    return p


# In[5]:


def ProbDetect(Sensors, Point):
    distances = Dist2Sens(Sensors, Point)
    twoSens = np.argsort(distances)[:2]
    p1 = PropSens(distances[twoSens[0]])
    p2 = PropSens(distances[twoSens[1]])
    P = p1*p2
    return P


# In[6]:


def StartSolPoints(Start, End):
    points = []
    direc = End-Start
    dist = np.linalg.norm(direc)
    numPoints = (20)+1
    round(dist)
    stepSize = dist/numPoints
    for Lambda in range(numPoints+1):
        point = Start + stepSize*Lambda*(1/dist)*direc
        points.append(point)
    return points


# In[7]:


def Points2Velocity(Points):
    V = []
    numVelocity = len(Points)-1
    for i in range(numVelocity):
        Start = Points[i]
        End = Points[i+1]
        direction= End-Start
        r = np.linalg.norm(direction)
        theta = np.arctan(direction[1]/direction[0])
        V.append((r, theta))
    return V


# In[8]:


def Plot(Start, End, Points, Sensors, Speeds):
    # Create a figure and axes with larger size
    fig, ax = plt.subplots(figsize=(15, 10))

    # Define the optimized path as a list of points
    optimized_path = Points

    # Extract x and y coordinates from the optimized path
    x_coords = [point[0] for point in optimized_path]
    y_coords = [point[1] for point in optimized_path]

    # Define the colors for the path based on speeds
    speed_values = np.array(Speeds)
    normalized_speeds = (speed_values - np.min(speed_values)) / (np.max(speed_values) - np.min(speed_values))
    cmap_speed = plt.cm.get_cmap('viridis_r')
    colors_speed = cmap_speed(normalized_speeds)

    # Compute the range of x and y coordinates for the entire graph
    x_min = min(np.min(x_coords), np.min(Sensors[:, 0]), Start[0], End[0])
    x_max = max(np.max(x_coords), np.max(Sensors[:, 0]), Start[0], End[0])
    y_min = min(np.min(y_coords), np.min(Sensors[:, 1]), Start[1], End[1])
    y_max = max(np.max(y_coords), np.max(Sensors[:, 1]), Start[1], End[1])

    # Add space to the x and y coordinate ranges
    x_range = np.linspace(x_min - 1, x_max + 1, 100)
    y_range = np.linspace(y_min - 1, y_max + 1, 100)

    # Calculate detection probabilities for the heatmap
    heatmap_resolution = 100
    heatmap = np.zeros((heatmap_resolution, heatmap_resolution))
    for x_idx, x in enumerate(x_range):
        for y_idx, y in enumerate(y_range):
            detection = ProbDetect(Sensors, np.array([x, y]))
            heatmap[y_idx, x_idx] = detection

    norm_heatmap = Normalize(vmin=0, vmax=1)
    heatmap_plot = ax.imshow(heatmap, extent=[x_min - 1, x_max + 1, y_min - 1, y_max + 1], cmap='RdYlBu',
                             alpha=0.8, norm=norm_heatmap)

    # Plot the sensors
    ax.scatter(Sensors[:, 0], Sensors[:, 1], color='black', label='Sensors')

    ax.scatter(Start[0], Start[1], color='blue', label='Start')
    ax.scatter(End[0], End[1], color='green', label='End')

    # Plot the path with different colors based on speeds
    for i in range(len(x_coords) - 1):
        ax.plot([x_coords[i], x_coords[i + 1]], [y_coords[i], y_coords[i + 1]], color=colors_speed[i])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Optimal Path')
    ax.grid(True)

    # Add legends for sensors, start, and end
    ax.legend(loc='upper right', bbox_to_anchor=(1, 1))

    # Create a custom colorbar for the speed scale
    cax_speed = fig.add_axes([0.15, 0.1, 0.7, 0.03])  # Adjust the position and size of the speed colorbar
    cbar_speed = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap_speed), cax=cax_speed, orientation='horizontal')
    cbar_speed.set_ticks([0, 1])  # Set the colorbar ticks
    cbar_speed.set_ticklabels(['Low', 'High'])  # Set the tick labels
    cbar_speed.ax.xaxis.set_label_coords(0.5, -1)  # Adjust the position of the colorbar label
    cbar_speed.ax.set_xlabel('Speed')  # Set the colorbar label

    # Create a custom colorbar for the sensor detection heatmap
    cax_heatmap = fig.add_axes([1, 0.15, 0.03, 0.7])  # Adjust the position and size of the heatmap colorbar
    cbar_heatmap = plt.colorbar(heatmap_plot, cax=cax_heatmap)
    cbar_heatmap.ax.set_ylabel('Detection')

    plt.tight_layout()

    return fig


# In[9]:


def PlotE(Iter_E):
    # Sample energy values for each iteration (replace with your own data)
    energy_values = iter_E

    # Generate x-axis values for iterations
    iterations = range(1, len(energy_values) + 1)

    # Adjusting figure size
    plt.figure(figsize=(8, 6))

    # Plotting the line graph with thin lines
    plt.plot(iterations, energy_values, linestyle='-', color='b', linewidth=1)

    # Adding labels and title
    plt.xlabel('Iteration Number')
    plt.ylabel('Energy')
    plt.title('Energy Through Iterations')

    # Adding a curve of best fit (quadratic curve)
    best_fit_curve = np.polyfit(iterations, energy_values, 2)  # Fit a 2nd-degree polynomial (quadratic fit)
    curve_fit_values = np.polyval(best_fit_curve, iterations)
    plt.plot(iterations, curve_fit_values, linestyle='--', color='r', linewidth=1)

    # Display the line graph
    plt.tight_layout()
    plt.show()


# In[10]:


def PlotE(Iter_E):
    # Sample energy values for each iteration (replace with your own data)
    energy_values = Iter_E

    # Generate x-axis values for iterations
    iterations = range(1, len(energy_values) + 1)

    # Adjusting figure size and creating the figure and axis objects
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plotting the line graph with thin lines
    ax.plot(iterations, energy_values, linestyle='-', color='b', linewidth=1)

    # Adding labels and title
    ax.set_xlabel('Iteration Number')
    ax.set_ylabel('Energy')
    ax.set_title('Energy Through Iterations')

    # Adding a curve of best fit (quadratic curve)
    best_fit_curve = np.polyfit(iterations, energy_values, 2)  # Fit a 2nd-degree polynomial (quadratic fit)
    curve_fit_values = np.polyval(best_fit_curve, iterations)
    ax.plot(iterations, curve_fit_values, linestyle='--', color='r', linewidth=1)

    # Display the line graph
    plt.tight_layout()

    return fig

# Example usage:
# Assuming you have your data in the list `energy_values`, call the function like this:
# figure = PlotE(energy_values)
# plt.show()  # To display the figure if needed


# In[11]:


def update(frame, figures):
    # Clear the current axes
    plt.clf()
    
    # Get the current figure
    fig = figures[frame]
    
    # Convert figure to image
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    # Display the image
    plt.imshow(image)
    plt.axis('off')
    
def iterations(list_points, list_speeds, slicing=1000):
    figures = []
    for i in range(len(list_points[::slicing])):
        points = list_points[::slicing]
        speeds = list_speeds[::slicing]
        point = points[i]
        speed = speeds[i]
        figure = Plot(start, end, point, sensors, speed)
        figures.append(figure)
        plt.close()
    
    # Create a new figure
    fig, ax = plt.subplots()

    # Create the animation
    ani = animation.FuncAnimation(fig, update, frames=len(figures), fargs=(figures,), interval=200)
    ani.save('animation.mp4', dpi=500, writer='ffmpeg')
    plt.close()


# In[12]:


def TotalT_ThreshCalc(Start, End, Min_Speed = 1.5):
    Dist = np.linalg.norm(End - Start)
    Time = Dist/Min_Speed
    return Time


# In[13]:


iter_Vectors = []
iter_Points = []
iter_T = []
iter_E = []

# Define the problem class
class SimulatedAnnealer(Annealer):
    def __init__(self, intial_temp):
        initial_state = self.random_state()  # Generate initial state
        super().__init__(initial_state)
        self.Tmax = initial_temp
        
        
    def move(self):
        # Iterate through each point in the state and modify its direction and magnitude
        Vectors = self.state[0]
        Points = self.state[1]
        T = self.state[2]
        
        i = random.randint(1, len(Points)-2)
        
        TotalT_Curr = TotalT_Thresh + 10

        while TotalT_Curr > TotalT_Thresh:
            
            # Randomly adjust the direction of the point by a small angle
            adjustTheta = 1
            angle = np.random.uniform(-np.pi*adjustTheta, np.pi*adjustTheta)  # Modify this range as needed


            # Randomly adjust the magnitude of the point by a small factor
            adjustR = 0.1
            distR = np.random.uniform(0, adjustR)  # Modify this range as needed


            # Update the point with the new polar coordinates
            Points[i] = Points[i] + np.array([distR*np.cos(angle), 
                                              distR*np.sin(angle)])

            Vectors = Points2Velocity(Points)
        
        
            adjustT = 0.1
            magnitude_factorT = np.random.uniform(1-adjustT, 1+adjustT)
            T = T*magnitude_factorT
            
            TotalT_Curr = T*len(Vectors)
        
        iter_Vectors.append(Vectors)
        iter_Points.append(Points)
        iter_T.append(T)

        self.state = Vectors, Points, T
        
    
    def energy(self):
        Vectors = self.state[0]
        Points = self.state[1]
        T = self.state[2]
        
        P = 0
        for i in range(len(Points)):
            P += ProbDetect(sensors, Points[i])
        
        S = 0
        for i in range(len(Vectors)):
            S += (Vectors[i][0])**2/T

        TotalT = T*len(Vectors)
        
        E = float(lambda0*P*T + lambda1*S + lambda2*TotalT)
        
        iter_E.append(E)

        return E


    def random_state(self):
        points = StartSolPoints(start, end)
        vectors = Points2Velocity(points)
        T = TotalT_ThreshCalc(start, end, Min_Speed = 2.5)/len(vectors)
        return vectors, points, T


# In[14]:


def RunAnnealer(Initial_temp = 25000):

    # Create an instance of SimulatedAnnealer
    annealer = SimulatedAnnealer(initial_temp)

    # Set initial state
    initial_state = annealer.random_state()

     # Run the annealing process
    best_state, best_energy = annealer.anneal()

    iter_Speeds = []
    for vectors in iter_Vectors:
        speeds = []
        for speed in vectors:
            speeds.append(np.abs(speed[0])/best_state[2])
        iter_Speeds.append(speeds)
    
    return best_state, best_energy, iter_Vectors, iter_Points, iter_T, iter_Speeds


# In[15]:


def Visialise(Iter_Points, Iter_Speeds, Iter_E, slicing = 5000):
    length = len(iter_Points)
    points = iter_Points[length-1]
    speeds = iter_Speeds[length-1]
    #iterations(iter_Points, iter_Speeds, slicing)
    a = Plot(start, end, points, sensors, speeds)
    b = PlotE(Iter_E)


# In[42]:


#StartEnd
start = np.array([ 1, 5])
end = np.array([ 7, 5])

#Sensors
sensors = np.array([[5, 4], [4, 6.5], [2, 5]])
initial_temp = 25000
lambda0 = 11500
lambda1 = 10000
lambda2 = 1000
TotalT_Thresh = TotalT_ThreshCalc(start, end, Min_Speed = 0.01)+5


# In[45]:


best_state, best_energy, iter_Vectors, iter_Points, iter_T, iter_Speeds = RunAnnealer()


# In[46]:


Visialise(iter_Points, iter_Speeds, iter_E, slicing = 5000)


# In[21]:


def find_convergence_time(iter_E, threshold=0.001):
    """
    Find the time step where the energy (or cost) becomes stable.
    
    Parameters:
        iter_E (list): List of energy (or cost) values at each time step.
        threshold (float): The threshold for the rate of change of energy.
        
    Returns:
        int: The time step where the energy becomes stable. 
            Returns -1 if the energy never stabilizes within the given threshold.
    """
    for i in range(1, len(iter_E) - 1):
        # Calculate the rate of change of energy
        rate_of_change = abs(iter_E[i] - iter_E[i - 1]) / iter_E[i]
        
        # Check if the rate of change is below the threshold
        if rate_of_change < threshold:
            return i
    return -1  # Return -1 if never converges within the given threshold


# In[22]:


find_convergence_time(iter_E, threshold=0.001)


# In[53]:


import numpy as np
import matplotlib.pyplot as plt

# Create an array of x values
x = np.linspace(-10, 10, 400)  # Adjust the range as needed

# Calculate corresponding y values
y = 2 + (2/np.pi) * np.arctan(x)

# Create the plot
plt.figure(figsize=(8, 6))  # Adjust the figure size as needed
plt.plot(x, y, label='y = 2 + (2/$\pi$)arctan(x)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Plot of y = 2 + (2/$\pi$)arctan(x)')
plt.grid(True)
plt.legend()
plt.show()


# In[ ]:





# In[ ]:




