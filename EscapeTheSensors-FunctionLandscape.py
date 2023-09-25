#!/usr/bin/env python
# coding: utf-8

# In[73]:


import os
import numpy as np
from simanneal import Annealer
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap, Normalize
import random


# In[74]:


def PropSens(dist, sig = 15):
    p = 1/(1+((dist/sig)**2))
    return p


# In[75]:


def ProbDetect(Sensors, Point):
    Point = Height(Point)
    probs = []
    for Sensor in Sensors:
        Sensor = Height(Sensor)
        dist = np.linalg.norm(np.array(Sensor) - np.array(Point))
        probs.append(PropSens(dist)*LineOfSightResistance(Point, Sensor)) 
        #print(LineOfSightResistance(Point, Sensor))
    probs_sorted = sorted(probs, reverse=True)
    p1 = probs_sorted[0]
    p2 = probs_sorted[1]
    P = p1*p2
    return P


# In[76]:


def LineOfSightResistance(Position, Sensor, interval = 0.1, strength = 0.8): #0<=strength<=1
    Position = Height(Position)
    Sensor = Height(Sensor)
    dist = np.linalg.norm(np.array(Sensor) - np.array(Position))
    if dist == 0:
        Resistance = 0
        return Resistance 
    Direc = (np.array(Sensor) - np.array(Position))/dist
    Lambda = 0
    CheckP = 0
    NLoSCheckP = 0
    while Lambda <= dist:
        CheckP += 1
        L = Position + Lambda*Direc
        HeightL = L[2]
        HeightS = Height([L[0],L[1]])[2]
        if HeightS > HeightL:
            NLoSCheckP += 1
        Lambda += interval
    NLoSPercentage = NLoSCheckP/CheckP
    Resistance = (-strength*NLoSPercentage)+1
    return Resistance**2


# In[77]:


def StartSolPoints(Start, End):
    points = []
    direc = np.array(End)-np.array(Start)
    dist = np.linalg.norm(direc)
    numPoints = (20)+1
    round(dist)
    stepSize = dist/numPoints
    for Lambda in range(numPoints+1):
        point = Start + stepSize*Lambda*(1/dist)*direc
        points.append(point)
    return points


# In[78]:


def Points2VelocityNGrad(Points):
    V = []
    Grads = []
    numVelocity = len(Points)-1
    for i in range(numVelocity):
        Start = np.array(Height(Points[i]))
        End = np.array(Height(Points[i+1]))
        Vector = End-Start
        L = np.sqrt(Vector[0]**2 + Vector[1]**2)
        Z = Vector[2]
        Grad = Z/L
        V.append(Vector)
        Grads.append(Grad)
    return V, Grads


# In[79]:


def Plot(Start, End, Points, Sensors, Speeds):
    Start = np.array(Start)
    End = np.array(End)
    Sensors = np.array(Sensors)
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
    
    # Add this right after defining the x_range and y_range in the Plot function:
    X, Y = np.meshgrid(x_range, y_range)
    
    H = np.zeros(X.shape)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            H[i, j] = Height((X[i, j], Y[i, j]))[2]

    Z = np.zeros(X.shape)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = ProbDetect(Sensors, (X[i, j], Y[i, j]))

    # Create a heatmap using the detection probabilities
    cmap_heatmap = plt.cm.RdYlBu
    ax.imshow(Z, extent=[x_min-1, x_max+1, y_min-1, y_max+1], origin='lower', aspect='auto', cmap=cmap_heatmap, alpha=0.5)
    
    contours = ax.contour(X, Y, H, colors='black', linestyles='solid')
    ax.clabel(contours, inline=True, fontsize=8)

    
    # Add the heatmap colorbar
    cax_heatmap = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar_heatmap = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap_heatmap), cax=cax_heatmap)
    cbar_heatmap.ax.set_ylabel('Detection Probability')


    # Create a custom colorbar for the speed scale
    cax_speed = fig.add_axes([0.15, 0.1, 0.7, 0.03])  # Adjust the position and size of the speed colorbar
    cbar_speed = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap_speed), cax=cax_speed, orientation='horizontal')
    cbar_speed.set_ticks([0, 1])  # Set the colorbar ticks
    cbar_speed.set_ticklabels(['Low', 'High'])  # Set the tick labels
    cbar_speed.ax.xaxis.set_label_coords(0.5, -1)  # Adjust the position of the colorbar label
    cbar_speed.ax.set_xlabel('Speed')  # Set the colorbar label
    plt.tight_layout()

    return fig


# In[80]:


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


# In[81]:


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


# In[82]:


def TotalT_ThreshCalc(Start, End, Min_Speed = 1.5):
    Dist = np.linalg.norm(np.array(End) - np.array(Start))
    Time = Dist/Min_Speed
    return Time


# In[83]:


iter_Vectors = []
iter_Points = []
iter_T = []
iter_E = []
iter_Grads = []

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

            Vectors, Grads = Points2VelocityNGrad(Points)
        
        
            adjustT = 0.1
            magnitude_factorT = np.random.uniform(1-adjustT, 1+adjustT)
            T = T*magnitude_factorT
            
            TotalT_Curr = T*len(Vectors)
        
        iter_Vectors.append(Vectors)
        iter_Points.append(Points)
        iter_T.append(T)
        iter_Grads.append(Grads)

        self.state = Vectors, Points, T
        
    
    def energy(self):
        Vectors = self.state[0]
        Points = self.state[1]
        T = self.state[2]
        
        P = 0
        for Point in Points:
            P += ProbDetect(sensors, Point)
        
        S = 0
        for Vector in range(len(Vectors)):
            Grad = iter_Grads[-1][Vector]
            EXP = lambda3+(2/np.pi)*np.arctan(Grad)
            S += (np.linalg.norm(Vectors[Vector]))**EXP/T

        TotalT = T*len(Vectors)
        
        E = float(lambda0*P*T + lambda1*S + lambda2*TotalT)
        
        iter_E.append(E)

        return E


    def random_state(self):
        points = StartSolPoints(start, end)
        vectors, grads = Points2VelocityNGrad(points)
        iter_Grads.append(grads)
        T = TotalT_ThreshCalc(start, end, Min_Speed = 2.5)/len(vectors)
        return vectors, points, T


# In[84]:


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


# In[85]:


def Visialise(Iter_Points, Iter_Speeds, Iter_E, slicing = 5000):
    length = len(iter_Points)
    points = iter_Points[length-1]
    speeds = iter_Speeds[length-1]
    #iterations(iter_Points, iter_Speeds, slicing)
    a = Plot(start, end, points, sensors, speeds)
    b = PlotE(Iter_E)


# In[86]:


def Height(Position):
    x = Position[0]
    y = Position[1]
    h = np.exp(-((x-2.5)**2+(y-2)**2)*2)*2+np.exp(-((x+1.5)**2+(y+1)**2)*2)*2
    #np.sin(5*x)*(np.cos(5*y)/5)
    return [x, y, h]


# In[87]:


#StartEnd
start = [ -3, -1]
end = [ -0.2, 0.2]

#Sensors
sensors = [[-1.8, -0.8], [-1.5, -0.8]]
initial_temp = 25000
lambda0 = 10000
lambda1 = 100
lambda2 = 100
lambda3 = 2
TotalT_Thresh = TotalT_ThreshCalc(start, end, Min_Speed = 0.1)


# In[88]:


best_state, best_energy, iter_Vectors, iter_Points, iter_T, iter_Speeds = RunAnnealer()


# In[89]:


Visialise(iter_Points, iter_Speeds, iter_E, slicing = 5000)


# In[ ]:




