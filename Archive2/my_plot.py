import numpy as np
import plotly.graph_objects as go


class Plotter():
    def __init__(self):
        self.fig = go.Figure()
        self.plotly_colors = [
                        'blue', 'green', 'red', 'purple', 'orange',
                        'yellow', 'cyan', 'magenta', 'black', 'pink',
                        'gray', 'lightblue', 'lightgreen', 'lightgray'
                    ]
        self.left_colors = self.plotly_colors
        
    def random_color(self):
        import random
        if len(self.left_colors) == 0:
            self.left_colors = self.plotly_colors

        color = random.choice(self.plotly_colors)
        self.left_colors.remove(color)
        # print(len(self.left_colors))

        return color

    def gen_ellipse_points(self, S, R, T):
  # generate sphere
        theta = np.linspace(0, 2*np.pi, 72)
        phi = np.linspace(-np.pi/2, np.pi/2, 36)

        # create angles sphere
        theta, phi = np.meshgrid(theta, phi)
        x = np.cos(theta) * np.cos(phi)
        y = np.sin(theta) * np.cos(phi)
        z = np.sin(phi)

        # create separate points
        x_, y_, z_ = np.zeros_like(np.array([x, y, z]))

        # transform to ellipse
        for i in range(len(x)):
            for j in range(len(x[i])):
                vec = np.array([x[i, j], y[i, j], z[i, j]])
                new_vec = vec @ S @ R + T

                x_[i, j], y_[i, j], z_[i, j] = new_vec

        return np.array([x_, y_, z_])
    
    def plot_ellipsoid(self, S, R, T, name='', alpha=1):

        ellipsoid_points = self.gen_ellipse_points(S, R, T)
        X_0, Y_0, Z_0 = T + S @ R

        colorscales = [
                        'Viridis', 'Cividis', 'Plasma', 'Inferno', 'Magma',
                        'Blues', 'Greens', 'Reds', 'Greys', 'YlGnBu',
                        'YlOrRd', 'Rainbow', 'Jet', 'Hot', 'Cool'
                        ]

        random_color = self.random_color()
        self.fig.add_trace(go.Surface(x=ellipsoid_points[0], y=ellipsoid_points[1], z=ellipsoid_points[2],
                                opacity=alpha,
                                showscale=False,
                                colorscale=[[0, random_color], [1,random_color]], 
                                # colorscale=random.choice(colorscales),
                                showlegend=True,
                                name=f'{name} Surface'))

        # self.fig.update_traces(showlegend=True, showscale=False)

        self.fig.add_trace(go.Scatter3d(x=[T[0], X_0[0]], y=[T[1], X_0[1]], z=[T[2], X_0[2]],
                                mode='lines',
                                line=dict(color='red', width=4),
                                name=f'{name} X',
                                showlegend=False)
                                )

        self.fig.add_trace(go.Scatter3d(x=[T[0], Y_0[0]], y=[T[1], Y_0[1]], z=[T[2], Y_0[2]],
                                mode='lines',
                                line=dict(color='green', width=4),
                                name=f'{name} Y',
                                showlegend=False),
                                )

        self.fig.add_trace(go.Scatter3d(x=[T[0], Z_0[0]], y=[T[1], Z_0[1]], z=[T[2], Z_0[2]],
                                mode='lines',
                                line=dict(color='blue', width=4),
                                name=f'{name} Z',
                                showlegend=False),
                                )

        self.fig.add_trace(go.Scatter3d(x=[T[0], 0], y=[T[1], 0], z=[T[2], 0],
                                mode='lines',
                                line=dict(color='grey', 
                                          width=4, 
                                          dash='longdash'),
                                name=f'{name} T'))

        
        self.plot_points(points=T, name=f'{name} center')
        
    def plot_points(self, points, name=None, color=None, alpha=1, size=4):
        if color == None:
            color = self.random_color()

        if points.shape == (3,):
            points = points.reshape((3, 1))

        self.fig.add_trace(go.Scatter3d(x=points[0], y=points[1], z=points[2],
                                mode='markers',
                                marker=dict(color=color, 
                                            size=size,
                                            opacity=alpha),
                                name=name))


    def show(self):
    
        self.fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[0],
                                mode='markers',
                                marker=dict(color='grey', size=4),
                                name=f'zero'))
        self.fig.show()