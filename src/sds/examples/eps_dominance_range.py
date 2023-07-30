import numpy as np
import plotly.io as pio
import plotly.graph_objects as go
from numpy.linalg import norm
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from math import cos, sin, acos

from src.utils.plot import plotly_colors

pio.renderers.default = "browser"

if __name__ == '__main__':
    # %%
    a = -np.array([0.5, 0.5])

    fx = np.array([1, 1]) * 5

    eps = -0.2
    valid_fys = []

    fy = np.array([1, 1]) * 4
    n = norm(fy - fx)


    def add_quiver(x, y, fig, color_ix=0, name=None, showlegend=False, line_width=4):
        p_quiver_fig = ff.create_quiver([x[0]], [x[1]], [(y - x)[0]], [(y - x)[1]],
                                        scale=1, arrow_scale=.1,
                                        marker=dict(color=plotly_colors[color_ix]),
                                        name=name, showlegend=showlegend,
                                        line=dict(width=line_width))
        fig.add_traces(data=p_quiver_fig.data)


    fig = make_subplots(rows=1, cols=1)
    fig.append_trace(
        go.Scatter(
            x=[fx[0]],
            y=[fx[1]],
            marker=dict(color=plotly_colors[0])),
        row=1, col=1)

    fig.append_trace(
        go.Scatter(
            x=[fy[0]],
            y=[fy[1]],
            marker=dict(color=plotly_colors[0])),
        row=1, col=1)

    step = 5
    for deg in range(0, 270 + step, step):
        theta = np.deg2rad(deg)
        rot = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])
        d = np.dot(rot, np.array([0, 1])) * n
        fy2 = fx + d
        angle = np.rad2deg(acos(np.dot(fy2 - fx, a) / (norm(fy2 - fx) * norm(a))))
        # ix = 0 if np.all((fy2 - fx) * a > -eps) else 1
        ix = 0 if angle < 135 else 1
        add_quiver(fx, fy2, fig, color_ix=ix, line_width=3 if ix == 0 else 1)

    ref = fx + a / norm(a) * n
    add_quiver(fx, ref, fig, color_ix=3, name="a", showlegend=True, line_width=5)
    fig.show()
