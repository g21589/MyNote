# -*- coding: utf-8 -*-

import plotly.graph_objects as go
import numpy as np

N = 10     # Number of boxes

# generate an array of rainbow colors by fixing the saturation and lightness of the HSL
# representation of colour and marching around the hue.
# Plotly accepts any CSS color format, see e.g. http://www.w3schools.com/cssref/css_colors_legal.asp.
c = ['hsl('+str(h)+',50%'+',50%)' for h in np.linspace(0, 360, N * 3)]

# Each box is represented by a dict that contains the data, the type, and the colour.
# Use list comprehension to describe N boxes, each with a different colour and with different randomly generated data:
fig = go.Figure(
    data=[
        go.Box(
            q1=[1,2,0.5] * N, 
            median=[2,2,2] * N, 
            q3=[4,4,4] * N,
            upperfence=[6,7,8] * N,
            lowerfence=[0,-2,-1] * N,
            x=list(map(lambda x: 'A' + str(x), range(3 * N))),
            marker_color='rgb(50,180,40)'
        ),
        go.Box(
            q1=[1,2,0.8] * N, 
            median=[2,2,2] * N, 
            q3=[4,4,4] * N,
            upperfence=[6,7,8] * N,
            lowerfence=[0,-2,-1] * N,
            x=list(map(lambda x: 'B' + str(x), range(3 * N))),
            marker_color='rgb(80,120,40)'
        ),
        go.Box(
            q1=[1,2,0.8] * N, 
            median=[2,2,2] * N, 
            q3=[4,4,4] * N,
            upperfence=[6,7,8] * N,
            lowerfence=[0,-2,-1] * N,
            x=list(map(lambda x: 'C' + str(x), range(3 * N))),
            marker_color='rgb(190,120,180)'
        ),
    ]
)

# format the layout
fig.update_layout(
    autosize=False,
    width=1200,
    height=300,
    margin=dict(l=0, r=0, t=0, b=20),
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=True),
    yaxis=dict(showgrid=False, zeroline=False, gridcolor='gray'),
    paper_bgcolor='rgb(255,255,255)',
    plot_bgcolor='rgb(255,255,255)',
    
)

#fig.show()
fig.write_image("fig1.png")
#fig.write_image("fig1.svg")
