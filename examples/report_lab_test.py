# -*- coding: utf-8 -*-

from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import *
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.graphics.shapes import Drawing, Rect, Circle
import pandas as pd
import random

PATH_OUT = "./"

elements = []
styles = getSampleStyleSheet()

elements.append(Paragraph("Report Title", styles['Title']))

data = [[random.random() for i in range(1,4)] for j in range (1,8)]
df = pd.DataFrame(data)
lista = [df.columns[:,].values.astype(str).tolist()] + df.values.tolist()

ts = [
    ('ALIGN', (1,1), (-1,-1), 'CENTER'),
    ('LINEABOVE', (0,0), (-1,0), 1, colors.purple),
    ('LINEBELOW', (0,0), (-1,0), 1, colors.purple),
    ('FONT', (0,0), (-1,0), 'Times-Bold'),
    ('LINEABOVE', (0,-1), (-1,-1), 1, colors.purple),
    ('LINEBELOW', (0,-1), (-1,-1), 0.5, colors.purple, 1, None, None, 4,1),
    ('LINEBELOW', (0,-1), (-1,-1), 1, colors.red),
    ('FONT', (0,-1), (-1,-1), 'Times-Bold'),
    ('BACKGROUND',(1,1),(-2,-2),colors.green),
    ('TEXTCOLOR',(0,0),(1,-1),colors.red)
]

table = Table(lista, style=ts)
elements.append(table)

d = Drawing(20, 20)
d.add(Circle(10, 10, 5, strokeColor="#FF0000", fillColor='#FF0000'))
d.background = Rect(0, 0, 20, 20, strokeWidth=0.25, strokeColor="#868686", fillColor=None)

table2 = Table([
    ['AAA', d] 
]* 10, style=[
    ('INNERGRID', (0, 0), (-1, -1), 0.25, colors.black),
    ('BOX', (0, 0), (-1, -1), 0.25, colors.black)    
])

elements.append(table2)

doc = SimpleDocTemplate(PATH_OUT + 'Report_File.pdf')
doc.build(elements)
