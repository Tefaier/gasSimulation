from sympy import *

# calculate distance
# || x1 + v1 * (t - t1) - x2 - v2 * (t - t2) || = r1 + r2
x1 = Symbol('x1')
y1 = Symbol('y1')
z1 = Symbol('z1')
x2 = Symbol('x2')
y2 = Symbol('y2')
z2 = Symbol('z2')
vx1 = Symbol('vx1')
vy1 = Symbol('vy1')
vz1 = Symbol('vz1')
vx2 = Symbol('vx2')
vy2 = Symbol('vy2')
vz2 = Symbol('vz2')
r1 = Symbol('r1')
r2 = Symbol('r2')
t1 = Symbol('t1')
t2 = Symbol('t2')
t = Symbol('t')

diffx = Symbol('diffx')
diffy = Symbol('diffy')
diffz = Symbol('diffz')

eq_main = diffx**2 - (r1 + r2)**2
substX = x1 + vx1 * (t - t1) - x2 - vx2 * (t - t2)
substY = y1 + vy1 * (t - t1) - y2 - vy2 * (t - t2)
substZ = z1 + vz1 * (t - t1) - z2 - vz2 * (t - t2)

final_eq = eq_main.subs(diffx, substX)
final_eq = collect(expand(final_eq), t)
print(final_eq)
new_eq = final_eq - t * (2 * (vx2-vx1) * (x2-x1+t1*vx1-t2*vx2))
new_eq = collect(expand(new_eq), t)
print(new_eq)
