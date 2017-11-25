import math


# 求出二维两点之间距离
def point_distance(x1, y1, x2, y2):
    d = math.sqrt(math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2))
    return d


d_ag = point_distance(3, 104, 18, 90)
d_bg = point_distance(2, 100, 18, 90)
d_cg = point_distance(1, 81, 18, 90)
d_dg = point_distance(101, 10, 18, 90)
d_eg = point_distance(99, 5, 18, 90)
d_fg = point_distance(98, 2, 18, 90)

print(d_ag)
print(d_bg)
print(d_cg)
print(d_dg)
print(d_eg)
print(d_fg)
