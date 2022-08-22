

Task = input('Введите тип задачи (1 - боковая задача; 2 - торцевая задача): ')
R = float(input('Radius of the tank, m: '))
H = float(input('Height of the tank, m: '))
a = float(input('Distance from the tank axis to the reference point, m: '))
if Task==1:
    h = float(input('Height level of the reference point, m: '))
d = float(input('Protection thickness: '))
E = input('Radiation energy, MeV: ')
p = a/R
#mus - for water
#musR = mus*R*100
k = H/R
k1 = (H-h)/R
k2 = h/R
#mu - for protection
#b = mu*d
