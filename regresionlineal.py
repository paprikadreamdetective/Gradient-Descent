import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation
# Open the file in read mode
list_values = []

with open('ex1data1.txt.', 'r') as file:
    # Read each line in the file
    for line in file:
        # Print each line
        #print(line.rstrip().split(','))
        list_values.append(line.rstrip().split(','))

df_list_products = pd.DataFrame(list_values, columns=['X', 'Y'])
#plt.scatter(df_list_products['X'])
x_i = df_list_products['X'].astype('float64')
y_i = df_list_products['Y'].astype('float64')
#plt.scatter(x_i, y_i)
rng = np.random.RandomState(0)
colors = rng.rand(97)
sizes = 1000 * rng.rand(97)
plt.scatter(x_i, y_i, c=colors,  alpha=0.4, cmap='viridis')
plt.colorbar()
plt.title('Ganancias en función de la población de una ciudad')
plt.xlabel('Poblacion por ciudad (en miles de habitantes)')
plt.ylabel('Ganacias generadas por ciudad (en miles)')
plt.show()

# Parámetros del gradiente descendente
data_matrix = df_list_products.to_numpy().astype('float64')
print(data_matrix)

alpha = 0.02  # Tasa de aprendizaje
iterations = 1500  # Número de iteraciones
Xvector = data_matrix[:, :1]
Ytags = data_matrix[:, 1][:, np.newaxis]
print('Vector X ')
print(Xvector)
print('Vector Y ')
print(Ytags)

# Añadimos una columna de unos a demanda_total para el término de sesgo (intercepto)
X = np.hstack((np.ones((Xvector.shape[0], 1)), Xvector))
#print(X)
y = Ytags

# Inicialización de theta (pendiente e intersección)
theta = np.array([[25], [-12]])
print("Vector Theta")

print(theta)

theta_values = []

for i in range(iterations):
    # Calcular la predicción
    predictions = np.dot(X, theta)
    
    # Calcular el error
    error = predictions - y
    
    # Calcular el gradiente
    gradient = np.dot(X.T, error)
    
    # Actualizar theta
    theta = theta - (alpha / len(y)) * gradient
    #theta_values.append(theta)
    theta_values.append(theta.copy().reshape(-1, 1))
    
# Imprimir los parámetros optimizados

print("Parámetros optimizados (theta):")
print("Intersección (theta0):", theta[0, 0])
print("Pendiente (theta1):", theta[1, 0])

print(theta_values[0])
print(theta_values[1])

y_pred = np.dot(X, theta)

# Graficar los puntos de datos y la línea de regresión
plt.scatter(Xvector, Ytags, color='blue', label='Datos reales')
plt.plot(Xvector, y_pred, color='red', label=f'Regresión lineal: y = {theta[0, 0]:.2f} + {theta[1, 0]:.2f}x')
plt.xlabel("Poblacion por ciudad (en miles de habitantes)")
plt.ylabel("Ganacias generadas por ciudad (en miles)")
plt.title("Ganancias en función de la población de una ciudad")
plt.legend()
plt.show()

# Función de costo
def compute_cost(X, y, theta):
    m = len(y)
    predictions = np.dot(X, theta)
    cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
    return cost

# Generar datos para la visualización
theta0_vals = np.linspace(-50, 50, 200)
theta1_vals = np.linspace(-15, 15, 200)
T0, T1 = np.meshgrid(theta0_vals, theta1_vals)
J_vals = np.zeros_like(T0)

for i in range(T0.shape[0]):
    for j in range(T0.shape[1]):
        t = np.array([[T0[i, j]], [T1[i, j]]])
        J_vals[i, j] = compute_cost(X, y, t)

# Configurar la animación
fig, ax = plt.subplots()
contour = ax.contour(T0, T1, J_vals, levels=np.logspace(-2, 3, 20), cmap='viridis')
ax.set_xlabel(r'$\theta_0$')
ax.set_ylabel(r'$\theta_1$')
ax.set_title('Curvas de nivel de la función de costo')

# Inicializar el punto de gradiente descendente
point, = ax.plot([], [], 'ro', markersize=5)
line, = ax.plot([], [], 'r--')

def init():
    point.set_data([], [])
    line.set_data([], [])
    return point, line

def update(frame):
    t_vals = np.array(theta_values[:frame+1])  # Convertir a una matriz
    if t_vals.ndim == 2:  # Si solo tiene un valor (1D), ajusta
        t_vals = t_vals[:, np.newaxis]
    point.set_data(t_vals[-1, 0], t_vals[-1, 1])
    line.set_data(t_vals[:, 0], t_vals[:, 1])
    return point, line

frames = 200
interval = 50
ani = FuncAnimation(fig, update, frames=len(theta_values), init_func=init, blit=True, interval=50)

# Mostrar la animación
plt.show()
#ani.save('gradient_descent_2d_line_v2.gif', writer='imagemagick')