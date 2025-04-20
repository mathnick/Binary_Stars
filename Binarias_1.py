import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# =============================================
# PARÂMETROS FÍSICOS
# =============================================
N1 = 100  # Partículas estrela primária
N2 = 100  # Partículas estrela secundária
M1, M2 = 6.0, 3.0  # Massas (em unidades solares)
R1, R2 = 1.0, 1.0  # Raios iniciais
separacao_inicial = 5.0  # Distância entre estrelas
h = 1.0  # Suavização do kernel SPH
dt = 0.05  # Passo de tempo (reduzido para estabilidade)
k = 0.5  # Constante de pressão (aumentada)
gamma = 5 / 3  # Índice adiabático para gás ideal

# Constantes físicas
G = 1.0  # Simplificado para unidades normalizadas


# =============================================
# FUNÇÕES AUXILIARES
# =============================================
def gerar_particulas(N, R, centro):
    """Gera partículas em uma esfera com distribuição uniforme."""
    theta = np.random.uniform(0, np.pi, N)
    phi = np.random.uniform(0, 2 * np.pi, N)
    r = R * np.random.uniform(0, 1, N) ** (1 / 3)
    x = centro[0] + r * np.sin(theta) * np.cos(phi)
    y = centro[1] + r * np.sin(theta) * np.sin(phi)
    z = centro[2] + r * np.cos(theta)
    return x, y, z


def kernel(r, h):
    """Kernel SPH Monaghan-Lattanzio padrão."""
    q = r / h
    sigma = 1 / np.pi  # Normalização para 3D
    W = np.zeros_like(q)
    mask = (q <= 2.0)
    W[mask] = sigma / h ** 3 * (1 - 1.5 * q[mask] ** 2 + 0.75 * q[mask] ** 3)
    return W


def grad_kernel(r, h):
    """Gradiente do kernel SPH."""
    q = r / h
    sigma = 1 / np.pi
    grad_W = np.zeros_like(q)
    mask = (q <= 2.0)
    grad_W[mask] = sigma / h ** 4 * (-3 * q[mask] + 2.25 * q[mask] ** 2)
    return grad_W


# =============================================
# INICIALIZAÇÃO
# =============================================
# Posições e massas iniciais
x1, y1, z1 = gerar_particulas(N1, R1, [-separacao_inicial / 2, 0, 0])
x2, y2, z2 = gerar_particulas(N2, R2, [separacao_inicial / 2, 0, 0])

# Concatenar todas as partículas
x = np.concatenate([x1, x2])
y = np.concatenate([y1, y2])
z = np.concatenate([z1, z2])
M = np.concatenate([np.ones(N1) * M1 / N1, np.ones(N2) * M2 / N2])  # Massas normalizadas

# Velocidades iniciais (incluindo orbital)
vx, vy, vz = np.zeros_like(x), np.zeros_like(y), np.zeros_like(z)
v_orbital = np.sqrt(G * (M1 + M2) / separacao_inicial)
vx[N1:] = -v_orbital * np.sign(y[N1:])  # Secundária orbita no plano x-y
vy[N1:] = v_orbital * np.sign(x[N1:])

# =============================================
# FUNÇÕES DE CÁLCULO
# =============================================


def calcular_densidade(x, y, z, M, h):
    """Calcula a densidade via SPH."""
    n = len(x)
    rho = np.zeros(n)
    for i in range(n):
        dx = x[i] - x
        dy = y[i] - y
        dz = z[i] - z
        r = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        rho[i] = np.sum(M * kernel(r, h))
    return rho


def calcular_forcas(x, y, z, M, h, rho):
    """Calcula forças gravitacionais e de pressão."""
    n = len(x)
    Fx, Fy, Fz = np.zeros(n), np.zeros(n), np.zeros(n)
    pressao = k * rho ** gamma  # Equação de estado adiabática

    for i in range(n):
        dx = x[i] - x
        dy = y[i] - y
        dz = z[i] - z
        r = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        r[r < 1e-5] = 1e-5  # Evitar divisão por zero

        # Gravidade (suavizada)
        F_grav = -G * M[i] * M / (r ** 2 + 0.1 * h ** 2) ** 1.5  # Suavização de Plummer
        Fx[i] += np.sum(F_grav * dx)
        Fy[i] += np.sum(F_grav * dy)
        Fz[i] += np.sum(F_grav * dz)

        # Pressão SPH (corrigida)
        grad_W = grad_kernel(r, h)
        F_press = -M * (pressao[i] / rho[i] ** 2 + pressao / rho ** 2) * grad_W  # Força simétrica
        Fx[i] += np.sum(F_press * dx / r)
        Fy[i] += np.sum(F_press * dy / r)
        Fz[i] += np.sum(F_press * dz / r)

    return Fx, Fy, Fz


# =============================================
# ANIMAÇÃO
# =============================================
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-8, 8)
ax.set_ylim(-8, 8)
ax.set_zlim(-8, 8)
particulas, = ax.plot([], [], [], 'bo', ms=3)


def init():
    particulas.set_data([], [])
    particulas.set_3d_properties([])
    return particulas,


def update(frame):
    global x, y, z, vx, vy, vz

    # Calcular densidade e forças
    rho = calcular_densidade(x, y, z, M, h)
    Fx, Fy, Fz = calcular_forcas(x, y, z, M, h, rho)

    # Integração (Euler semi-implícito para estabilidade)
    vx += Fx / M * dt
    vy += Fy / M * dt
    vz += Fz / M * dt
    x += vx * dt
    y += vy * dt
    z += vz * dt

    # Atualizar plot
    particulas.set_data(x, y)
    particulas.set_3d_properties(z)
    ax.set_title(f"Frame: {frame}, Tempo: {frame * dt:.2f}")

    return particulas,


# Configuração da animação
ani = FuncAnimation(fig, update, frames=200, init_func=init, blit=True, interval=50)
plt.show()
