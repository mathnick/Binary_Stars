import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import cKDTree

# =============================================
# PARÂMETROS FÍSICOS E DE SIMULAÇÃO
# =============================================
N1 = 1000
N2 = 100
M1, M2 = 300.0, 30.0
R1, R2 = 2.15, 1.0
separacao_inicial = 6.0
h = 1.5
dt = 0.01
T_MAX = 3.0
k = 2.0
gamma = 5/3
G = 1.0

# Parâmetro para o deslocamento vertical
deslocamento_z = 1.5  # Ajuste este valor para controlar a diferença de altura

N_total = N1 + N2
m_particula = (M1 + M2) / N_total

# =============================================
# KERNEL DE MONAGHAN (1984) - CORRIGIDO
# =============================================
def kernel_monaghan(r, h):
    """
    Kernel cúbico de Monaghan (1984) com erro O(h^3)
    W(r,h) = (3/(4πh³)) × [ (2-v)² × (5-4v)/3 ] para 1 ≤ v ≤ 2
    Onde v = r/h
    """
    v = r / h
    sigma = 3.0 / (4.0 * np.pi * h**3)  # Fator de normalização
    
    W = np.zeros_like(v)
    
    # Regime 1: 0 ≤ v ≤ 1
    mask1 = (v >= 0) & (v <= 1)
    W[mask1] = (10/3 - 7*v[mask1]**2 + 4*v[mask1]**3)
    
    # Regime 2: 1 < v ≤ 2  
    mask2 = (v > 1) & (v <= 2)
    W[mask2] = (2.0 - v[mask2])**2 * (5.0 - 4.0 * v[mask2]) / 3.0
    
    # Aplica normalização
    W = W * sigma
    
    return W

def grad_kernel_monaghan(r, h):
    """
    Gradiente do kernel cúbico de Monaghan: dW/dr
    """
    v = r / h
    sigma = 3.0 / (4.0 * np.pi * h**3)
    
    grad_W = np.zeros_like(v)
    
    # Regime 1: 0 ≤ v ≤ 1
    mask1 = (v >= 0) & (v <= 1)
    # Derivada de (10/3 - 7v² + 4v³) é -14v + 12v²
    grad_W[mask1] = (-14 * v[mask1] + 12 * v[mask1]**2)
    
    # Regime 2: 1 < v ≤ 2
    mask2 = (v > 1) & (v <= 2)
    # Derivada de [(2-v)²(5-4v)/3]
    # = -2(2-v)(3-2v)
    grad_W[mask2] = -2.0 * (2.0 - v[mask2]) * (3.0 - 2.0 * v[mask2])
    
    # Aplica normalização e regra da cadeia: dW/dr = (dW/dv) × (1/h)
    grad_W = grad_W * sigma * (1.0 / h)
    
    return grad_W

# =============================================
# FUNÇÕES AUXILIARES
# =============================================
def gerar_particulas(N, R, centro):
    theta = np.random.uniform(0, np.pi, N)
    phi = np.random.uniform(0, 2*np.pi, N)
    r = R * np.random.uniform(0, 1, N)**(1/3)
    
    x = centro[0] + r * np.sin(theta) * np.cos(phi)
    y = centro[1] + r * np.sin(theta) * np.sin(phi)
    z = centro[2] + r * np.cos(theta)
    
    return x, y, z

def calcular_cm(x, y, z, M, indices):
    if np.sum(M[indices]) == 0:
        return np.array([x[indices[0]], y[indices[0]], z[indices[0]]]) if len(indices) > 0 else np.array([0, 0, 0])
        
    x_cm = np.sum(x[indices] * M[indices]) / np.sum(M[indices])
    y_cm = np.sum(y[indices] * M[indices]) / np.sum(M[indices])
    z_cm = np.sum(z[indices] * M[indices]) / np.sum(M[indices])
    return np.array([x_cm, y_cm, z_cm])

def definir_afiliacao(x, y, z, M, N1_original):
    indices_1_init = np.arange(N1_original)
    indices_2_init = np.arange(N1_original, len(x))
    
    CM1 = calcular_cm(x, y, z, M, indices_1_init)
    CM2 = calcular_cm(x, y, z, M, indices_2_init)
    
    r1 = np.sqrt((x - CM1[0])**2 + (y - CM1[1])**2 + (z - CM1[2])**2)
    r2 = np.sqrt((x - CM2[0])**2 + (y - CM2[1])**2 + (z - CM2[2])**2)
    
    afiliacao = np.where(r1 < r2, 1, 2)
    return afiliacao, CM1, CM2

# =============================================
# FUNÇÕES SPH COM NOVO KERNEL
# =============================================
def calcular_densidade_otimizado(x, y, z, M, h):
    """Calcula densidade usando o kernel de Monaghan"""
    n = len(x)
    rho = np.zeros(n)
    
    points = np.column_stack([x, y, z])
    tree = cKDTree(points)
    indices = tree.query_ball_tree(tree, 2*h)
    
    for i in range(n):
        if len(indices[i]) > 0:
            dx = x[i] - x[indices[i]]
            dy = y[i] - y[indices[i]]
            dz = z[i] - z[indices[i]]
            r = np.sqrt(dx**2 + dy**2 + dz**2)
            rho[i] = np.sum(M[indices[i]] * kernel_monaghan(r, h))
        else:
            rho[i] = M[i] * kernel_monaghan(0, h)
    
    return rho

def calcular_forcas_otimizado(x, y, z, M, h, rho):
    """Calcula forças usando o gradiente do kernel de Monaghan"""
    n = len(x)
    Fx, Fy, Fz = np.zeros(n), np.zeros(n), np.zeros(n)
    pressao = k * rho**gamma
    
    points = np.column_stack([x, y, z])
    tree = cKDTree(points)
    indices = tree.query_ball_tree(tree, 2*h)
    
    for i in range(n):
        if len(indices[i]) == 0:
            continue
            
        j_list = indices[i]
        j_list = [j for j in j_list if j != i]
        
        if not j_list:
            continue
            
        dx = x[i] - x[j_list]
        dy = y[i] - y[j_list]
        dz = z[i] - z[j_list]
        r = np.sqrt(dx**2 + dy**2 + dz**2)
        r[r < 1e-5] = 1e-5
        
        # Força gravitacional
        F_grav = -G * M[i] * M[j_list] / (r**2 + 0.1 * h**2)**1.5
        Fx[i] += np.sum(F_grav * dx)
        Fy[i] += np.sum(F_grav * dy)
        Fz[i] += np.sum(F_grav * dz)
        
        # Força de pressão com novo kernel
        grad_W = grad_kernel_monaghan(r, h)
        F_press = -M[j_list] * (pressao[i]/rho[i]**2 + pressao[j_list]/rho[j_list]**2) * grad_W
        
        Fx[i] += np.sum(F_press * dx / r)
        Fy[i] += np.sum(F_press * dy / r)
        Fz[i] += np.sum(F_press * dz / r)
    
    return Fx, Fy, Fz

# =============================================
# INTEGRAÇÃO RK4
# =============================================
def derivadas(X, t, M, h, k, gamma):
    n = len(M)
    x, y, z = X[0:n], X[n:2*n], X[2*n:3*n]
    vx, vy, vz = X[3*n:4*n], X[4*n:5*n], X[5*n:6*n]
    
    dx_dt, dy_dt, dz_dt = vx, vy, vz
    
    rho = calcular_densidade_otimizado(x, y, z, M, h)
    Fx, Fy, Fz = calcular_forcas_otimizado(x, y, z, M, h, rho)
    
    dvx_dt, dvy_dt, dvz_dt = Fx / M, Fy / M, Fz / M
    
    return np.concatenate([dx_dt, dy_dt, dz_dt, dvx_dt, dvy_dt, dvz_dt])

def rk4_step(X_in, t, dt, M, h, k, gamma):
    dX1 = derivadas(X_in, t, M, h, k, gamma)
    X2 = X_in + 0.5 * dt * dX1
    dX2 = derivadas(X2, t + 0.5 * dt, M, h, k, gamma)
    X3 = X_in + 0.5 * dt * dX2
    dX3 = derivadas(X3, t + 0.5 * dt, M, h, k, gamma)
    X4 = X_in + dt * dX3
    dX4 = derivadas(X4, t + dt, M, h, k, gamma)
    
    X_out = X_in + (dt / 6.0) * (dX1 + 2 * dX2 + 2 * dX3 + dX4)
    return X_out

# =============================================
# INICIALIZAÇÃO COM ALTURAS DIFERENTES
# =============================================

# Estrela primária: centro em [-separacao_inicial/2, 0, -deslocamento_z/2]
# Estrela secundária: centro em [separacao_inicial/2, 0, deslocamento_z/2]
x1, y1, z1 = gerar_particulas(N1, R1, [-separacao_inicial/2, 0, -deslocamento_z/2])
x2, y2, z2 = gerar_particulas(N2, R2, [separacao_inicial/2, 0, deslocamento_z/2])

x = np.concatenate([x1, x2])
y = np.concatenate([y1, y2])
z = np.concatenate([z1, z2])
M = np.full(N_total, m_particula)

# Velocidades orbitais ajustadas para órbitas inclinadas
v_orbital = np.sqrt(G * (M1 + M2) / separacao_inicial)
vx = np.zeros(N_total)
vy = np.zeros(N_total)
vz = np.zeros(N_total)

ratio = M2 / (M1 + M2)

# Aplica velocidades orbitais considerando a inclinação
# Componentes X e Y para criar órbita inclinada
inclinacao = np.radians(10)  # Pequena inclinação de 10 graus

vy[:N1] = -v_orbital * ratio * np.cos(inclinacao)  # Estrela primária
vz[:N1] = v_orbital * ratio * np.sin(inclinacao)   # Pequena componente Z

vy[N1:] = v_orbital * (1 - ratio) * np.cos(inclinacao)  # Estrela secundária  
vz[N1:] = -v_orbital * (1 - ratio) * np.sin(inclinacao) # Pequena componente Z

X = np.concatenate([x, y, z, vx, vy, vz])

# =============================================
# LOOP DE SIMULAÇÃO
# =============================================
tempo_historico = [0.0]
mdot_historico = [0.0]
historico_X = [X.copy()]

afiliacao_antiga, _, _ = definir_afiliacao(x, y, z, M, N1)
afiliacao_historico = [afiliacao_antiga.copy()]

n_steps = int(T_MAX / dt)
print(f"Iniciando simulação com estrelas em alturas diferentes...")
print(f"Deslocamento Z: {deslocamento_z}")
print(f"Inclinação orbital: {np.degrees(inclinacao):.1f}°")

for step in range(n_steps):
    t = (step + 1) * dt
    
    X = rk4_step(X, t, dt, M, h, k, gamma)
    
    x_novo = X[0:N_total]
    y_novo = X[N_total:2*N_total]
    z_novo = X[2*N_total:3*N_total]
    
    afiliacao_nova, CM1, CM2 = definir_afiliacao(x_novo, y_novo, z_novo, M, N1)
    
    transferencia_1_para_2 = np.sum((afiliacao_antiga == 1) & (afiliacao_nova == 2)) * m_particula
    transferencia_2_para_1 = np.sum((afiliacao_antiga == 2) & (afiliacao_nova == 1)) * m_particula
    delta_M_liquido = transferencia_1_para_2 - transferencia_2_para_1
    M_dot = delta_M_liquido / dt
    
    tempo_historico.append(t)
    mdot_historico.append(M_dot)
    historico_X.append(X.copy())
    afiliacao_historico.append(afiliacao_nova.copy())
    
    afiliacao_antiga = afiliacao_nova
    
    if step % 20 == 0:
        separacao = np.linalg.norm(CM1 - CM2)
        altura_relativa = CM1[2] - CM2[2]
        print(f"Tempo: {t:.2f} | M_dot: {M_dot:.6f} | Sep: {separacao:.2f} | ΔZ: {altura_relativa:.2f}")

# =============================================
# VISUALIZAÇÃO DOS RESULTADOS
# =============================================

# Gráfico da taxa de transferência de massa
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(tempo_historico, mdot_historico, 'b-', linewidth=2, label=r'$\dot{M}_{1 \to 2}$')
plt.axhline(0, color='red', linestyle='--', alpha=0.5)
plt.xlabel("Tempo")
plt.ylabel("Taxa de Transferência de Massa")
plt.title(f"Transferência de Massa\nDeslocamento Z = {deslocamento_z}")
plt.legend()
plt.grid(True)

# Gráfico da separação e altura relativa
plt.subplot(1, 2, 2)
separacoes = []
alturas = []
for i, current_X in enumerate(historico_X):
    x_f = current_X[0:N_total]
    y_f = current_X[N_total:2*N_total]
    z_f = current_X[2*N_total:3*N_total]
    afiliacao = afiliacao_historico[i]
    
    indices_1 = np.where(afiliacao == 1)[0]
    indices_2 = np.where(afiliacao == 2)[0]
    
    CM1 = calcular_cm(x_f, y_f, z_f, M, indices_1)
    CM2 = calcular_cm(x_f, y_f, z_f, M, indices_2)
    
    separacoes.append(np.linalg.norm(CM1 - CM2))
    alturas.append(CM1[2] - CM2[2])

plt.plot(tempo_historico, separacoes, 'g-', label='Separação CM')
plt.plot(tempo_historico, alturas, 'm-', label='ΔZ (CM1 - CM2)')
plt.xlabel("Tempo")
plt.ylabel("Distância")
plt.title("Separação e Altura Relativa")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# =============================================
# ANIMAÇÃO 3D
# =============================================
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

def init():
    ax.clear()
    # Ajusta os limites para acomodar o movimento 3D
    max_range = separacao_inicial * 1.2
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-max_range/2, max_range/2)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Sistema Binário com Estrelas em Diferentes Alturas')
    return []

def update(frame):
    ax.clear()
    current_X = historico_X[frame]
    current_afiliacao = afiliacao_historico[frame]
    
    x_f = current_X[0:N_total]
    y_f = current_X[N_total:2*N_total]
    z_f = current_X[2*N_total:3*N_total]
    
    # Cores diferentes para cada estrela
    cores = np.where(current_afiliacao == 1, 'red', 'blue')
    tamanhos = np.where(current_afiliacao == 1, 8, 6)
    
    # Plot das partículas
    ax.scatter(x_f, y_f, z_f, c=cores, s=tamanhos, alpha=0.6)
    
    # Plot dos centros de massa
    indices_1 = np.where(current_afiliacao == 1)[0]
    indices_2 = np.where(current_afiliacao == 2)[0]
    
    CM1 = calcular_cm(x_f, y_f, z_f, M, indices_1)
    CM2 = calcular_cm(x_f, y_f, z_f, M, indices_2)
    
    ax.scatter(CM1[0], CM1[1], CM1[2], c='darkred', s=100, marker='*', label='CM Estrela 1')
    ax.scatter(CM2[0], CM2[1], CM2[2], c='darkblue', s=100, marker='*', label='CM Estrela 2')
    
    # Linha conectando os CMs
    ax.plot([CM1[0], CM2[0]], [CM1[1], CM2[1]], [CM1[2], CM2[2]], 
            'k--', alpha=0.5, linewidth=1)
    
    # Configurações do gráfico
    max_range = separacao_inicial * 1.2
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-max_range/2, max_range/2)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    massa_1 = np.sum(current_afiliacao == 1) * m_particula
    massa_2 = np.sum(current_afiliacao == 2) * m_particula
    separacao = np.linalg.norm(CM1 - CM2)
    
    ax.set_title(f'Tempo: {frame * dt:.2f}\nM1: {massa_1:.1f} | M2: {massa_2:.1f} | Sep: {separacao:.2f}')
    ax.legend()
    
    return []

# Cria a animação
ani = FuncAnimation(fig, update, frames=min(200, len(historico_X)), 
                   init_func=init, blit=False, interval=50)

plt.tight_layout()
plt.show()

print(f"\nSimulação concluída!")
print(f"Configuração: Estrelas inicialmente separadas por {separacao_inicial} em X")
print(f"e com diferença de altura de {deslocamento_z} em Z")
