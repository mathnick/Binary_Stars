import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# =============================================
# PARÂMETROS FÍSICOS E DE SIMULAÇÃO
# =============================================
N1 = 2000  # Partículas estrela primária
N2 = 200  # Partículas estrela secundária
M1, M2 = 300.0, 30.0  # Massas totais (em unidades normalizadas)
R1, R2 = 2.15, 1.0  # Raios iniciais
separacao_inicial = 3.0  # Distância entre estrelas
h = 1.5  # Suavização do kernel SPH
dt = 0.025  # Passo de tempo
T_MAX = 5.0  # Tempo total de simulação
k = 0.5  # Constante de pressão
gamma = 5 / 3  # Índice adiabático
G = 1.0  # Constante gravitacional

N_total = N1 + N2
m_particula = (M1 + M2) / (N1 + N2)  # Massa de uma partícula 

# =============================================
# FUNÇÕES AUXILIARES SPH
# =============================================
def gerar_particulas(N, R, centro):
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

# def kernel(r, h):
#     """
#     Kernel SPH Monaghan (Cubic Spline), Eq. (22) em 3D.
#     v = r/h.
#     """
#     # 1. Distância relativa normalizada (v)
#     v = r / h

#     # 2. Fator de normalização constante (3 / (4 * pi * h^3))
#     sigma_prefactor = 3.0 / (4.0 * np.pi * h**3)

#     # 3. Inicializa o array de resultados com zeros (atende ao regime v > 2)
#     W = np.zeros_like(v) 

#     # --- Regime 1: 0 <= v <= 1 ---
#     # Cria a máscara para o primeiro regime de distância
#     mask1 = (v <= 1.0)
    
#     # Aplica a fórmula do regime 1 apenas onde a máscara é True
#     # Fórmula interna: (10/3 - 7v^2 + 4v^3)
#     W[mask1] = (10/3 - 7 * v[mask1]**2 + 4 * v[mask1]**3)

#     # --- Regime 2: 1 < v <= 2 ---
#     # Cria a máscara para o segundo regime de distância
#     # É um 'e' lógico (AND) entre v > 1 e v <= 2
#     mask2 = (v > 1.0) & (v <= 2.0)

#     # Aplica a fórmula do regime 2 apenas onde a máscara é True
#     # Fórmula interna: (2-v)^2 * (5-4v)/3
#     W[mask2] = (2.0 - v[mask2])**2 * (5.0 - 4.0 * v[mask2]) / 3.0
    
#     # 4. Multiplica pelo fator de normalização
#     W = W * sigma_prefactor
    
#     # 5. Retorna o kernel
#     return W

def kernel(r, h, dim=3):
    q = r / h
    sigma = {1: 2 / 3, 2: 10 / (7 * np.pi), 3: 1 / np.pi}[dim]  # Constante de normalização

    W = np.zeros_like(q)
    mask1 = (q >= 0) & (q <= 1)
    mask2 = (q > 1) & (q <= 2)

    W[mask1] = sigma / (h ** dim) * (1 - 1.5 * q[mask1] ** 2 + 0.75 * q[mask1] ** 3)
    W[mask2] = sigma / (h ** dim) * 0.25 * (2 - q[mask2]) ** 3

    return W


# def grad_kernel(r, h):
#     """Gradiente do kernel SPH."""
#     q = r / h
#     sigma = 1 / np.pi
#     grad_W = np.zeros_like(q)
#     mask = (q <= 2.0)
#     grad_W[mask] = sigma / h ** 4 * (-3 * q[mask] + 2.25 * q[mask] ** 2)
#     return grad_W

# def grad_kernel(r, h):
#     """
#     Gradiente do Kernel SPH Monaghan, dW/dr.
#     Retorna o valor escalar do gradiente |dW/dr|.
#     """
#     # 1. Distância relativa normalizada (v)
#     v = r / h

#     # 2. Constante de Normalização para a DERIVADA (3 / (4 * pi * h^3))
#     sigma_prefactor = 3.0 / (4.0 * np.pi * h**3)

#     # 3. Inicializa o array de resultados com zeros (atende ao regime v > 2)
#     dW_dv = np.zeros_like(v) 

#     # --- Regime 1: 0 <= v <= 1 ---
#     mask1 = (v <= 1.0)
    
#     # Derivada da fórmula interna: (-14v + 12v^2)
#     dW_dv[mask1] = (-14 * v[mask1] + 12 * v[mask1]**2)

#     # --- Regime 2: 1 < v <= 2 ---
#     mask2 = (v > 1.0) & (v <= 2.0)

#     # Derivada da fórmula interna (simplificada): -4(2-v)(5-4v) + (2-v)^2(-4)
#     # Forma mais comum e simplificada:
#     dW_dv[mask2] = - (2.0 - v[mask2]) * (7.0 - 5.0 * v[mask2])

#     # 4. Multiplica pelo fator de normalização e pelo 1/h (Regra da Cadeia)
#     # dW/dr = (dW/dv) * (1/h)
#     dW_dr = dW_dv * sigma_prefactor * (1.0 / h)

#     # 5. O gradiente retornado é o valor escalar do dW/dr
#     return dW_dr

def grad_kernel(r, h, dim=3):
    q = r / h
    sigma = {1: 2 / 3, 2: 10 / (7 * np.pi), 3: 1 / np.pi}[dim]  # Constante de normalização

    grad_W = np.zeros_like(q)
    mask1 = (q >= 0) & (q <= 1)
    mask2 = (q > 1) & (q <= 2)

    grad_W[mask1] = sigma / (h ** (dim + 1)) * (-3 * q[mask1] + 2.25 * q[mask1] ** 2)
    grad_W[mask2] = sigma / (h ** (dim + 1)) * (-0.75 * (2 - q[mask2]) ** 2)

    return grad_W


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
    """Calcula forças gravitacionais e de pressão (usada na função de derivadas)."""
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
# CÁLCULO DE CM E AFILIAÇÃO (M_DOT)
# =============================================

def calcular_cm(x, y, z, M, indices):
    if np.sum(M[indices]) == 0:
         # Fallback se a massa do subconjunto for zero (evita ZeroDivisionError)
        return np.array([x[indices[0]], y[indices[0]], z[indices[0]]]) if len(indices) > 0 else np.array([0, 0, 0])
        
    x_cm = np.sum(x[indices] * M[indices]) / np.sum(M[indices])
    y_cm = np.sum(y[indices] * M[indices]) / np.sum(M[indices])
    z_cm = np.sum(z[indices] * M[indices]) / np.sum(M[indices])
    return np.array([x_cm, y_cm, z_cm])

def definir_afiliacao(x, y, z, M, N1_original):
    # Índices iniciais: usados para determinar quem é o "CM1" e o "CM2" de referência
    indices_1_init = np.arange(N1_original)
    indices_2_init = np.arange(N1_original, len(x))
    
    # CM baseado nas posições atuais das partículas originais
    CM1 = calcular_cm(x, y, z, M, indices_1_init)
    CM2 = calcular_cm(x, y, z, M, indices_2_init)
    
    # Distância de cada partícula ao CM1 e CM2
    r1 = np.sqrt((x - CM1[0])**2 + (y - CM1[1])**2 + (z - CM1[2])**2)
    r2 = np.sqrt((x - CM2[0])**2 + (y - CM2[1])**2 + (z - CM2[2])**2)
    
    # Atribui afiliação: 1 se mais próximo do CM1, 2 se mais próximo do CM2
    afiliacao = np.where(r1 < r2, 1, 2)
    return afiliacao, CM1, CM2

# =============================================
# FUNÇÕES RK4
# =============================================

def derivadas(X, t, M, h, k, gamma):
    """
    Calcula as derivadas de posição (velocidade) e velocidade (aceleração).
    X é o vetor de estado: [x, y, z, vx, vy, vz]
    """
    n = len(M)
    x, y, z = X[0:n], X[n:2*n], X[2*n:3*n]
    
    # Derivadas de Posição (Velocidades)
    vx, vy, vz = X[3*n:4*n], X[4*n:5*n], X[5*n:6*n]
    dx_dt, dy_dt, dz_dt = vx, vy, vz
    
    # Derivadas de Velocidade (Acelerações = Forças / Massa)
    rho = calcular_densidade(x, y, z, M, h)
    Fx, Fy, Fz = calcular_forcas(x, y, z, M, h, rho)
    
    dvx_dt, dvy_dt, dvz_dt = Fx / M, Fy / M, Fz / M
    
    return np.concatenate([dx_dt, dy_dt, dz_dt, dvx_dt, dvy_dt, dvz_dt])

def rk4_step(X_in, t, dt, M, h, k, gamma):
    """Um passo de Runge-Kutta de 4ª Ordem."""
    
    # k1
    dX1 = derivadas(X_in, t, M, h, k, gamma)
    
    # k2
    X2 = X_in + 0.5 * dt * dX1
    dX2 = derivadas(X2, t + 0.5 * dt, M, h, k, gamma)
    
    # k3
    X3 = X_in + 0.5 * dt * dX2
    dX3 = derivadas(X3, t + 0.5 * dt, M, h, k, gamma)
    
    # k4
    X4 = X_in + dt * dX3
    dX4 = derivadas(X4, t + dt, M, h, k, gamma)
    
    # Combinação
    X_out = X_in + (dt / 6.0) * (dX1 + 2 * dX2 + 2 * dX3 + dX4)
    return X_out

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
# Aplicar velocidade orbital de Kepler simples (estimativa)
vx[N1:] = 0 # Inicialmente zero
vy[N1:] = v_orbital / 2 # Dividido por 2 pois o CM não está na origem
vx[:N1] = 0
vy[:N1] = -v_orbital / 2

# Vetor de estado inicial: [x, y, z, vx, vy, vz]
X = np.concatenate([x, y, z, vx, vy, vz])

# =============================================
# LOOP DE SIMULAÇÃO E CÁLCULO DE M_DOT
# =============================================

# Variáveis para registro histórico
tempo_historico = [0.0]
mdot_historico = [0.0]
historico_X = [X.copy()] # Armazenar o estado completo para a animação

# Determinar a afiliação inicial
afiliacao_antiga, _, _ = definir_afiliacao(x, y, z, M, N1)
afiliacao_historico = [afiliacao_antiga.copy()]

n_steps = int(T_MAX / dt)
print(f"Iniciando simulação de {n_steps} passos...")

for step in range(n_steps):
    t = (step + 1) * dt
    
    # 1. Passo RK4
    X = rk4_step(X, t, dt, M, h, k, gamma)
    
    # 2. Desempacotar o novo estado (posições)
    x_novo, y_novo, z_novo = X[0:N_total], X[N_total:2*N_total], X[2*N_total:3*N_total]
    
    # 3. Determinar afiliação após o passo (ID_Novo)
    afiliacao_nova, CM1, CM2 = definir_afiliacao(x_novo, y_novo, z_novo, M, N1)
    
    # 4. Cálculo da Transferência de Massa
    
    # Partículas que mudaram de Estrela 1 para Estrela 2 (M1 -> M2)
    transferencia_1_para_2 = np.sum((afiliacao_antiga == 1) & (afiliacao_nova == 2)) * m_particula
    
    # Partículas que mudaram de Estrela 2 para Estrela 1 (M2 -> M1)
    transferencia_2_para_1 = np.sum((afiliacao_antiga == 2) & (afiliacao_nova == 1)) * m_particula
    
    # Taxa de Transferência Líquida
    delta_M_liquido = transferencia_1_para_2 - transferencia_2_para_1
    M_dot = delta_M_liquido / dt
    
    # 5. Salvar e Atualizar
    tempo_historico.append(t)
    mdot_historico.append(M_dot)
    historico_X.append(X.copy()) # Salva o estado para a animação
    afiliacao_historico.append(afiliacao_nova.copy())
    
    # Prepara para o próximo passo
    afiliacao_antiga = afiliacao_nova
    
    if step % 50 == 0:
        print(f"Tempo: {t:.2f} | M_dot (1->2): {M_dot:.4f} | Separação CM: {np.linalg.norm(CM1-CM2):.2f}")


print("\nSimulação concluída. Gerando gráficos.")

# =============================================
# PLOTAGEM DO GRÁFICO M_DOT (Taxa de Transferência)
# =============================================
fig_mdot, ax_mdot = plt.subplots(figsize=(10, 5))
ax_mdot.plot(tempo_historico, mdot_historico, label=r'Taxa Líquida $\dot{M}_{1 \to 2}$')
ax_mdot.hlines(0, 0, T_MAX, color='red', linestyle='--', label='Transferência Zero')
ax_mdot.set_xlabel("Tempo")
ax_mdot.set_ylabel(r"Taxa de Transferência de Massa ($\dot{M}$)")
ax_mdot.set_title(f"Taxa de Transferência de Massa (N1={N1}, N2={N2}, dt={dt})")
ax_mdot.legend()
ax_mdot.grid(True)
plt.show()

# =============================================
# ANIMAÇÃO (VISUALIZAÇÃO 3D)
# =============================================
fig_anim = plt.figure(figsize=(10, 8))
ax_anim = fig_anim.add_subplot(111, projection='3d')
max_range = np.max(np.abs(historico_X[0][0:N_total*3])) * 1.5
ax_anim.set_xlim(-max_range, max_range)
ax_anim.set_ylim(-max_range, max_range)
ax_anim.set_zlim(-max_range, max_range)
ax_anim.set_xlabel('X')
ax_anim.set_ylabel('Y')
ax_anim.set_zlabel('Z')
particulas = ax_anim.scatter([], [], [], s=5) 

def init_anim():
    particulas._offsets3d = ([], [], [])
    return particulas,

def update_anim(frame):
    current_X = historico_X[frame]
    current_afiliacao = afiliacao_historico[frame]
    
    x_f = current_X[0:N_total]
    y_f = current_X[N_total:2*N_total]
    z_f = current_X[2*N_total:3*N_total]

    # Cores: Estrela 1 e Estrela 2
    cores = np.where(current_afiliacao == 1, 'red', 'purple')
    
    particulas._offsets3d = (x_f, y_f, z_f)
    particulas.set_facecolors(cores)
    
    # Título mostra a massa da estrela 1 no tempo atual
    massa_1_atual = np.sum(current_afiliacao == 1) * m_particula
    
    ax_anim.set_title(f"Tempo: {frame * dt:.2f} | Massa Estrela 1: {massa_1_atual:.2f}")

    return particulas,

# Criação da Animação 
ani = FuncAnimation(fig_anim, update_anim, frames=len(tempo_historico), init_func=init_anim, blit=False, interval=50)
# print("Salvando animação...")
# ani.save("animation_binary_stars1.gif", writer='pillow', fps=10)
# print("Animação salva.")

plt.show()
# Configuração da animação
ani = FuncAnimation(fig, update, frames=200, init_func=init, blit=True, interval=50)
plt.show()
