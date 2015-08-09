%%cython
import numpy as np
cimport numpy as np
cimport cython
import sys
@cython.boundscheck(False)
def dS_multi(float S, float I, float beta, int node,
             np.ndarray[np.float32_t, ndim=1] St,
             np.ndarray[np.float32_t, ndim=1] It,
             np.ndarray[np.float32_t, ndim=1] C):
    '''
        St, It: vector
        node: 更新するノード
    '''
    # 和のファクタの計算
    cdef int other_node
    cdef float summation = 0
    for other_node in range(len(St)):
        if other_node != node:
            summation += S * It[other_node] * C[other_node]
    return -beta * I  * S - beta * summation

def dI_multi(float I, float S, float beta, float gamma, int node,
             np.ndarray[np.float32_t, ndim=1] St,
             np.ndarray[np.float32_t, ndim=1] It,
             np.ndarray[np.float32_t, ndim=1] C):
    '''
        St, It: vector
        node: 更新するノード
    '''
    # 和のファクタの計算
    cdef int other_node
    cdef float summation = 0
    for other_node in range(len(St)):
        if other_node != node:
            summation += S * It[other_node] * C[other_node]
    return beta * I  * S + beta * summation - gamma * I 

def rk_dS(float S,float h, float I, float beta, int node,
             np.ndarray[np.float32_t, ndim=1] St,
             np.ndarray[np.float32_t, ndim=1] It,
             np.ndarray[np.float32_t, ndim=1] C):
    '''
    ルンゲクッタ法を実行して，微分方程式から次の実行ステップの微分方程式
    の行き先を返す関数
    y: その地点での微分方程式の解の値
    h: 刻み幅(dt)
    func: 解く微分方程式
    *static_args: 微分方程式に代入する他の変数
    '''
    
    cdef float y0d = dS_multi(S, I, beta, node, St, It, C)
    if abs(y0d) < 1e-8:
        y0d = 0
    cdef float y1=S+y0d*h/2.0
    cdef float y1d = dS_multi(y1, I, beta, node, St, It, C)
    if abs(y1d) < 1e-8:
        y1d = 0
    cdef float y2=S+y1d*h/2.0
    cdef float y2d = dS_multi(y2, I, beta, node, St, It, C)
    if abs(y2d) < 1e-8:
        y2d = 0
    cdef float y3=S+y2d*h/2.0
    cdef float y3d = dS_multi(y3, I, beta, node, St, It, C)
    if abs(y3d) < 1e-8:
        y3d = 0
    cdef float next_S = S+(y0d+2.0*y1d+2.0*y2d+y3d)*h/6.0
    if abs(next_S) < 1e-8:
        return 0
    return next_S



def rk_dI(float I,float h, float S, float beta, float gamma, int node,
             np.ndarray[np.float32_t, ndim=1] St,
             np.ndarray[np.float32_t, ndim=1] It,
             np.ndarray[np.float32_t, ndim=1] C):
    '''
    ルンゲクッタ法を実行して，微分方程式から次の実行ステップの微分方程式
    の行き先を返す関数
    y: その地点での微分方程式の解の値
    h: 刻み幅(dt)
    func: 解く微分方程式
    *static_args: 微分方程式に代入する他の変数
    '''
    
    cdef float y0d = dI_multi(I, S, beta, gamma, node, St, It, C)
    if abs(y0d) < 1e-8:
        y0d = 0
    cdef float y1=I+y0d*h/2.0
    cdef float y1d = dI_multi(y1, S, beta, gamma, node, St, It, C)
    if abs(y1d) < 1e-8:
        y1d = 0
    cdef float y2=I+y1d*h/2.0
    cdef float y2d = dI_multi(y2, S, beta, gamma, node, St, It, C)
    if abs(y2d) < 1e-8:
        y2d = 0
    cdef float y3=I+y2d*h/2.0
    cdef float y3d = dI_multi(y3, S, beta, gamma, node, St, It, C)
    if abs(y3d) < 1e-8:
        y3d = 0
    cdef float next_I = I+(y0d+2.0*y1d+2.0*y2d+y3d)*h/6.0
    if abs(next_I) < 1e-8:
        return 0
    return next_I

def solve_ODE_SIR(np.ndarray[np.float32_t, ndim=2] Ss,
                  np.ndarray[np.float32_t, ndim=2] Is, 
                  np.ndarray[np.float32_t, ndim=2] Rs,
                  np.ndarray[np.float32_t, ndim=1] Ns,
                  np.ndarray[np.float32_t, ndim=2] Ct,
                  int duration, float dt, float beta, float gamma):
    cdef int i, j
    cdef int num_node = Ss.shape[0]
    cdef np.ndarray[np.float32_t, ndim=1] times = np.zeros(duration, dtype=np.float32)
    for j in range(1, duration):
        # 微分方程式に従って値を更新
        for i in range(num_node):        
            Ss[i, j] = rk_dS(Ss[i, j-1], dt, Is[i, j-1], beta, i,Ss[:, j], Is[:, j], Ct[i, :])
            Is[i, j] = rk_dI(Is[i, j-1], dt, Ss[i, j-1], beta, gamma, i, Ss[:, j], Is[:, j], Ct[i, :])
            Rs[i, j] = Ns[i] - (Is[i, j] + Ss[i, j])
            # 時間を更新
        times[j] = times[j-1] + dt
    return Ss, Is, Rs, times