import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64

def print_tableau(tableau):
    """Función para imprimir la tabla simplex."""
    print("Tabla Simplex:")
    print(tableau)
    print()

def simplex(c, A, b):
    m, n = A.shape
    tableau = np.hstack((A, np.eye(m), b.reshape(-1, 1)))
    tableau = np.vstack((tableau, np.hstack((c, np.zeros(m + 1)))))
    
    while np.any(tableau[-1, :-1] > 0):
        pivot_col = np.argmax(tableau[-1, :-1])
        ratios = np.divide(tableau[:-1, -1], tableau[:-1, pivot_col], out=np.full(m, np.inf), where=tableau[:-1, pivot_col] > 0)
        pivot_row = np.argmin(ratios)
        
        pivot_element = tableau[pivot_row, pivot_col]
        tableau[pivot_row, :] /= pivot_element
        for i in range(m + 1):
            if i != pivot_row:
                tableau[i, :] -= tableau[pivot_row, :] * tableau[i, pivot_col]
    
    optimal_value = tableau[-1, -1]
    solution = tableau[:-1, -1]
    
    return optimal_value, solution

def plot_graph(c, A, b):
    fig, ax = plt.subplots()
    x = np.linspace(0, 10, 400)
    
    for i in range(len(A)):
        y = (b[i] - A[i, 0] * x) / A[i, 1]
        ax.plot(x, y, label=f'Restricción {i+1}')
    
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.legend()
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plot_url = base64.b64encode(image_png).decode('utf-8')
    
    return f'data:image/png;base64,{plot_url}'
