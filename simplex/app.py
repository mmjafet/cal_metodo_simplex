from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

def simplex_step_by_step(c, A, b, maximize=True):
    num_vars = len(c)
    num_constraints = len(b)
    
    tableau = np.zeros((num_constraints + 1, num_vars + num_constraints + 1))
    
    tableau[0, :num_vars] = -np.array(c) if maximize else np.array(c)
    
    for i in range(num_constraints):
        tableau[i + 1, :num_vars] = A[i]
        tableau[i + 1, num_vars + i] = 1
        tableau[i + 1, -1] = b[i]
    
    socketio.emit('update_tableau', {'tableau': tableau.tolist()})
    
    while True:
        pivot_col = np.argmin(tableau[0, :-1]) if maximize else np.argmax(tableau[0, :-1])
        if (tableau[0, pivot_col] >= 0 and maximize) or (tableau[0, pivot_col] <= 0 and not maximize):
            break
        
        ratios = tableau[1:, -1] / tableau[1:, pivot_col]
        ratios[tableau[1:, pivot_col] <= 0] = np.inf
        pivot_row = np.argmin(ratios) + 1
        
        pivot_element = tableau[pivot_row, pivot_col]
        tableau[pivot_row] /= pivot_element
        
        for i in range(len(tableau)):
            if i != pivot_row:
                tableau[i] -= tableau[i, pivot_col] * tableau[pivot_row]
        
        socketio.emit('update_tableau', {'tableau': tableau.tolist()})
    
    solution = np.zeros(num_vars)
    for i in range(num_vars):
        col = tableau[1:, i]
        if np.sum(col == 1) == 1 and np.sum(col == 0) == num_constraints - 1:
            solution[i] = tableau[1 + np.argmax(col), -1]
    
    z = tableau[0, -1]
    return z, solution

def graphical_method(c, A, b):
    num_vars = len(c)
    num_constraints = len(b)
    
    intersections = []
    for i in range(num_constraints):
        for j in range(i + 1, num_constraints):
            if np.linalg.matrix_rank(np.array([A[i], A[j]])) == num_vars:
                A_intersect = np.array([A[i], A[j]])
                b_intersect = np.array([b[i], b[j]])
                try:
                    intersection_point = np.linalg.solve(A_intersect, b_intersect)
                    if np.all(intersection_point >= 0):  # Ensure non-negativity
                        intersections.append((intersection_point, np.dot(c, intersection_point)))
                except np.linalg.LinAlgError:
                    pass
    
    if intersections:
        optimal_solutions = [x for x in intersections if np.isclose(x[1], max(intersections, key=lambda x: x[1])[1])]
        if len(optimal_solutions) > 1:
            z = None
            solution = None
            message = 'El problema tiene múltiples soluciones óptimas.'
        else:
            best_intersection = max(intersections, key=lambda x: x[1])
            z = best_intersection[1]
            solution = best_intersection[0]
            message = None
    else:
        z = None
        solution = None
        message = 'No se pudo encontrar una solución válida.'
    
    return z, solution, message

def plot_graph(c, A, b):
    n = len(c)
    x = np.linspace(0, 50, 400)
    y = np.linspace(0, 50, 400)
    X, Y = np.meshgrid(x, y)
    
    plt.figure()
    for i in range(len(A)):
        plt.plot(x, (b[i] - A[i][0] * x) / A[i][1], label=f'Restricción {i+1}')

    z = c[0] * X + c[1] * Y
    plt.contour(X, Y, z, levels=[-100, -50, 0, 50, 100, 150], colors='k', linestyles='dashed')
    
    plt.xlim((0, 50))
    plt.ylim((0, 50))
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Gráfica de las restricciones y función objetivo')
    plt.legend()
    plt.grid(True)
    
    img = BytesIO()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    return plot_url

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/graphical_solution', methods=['POST'])
def graphical_solution():
    data = request.get_json()

    c = list(map(float, data['c']))
    A = [list(map(float, row)) for row in data['A']]
    b = list(map(float, data['b']))

    z, solution, message = graphical_method(c, A, b)

    if z is not None and solution is not None:
        plot_url = plot_graph(c, A, b)
        return jsonify({'success': True, 'solution': solution.tolist(), 'plot_url': plot_url, 'message': message})
    else:
        return jsonify({'success': False, 'error': message})

@socketio.on('solve')
def handle_solve(data):
    maximize = data['objective'] == 'Maximizar'
    c = list(map(float, data['c']))
    A = [list(map(float, row)) for row in data['A']]
    b = list(map(float, data['b']))
    
    method = data.get('method', 'simplex')

    if method == 'simplex':
        try:
            z, solution = simplex_step_by_step(c, A, b, maximize)
            emit('solution', {'z': z, 'solution': solution.tolist()})
        except Exception as e:
            emit('solution', {'error': str(e)})
    elif method == 'graphical':
        try:
            z, solution, message = graphical_method(c, A, b)
            if z is not None and solution is not None:
                plot_url = plot_graph(c, A, b)
                emit('graphical_solution', {'z': z, 'solution': solution.tolist(), 'plot_url': plot_url, 'message': message})
            else:
                emit('graphical_solution', {'error': message})
        except Exception as e:
            emit('graphical_solution', {'error': str(e)})

if __name__ == '__main__':
    socketio.run(app, debug=True)
