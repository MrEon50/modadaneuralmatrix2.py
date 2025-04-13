"""
ModularAdaptiveNeuralMatrix - Wydajny, modułowy, adaptacyjny system sieci neuronowych
bez zewnętrznych zależności.

Główne cechy:
- Operacje macierzowe dla wysokiej wydajności
- Modułowa architektura z automatycznym skalowaniem
- Samoadaptujące się warstwy i optymalizatory
- Możliwość łączenia modeli i transferu wiedzy
- Kompaktowa implementacja (~2000 linii)
"""

import math
import random
import time
import pickle
from typing import List, Tuple, Dict, Callable, Optional, Union, Any

# ---------------------- PODSTAWOWE OPERACJE MACIERZOWE ---------------------- #

class Matrix:
    """Klasa implementująca podstawowe operacje macierzowe."""
    
    @staticmethod
    def zeros(rows: int, cols: int) -> List[List[float]]:
        """Tworzy macierz wypełnioną zerami."""
        return [[0.0 for _ in range(cols)] for _ in range(rows)]
    
    @staticmethod
    def ones(rows: int, cols: int) -> List[List[float]]:
        """Tworzy macierz wypełnioną jedynkami."""
        return [[1.0 for _ in range(cols)] for _ in range(rows)]
    
    @staticmethod
    def random(rows: int, cols: int, low: float = -1.0, high: float = 1.0) -> List[List[float]]:
        """Tworzy macierz wypełnioną losowymi wartościami."""
        return [[random.uniform(low, high) for _ in range(cols)] for _ in range(rows)]
    
    @staticmethod
    def eye(size: int) -> List[List[float]]:
        """Tworzy macierz jednostkową."""
        return [[1.0 if i == j else 0.0 for j in range(size)] for i in range(size)]
    
    @staticmethod
    def transpose(matrix: List[List[float]]) -> List[List[float]]:
        """Transponuje macierz."""
        rows, cols = len(matrix), len(matrix[0]) if matrix else 0
        return [[matrix[j][i] for j in range(rows)] for i in range(cols)]
    
    @staticmethod
    def add(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
        """Dodaje dwie macierze."""
        rows, cols = len(a), len(a[0])
        return [[a[i][j] + b[i][j] for j in range(cols)] for i in range(rows)]
    
    @staticmethod
    def subtract(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
        """Odejmuje macierz b od macierzy a."""
        rows, cols = len(a), len(a[0])
        return [[a[i][j] - b[i][j] for j in range(cols)] for i in range(rows)]
    
    @staticmethod
    def multiply(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
        """Mnoży dwie macierze."""
        a_rows, a_cols = len(a), len(a[0])
        b_rows, b_cols = len(b), len(b[0])
        
        if a_cols != b_rows:
            raise ValueError(f"Nieprawidłowe wymiary macierzy: {a_rows}x{a_cols} i {b_rows}x{b_cols}")
        
        result = Matrix.zeros(a_rows, b_cols)
        for i in range(a_rows):
            for j in range(b_cols):
                for k in range(a_cols):
                    result[i][j] += a[i][k] * b[k][j]
        
        return result
    
    @staticmethod
    def hadamard_product(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
        """Iloczyn Hadamarda (mnożenie element po elemencie)."""
        rows, cols = len(a), len(a[0])
        return [[a[i][j] * b[i][j] for j in range(cols)] for i in range(rows)]
    
    @staticmethod
    def scale(matrix: List[List[float]], scalar: float) -> List[List[float]]:
        """Mnoży macierz przez skalar."""
        rows, cols = len(matrix), len(matrix[0])
        return [[matrix[i][j] * scalar for j in range(cols)] for i in range(rows)]
    
    @staticmethod
    def apply(matrix: List[List[float]], func: Callable[[float], float]) -> List[List[float]]:
        """Aplikuje funkcję do każdego elementu macierzy."""
        rows, cols = len(matrix), len(matrix[0])
        return [[func(matrix[i][j]) for j in range(cols)] for i in range(rows)]
    
    @staticmethod
    def flatten(matrix: List[List[float]]) -> List[float]:
        """Spłaszcza macierz do jednowymiarowej listy."""
        return [element for row in matrix for element in row]
    
    @staticmethod
    def reshape(flat_array: List[float], rows: int, cols: int) -> List[List[float]]:
        """Przekształca płaską tablicę w macierz o podanych wymiarach."""
        if len(flat_array) != rows * cols:
            raise ValueError("Nieprawidłowy rozmiar tablicy dla podanych wymiarów")
        return [flat_array[i*cols:(i+1)*cols] for i in range(rows)]
    
    @staticmethod
    def sum(matrix: List[List[float]]) -> float:
        """Sumuje wszystkie elementy macierzy."""
        return sum(sum(row) for row in matrix)
    
    @staticmethod
    def mean(matrix: List[List[float]]) -> float:
        """Oblicza średnią wszystkich elementów macierzy."""
        flat = Matrix.flatten(matrix)
        return sum(flat) / len(flat) if flat else 0.0
    
    @staticmethod
    def std(matrix: List[List[float]]) -> float:
        """Oblicza odchylenie standardowe wszystkich elementów macierzy."""
        flat = Matrix.flatten(matrix)
        if not flat:
            return 0.0
        mean = sum(flat) / len(flat)
        return math.sqrt(sum((x - mean) ** 2 for x in flat) / len(flat))
    
    @staticmethod
    def normalize(matrix: List[List[float]]) -> List[List[float]]:
        """Normalizuje macierz (wartości w zakresie [0,1])."""
        flat = Matrix.flatten(matrix)
        if not flat:
            return matrix
        min_val, max_val = min(flat), max(flat)
        range_val = max_val - min_val
        if range_val == 0:
            return matrix
        rows, cols = len(matrix), len(matrix[0])
        return [[(matrix[i][j] - min_val) / range_val for j in range(cols)] for i in range(rows)]


# ---------------------- FUNKCJE AKTYWACJI I ICH POCHODNE ---------------------- #

class Activations:
    """Klasa zawierająca funkcje aktywacji i ich pochodne."""
    
    @staticmethod
    def sigmoid(x: float) -> float:
        """Funkcja sigmoidalna."""
        if x < -100: return 0
        if x > 100: return 1
        return 1.0 / (1.0 + math.exp(-x))
    
    @staticmethod
    def sigmoid_derivative(x: float) -> float:
        """Pochodna funkcji sigmoidalnej."""
        s = Activations.sigmoid(x)
        return s * (1 - s)
    
    @staticmethod
    def tanh(x: float) -> float:
        """Funkcja tanh."""
        if x < -20: return -1
        if x > 20: return 1
        return math.tanh(x)
    
    @staticmethod
    def tanh_derivative(x: float) -> float:
        """Pochodna funkcji tanh."""
        t = Activations.tanh(x)
        return 1 - t * t
    
    @staticmethod
    def relu(x: float) -> float:
        """Funkcja ReLU."""
        return max(0.0, x)
    
    @staticmethod
    def relu_derivative(x: float) -> float:
        """Pochodna funkcji ReLU."""
        return 1.0 if x > 0 else 0.0
    
    @staticmethod
    def leaky_relu(x: float, alpha: float = 0.01) -> float:
        """Funkcja Leaky ReLU."""
        return max(alpha * x, x)
    
    @staticmethod
    def leaky_relu_derivative(x: float, alpha: float = 0.01) -> float:
        """Pochodna funkcji Leaky ReLU."""
        return 1.0 if x > 0 else alpha
    
    @staticmethod
    def softmax(vector: List[float]) -> List[float]:
        """Funkcja softmax dla wektora."""
        max_val = max(vector)
        exp_values = [math.exp(x - max_val) for x in vector]
        sum_exp = sum(exp_values)
        return [x / sum_exp for x in exp_values]
    
    @staticmethod
    def softmax_matrix(matrix: List[List[float]]) -> List[List[float]]:
        """Funkcja softmax dla macierzy (każdy wiersz traktowany jako oddzielny wektor)."""
        rows, cols = len(matrix), len(matrix[0])
        result = Matrix.zeros(rows, cols)
        
        for i in range(rows):
            row = [matrix[i][j] for j in range(cols)]
            softmax_row = Activations.softmax(row)
            for j in range(cols):
                result[i][j] = softmax_row[j]
        
        return result


# ---------------------- FUNKCJE STRATY ---------------------- #

class Loss:
    """Klasa zawierająca funkcje straty i ich pochodne."""
    
    @staticmethod
    def mse(predictions: List[List[float]], targets: List[List[float]]) -> float:
        """Mean Squared Error (MSE)."""
        error_sum = 0.0
        count = 0
        
        for i in range(len(predictions)):
            for j in range(len(predictions[0])):
                error_sum += (predictions[i][j] - targets[i][j]) ** 2
                count += 1
        
        return error_sum / count if count > 0 else 0.0
    
    @staticmethod
    def mse_derivative(predictions: List[List[float]], targets: List[List[float]]) -> List[List[float]]:
        """Pochodna MSE."""
        rows, cols = len(predictions), len(predictions[0])
        result = Matrix.zeros(rows, cols)
        
        for i in range(rows):
            for j in range(cols):
                result[i][j] = 2 * (predictions[i][j] - targets[i][j]) / (rows * cols)
        
        return result
    
    @staticmethod
    def binary_cross_entropy(predictions: List[List[float]], targets: List[List[float]]) -> float:
        """Binary Cross Entropy."""
        epsilon = 1e-15
        error_sum = 0.0
        count = 0
        
        for i in range(len(predictions)):
            for j in range(len(predictions[0])):
                p = max(min(predictions[i][j], 1 - epsilon), epsilon)
                t = targets[i][j]
                error_sum += -(t * math.log(p) + (1 - t) * math.log(1 - p))
                count += 1
        
        return error_sum / count if count > 0 else 0.0
    
    @staticmethod
    def binary_cross_entropy_derivative(predictions: List[List[float]], targets: List[List[float]]) -> List[List[float]]:
        """Pochodna Binary Cross Entropy."""
        epsilon = 1e-15
        rows, cols = len(predictions), len(predictions[0])
        result = Matrix.zeros(rows, cols)
        
        for i in range(rows):
            for j in range(cols):
                p = max(min(predictions[i][j], 1 - epsilon), epsilon)
                t = targets[i][j]
                result[i][j] = -((t / p) - (1 - t) / (1 - p)) / (rows * cols)
        
        return result
    
    @staticmethod
    def categorical_cross_entropy(predictions: List[List[float]], targets: List[List[float]]) -> float:
        """Categorical Cross Entropy."""
        epsilon = 1e-15
        error_sum = 0.0
        
        for i in range(len(predictions)):
            for j in range(len(predictions[0])):
                p = max(min(predictions[i][j], 1 - epsilon), epsilon)
                t = targets[i][j]
                if t > 0:  # Pomijamy zera, aby uniknąć mnożenia przez log(p) = 0
                    error_sum += -t * math.log(p)
        
        return error_sum / len(predictions) if len(predictions) > 0 else 0.0
    
    @staticmethod
    def categorical_cross_entropy_derivative(predictions: List[List[float]], targets: List[List[float]]) -> List[List[float]]:
        """Pochodna Categorical Cross Entropy."""
        epsilon = 1e-15
        rows, cols = len(predictions), len(predictions[0])
        result = Matrix.zeros(rows, cols)
        
        for i in range(rows):
            for j in range(cols):
                p = max(min(predictions[i][j], 1 - epsilon), epsilon)
                t = targets[i][j]
                result[i][j] = -(t / p) / rows
        
        return result


# ---------------------- PODSTAWOWE MODUŁY SIECI ---------------------- #

class Module:
    """Abstrakcyjna klasa bazowa dla wszystkich modułów sieci."""
    
    def forward(self, inputs: List[List[float]]) -> List[List[float]]:
        """Przeprowadza propagację w przód."""
        raise NotImplementedError("Metoda forward musi być zaimplementowana przez klasę pochodną.")
    
    def backward(self, gradient: List[List[float]]) -> List[List[float]]:
        """Przeprowadza propagację wstecz."""
        raise NotImplementedError("Metoda backward musi być zaimplementowana przez klasę pochodną.")
    
    def update(self, learning_rate: float, optimizer: str = "adam", 
               beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8) -> None:
        """Aktualizuje parametry modułu."""
        pass
    
    def get_parameters(self) -> List[Tuple[List[List[float]], List[List[float]]]]:
        """Zwraca parametry modułu (wagi i gradienty)."""
        return []
    
    def set_parameters(self, parameters: List[Tuple[List[List[float]], List[List[float]]]]) -> None:
        """Ustawia parametry modułu."""
        pass
    
    def reset(self) -> None:
        """Resetuje stan modułu."""
        pass


class Linear(Module):
    """Warstwa liniowa (w pełni połączona)."""
    
    def __init__(self, input_size: int, output_size: int, weight_init: str = "xavier"):
        """Inicjalizuje warstwę liniową."""
        self.input_size = input_size
        self.output_size = output_size
        
        # Inicjalizacja wag
        if weight_init == "xavier":
            # Inicjalizacja Glorota/Xaviera
            std = math.sqrt(2.0 / (input_size + output_size))
            self.weights = Matrix.random(output_size, input_size, -std, std)
        elif weight_init == "he":
            # Inicjalizacja He (dla ReLU)
            std = math.sqrt(2.0 / input_size)
            self.weights = Matrix.random(output_size, input_size, -std, std)
        else:
            # Standardowa inicjalizacja równomierna
            self.weights = Matrix.random(output_size, input_size, -0.1, 0.1)
        
        # Inicjalizacja biasów zerami
        self.biases = Matrix.zeros(output_size, 1)
        
        # Gradienty dla wag i biasów
        self.weights_grad = Matrix.zeros(output_size, input_size)
        self.biases_grad = Matrix.zeros(output_size, 1)
        
        # Zapamiętanie ostatniego wejścia dla propagacji wstecznej
        self.last_input = None
        
        # Parametry dla optymalizatorów
        self.weights_momentum = Matrix.zeros(output_size, input_size)
        self.biases_momentum = Matrix.zeros(output_size, 1)
        self.weights_v = Matrix.zeros(output_size, input_size)
        self.biases_v = Matrix.zeros(output_size, 1)
        self.t = 0  # Licznik iteracji dla optymalizatorów
    
    def forward(self, inputs: List[List[float]]) -> List[List[float]]:
        """Przeprowadza propagację w przód."""
        self.last_input = inputs
        batch_size = len(inputs)
        
        # Obliczenie wyjścia: output = inputs * weights^T + biases
        output = Matrix.multiply(inputs, Matrix.transpose(self.weights))
        
        # Dodanie biasów do każdego przykładu w batchu
        biases_batch = [self.biases[i][0] for i in range(len(self.biases))]
        
        result = []
        for i in range(batch_size):
            row = []
            for j in range(self.output_size):
                row.append(output[i][j] + biases_batch[j])
            result.append(row)
        
        return result
    
    def backward(self, gradient: List[List[float]]) -> List[List[float]]:
        """Przeprowadza propagację wsteczną."""
        batch_size = len(gradient)
        
        # Obliczenie gradientu dla wag: weights_grad = gradient^T * inputs
        self.weights_grad = Matrix.zeros(self.output_size, self.input_size)
        
        for i in range(batch_size):
            for j in range(self.output_size):
                for k in range(self.input_size):
                    self.weights_grad[j][k] += gradient[i][j] * self.last_input[i][k]
        
        # Normalizacja gradientu przez rozmiar batcha
        self.weights_grad = Matrix.scale(self.weights_grad, 1.0 / batch_size)
        
        # Obliczenie gradientu dla biasów: biases_grad = sum(gradient, axis=0)
        self.biases_grad = Matrix.zeros(self.output_size, 1)
        
        for i in range(batch_size):
            for j in range(self.output_size):
                self.biases_grad[j][0] += gradient[i][j]
        
        # Normalizacja gradientu przez rozmiar batcha
        self.biases_grad = Matrix.scale(self.biases_grad, 1.0 / batch_size)
        
        # Obliczenie gradientu dla wejścia: input_grad = gradient * weights
        input_grad = Matrix.multiply(gradient, self.weights)
        
        return input_grad
    
    def update(self, learning_rate: float, optimizer: str = "adam", 
               beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8) -> None:
        """Aktualizuje parametry warstwy przy użyciu wybranego optymalizatora."""
        self.t += 1  # Inkrementacja licznika iteracji
        
        if optimizer == "sgd":
            # Standardowy SGD
            for i in range(self.output_size):
                for j in range(self.input_size):
                    self.weights[i][j] -= learning_rate * self.weights_grad[i][j]
                self.biases[i][0] -= learning_rate * self.biases_grad[i][0]
        
        elif optimizer == "momentum":
            # SGD z momentum
            momentum = 0.9
            
            for i in range(self.output_size):
                for j in range(self.input_size):
                    self.weights_momentum[i][j] = momentum * self.weights_momentum[i][j] + learning_rate * self.weights_grad[i][j]
                    self.weights[i][j] -= self.weights_momentum[i][j]
                
                self.biases_momentum[i][0] = momentum * self.biases_momentum[i][0] + learning_rate * self.biases_grad[i][0]
                self.biases[i][0] -= self.biases_momentum[i][0]
        
        elif optimizer == "adam":
            # Adam
            for i in range(self.output_size):
                for j in range(self.input_size):
                    # Aktualizacja momentów pierwszego rzędu
                    self.weights_momentum[i][j] = beta1 * self.weights_momentum[i][j] + (1 - beta1) * self.weights_grad[i][j]
                    
                    # Aktualizacja momentów drugiego rzędu
                    self.weights_v[i][j] = beta2 * self.weights_v[i][j] + (1 - beta2) * (self.weights_grad[i][j] ** 2)
                    
                    # Korekta obciążenia
                    m_corrected = self.weights_momentum[i][j] / (1 - beta1 ** self.t)
                    v_corrected = self.weights_v[i][j] / (1 - beta2 ** self.t)
                    
                    # Aktualizacja wag
                    self.weights[i][j] -= learning_rate * m_corrected / (math.sqrt(v_corrected) + epsilon)
                
                # Aktualizacja biasów
                self.biases_momentum[i][0] = beta1 * self.biases_momentum[i][0] + (1 - beta1) * self.biases_grad[i][0]
                self.biases_v[i][0] = beta2 * self.biases_v[i][0] + (1 - beta2) * (self.biases_grad[i][0] ** 2)
                
                m_corrected = self.biases_momentum[i][0] / (1 - beta1 ** self.t)
                v_corrected = self.biases_v[i][0] / (1 - beta2 ** self.t)
                
                self.biases[i][0] -= learning_rate * m_corrected / (math.sqrt(v_corrected) + epsilon)
    
    def get_parameters(self) -> List[Tuple[List[List[float]], List[List[float]]]]:
        """Zwraca parametry warstwy (wagi i gradienty)."""
        return [(self.weights, self.weights_grad), (self.biases, self.biases_grad)]
    
    def set_parameters(self, parameters: List[Tuple[List[List[float]], List[List[float]]]]) -> None:
        """Ustawia parametry warstwy."""
        if len(parameters) >= 2:
            self.weights, self.weights_grad = parameters[0]
            self.biases, self.biases_grad = parameters[1]
    
    def reset(self) -> None:
        """Resetuje stan warstwy."""
        self.last_input = None
        self.weights_grad = Matrix.zeros(self.output_size, self.input_size)
        self.biases_grad = Matrix.zeros(self.output_size, 1)
        self.weights_momentum = Matrix.zeros(self.output_size, self.input_size)
        self.biases_momentum = Matrix.zeros(self.output_size, 1)
        self.weights_v = Matrix.zeros(self.output_size, self.input_size)
        self.biases_v = Matrix.zeros(self.output_size, 1)
        self.t = 0


class Activation(Module):
    """Warstwa aktywacji."""
    
    def __init__(self, activation_type: str = "relu", alpha: float = 0.01):
        """Inicjalizuje warstwę aktywacji."""
        self.activation_type = activation_type
        self.alpha = alpha
        self.last_input = None
        
        # Wybór odpowiedniej funkcji aktywacji i jej pochodnej
        if activation_type == "sigmoid":
            self.activation_func = Activations.sigmoid
            self.activation_derivative = Activations.sigmoid_derivative
        elif activation_type == "tanh":
            self.activation_func = Activations.tanh
            self.activation_derivative = Activations.tanh_derivative
        elif activation_type == "relu":
            self.activation_func = Activations.relu
            self.activation_derivative = Activations.relu_derivative
        elif activation_type == "leaky_relu":
            self.activation_func = lambda x: Activations.leaky_relu(x, alpha)
            self.activation_derivative = lambda x: Activations.leaky_relu_derivative(x, alpha)
        elif activation_type == "softmax":
            self.activation_func = lambda x: x  # Placeholder, softmax jest obsługiwany specjalnie
            self.activation_derivative = lambda x: 1.0  # Placeholder
        else:
            raise ValueError(f"Nieznany typ aktywacji: {activation_type}")
    
    def forward(self, inputs: List[List[float]]) -> List[List[float]]:
        """Przeprowadza propagację w przód."""
        self.last_input = inputs
        
        if self.activation_type == "softmax":
            return Activations.softmax_matrix(inputs)
        else:
            return Matrix.apply(inputs, self.activation_func)
    
    def backward(self, gradient: List[List[float]]) -> List[List[float]]:
        """Przeprowadza propagację wsteczną."""
        if self.activation_type == "softmax":
            # Uproszczona implementacja - zakładamy, że softmax jest ostatnią warstwą 
            # i jest używany z cross-entropy, co upraszcza gradient
            return gradient
        else:
            batch_size, features = len(self.last_input), len(self.last_input[0])
            result = []
            
            for i in range(batch_size):
                row = []
                for j in range(features):
                    derivative = self.activation_derivative(self.last_input[i][j])
                    row.append(gradient[i][j] * derivative)
                result.append(row)
            
            return result
    
    def reset(self) -> None:
        """Resetuje stan warstwy."""
        self.last_input = None


class Dropout(Module):
    """Warstwa Dropout dla regularyzacji."""
    
    def __init__(self, drop_probability: float = 0.5):
        """Inicjalizuje warstwę Dropout."""
        self.drop_probability = drop_probability
        self.mask = None
        self.training = True
    
    def forward(self, inputs: List[List[float]]) -> List[List[float]]:
        """Przeprowadza propagację w przód."""
        if not self.training or self.drop_probability == 0:
            return inputs
        
        batch_size, features = len(inputs), len(inputs[0])
        
        # Tworzenie maski dropout
        self.mask = []
        for i in range(batch_size):
            row = []
            for j in range(features):
                # Neuron pozostaje aktywny z prawdopodobieństwem (1 - drop_probability)
                row.append(1.0 if random.random() > self.drop_probability else 0.0)
            self.mask.append(row)
        
        # Skalowanie wyjścia przez 1/(1-p) dla zachowania oczekiwanej wartości
        scale = 1.0 / (1.0 - self.drop_probability)
        
        # Zastosowanie maski i skalowanie
        result = []
        for i in range(batch_size):
            row = []
            for j in range(features):
                row.append(inputs[i][j] * self.mask[i][j] * scale)
            result.append(row)
        
        return result
    
    def backward(self, gradient: List[List[float]]) -> List[List[float]]:
        """Przeprowadza propagację wsteczną."""
        if not self.training or self.drop_probability == 0:
            return gradient
        
        batch_size, features = len(gradient), len(gradient[0])
        scale = 1.0 / (1.0 - self.drop_probability)
        
        # Przemnożenie gradientu przez maskę i współczynnik skalowania
        result = []
        for i in range(batch_size):
            row = []
            for j in range(features):
                row.append(gradient[i][j] * self.mask[i][j] * scale)
            result.append(row)
        
        return result
    
    def train(self) -> None:
        """Przełącza warstwę w tryb treningowy."""
        self.training = True
    
    def eval(self) -> None:
        """Przełącza warstwę w tryb ewaluacji."""
        self.training = False
    
    def reset(self) -> None:
        """Resetuje stan warstwy."""
        self.mask = None


class BatchNormalization(Module):
    """Warstwa Batch Normalization."""
    
    def __init__(self, num_features: int, epsilon: float = 1e-5, momentum: float = 0.1):
        """Inicjalizuje warstwę Batch Normalization."""
        self.num_features = num_features
        self.epsilon = epsilon
        self.momentum = momentum
        
        # Parametry do nauczenia
        self.gamma = [1.0] * num_features  # Skalowanie
        self.beta = [0.0] * num_features   # Przesunięcie
        
        # Gradienty
        self.gamma_grad = [0.0] * num_features
        self.beta_grad = [0.0] * num_features
        
        # Średnie ruchome dla fazy ewaluacji
        self.running_mean = [0.0] * num_features
        self.running_var = [1.0] * num_features
        
        # Stan warstwy
        self.batch_mean = None
        self.batch_var = None
        self.normalized = None
        self.last_input = None
        self.training = True
    
    def forward(self, inputs: List[List[float]]) -> List[List[float]]:
        """Przeprowadza propagację w przód."""
        batch_size = len(inputs)
        self.last_input = inputs
        
        if self.training:
            # Obliczenie średniej dla każdej cechy
            self.batch_mean = [0.0] * self.num_features
            for i in range(batch_size):
                for j in range(self.num_features):
                    self.batch_mean[j] += inputs[i][j]
            
            for j in range(self.num_features):
                self.batch_mean[j] /= batch_size
            
            # Obliczenie wariancji dla każdej cechy
            self.batch_var = [0.0] * self.num_features
            for i in range(batch_size):
                for j in range(self.num_features):
                    self.batch_var[j] += (inputs[i][j] - self.batch_mean[j]) ** 2
            
            for j in range(self.num_features):
                self.batch_var[j] /= batch_size
            
            # Aktualizacja średnich ruchomych
            for j in range(self.num_features):
                self.running_mean[j] = self.momentum * self.batch_mean[j] + (1 - self.momentum) * self.running_mean[j]
                self.running_var[j] = self.momentum * self.batch_var[j] + (1 - self.momentum) * self.running_var[j]
            
            # Normalizacja
            self.normalized = []
            for i in range(batch_size):
                normalized_row = []
                for j in range(self.num_features):
                    normalized_val = (inputs[i][j] - self.batch_mean[j]) / math.sqrt(self.batch_var[j] + self.epsilon)
                    normalized_row.append(normalized_val)
                self.normalized.append(normalized_row)
            
            # Skalowanie i przesunięcie
            result = []
            for i in range(batch_size):
                result_row = []
                for j in range(self.num_features):
                    result_row.append(self.gamma[j] * self.normalized[i][j] + self.beta[j])
                result.append(result_row)
            
            return result
        
        else:  # Tryb ewaluacji
            # Używanie średnich ruchomych
            result = []
            for i in range(batch_size):
                result_row = []
                for j in range(self.num_features):
                    normalized_val = (inputs[i][j] - self.running_mean[j]) / math.sqrt(self.running_var[j] + self.epsilon)
                    result_row.append(self.gamma[j] * normalized_val + self.beta[j])
                result.append(result_row)
            
            return result
    
    def backward(self, gradient: List[List[float]]) -> List[List[float]]:
        """Przeprowadza propagację wsteczną."""
        batch_size = len(gradient)
        
        # Obliczenie gradientu dla gamma i beta
        for j in range(self.num_features):
            self.gamma_grad[j] = 0.0
            self.beta_grad[j] = 0.0
            
            for i in range(batch_size):
                self.gamma_grad[j] += gradient[i][j] * self.normalized[i][j]
                self.beta_grad[j] += gradient[i][j]
        
        # Normalizacja gradientów przez rozmiar batcha
        for j in range(self.num_features):
            self.gamma_grad[j] /= batch_size
            self.beta_grad[j] /= batch_size
        
        # Obliczenie gradientu dla wejścia
        # Implementacja algorytmu propagacji wstecznej dla Batch Normalization
        dxhat = []
        for i in range(batch_size):
            dxhat_row = []
            for j in range(self.num_features):
                dxhat_row.append(gradient[i][j] * self.gamma[j])
            dxhat.append(dxhat_row)
        
        # Obliczenie gradientu względem wariancji
        dvar = [0.0] * self.num_features
        for j in range(self.num_features):
            for i in range(batch_size):
                dvar[j] += dxhat[i][j] * (self.last_input[i][j] - self.batch_mean[j]) * (-0.5) * (self.batch_var[j] + self.epsilon) ** (-1.5)
        
        # Obliczenie gradientu względem średniej
        dmean = [0.0] * self.num_features
        for j in range(self.num_features):
            for i in range(batch_size):
                dmean[j] += dxhat[i][j] * (-1.0 / math.sqrt(self.batch_var[j] + self.epsilon))
            dmean[j] += dvar[j] * (-2.0 / batch_size) * sum(self.last_input[i][j] - self.batch_mean[j] for i in range(batch_size))
        
        # Obliczenie gradientu względem wejścia
        dx = []
        for i in range(batch_size):
            dx_row = []
            for j in range(self.num_features):
                dx_value = dxhat[i][j] / math.sqrt(self.batch_var[j] + self.epsilon)
                dx_value += dvar[j] * (2.0 / batch_size) * (self.last_input[i][j] - self.batch_mean[j])
                dx_value += dmean[j] / batch_size
                dx_row.append(dx_value)
            dx.append(dx_row)
        
        return dx
    
    def update(self, learning_rate: float, **kwargs) -> None:
        """Aktualizuje parametry warstwy."""
        for j in range(self.num_features):
            self.gamma[j] -= learning_rate * self.gamma_grad[j]
            self.beta[j] -= learning_rate * self.beta_grad[j]
    
    def train(self) -> None:
        """Przełącza warstwę w tryb treningowy."""
        self.training = True
    
    def eval(self) -> None:
        """Przełącza warstwę w tryb ewaluacji."""
        self.training = False
    
    def reset(self) -> None:
        """Resetuje stan warstwy."""
        self.batch_mean = None
        self.batch_var = None
        self.normalized = None
        self.last_input = None
        self.gamma_grad = [0.0] * self.num_features
        self.beta_grad = [0.0] * self.num_features


class SelfAdaptingLayer(Module):
    """Samoadaptująca się warstwa, która dostosowuje swoją strukturę w trakcie treningu."""
    
    def __init__(self, input_size: int, output_size: int, adaptation_rate: float = 0.01):
        """Inicjalizuje samoadaptującą się warstwę."""
        self.input_size = input_size
        self.output_size = output_size
        self.adaptation_rate = adaptation_rate
        
        # Główna warstwa liniowa
        self.linear = Linear(input_size, output_size)
        
        # Metryki wydajności
        self.activation_levels = [0.0] * output_size
        self.gradient_magnitudes = [0.0] * output_size
        self.importance_scores = [0.0] * output_size
        
        # Licznik iteracji
        self.iterations = 0
    
    def forward(self, inputs: List[List[float]]) -> List[List[float]]:
        """Przeprowadza propagację w przód."""
        outputs = self.linear.forward(inputs)
        
        # Aktualizacja poziomów aktywacji
        batch_size = len(outputs)
        for j in range(self.output_size):
            activation_sum = sum(abs(outputs[i][j]) for i in range(batch_size))
            self.activation_levels[j] = 0.9 * self.activation_levels[j] + 0.1 * (activation_sum / batch_size)
        
        return outputs
    
    def backward(self, gradient: List[List[float]]) -> List[List[float]]:
        """Przeprowadza propagację wsteczną."""
        # Aktualizacja magnitud gradientów
        batch_size = len(gradient)
        for j in range(self.output_size):
            gradient_sum = sum(abs(gradient[i][j]) for i in range(batch_size))
            self.gradient_magnitudes[j] = 0.9 * self.gradient_magnitudes[j] + 0.1 * (gradient_sum / batch_size)
        
        # Obliczenie ważności neuronów
        for j in range(self.output_size):
            self.importance_scores[j] = self.activation_levels[j] * self.gradient_magnitudes[j]
        
        # Propagacja wsteczna przez warstwę liniową
        return self.linear.backward(gradient)
    
    def update(self, learning_rate: float, **kwargs) -> None:
        """Aktualizuje parametry warstwy i adaptuje jej strukturę."""
        self.iterations += 1
        
        # Aktualizacja parametrów warstwy liniowej
        self.linear.update(learning_rate, **kwargs)
        
        # Co 100 iteracji sprawdzamy, czy adaptować strukturę
        if self.iterations % 100 == 0:
            self._adapt_structure()
    
    def _adapt_structure(self) -> None:
        """Adaptuje strukturę warstwy na podstawie ważności neuronów."""
        # Sortowanie neuronów według ważności
        sorted_indices = sorted(range(self.output_size), key=lambda i: self.importance_scores[i])
        
        # Identyfikacja najmniej ważnych neuronów
        least_important = sorted_indices[:int(self.output_size * 0.1)]  # 10% najmniej ważnych
        most_important = sorted_indices[-int(self.output_size * 0.1):]  # 10% najbardziej ważnych
        
        # Adaptacja wag dla najmniej ważnych neuronów
        for i in least_important:
            # Losowa inicjalizacja wag dla nieaktywnych neuronów
            if self.importance_scores[i] < 0.01:
                for j in range(self.input_size):
                    self.linear.weights[i][j] = random.uniform(-0.1, 0.1)
        
        # Wzmocnienie wag dla najbardziej ważnych neuronów
        for i in most_important:
            # Zwiększenie wag dla aktywnych neuronów
            for j in range(self.input_size):
                self.linear.weights[i][j] *= (1.0 + self.adaptation_rate)
    
    def get_parameters(self) -> List[Tuple[List[List[float]], List[List[float]]]]:
        """Zwraca parametry warstwy."""
        return self.linear.get_parameters()
    
    def reset(self) -> None:
        """Resetuje stan warstwy."""
        self.linear.reset()
        self.activation_levels = [0.0] * self.output_size
        self.gradient_magnitudes = [0.0] * self.output_size
        self.importance_scores = [0.0] * self.output_size
        self.iterations = 0


# ---------------------- KLASA SEKWENCYJNA ŁĄCZĄCA MODUŁY ---------------------- #

class Sequential(Module):
    """Sekwencyjny kontener modułów."""
    
    def __init__(self, modules: List[Module] = None):
        """Inicjalizuje kontener sekwencyjny."""
        self.modules = modules if modules is not None else []
    
    def add(self, module: Module) -> None:
        """Dodaje moduł do kontenera."""
        self.modules.append(module)
    
    def forward(self, inputs: List[List[float]]) -> List[List[float]]:
        """Przeprowadza propagację w przód przez wszystkie moduły."""
        x = inputs
        for module in self.modules:
            x = module.forward(x)
        return x
    
    def backward(self, gradient: List[List[float]]) -> List[List[float]]:
        """Przeprowadza propagację wsteczną przez wszystkie moduły."""
        grad = gradient
        for module in reversed(self.modules):
            grad = module.backward(grad)
        return grad
    
    def update(self, learning_rate: float, optimizer: str = "adam", 
               beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8) -> None:
        """Aktualizuje parametry wszystkich modułów."""
        for module in self.modules:
            if hasattr(module, 'update'):
                module.update(learning_rate, optimizer=optimizer, beta1=beta1, beta2=beta2, epsilon=epsilon)
    
    def train(self) -> None:
        """Przełącza wszystkie moduły w tryb treningowy."""
        for module in self.modules:
            if hasattr(module, 'train'):
                module.train()
    
    def eval(self) -> None:
        """Przełącza wszystkie moduły w tryb ewaluacji."""
        for module in self.modules:
            if hasattr(module, 'eval'):
                module.eval()
    
    def reset(self) -> None:
        """Resetuje stan wszystkich modułów."""
        for module in self.modules:
            module.reset()
    
    def get_parameters(self) -> List[Tuple[List[List[float]], List[List[float]]]]:
        """Zwraca parametry wszystkich modułów."""
        params = []
        for module in self.modules:
            if hasattr(module, 'get_parameters'):
                params.extend(module.get_parameters())
        return params
    
    def set_parameters(self, parameters: List[Tuple[List[List[float]], List[List[float]]]]) -> None:
        """Ustawia parametry wszystkich modułów."""
        index = 0
        for module in self.modules:
            if hasattr(module, 'get_parameters'):
                num_params = len(module.get_parameters())
                if index + num_params <= len(parameters):
                    module_params = parameters[index:index+num_params]
                    module.set_parameters(module_params)
                    index += num_params


# ---------------------- KLASA MODELU SIECI NEURONOWEJ ---------------------- #

class NeuralNetwork:
    """Model sieci neuronowej."""
    
    def __init__(self, sequential: Sequential = None):
        """Inicjalizuje model sieci neuronowej."""
        self.sequential = sequential if sequential is not None else Sequential()
        self.loss_func = None
        self.loss_derivative = None
        self.current_loss = 0.0
        
        # Domyślne ustawienia treningowe
        self.learning_rate = 0.01
        self.optimizer = "adam"
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.batch_size = 32
        self.epochs = 10
        
        # Metryki wydajności
        self.metrics_history = {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": []}
        
        # Flaga adaptacyjnego uczenia
        self.adaptive_learning = False
        self.min_learning_rate = 0.0001
        self.max_learning_rate = 0.1
    
    def add(self, module: Module) -> None:
        """Dodaje moduł do sieci."""
        self.sequential.add(module)
    
    def forward(self, inputs: List[List[float]]) -> List[List[float]]:
        """Przeprowadza propagację w przód."""
        return self.sequential.forward(inputs)
    
    def compile(self, loss: str = "mse", learning_rate: float = 0.01, 
                optimizer: str = "adam", batch_size: int = 32, 
                adaptive_learning: bool = False) -> None:
        """Konfiguruje model do treningu."""
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.adaptive_learning = adaptive_learning
        
        # Wybór odpowiedniej funkcji straty i jej pochodnej
        if loss == "mse":
            self.loss_func = Loss.mse
            self.loss_derivative = Loss.mse_derivative
        elif loss == "binary_cross_entropy":
            self.loss_func = Loss.binary_cross_entropy
            self.loss_derivative = Loss.binary_cross_entropy_derivative
        elif loss == "categorical_cross_entropy":
            self.loss_func = Loss.categorical_cross_entropy
            self.loss_derivative = Loss.categorical_cross_entropy_derivative
        else:
            raise ValueError(f"Nieznana funkcja straty: {loss}")
    
    def train_batch(self, inputs: List[List[float]], targets: List[List[float]]) -> float:
        """Trenuje model na jednym batchu danych."""
        # Propagacja w przód
        predictions = self.forward(inputs)
        
        # Obliczenie straty
        loss = self.loss_func(predictions, targets)
        self.current_loss = loss
        
        # Obliczenie gradientu straty
        loss_grad = self.loss_derivative(predictions, targets)
        
        # Propagacja wsteczna
        self.sequential.backward(loss_grad)
        
        # Aktualizacja parametrów
        self.sequential.update(self.learning_rate, self.optimizer, self.beta1, self.beta2, self.epsilon)
        
        return loss
    
    def fit(self, x_train: List[List[float]], y_train: List[List[float]], 
            epochs: int = None, batch_size: int = None, validation_split: float = 0.0, 
            validation_data: Tuple[List[List[float]], List[List[float]]] = None,
            verbose: bool = True) -> Dict[str, List[float]]:
        """Trenuje model na danych treningowych."""
        # Ustawienie parametrów treningu
        epochs = epochs if epochs is not None else self.epochs
        batch_size = batch_size if batch_size is not None else self.batch_size
        
        # Przygotowanie danych walidacyjnych
        x_val, y_val = None, None
        
        if validation_data is not None:
            x_val, y_val = validation_data
        elif validation_split > 0:
            split_idx = int(len(x_train) * (1 - validation_split))
            x_val = x_train[split_idx:]
            y_val = y_train[split_idx:]
            x_train = x_train[:split_idx]
            y_train = y_train[:split_idx]
        
        # Historia treningu
        history = {
            "loss": [],
            "val_loss": [] if (x_val is not None and y_val is not None) else None,
            "accuracy": [],
            "val_accuracy": [] if (x_val is not None and y_val is not None) else None
        }
        
        # Przełączenie modelu w tryb treningowy
        self.sequential.train()
        
        # Główna pętla treningowa
        for epoch in range(epochs):
            start_time = time.time()
            
            # Przemieszanie danych treningowych
            combined = list(zip(x_train, y_train))
            random.shuffle(combined)
            x_shuffled, y_shuffled = zip(*combined)
            x_shuffled, y_shuffled = list(x_shuffled), list(y_shuffled)
            
            # Przetwarzanie po batchach
            num_batches = (len(x_shuffled) + batch_size - 1) // batch_size
            epoch_loss = 0.0
            
            for batch in range(num_batches):
                start_idx = batch * batch_size
                end_idx = min(start_idx + batch_size, len(x_shuffled))
                
                x_batch = x_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                batch_loss = self.train_batch(x_batch, y_batch)
                epoch_loss += batch_loss
                
                # Wyświetlanie postępu
                if verbose and batch % max(1, num_batches // 10) == 0:
                    print(f"Epoka {epoch+1}/{epochs}, Batch {batch+1}/{num_batches}, Loss: {batch_loss:.4f}")
            
            # Średnia strata dla epoki
            epoch_loss /= num_batches
            history["loss"].append(epoch_loss)
            
            # Obliczenie dokładności na zbiorze treningowym
            train_accuracy = self.evaluate(x_shuffled[:min(1000, len(x_shuffled))], 
                                          y_shuffled[:min(1000, len(y_shuffled))], 
                                          metrics=["accuracy"])["accuracy"]
            history["accuracy"].append(train_accuracy)
            
            # Walidacja
            if x_val is not None and y_val is not None:
                # Przełączenie modelu w tryb ewaluacji
                self.sequential.eval()
                
                # Obliczenie straty i dokładności na zbiorze walidacyjnym
                val_metrics = self.evaluate(x_val, y_val, metrics=["loss", "accuracy"])
                val_loss = val_metrics["loss"]
                val_accuracy = val_metrics["accuracy"]
                
                history["val_loss"].append(val_loss)
                history["val_accuracy"].append(val_accuracy)
                
                # Przełączenie modelu z powrotem w tryb treningowy
                self.sequential.train()
                
                # Adaptacyjne dostosowanie współczynnika uczenia
                if self.adaptive_learning:
                    self._adjust_learning_rate(val_loss, epoch)
                
                if verbose:
                    print(f"Epoka {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {train_accuracy:.4f}, "
                          f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, "
                          f"Czas: {time.time() - start_time:.2f}s, LR: {self.learning_rate:.6f}")
            else:
                if verbose:
                    print(f"Epoka {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {train_accuracy:.4f}, "
                          f"Czas: {time.time() - start_time:.2f}s, LR: {self.learning_rate:.6f}")
        
        # Aktualizacja historii metryk
        for key in history:
            if history[key] is not None:
                self.metrics_history[key].extend(history[key])
        
        return history
    
    def _adjust_learning_rate(self, val_loss: float, epoch: int) -> None:
        """Dostosowuje współczynnik uczenia na podstawie wydajności walidacji."""
        if epoch > 0:
            # Jeśli strata walidacyjna wzrosła, zmniejszamy learning rate
            if len(self.metrics_history["val_loss"]) > 1 and val_loss > self.metrics_history["val_loss"][-1]:
                self.learning_rate = max(self.learning_rate * 0.5, self.min_learning_rate)
            # Jeśli strata walidacyjna spadła, możemy zwiększyć learning rate
            elif len(self.metrics_history["val_loss"]) > 2 and val_loss < self.metrics_history["val_loss"][-1] < self.metrics_history["val_loss"][-2]:
                self.learning_rate = min(self.learning_rate * 1.1, self.max_learning_rate)
    
    def evaluate(self, x_test: List[List[float]], y_test: List[List[float]], 
                metrics: List[str] = ["loss"]) -> Dict[str, float]:
        """Ewaluuje model na danych testowych."""
        # Przełączenie modelu w tryb ewaluacji
        self.sequential.eval()
        
        # Propagacja w przód
        predictions = self.forward(x_test)
        
        # Inicjalizacja wyników
        results = {}
        
        # Obliczenie metryk
        if "loss" in metrics:
            results["loss"] = self.loss_func(predictions, y_test)
        
        if "accuracy" in metrics:
            # Obliczenie dokładności
            correct = 0
            for i in range(len(predictions)):
                pred_class = predictions[i].index(max(predictions[i]))
                true_class = y_test[i].index(max(y_test[i]))
                if pred_class == true_class:
                    correct += 1
            
            results["accuracy"] = correct / len(predictions) if len(predictions) > 0 else 0.0
        
        # Przełączenie modelu z powrotem w tryb treningowy
        self.sequential.train()
        
        return results
    
    def predict(self, x: List[List[float]]) -> List[List[float]]:
        """Generuje predykcje dla danych wejściowych."""
        # Przełączenie modelu w tryb ewaluacji
        self.sequential.eval()
        
        # Propagacja w przód
        predictions = self.forward(x)
        
        # Przełączenie modelu z powrotem w tryb treningowy
        self.sequential.train()
        
        return predictions
    
    def save(self, filepath: str) -> None:
        """Zapisuje model do pliku."""
        model_data = {
            "parameters": self.sequential.get_parameters(),
            "config": {
                "learning_rate": self.learning_rate,
                "optimizer": self.optimizer,
                "batch_size": self.batch_size,
                "epochs": self.epochs,
                "beta1": self.beta1,
                "beta2": self.beta2,
                "epsilon": self.epsilon,
                "adaptive_learning": self.adaptive_learning
            },
            "metrics_history": self.metrics_history
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load(self, filepath: str) -> None:
        """Wczytuje model z pliku."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Wczytanie parametrów
        self.sequential.set_parameters(model_data["parameters"])
        
        # Wczytanie konfiguracji
        config = model_data["config"]
        self.learning_rate = config["learning_rate"]
        self.optimizer = config["optimizer"]
        self.batch_size = config["batch_size"]
        self.epochs = config["epochs"]
        self.beta1 = config["beta1"]
        self.beta2 = config["beta2"]
        self.epsilon = config["epsilon"]
        self.adaptive_learning = config["adaptive_learning"]
        
        # Wczytanie historii metryk
        self.metrics_history = model_data["metrics_history"]
    
    def summary(self) -> None:
        """Wyświetla podsumowanie modelu."""
        print("=" * 80)
        print("Model Summary")
        print("=" * 80)
        
        total_params = 0
        trainable_params = 0
        
        print(f"{'Layer (type)':<30} {'Output Shape':<20} {'Param #':<10}")
        print("-" * 80)
        
        for i, module in enumerate(self.sequential.modules):
            module_name = module.__class__.__name__
            
            # Obliczenie liczby parametrów
            params = 0
            if hasattr(module, 'get_parameters'):
                module_params = module.get_parameters()
                for weights, _ in module_params:
                    if isinstance(weights, list):
                        if isinstance(weights[0], list):
                            params += len(weights) * len(weights[0])
                        else:
                            params += len(weights)
            
            # Określenie kształtu wyjścia
            output_shape = "Unknown"
            if isinstance(module, Linear):
                output_shape = f"({module.output_size})"
            elif isinstance(module, BatchNormalization):
                output_shape = f"({module.num_features})"
            
            print(f"{i+1} {module_name:<25} {'Output: ' + output_shape:<20} {params:<10}")
            
            total_params += params
            trainable_params += params
        
        print("-" * 80)
        print(f"Total params: {total_params}")
        print(f"Trainable params: {trainable_params}")
        print(f"Non-trainable params: {total_params - trainable_params}")
        print("=" * 80)
        
        # Wyświetlenie konfiguracji treningu
        print("Training Configuration:")
        print(f"  Learning Rate: {self.learning_rate}")
        print(f"  Optimizer: {self.optimizer}")
        print(f"  Batch Size: {self.batch_size}")
        print(f"  Adaptive Learning: {self.adaptive_learning}")
        print("=" * 80)
    
    def transfer_learning(self, source_model: 'NeuralNetwork', freeze_layers: int = 0) -> None:
        """Implementuje transfer learning z modelu źródłowego."""
        # Kopiowanie parametrów
        source_params = source_model.sequential.get_parameters()
        
        # Licznik warstw z parametrami
        param_layer_count = 0
        transferred_params = []
        
        for i, module in enumerate(self.sequential.modules):
            if hasattr(module, 'get_parameters') and module.get_parameters():
                if param_layer_count < len(source_params):
                    # Kopiowanie parametrów
                    module_params = module.get_parameters()
                    source_module_params = source_params[param_layer_count:param_layer_count+len(module_params)]
                    
                    # Sprawdzenie kompatybilności kształtów
                    compatible = True
                    for j, ((target_w, target_g), (source_w, source_g)) in enumerate(zip(module_params, source_module_params)):
                        if isinstance(target_w, list) and isinstance(source_w, list):
                            if len(target_w) != len(source_w) or (len(target_w) > 0 and len(target_w[0]) != len(source_w[0])):
                                compatible = False
                                break
                    
                    if compatible:
                        module.set_parameters(source_module_params)
                        transferred_params.append(param_layer_count)
                        
                        # Zamrożenie warstwy, jeśli potrzebne
                        if param_layer_count < freeze_layers:
                            setattr(module, '_frozen', True)
                            
                            # Nadpisanie metody update, aby nie aktualizować zamrożonych warstw
                            original_update = module.update
                            def frozen_update(*args, **kwargs):
                                pass
                            module.update = frozen_update
                    
                param_layer_count += len(module_params)
        
        print(f"Transfer learning: przeniesiono parametry z {len(transferred_params)} warstw")
        print(f"Zamrożone warstwy: {min(freeze_layers, len(transferred_params))}")


# ---------------------- PRZYKŁADOWE ARCHITEKTURY ---------------------- #

class ModelFactory:
    """Fabryka modeli - tworzy popularne architektury sieci neuronowych."""
    
    @staticmethod
    def create_mlp(input_size: int, hidden_sizes: List[int], output_size: int, 
                  dropout_rate: float = 0.0, batch_norm: bool = False) -> NeuralNetwork:
        """Tworzy wielowarstwowy perceptron (MLP)."""
        model = NeuralNetwork()
        
        # Pierwsza warstwa
        model.add(Linear(input_size, hidden_sizes[0]))
        if batch_norm:
            model.add(BatchNormalization(hidden_sizes[0]))
        model.add(Activation("relu"))
        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))
        
        # Warstwy ukryte
        for i in range(1, len(hidden_sizes)):
            model.add(Linear(hidden_sizes[i-1], hidden_sizes[i]))
            if batch_norm:
                model.add(BatchNormalization(hidden_sizes[i]))
            model.add(Activation("relu"))
            if dropout_rate > 0:
                model.add(Dropout(dropout_rate))
        
        # Warstwa wyjściowa
        model.add(Linear(hidden_sizes[-1], output_size))
        
        return model
    
    @staticmethod
    def create_adaptive_network(input_size: int, hidden_sizes: List[int], output_size: int) -> NeuralNetwork:
        """Tworzy sieć z samoadaptującymi się warstwami."""
        model = NeuralNetwork()
        
        # Pierwsza warstwa
        model.add(SelfAdaptingLayer(input_size, hidden_sizes[0]))
        model.add(Activation("relu"))
        model.add(BatchNormalization(hidden_sizes[0]))
        
        # Warstwy ukryte
        for i in range(1, len(hidden_sizes)):
            model.add(SelfAdaptingLayer(hidden_sizes[i-1], hidden_sizes[i]))
            model.add(Activation("relu"))
            model.add(BatchNormalization(hidden_sizes[i]))
        
        # Warstwa wyjściowa
        model.add(Linear(hidden_sizes[-1], output_size))
        
        return model


# ---------------------- NARZĘDZIA DO PRZETWARZANIA DANYCH ---------------------- #

class DataProcessor:
    """Klasa do przetwarzania danych dla sieci neuronowych."""
    
    @staticmethod
    def normalize(data: List[List[float]]) -> Tuple[List[List[float]], List[float], List[float]]:
        """Normalizuje dane (średnia 0, odchylenie standardowe 1)."""
        num_features = len(data[0])
        means = [0.0] * num_features
        stds = [0.0] * num_features
        
        # Obliczenie średniej dla każdej cechy
        for j in range(num_features):
            feature_sum = sum(data[i][j] for i in range(len(data)))
            means[j] = feature_sum / len(data)
        
        # Obliczenie odchylenia standardowego dla każdej cechy
        for j in range(num_features):
            variance_sum = sum((data[i][j] - means[j]) ** 2 for i in range(len(data)))
            stds[j] = math.sqrt(variance_sum / len(data))
            if stds[j] == 0:
                stds[j] = 1.0  # Unikamy dzielenia przez zero
        
        # Normalizacja danych
        normalized_data = []
        for i in range(len(data)):
            normalized_row = [(data[i][j] - means[j]) / stds[j] for j in range(num_features)]
            normalized_data.append(normalized_row)
        
        return normalized_data, means, stds
    
    @staticmethod
    def apply_normalization(data: List[List[float]], means: List[float], stds: List[float]) -> List[List[float]]:
        """Stosuje wcześniej obliczoną normalizację do nowych danych."""
        num_features = len(data[0])
        normalized_data = []
        
        for i in range(len(data)):
            normalized_row = [(data[i][j] - means[j]) / stds[j] for j in range(num_features)]
            normalized_data.append(normalized_row)
        
        return normalized_data
    
    @staticmethod
    def one_hot_encode(labels: List[int], num_classes: int) -> List[List[float]]:
        """Konwertuje etykiety na kodowanie one-hot."""
        one_hot = []
        for label in labels:
            encoding = [0.0] * num_classes
            encoding[label] = 1.0
            one_hot.append(encoding)
        
        return one_hot
    
    @staticmethod
    def train_test_split(data: List[List[float]], labels: List[List[float]], 
                         test_size: float = 0.2, shuffle: bool = True) -> Tuple[List[List[float]], List[List[float]], List[List[float]], List[List[float]]]:
        """Dzieli dane na zbiory treningowy i testowy."""
        if shuffle:
            # Przemieszanie danych
            combined = list(zip(data, labels))
            random.shuffle(combined)
            data, labels = zip(*combined)
            data, labels = list(data), list(labels)
        
        # Podział na zbiory
        split_idx = int(len(data) * (1 - test_size))
        x_train = data[:split_idx]
        x_test = data[split_idx:]
        y_train = labels[:split_idx]
        y_test = labels[split_idx:]
        
        return x_train, x_test, y_train, y_test
    
    @staticmethod
    def batch_generator(x: List[List[float]], y: List[List[float]], batch_size: int, shuffle: bool = True) -> Tuple[List[List[float]], List[List[float]]]:
        """Generator batchów danych."""
        if shuffle:
            # Przemieszanie danych
            combined = list(zip(x, y))
            random.shuffle(combined)
            x, y = zip(*combined)
            x, y = list(x), list(y)
        
        # Generowanie batchów
        for i in range(0, len(x), batch_size):
            yield x[i:i+batch_size], y[i:i+batch_size]


# ---------------------- PRZYKŁAD UŻYCIA ---------------------- #

def example_xor():
    """Przykład uczenia sieci na problemie XOR."""
    # Dane XOR
    x_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y_data = [[0], [1], [1], [0]]
    
    # Tworzenie modelu
    model = NeuralNetwork()
    model.add(Linear(2, 4))
    model.add(Activation("relu"))
    model.add(Linear(4, 1))
    model.add(Activation("sigmoid"))
    
    # Kompilacja modelu
    model.compile(loss="binary_cross_entropy", learning_rate=0.05, optimizer="adam")
    
    # Trening modelu
    history = model.fit(x_data, y_data, epochs=1000, batch_size=4, verbose=False)
    
    # Testowanie modelu
    predictions = model.predict(x_data)
    print("\nXOR Problem Results:")
    for i in range(len(x_data)):
        print(f"Input: {x_data[i]}, Target: {y_data[i][0]}, Prediction: {predictions[i][0]:.4f}")
    
    return model


if __name__ == "__main__":
    # Przykład użycia
    print("ModularAdaptiveNeuralMatrix - Przykład użycia")
    model = example_xor()
    model.summary()