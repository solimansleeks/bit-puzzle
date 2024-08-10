import math
import random
import numpy as np
from sklearn.neural_network import MLPClassifier
from src.utils.bitcoin_utils import public_key_to_address, private_key_to_public_key

class MathematicalConstantsSolver:
    def __init__(self):
        self.constants = [math.pi, math.e, math.sqrt(2), (1 + math.sqrt(5)) / 2]  # Golden ratio
        self.model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000)

    def solve(self, target_address, public_key, difficulty):
        X_train, y_train = self.prepare_training_data(target_address, public_key, difficulty)
        self.model.fit(X_train, y_train)
        
        for _ in range(1000000):  # Increase the number of attempts
            candidate = random.randint(0, 2**difficulty - 1)
            features = self.extract_features(candidate)
            if self.model.predict([features])[0] == 1:
                candidate_public_key = private_key_to_public_key(hex(candidate)[2:].zfill(64))
                if public_key_to_address(candidate_public_key) == target_address:
                    return candidate
        return None

    def prepare_training_data(self, target_address, public_key, difficulty):
        X = []
        y = []
        for _ in range(10000):  # Increase the number of samples
            private_key = random.randint(0, 2**difficulty - 1)
            features = self.extract_features(private_key)
            X.append(features)
            candidate_public_key = private_key_to_public_key(hex(private_key)[2:].zfill(64))
            generated_address = public_key_to_address(candidate_public_key)
            y.append(1 if generated_address == target_address else 0)
        return np.array(X), np.array(y)

    def extract_features(self, number):
        return [
            number / self.constants[0],
            number / self.constants[1],
            number / self.constants[2],
            number / self.constants[3],
            int(math.log2(number + 1)),
            number % 2,  # Parity
            sum(int(digit) for digit in str(number)),  # Digital sum
        ]