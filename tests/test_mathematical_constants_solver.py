import unittest
from src.solvers.mathematical_constants_solver import MathematicalConstantsSolver
from src.utils.bitcoin_utils import public_key_to_address, private_key_to_public_key

class TestMathematicalConstantsSolver(unittest.TestCase):
    def setUp(self):
        self.solver = MathematicalConstantsSolver()
        
        # Data for Bitcoin puzzle #64 (solved)
        self.private_key = "0000000000000000000000000000000000000000000000000000000000001249"
        self.public_key = "0456b3817434935db42afda0165de529b938cf67c7510168dbbe075f6b4da00f7769133b7c47ba4aba5eca84d4d7cdc644e231f5bb0adb7af34d1aec5c0891add9"
        self.target_address = "1BgGZ9tcN4rm9KBzDn7KprQz87SZ26SAMH"
        self.difficulty = 64

    def test_solve(self):
        result = self.solver.solve(self.target_address, self.public_key, self.difficulty)
        self.assertIsNotNone(result)
        self.assertEqual(public_key_to_address(private_key_to_public_key(hex(result)[2:].zfill(64))), self.target_address)

    def test_prepare_training_data(self):
        X_train, y_train = self.solver.prepare_training_data(self.target_address, self.public_key, self.difficulty)
        self.assertEqual(X_train.shape[1], 7)  # 7 features
        self.assertEqual(len(y_train), 10000)  # 10000 samples

    def test_public_key_to_address(self):
        address = public_key_to_address(self.public_key)
        self.assertEqual(address, self.target_address)

    def test_extract_features(self):
        number = int(self.private_key, 16)
        features = self.solver.extract_features(number)
        self.assertEqual(len(features), 7)  # 7 features
        self.assertIsInstance(features, list)
        for feature in features:
            self.assertIsInstance(feature, (int, float))

if __name__ == '__main__':
    unittest.main()