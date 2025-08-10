import unittest
from calculator import add, divide


class TestCalculator(unittest.TestCase):
    def test_add(self):
        self.assertEqual(add(2, 3), 5)
        self.assertEqual(add(-1, 1), 0)

    def test_divide(self):
        self.assertEqual(divide(10, 2), 5)
        self.assertAlmostEqual(divide(3, 2), 1.5)
        with self.assertRaises(ValueError):
            divide(4, 0)


if __name__ == '__main__':
    unittest.main()
