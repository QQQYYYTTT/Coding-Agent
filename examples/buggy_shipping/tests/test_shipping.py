from decimal import Decimal
import unittest

from shipping import calculate_shipping_fee


class CalculateShippingFeeTests(unittest.TestCase):
    def test_small_package_uses_standard_fee(self) -> None:
        self.assertEqual(calculate_shipping_fee("0.5"), Decimal("5.00"))

    def test_exactly_five_kg_stays_in_standard_tier(self) -> None:
        self.assertEqual(calculate_shipping_fee(5), Decimal("5.00"))

    def test_middle_weight_uses_heavy_fee(self) -> None:
        self.assertEqual(calculate_shipping_fee("12.75"), Decimal("12.00"))

    def test_exactly_twenty_kg_stays_in_heavy_tier(self) -> None:
        self.assertEqual(calculate_shipping_fee(20), Decimal("12.00"))

    def test_over_twenty_kg_uses_oversize_fee(self) -> None:
        self.assertEqual(calculate_shipping_fee(Decimal("20.01")), Decimal("20.00"))

    def test_zero_weight_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "greater than zero"):
            calculate_shipping_fee(0)

    def test_negative_weight_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "greater than zero"):
            calculate_shipping_fee(-1)

    def test_non_numeric_weight_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be a number"):
            calculate_shipping_fee("not-a-number")


if __name__ == "__main__":
    unittest.main()
