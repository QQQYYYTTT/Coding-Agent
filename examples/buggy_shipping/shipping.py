"""Shipping fee calculation used by the coding-agent demo project."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation


STANDARD_LIMIT_KG = Decimal("5")
HEAVY_LIMIT_KG = Decimal("20")

STANDARD_FEE = Decimal("5.00")
HEAVY_FEE = Decimal("12.00")
OVERSIZE_FEE = Decimal("20.00")


def _parse_weight(weight_kg: int | float | str | Decimal) -> Decimal:
    """Convert one supported weight value to Decimal and validate its range."""

    try:
        weight = Decimal(str(weight_kg))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError("weight_kg must be a number") from exc
    if not weight.is_finite() or weight <= 0:
        raise ValueError("weight_kg must be greater than zero")
    return weight


def calculate_shipping_fee(
    weight_kg: int | float | str | Decimal,
) -> Decimal:
    """Return the shipping fee for a positive package weight in kilograms."""

    weight = _parse_weight(weight_kg)

    # Apply the configured tiers from lightest to heaviest.
    if weight < STANDARD_LIMIT_KG:
        return STANDARD_FEE
    if weight < HEAVY_LIMIT_KG:
        return HEAVY_FEE
    return OVERSIZE_FEE
