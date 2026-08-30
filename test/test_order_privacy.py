import unittest
from pathlib import Path
import json

from src.tools.order_lookup import (
    load_orders,
    sanitize_order_for_customer,
    lookup_order,
    audit_order_schema,
    INTERNAL_PII_FIELDS,
)


class TestOrderLookupPrivacyAndPrecedence(unittest.TestCase):
    """
    Automated regression and privacy test suite (Pattern C).
    Verifies that no PII, sensitive internal fields, or stale delivery ETAs leak.
    """

    def setUp(self):
        self.raw_orders = load_orders()

    def test_schema_audit_verifies_all_keys_are_classified(self):
        """Verify that every key currently in orders.json is properly classified."""
        audit = audit_order_schema()
        self.assertGreater(audit["total_orders"], 0)
        # All keys in database should be classified (zero unclassified keys in current db)
        self.assertEqual(
            audit["unclassified_keys"],
            [],
            f"Found unclassified keys in database: {audit['unclassified_keys']}",
        )

    def test_no_order_leaks_pii_or_internal_data(self):
        """Iterate over all orders in orders.json and assert zero PII leakage."""
        for raw_order in self.raw_orders:
            sanitized = sanitize_order_for_customer(raw_order)

            # Check top-level keys
            for key in sanitized.keys():
                self.assertNotIn(
                    key.lower(),
                    INTERNAL_PII_FIELDS,
                    f"Found forbidden key '{key}' in sanitized order",
                )

            # Check string representation to catch nested leaks
            serialized = json.dumps(sanitized).lower()
            for forbidden in [
                "risk_score",
                "warehouse_notes",
                "@example.test",
                "fraud review",
            ]:
                self.assertNotIn(
                    forbidden,
                    serialized,
                    f"Found forbidden string '{forbidden}' leaked in order {raw_order.get('order_id')}",
                )

    def test_status_precedence_cancelled_order(self):
        """
        ORD-1004 is cancelled but contains an old ETA in raw data.
        Sanitized output MUST suppress estimated_delivery and carrier.
        """
        res = lookup_order("ORD-1004")
        self.assertTrue(res["found"])
        order = res["order"]
        self.assertEqual(order["status"], "cancelled")
        self.assertIsNone(
            order["estimated_delivery"],
            "Cancelled order must not expose estimated_delivery",
        )
        self.assertIsNone(order["carrier"], "Cancelled order must not expose carrier")
        self.assertIsNone(
            order["tracking_number"],
            "Cancelled order must not expose tracking_number",
        )

    def test_input_normalization(self):
        """Order ID lookup must handle whitespace and case differences."""
        res_lower = lookup_order("  ord-1007  ")
        res_upper = lookup_order("ORD-1007")

        self.assertTrue(res_lower["found"])
        self.assertEqual(res_lower, res_upper)

    def test_missing_and_unknown_order_handling(self):
        """Missing IDs prompt for input; unknown IDs trigger human support flag."""
        res_empty = lookup_order("")
        self.assertFalse(res_empty["found"])
        self.assertIn("Missing order ID", res_empty["error"])
        self.assertFalse(res_empty["requires_human_support"])

        res_unknown = lookup_order("ORD-9999")
        self.assertFalse(res_unknown["found"])
        self.assertTrue(res_unknown["requires_human_support"])

    def test_new_unknown_field_quarantine_by_default(self):
        """
        Simulate a future scenario where a new unknown field is added to the database.
        The dynamic sanitization engine MUST quarantine it by default (fail-closed).
        """
        mock_order_with_new_fields = {
            "order_id": "ORD-TEST",
            "status": "processing",
            "future_experimental_field": "secret_data_123",
            "warehouse_shelf_number": "A-42",
        }
        sanitized = sanitize_order_for_customer(mock_order_with_new_fields)
        self.assertNotIn(
            "future_experimental_field",
            sanitized,
            "Unclassified new field was not quarantined!",
        )
        self.assertNotIn(
            "warehouse_shelf_number",
            sanitized,
            "Unclassified warehouse field was not quarantined!",
        )
        self.assertIn("order_id", sanitized)


if __name__ == "__main__":
    unittest.main()
