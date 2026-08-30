import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Set

# Configure logger for data governance alerts
logger = logging.getLogger("order_governance")

# Path to operational dataset
DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "orders.json"


# ============================================================================
# DATA GOVERNANCE & CLASSIFICATION REGISTRY (Derived from Data Dictionary)
# ============================================================================

# 1. Fields certified as safe to expose directly to the customer & LLM
CUSTOMER_SAFE_BASE_FIELDS: Set[str] = {
    "order_id",
    "status",
    "membership_tier",
    "placed_at",
    "status_updated_at",
    "customer_safe_message",
}

# 2. Delivery fields that are customer-safe BUT subject to Status Precedence rules
# (e.g., must be suppressed if order is cancelled or returned)
DELIVERY_FIELDS: Set[str] = {
    "carrier",
    "tracking_number",
    "shipped_at",
    "delivered_at",
    "estimated_delivery",
}

# 3. Item-level allowlist
ITEM_SAFE_FIELDS: Set[str] = {
    "name",
    "quantity",
    "final_sale",
}

# 4. Sensitive PII and Internal fields that must NEVER be exposed
INTERNAL_PII_FIELDS: Set[str] = {
    "customer",
    "internal",
    "risk_score",
    "warehouse_notes",
    "support_tags",
    "email",
    "shipping_address",
}


def load_orders() -> List[Dict[str, Any]]:
    # Loads raw order records from the operational dataset.
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Orders database not found at {DATA_PATH}")

    with open(DATA_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)

    return data.get("orders", [])


def sanitize_order_for_customer(order: Dict[str, Any]) -> Dict[str, Any]:
    
    # Policy-Driven Data Sanitization & Status Precedence Engine.
    # Instead of hardcoding manual dictionary keys, this engine dynamically
    # classifies every field in the order record:
    # - Passes CUSTOMER_SAFE_BASE_FIELDS directly.
    # - Evaluates DELIVERY_FIELDS against status precedence (cancelled/returned -> null).
    # - Blocks INTERNAL_PII_FIELDS permanently.
    # - Quarantines UNCLASSIFIED new fields by default and logs a governance warning.
    
    status = str(order.get("status", "")).lower()
    safe_order: Dict[str, Any] = {}
    unclassified_keys: List[str] = []

    # Dynamically classify and process every top-level field
    for key, value in order.items():
        if key in INTERNAL_PII_FIELDS:
            # Sensitive PII or internal field: strictly blocked
            continue

        elif key in CUSTOMER_SAFE_BASE_FIELDS:
            # Base customer-safe field: pass value
            safe_order[key] = value

        elif key in DELIVERY_FIELDS:
            # Delivery details: apply status precedence
            if status in ["cancelled", "returned"]:
                safe_order[key] = None
            else:
                safe_order[key] = value

        elif key == "items":
            # Dynamic item-level sanitization
            sanitized_items = []
            for raw_item in value if isinstance(value, list) else []:
                clean_item = {
                    item_k: item_v
                    for item_k, item_v in raw_item.items()
                    if item_k in ITEM_SAFE_FIELDS
                }
                # Default final_sale if not explicitly present
                clean_item.setdefault("final_sale", False)
                sanitized_items.append(clean_item)
            safe_order["items"] = sanitized_items

        else:
            # UNCLASSIFIED FIELD: A new field appeared in the database!
            # Fail-closed security: quarantine and alert
            unclassified_keys.append(key)
            logger.warning(
                f"[DATA GOVERNANCE ALERT] Unclassified field '{key}' detected in order {order.get('order_id')}. "
                f"Quarantined by default to prevent accidental data leaks."
            )

    # Status exception escalation flag
    if status == "exception":
        safe_order["requires_human_support"] = True

    return safe_order


def lookup_order(order_id: Optional[str]) -> Dict[str, Any]:
    """
    Main tool entrypoint for the agent to look up order status.
    Handles ID normalization, missing input prompts, and unknown order handoffs.
    """
    if not order_id or not str(order_id).strip():
        return {
            "found": False,
            "error": "Missing order ID. Please ask the customer to provide their order ID (e.g., ORD-1007).",
            "requires_human_support": False
        }

    normalized_id = str(order_id).strip().upper()
    orders = load_orders()

    for order in orders:
        if str(order.get("order_id", "")).strip().upper() == normalized_id:
            return {
                "found": True,
                "order": sanitize_order_for_customer(order)
            }

    return {
        "found": False,
        "error": f"Order {normalized_id} was not found. Please verify the order ID or contact customer support.",
        "requires_human_support": True
    }


def audit_order_schema() -> Dict[str, Any]:
    """
    Scans the entire database against our Classification Registry to verify all
    fields are accounted for and no unclassified fields exist.
    """
    orders = load_orders()
    all_top_level_keys = set()
    for order in orders:
        all_top_level_keys.update(order.keys())

    classified_keys = CUSTOMER_SAFE_BASE_FIELDS | DELIVERY_FIELDS | INTERNAL_PII_FIELDS | {"items"}
    unclassified = all_top_level_keys - classified_keys

    return {
        "total_orders": len(orders),
        "all_keys": sorted(list(all_top_level_keys)),
        "classified_keys": sorted(list(classified_keys)),
        "unclassified_keys": sorted(list(unclassified))
    }


if __name__ == "__main__":
    print("=== Dynamic Policy-Driven Order Lookup ===")

    # Test valid order lookup
    res = lookup_order(" ord-1007 ")
    print("\nLookup ' ord-1007 ':")
    print(json.dumps(res, indent=2))

    # Test cancelled order status precedence
    res_cancelled = lookup_order("ORD-1004")
    print("\nLookup 'ORD-1004' (Cancelled - ETA suppressed):")
    print(json.dumps(res_cancelled, indent=2))

    # Test Schema Audit
    print("\n=== Schema Audit Report ===")
    audit = audit_order_schema()
    print(f"Total orders: {audit['total_orders']}")
    print(f"All keys: {audit['all_keys']}")
    print(f"Unclassified keys: {audit['unclassified_keys']}") 