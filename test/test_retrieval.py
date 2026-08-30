import sys
from pathlib import Path
import unittest

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.retriever import HybridRetriever


class TestHybridRetriever(unittest.TestCase):
    """
    Automated regression tests for knowledge base retrieval,
    document authority precedence, and conflict detection.
    """

    @classmethod
    def setUpClass(cls):
        cls.retriever = HybridRetriever()

    def test_standard_return_window_retrieves_current_policy(self):
        """Must retrieve active returns policy (01) and NOT legacy policy (02)."""
        results = self.retriever.retrieve(
            "How long does a regular customer have to return an unused backpack?",
            top_k=3
        )
        sources = [r["file_name"] for r in results]
        self.assertIn("01-returns-policy-current.md", sources)
        self.assertNotIn("02-returns-policy-legacy.md", sources)
        self.assertNotIn("14-internal-content-migration-notes.md", sources)

    def test_trailplus_return_window(self):
        """Must prioritize TrailPlus membership policy."""
        results = self.retriever.retrieve(
            "My TrailPlus membership was active when I ordered. What is my return window?",
            top_k=2
        )
        self.assertEqual(results[0]["file_name"], "09-trailplus-membership.md")
        self.assertIn("Return window", results[0]["heading"])

    def test_damaged_final_sale_item_multi_source(self):
        """Must retrieve both final sale and damaged items policies."""
        results = self.retriever.retrieve(
            "A final-sale bag arrived with a broken zipper yesterday. Am I completely out of luck?",
            top_k=3
        )
        sources = [r["file_name"] for r in results]
        self.assertIn("04-damaged-or-wrong-items.md", sources)
        self.assertIn("03-final-sale-and-promotions.md", sources)

    def test_breeze_tumbler_conflict_sources(self):
        """Must retrieve both conflicting sources for dishwasher safety."""
        results = self.retriever.retrieve(
            "Can I put the entire Breeze Tumbler in the dishwasher?",
            top_k=2
        )
        sources = [r["file_name"] for r in results]
        self.assertIn("11-product-care.md", sources)
        self.assertIn("12-breeze-tumbler-product-card.md", sources)

    def test_unsupported_destination_germany(self):
        """Must retrieve international shipping destinations policy."""
        results = self.retriever.retrieve(
            "Can you ship an Atlas Weekender to Germany?",
            top_k=2
        )
        self.assertEqual(results[0]["file_name"], "06-international-shipping.md")
        self.assertEqual(results[0]["heading"], "Supported destinations")

    def test_warranty_periods(self):
        """Must retrieve limited warranty policy."""
        results = self.retriever.retrieve(
            "Do all Aster & Row products have a lifetime warranty?",
            top_k=2
        )
        self.assertEqual(results[0]["file_name"], "07-warranty.md")
        self.assertIn("Warranty periods", results[0]["heading"])


if __name__ == "__main__":
    unittest.main()

