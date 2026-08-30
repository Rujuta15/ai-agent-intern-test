from pathlib import Path
from typing import List, Dict, Any
import frontmatter

# Dynamic path to knowledge-base directory
KB_DIR = Path(__file__).resolve().parents[2] / "knowledge-base"


class MetadataPolicy:
    """
    Dynamic governance and classification policy for document frontmatter.
    Evaluates document authority and citation formatting without hardcoding logic across the app.
    """

    @staticmethod
    def is_customer_authoritative(metadata: Dict[str, Any]) -> bool:
        """
        Determines whether a document chunk is authoritative for public customer queries.
        Returns False for legacy policies, internal notes, or draft documents.
        """
        status = str(metadata.get("status", "active")).lower()
        audience = str(metadata.get("audience", "customer")).lower()
        authority = str(metadata.get("policy_authority", "official")).lower()
        customer_answering = str(metadata.get("customer_answering", "allowed")).lower()

        # Rule 1: Legacy / superseded policies are not current authority
        if status in ["legacy", "superseded", "archived"]:
            return False

        # Rule 2: Internal notes and disallowed content must not be used as customer authority
        if audience == "internal" or customer_answering == "disallowed":
            return False

        # Rule 3: Draft policies are not official
        if authority == "draft":
            return False

        return True

    @staticmethod
    def format_citation(chunk: Dict[str, Any]) -> str:
        """
        Dynamically formats the required citation string (file_name + heading).
        """
        return f"{chunk['file_name']} > {chunk['heading']}"


def chunk_document(file_path: Path) -> List[Dict[str, Any]]:
    
    # Parses a Markdown document with frontmatter and splits it into section-level chunks.
    # Preserves 100% of all existing and future YAML frontmatter metadata dynamically.
    
    post = frontmatter.load(file_path)

    chunks = []
    current_heading = None
    current_content = []

    for line in post.content.splitlines():
        line = line.strip()

        # Detect Markdown heading
        if line.startswith("#"):
            if current_heading and current_content:
                chunks.append({
                    "file_name": file_path.name,
                    "heading": current_heading.lstrip("#").strip(),
                    "raw_heading": current_heading,
                    "content": "\n".join(current_content).strip(),
                    # Store 100% of all frontmatter keys dynamically
                    "metadata": dict(post.metadata)
                })

            current_heading = line
            current_content = []
        elif line:
            current_content.append(line)

    # Save final section
    if current_heading and current_content:
        chunks.append({
            "file_name": file_path.name,
            "heading": current_heading.lstrip("#").strip(),
            "raw_heading": current_heading,
            "content": "\n".join(current_content).strip(),
            "metadata": dict(post.metadata)
        })

    return chunks


def load_all_chunks(kb_path: Path = KB_DIR) -> List[Dict[str, Any]]:
   
    # Loads and chunks all Markdown documents in the knowledge-base directory.
    
    if not kb_path.exists():
        raise FileNotFoundError(f"Knowledge base directory not found at: {kb_path}")

    all_chunks = []
    for file_path in sorted(kb_path.glob("*.md")):
        all_chunks.extend(chunk_document(file_path))

    return all_chunks


if __name__ == "__main__":
    print("=== Ingesting & Chunking Knowledge Base (Dynamic Metadata) ===")
    all_chunks = load_all_chunks()
    print(f"Total chunks created: {len(all_chunks)}\n")

    for i, chunk in enumerate(all_chunks[:2], start=1):
        print(f"--- Chunk {i} ---")
        print(f"Citation:     {MetadataPolicy.format_citation(chunk)}")
        print(f"Authoritative: {MetadataPolicy.is_customer_authoritative(chunk['metadata'])}")
        print(f"All Metadata Keys: {list(chunk['metadata'].keys())}")
        print(f"Content:      {chunk['content'][:100]}...\n")