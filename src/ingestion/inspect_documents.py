from pathlib import Path
import frontmatter


# Location of the knowledge-base folder
KB_PATH = Path("knowledge-base")


# Find all Markdown files
documents =  sorted(KB_PATH.glob("*.md"))


print(f"Found {len(documents)} documents\n")


# Read every document
for file_path in documents:

    post = frontmatter.load(file_path)
    print("=" * 70)
    print(f"File: {file_path.name}")
    print(f"Metadata: {post.metadata.get('title')}")
    print(f"Status: {post.metadata.get('status')}")
    print(f"Audience: {post.metadata.get('audience')}")
    print(f"Authority: {post.metadata.get('policy_authority')}")

    print("\nHeadings:")

    for line in post.content.splitlines():
        line = line.strip()
        if line.startswith("#"):
            print(f" {line}")

    print()