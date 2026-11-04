"""Extract renewable projects from a folder of PDFs via BAML."""

import base64
import json
import sys
from pathlib import Path

from baml_client import b
from baml_py import Pdf


def main() -> None:
    docs = Path(sys.argv[1] if len(sys.argv) > 1 else "docs_projects")
    out = Path("data/raw/projects_extracted.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for pdf_path in sorted(docs.glob("*.pdf")):
            b64 = base64.standard_b64encode(pdf_path.read_bytes()).decode()
            projects = b.ExtractRenewableProjects(Pdf.from_base64(b64))
            for proj in projects:
                rec = {"source_file": pdf_path.name, **proj.model_dump()}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"{pdf_path.name}: {len(projects)} projects")


if __name__ == "__main__":
    main()
