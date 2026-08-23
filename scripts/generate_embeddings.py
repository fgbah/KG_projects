# generate_embeddings.py
# Generate embeddings from text and PDF files using sentence-transformers
# Requirements:
#   pip install sentence-transformers numpy tqdm PyPDF2
# Usage:
#   python generate_embeddings.py --input-dir ./docs --output-dir ./embeddings --model all-MiniLM-L6-v2

import os
import argparse
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np

from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader


def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def read_pdf_file(path: Path) -> str:
    text = []
    try:
        reader = PdfReader(str(path))
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)
    except Exception:
        return ""
    return "\n".join(text)


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50):
    if not text:
        return []
    tokens = text.split()
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk = " ".join(tokens[start:end])
        chunks.append(chunk)
        if end == len(tokens):
            break
        start = end - overlap
    return chunks


def discover_files(input_dir: Path):
    ext_map = {".txt", ".md", ".pdf"}
    for p in input_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in ext_map:
            yield p


def main(args):
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = SentenceTransformer(args.model)
    metadata_out = output_dir / "metadata.jsonl"
    embeddings_out = output_dir / "embeddings.npy"

    metas = []
    all_embeddings = []

    file_paths = list(discover_files(input_dir))
    if not file_paths:
        print("No input files found in", input_dir)
        return

    for fp in tqdm(file_paths, desc="Files"):
        suffix = fp.suffix.lower()
        if suffix in {".txt", ".md"}:
            text = read_text_file(fp)
        elif suffix == ".pdf":
            text = read_pdf_file(fp)
        else:
            text = ""
        chunks = chunk_text(text, chunk_size=args.chunk_size, overlap=args.overlap)
        if not chunks:
            continue
        # process in batches
        for i in range(0, len(chunks), args.batch_size):
            batch = chunks[i:i + args.batch_size]
            emb = model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
            for j, e in enumerate(emb):
                chunk_index = i + j
                meta = {
                    "id": f"{fp.name}__{chunk_index}",
                    "source": str(fp.relative_to(input_dir)),
                    "chunk_index": chunk_index,
                    "text_snippet": batch[j][:1000]
                }
                metas.append(meta)
                all_embeddings.append(e)

    if all_embeddings:
        embeddings_arr = np.vstack(all_embeddings)
        np.save(embeddings_out, embeddings_arr)
        with metadata_out.open("w", encoding="utf-8") as f:
            for m in metas:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")
        print("Saved embeddings:", embeddings_out)
        print("Saved metadata:", metadata_out)
    else:
        print("No embeddings generated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate embeddings from text/pdf files using sentence-transformers")
    parser.add_argument("--input-dir", required=True, help="Directory containing .txt, .md, .pdf files")
    parser.add_argument("--output-dir", required=True, help="Directory to write embeddings and metadata")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="SentenceTransformer model name")
    parser.add_argument("--chunk-size", type=int, default=500, help="Tokens per chunk")
    parser.add_argument("--overlap", type=int, default=50, help="Token overlap between chunks")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for embedding model")
    args = parser.parse_args()
    main(args)
