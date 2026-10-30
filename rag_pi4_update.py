
#!/usr/bin/env python3
"""AAI RAG Incremental Update + Automatic OCR for Raspberry Pi 4."""

import argparse
import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import tempfile
from pathlib import Path

import rag_pi4 as rag

DOC_EXTENSIONS = {".pdf", ".txt", ".md"}
IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"
}
EXTENSIONS = DOC_EXTENSIONS | IMAGE_EXTENSIONS

PIPELINE_VERSION = "auto-ocr-v1-tha+eng"
CACHE_FOLDER = Path(__file__).resolve().with_name(".aai_ocr_cache")
MIN_TEXT_LENGTH = 25


class OCRFailure(RuntimeError):
    pass


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024), b""
        ):
            digest.update(block)
    return digest.hexdigest()


def files_to_update(path):
    path = path.expanduser().resolve()

    if not path.exists():
        raise ValueError(f"ไม่พบไฟล์หรือโฟลเดอร์: {path}")

    if path.is_dir():
        root = path
        files = sorted(
            (
                p for p in root.rglob("*")
                if p.is_file()
                and p.suffix.lower() in EXTENSIONS
            ),
            key=lambda p: p.as_posix().casefold(),
        )
    else:
        if path.suffix.lower() not in EXTENSIONS:
            raise ValueError("ไม่รองรับไฟล์ประเภทนี้")

        root = next(
            (
                parent for parent in path.parents
                if parent.name == "docs_aai"
            ),
            path.parent,
        )
        files = [path]

    return root, files


def load_manifest(root):
    path = root / "sources.json"

    if not path.exists():
        return {}

    manifest = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(manifest, dict):
        raise ValueError("sources.json ต้องเป็น JSON object")

    return manifest


def expected_source(path, root, manifest):
    relative = path.relative_to(root).as_posix()
    entry = manifest.get(relative, {})

    source = (
        entry.get("source_url", relative)
        if isinstance(entry, dict)
        else relative
    )

    if path.suffix.lower() in {".txt", ".md"}:
        with path.open("r", encoding="utf-8-sig") as handle:
            for _ in range(10):
                line = handle.readline()

                if not line:
                    break

                match = re.match(
                    r"^SOURCE_URL:\s*(https?://\S+)\s*$",
                    line.strip(),
                )

                if match:
                    return match.group(1)

    return source


def run_command(command, tool_name):
    if not shutil.which(command[0]):
        raise OCRFailure(
            f"ไม่พบ {tool_name}. "
            "ติดตั้งด้วย sudo apt install "
            "ocrmypdf tesseract-ocr-tha tesseract-ocr-eng"
        )

    try:
        return subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        detail = (
            exc.stderr or exc.stdout or "unknown error"
        ).strip()

        raise OCRFailure(
            f"{tool_name} ทำงานไม่สำเร็จ: {detail[-850:]}"
        ) from exc


def cache_path(path, root, digest, extension):
    relative = path.relative_to(root).as_posix()

    identity = hashlib.sha256(
        relative.encode("utf-8")
    ).hexdigest()[:18]

    version = hashlib.sha256(
        PIPELINE_VERSION.encode("utf-8")
    ).hexdigest()[:8]

    return (
        CACHE_FOLDER
        / f"{identity}_{digest[:20]}_{version}{extension}"
    )


def pdf_needs_ocr(path):
    from pypdf import PdfReader

    reader = PdfReader(str(path))

    if not reader.pages:
        return False

    for page in reader.pages:
        text = re.sub(
            r"\s+", "", page.extract_text() or ""
        )

        if len(text) < MIN_TEXT_LENGTH:
            return True

    return False


def ocr_pdf(path, root, digest, force=False):
    target = cache_path(path, root, digest, ".pdf")

    if (
        target.exists()
        and target.stat().st_size > 0
        and not force
    ):
        print(f"[ocr-cache] {path.name}")
        return target

    CACHE_FOLDER.mkdir(parents=True, exist_ok=True)

    fd, temp_name = tempfile.mkstemp(
        prefix="ocr_",
        suffix=".pdf",
        dir=CACHE_FOLDER,
    )
    os.close(fd)

    temp = Path(temp_name)
    temp.unlink(missing_ok=True)

    try:
        print(f"[ocr-pdf] {path.name}")
        print("  กำลัง OCR ภาษาไทยและอังกฤษ...")

        mode = "--force-ocr" if force else "--skip-text"

        run_command(
            [
                "ocrmypdf",
                "-l", "tha+eng",
                mode,
                "--rotate-pages",
                "--deskew",
                "--jobs", "1",
                "--optimize", "0",
                str(path),
                str(temp),
            ],
            "OCRmyPDF",
        )

        if not temp.is_file() or temp.stat().st_size == 0:
            raise OCRFailure("OCR ไม่สร้างไฟล์ผลลัพธ์")

        os.replace(temp, target)

    finally:
        temp.unlink(missing_ok=True)

    return target


def ocr_image(path, root, digest, force=False):
    target = cache_path(path, root, digest, ".txt")

    if target.exists() and not force:
        print(f"[ocr-cache] {path.name}")
        return target.read_text(encoding="utf-8")

    print(f"[ocr-image] {path.name}")

    result = run_command(
        [
            "tesseract",
            str(path),
            "stdout",
            "-l", "tha+eng",
        ],
        "Tesseract",
    )

    CACHE_FOLDER.mkdir(parents=True, exist_ok=True)

    fd, temp_name = tempfile.mkstemp(
        prefix="ocr_",
        suffix=".txt",
        dir=CACHE_FOLDER,
    )

    try:
        with os.fdopen(
            fd, "w", encoding="utf-8"
        ) as handle:
            handle.write(result.stdout)

        os.replace(temp_name, target)

    finally:
        Path(temp_name).unlink(missing_ok=True)

    return result.stdout


def content_records(
    path, root, manifest, digest, force_ocr=False
):
    source = expected_source(path, root, manifest)
    suffix = path.suffix.lower()

    records = []
    ocr_used = False

    if suffix in IMAGE_EXTENSIONS:
        pages = [
            (
                None,
                ocr_image(
                    path, root, digest, force=force_ocr
                ),
            )
        ]
        ocr_used = True

    else:
        read_path = path

        if suffix == ".pdf":
            if force_ocr or pdf_needs_ocr(path):
                read_path = ocr_pdf(
                    path, root, digest, force=force_ocr
                )
                ocr_used = True

        pages = rag.iter_pages(read_path)

    for page, text in pages:
        match = re.search(
            r"(?m)^SOURCE_URL:\s*(https?://\S+)\s*$",
            text,
        )

        actual_source = (
            match.group(1) if match else source
        )

        if actual_source != source and records:
            raise ValueError(
                "เอกสารหนึ่งไฟล์มีหลาย SOURCE_URL"
            )

        source = actual_source

        for chunk in rag.split_text(text):
            records.append((source, page, chunk))

    return source, records, ocr_used


def metadata_db(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS indexed_files (
            file_path TEXT PRIMARY KEY,
            sha256 TEXT NOT NULL,
            source TEXT NOT NULL,
            chunk_count INTEGER NOT NULL,
            pipeline_version TEXT
        )
    """)

    columns = {
        row[1]
        for row in conn.execute(
            "PRAGMA table_info(indexed_files)"
        )
    }

    if "pipeline_version" not in columns:
        conn.execute(
            "ALTER TABLE indexed_files "
            "ADD COLUMN pipeline_version TEXT"
        )

    conn.commit()


def verify_embedding_model(conn):
    row = conn.execute(
        "SELECT value FROM settings "
        "WHERE name='embed_model'"
    ).fetchone()

    count = conn.execute(
        "SELECT COUNT(*) FROM chunks"
    ).fetchone()[0]

    if row and row[0] != rag.EMBED_MODEL:
        raise RuntimeError(
            "Embedding Model ไม่ตรงกับฐานข้อมูลเดิม"
        )

    if not row and count:
        raise RuntimeError(
            "ฐานข้อมูลเดิมไม่มีข้อมูล Embedding Model"
        )

    if not row:
        with conn:
            conn.execute(
                "INSERT INTO settings(name,value) "
                "VALUES('embed_model',?)",
                (rag.EMBED_MODEL,),
            )


def legacy_matches(conn, source, records):
    current = conn.execute(
        "SELECT page, content FROM chunks "
        "WHERE source=? ORDER BY id",
        (source,),
    ).fetchall()

    expected = [
        (page, text)
        for _, page, text in records
    ]

    return bool(current) and current == expected


def update_documents(path, force_ocr=False):
    root, files = files_to_update(path)

    if not files:
        print("ไม่พบไฟล์ที่รองรับ")
        return

    manifest = load_manifest(root)

    if not rag.DB_PATH.exists():
        raise RuntimeError(
            "ไม่พบฐานข้อมูลเดิม aai_rag.sqlite3"
        )

    conn = rag.make_db()

    added = 0
    updated = 0
    adopted = 0
    skipped = 0
    empty = 0
    errors = 0
    ocr_count = 0

    try:
        metadata_db(conn)
        verify_embedding_model(conn)

        for path in files:
            relative = path.relative_to(root).as_posix()

            try:
                digest = file_hash(path)

                previous = conn.execute(
                    "SELECT sha256,source,chunk_count,"
                    "pipeline_version FROM indexed_files "
                    "WHERE file_path=?",
                    (relative,),
                ).fetchone()

                intended_source = expected_source(
                    path, root, manifest
                )

                count_existing = (
                    conn.execute(
                        "SELECT COUNT(*) FROM chunks "
                        "WHERE source=?",
                        (previous[1],),
                    ).fetchone()[0]
                    if previous else 0
                )

                if (
                    not force_ocr
                    and previous
                    and previous[0] == digest
                    and previous[1] == intended_source
                    and count_existing == previous[2]
                    and previous[3] == PIPELINE_VERSION
                ):
                    print(f"[skip] {relative}")
                    skipped += 1
                    continue

                source, records, ocr_used = content_records(
                    path,
                    root,
                    manifest,
                    digest,
                    force_ocr=force_ocr,
                )

                if ocr_used:
                    ocr_count += 1

                if not records:
                    print(
                        f"[warning] {relative} "
                        "ไม่มีข้อความที่อ่านได้"
                    )
                    empty += 1
                    continue

                collision = conn.execute(
                    "SELECT file_path FROM indexed_files "
                    "WHERE source=? AND file_path<>?",
                    (source, relative),
                ).fetchone()

                if collision:
                    print(
                        f"[warning] {relative} "
                        "SOURCE_URL ซ้ำ ข้ามไฟล์"
                    )
                    skipped += 1
                    continue

                if legacy_matches(conn, source, records):
                    with conn:
                        conn.execute(
                            "INSERT INTO indexed_files "
                            "(file_path,sha256,source,"
                            "chunk_count,pipeline_version) "
                            "VALUES(?,?,?,?,?) "
                            "ON CONFLICT(file_path) DO UPDATE SET "
                            "sha256=excluded.sha256,"
                            "source=excluded.source,"
                            "chunk_count=excluded.chunk_count,"
                            "pipeline_version="
                            "excluded.pipeline_version",
                            (
                                relative,
                                digest,
                                source,
                                len(records),
                                PIPELINE_VERSION,
                            ),
                        )

                    adopted += 1
                    print(
                        f"[existing] {relative}: "
                        f"{len(records)} chunks"
                    )
                    continue

                embedded = []

                for start in range(
                    0, len(records), rag.BATCH_SIZE
                ):
                    batch = records[
                        start:start + rag.BATCH_SIZE
                    ]

                    texts = [
                        f"title: {src} | text: {text}"
                        for src, _, text in batch
                    ]

                    last = (
                        start + rag.BATCH_SIZE
                        >= len(records)
                    )

                    vectors = rag.embed_inputs(
                        texts,
                        keep_alive=0 if last else "5m",
                    )

                    if len(vectors) != len(batch):
                        raise RuntimeError(
                            "Embedding จำนวนไม่ตรงกับ Chunk"
                        )

                    embedded.extend(
                        (
                            src,
                            page,
                            body,
                            json.dumps(vector),
                        )
                        for (src, page, body), vector
                        in zip(batch, vectors)
                    )

                    progress = min(
                        start + rag.BATCH_SIZE,
                        len(records),
                    )

                    print(
                        f"  {relative}: "
                        f"{progress}/{len(records)} chunks"
                    )

                with conn:
                    if previous and previous[1] != source:
                        conn.execute(
                            "DELETE FROM chunks WHERE source=?",
                            (previous[1],),
                        )

                    conn.execute(
                        "DELETE FROM chunks WHERE source=?",
                        (source,),
                    )

                    conn.executemany(
                        "INSERT INTO chunks "
                        "(source,page,content,embedding) "
                        "VALUES(?,?,?,?)",
                        embedded,
                    )

                    conn.execute(
                        "INSERT INTO indexed_files "
                        "(file_path,sha256,source,"
                        "chunk_count,pipeline_version) "
                        "VALUES(?,?,?,?,?) "
                        "ON CONFLICT(file_path) DO UPDATE SET "
                        "sha256=excluded.sha256,"
                        "source=excluded.source,"
                        "chunk_count=excluded.chunk_count,"
                        "pipeline_version="
                        "excluded.pipeline_version",
                        (
                            relative,
                            digest,
                            source,
                            len(records),
                            PIPELINE_VERSION,
                        ),
                    )

                if previous:
                    updated += 1
                    print(
                        f"[updated] {relative}: "
                        f"{len(records)} chunks"
                    )
                else:
                    added += 1
                    print(
                        f"[added] {relative}: "
                        f"{len(records)} chunks"
                    )

            except (
                OCRFailure,
                OSError,
                ValueError,
                RuntimeError,
                sqlite3.Error,
            ) as exc:
                print(f"[error] {relative}: {exc}")
                print("  ข้อมูลเดิมยังคงอยู่")
                errors += 1

    finally:
        conn.close()

    print("\n========== SUMMARY ==========")
    print(f"เพิ่มใหม่: {added}")
    print(f"อัปเดต: {updated}")
    print(f"มีอยู่แล้ว: {adopted}")
    print(f"ข้าม: {skipped}")
    print(f"อ่านข้อความไม่ได้: {empty}")
    print(f"ผิดพลาด: {errors}")
    print(f"ทำ OCR: {ocr_count}")
    print(f"Database: {rag.DB_PATH}")
    print("=============================")

    if errors:
        raise RuntimeError(
            f"มี {errors} ไฟล์ที่อัปเดตไม่สำเร็จ"
        )


def main():
    parser = argparse.ArgumentParser(
        description="AAI RAG Auto OCR Incremental Update"
    )

    parser.add_argument(
        "path",
        type=Path,
        help="โฟลเดอร์ docs_aai หรือไฟล์เดียว",
    )

    parser.add_argument(
        "--force-ocr",
        action="store_true",
        help="บังคับ OCR ใหม่",
    )

    args = parser.parse_args()

    try:
        update_documents(
            args.path,
            force_ocr=args.force_ocr,
        )

    except (
        ValueError,
        RuntimeError,
        OSError,
        sqlite3.Error,
    ) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
