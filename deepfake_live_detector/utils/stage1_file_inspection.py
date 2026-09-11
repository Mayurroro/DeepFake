import hashlib
import os

try:
    import magic
    MAGIC_AVAILABLE = True
except Exception:
    MAGIC_AVAILABLE = False


def inspect_file(file_path: str) -> dict:
    """
    Stage 1: Preserve and identify the file.
    - Hashes the file
    - Identifies container format using libmagic
    - Gets basic file stats
    """
    if not os.path.exists(file_path):
        return {"error": "File not found"}

    stats = os.stat(file_path)
    
    # Hash the original file
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
            
    try:
        if MAGIC_AVAILABLE:
            mime_type = magic.from_file(file_path, mime=True)
            file_desc = magic.from_file(file_path)
        else:
            mime_type = "unknown"
            file_desc = "python-magic not available"
    except Exception as e:
        mime_type = "unknown"
        file_desc = str(e)

    return {
        "file_name": os.path.basename(file_path),
        "file_size_bytes": stats.st_size,
        "sha256": sha256_hash.hexdigest(),
        "mime_type": mime_type,
        "description": file_desc,
        "creation_time": stats.st_ctime,
        "modification_time": stats.st_mtime
    }
