import os

try:
    import exiftool
    EXIFTOOL_AVAILABLE = True
except ImportError:
    EXIFTOOL_AVAILABLE = False


def extract_metadata(file_path: str) -> dict:
    """
    Stage 3: Extract metadata.
    - Uses ExifTool to extract EXIF, IPTC, XMP, ICC profiles, etc.
    """
    if not os.path.exists(file_path):
        return {"error": "File not found"}

    result = {
        "metadata_found": False,
        "metadata_keys": [],
        "exiftool_supported": EXIFTOOL_AVAILABLE,
        "raw_metadata": {}
    }

    if not EXIFTOOL_AVAILABLE:
        result["error"] = "PyExifTool is not installed or exiftool executable is not in PATH."
        return result

    try:
        with exiftool.ExifToolHelper() as et:
            metadata = et.get_metadata(file_path)
            if metadata and len(metadata) > 0:
                result["metadata_found"] = True
                # Return the first (and only) file's metadata
                file_meta = metadata[0]
                result["raw_metadata"] = file_meta
                result["metadata_keys"] = list(file_meta.keys())
    except Exception as e:
        result["error"] = str(e)

    return result
