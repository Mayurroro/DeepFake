import os
import json

try:
    from c2pa import Reader, C2paError
    C2PA_AVAILABLE = True
except ImportError:
    C2PA_AVAILABLE = False


def validate_provenance(file_path: str) -> dict:
    """
    Stage 2: Validate provenance (C2PA)
    - Parses C2PA manifests
    - Checks signatures and content bindings
    """
    if not os.path.exists(file_path):
        return {"error": "File not found"}

    result = {
        "c2pa_manifest_found": False,
        "is_valid": False,
        "manifest_data": None,
        "c2pa_supported": C2PA_AVAILABLE
    }

    if not C2PA_AVAILABLE:
        result["error"] = "c2pa-python is not installed"
        return result

    try:
        # c2pa-python 0.37+ API: Reader.try_create returns the reader or None
        reader = Reader.try_create(file_path)
        if reader is None:
            result["c2pa_manifest_found"] = False
            return result

        result["c2pa_manifest_found"] = True
        manifest_json = reader.json()
        if manifest_json:
            try:
                result["manifest_data"] = json.loads(manifest_json)
            except Exception:
                result["manifest_data"] = manifest_json

        # Built-in validity check + validation issues
        result["is_valid"] = bool(reader.is_valid)
        validation = reader.get_validation_results()
        if validation:
            issues = validation.get("validation_status") or validation.get("issues") or []
            if issues:
                result["validation_status"] = issues
                result["is_valid"] = len([x for x in issues
                                          if x.get("code") == "validation.sts.failed"]) == 0

        reader.close()

    except C2paError as e:
        result["c2pa_manifest_found"] = False
        result["error"] = str(e)
    except Exception as e:
        result["c2pa_manifest_found"] = False
        result["error"] = str(e)

    return result