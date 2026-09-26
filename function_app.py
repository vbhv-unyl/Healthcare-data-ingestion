import logging
import json
import uuid
import re
from urllib.parse import unquote, urlparse

import azure.functions as func

from domain.models import DocumentRecord
from settings import build_ingest_use_case, build_document_api_dependencies

app = func.FunctionApp()

_ingest_use_case = build_ingest_use_case()
_object_store, _structured_store = build_document_api_dependencies()

_UPLOAD_PATH_PATTERN = re.compile(r"^(?P<user_id>[^/]+)/(?P<doc_id>[^/]+)/(?P<filename>.+)$")


def _parse_blob_path(blob_url: str) -> tuple[str, str, str, str]:
    """
    (blob_path, user_id, doc_id, filename) from the full blob URL Event
    Grid reports in the event's data.url field. Returns blob_path relative
    to the container
    """
    parsed = urlparse(blob_url)
    _, _, blob_path = parsed.path.lstrip("/").partition("/")
    blob_path = unquote(blob_path)

    match = _UPLOAD_PATH_PATTERN.match(blob_path)
    if not match:
        raise ValueError(f"blob path does not match the uploads/ convention: {blob_path!r}")
    return blob_path, match.group("user_id"), match.group("doc_id"), match.group("filename")


@app.function_name(name="ingest_document_on_blob_created")
@app.event_grid_trigger(arg_name="event")
def ingest_document_on_blob_created(event: func.EventGridEvent) -> None:
    if event.event_type != "Microsoft.Storage.BlobCreated":
        logging.info("Ignoring event_type=%s (not a blob creation)", event.event_type)
        return

    data = event.get_json() or {}
    blob_url = data.get("url", "")

    try:
        blob_path, user_id, doc_id, filename = _parse_blob_path(blob_url)
    except ValueError as exc:
        logging.info("Skipping blob event: %s", exc)
        return

    logging.info("doc=%s user=%s: ingestion triggered by blob at %s", doc_id, user_id, blob_path)

    try:
        _ingest_use_case.execute(doc_id=doc_id, user_id=user_id, blob_path=blob_path, filename=filename)
    except Exception:
        logging.exception("doc=%s: ingestion failed", doc_id)
        raise

@app.function_name(name="upload_document")
@app.route(route="documents", methods=["POST"], auth_level=func.AuthLevel.FUNCTION)
def upload_document(req: func.HttpRequest) -> func.HttpResponse:
    user_id = req.params.get("user_id")
    filename = req.params.get("filename")

    if not user_id or not filename:
        return func.HttpResponse(
            json.dumps({"error": "user_id and filename query parameters are both required"}),
            status_code=400,
            mimetype="application/json",
        )

    file_bytes = req.get_body()
    if not file_bytes:
        return func.HttpResponse(
            json.dumps({"error": "request body must contain the file's raw bytes"}),
            status_code=400,
            mimetype="application/json",
        )

    doc_id = uuid.uuid4().hex
    blob_path = f"{user_id}/{doc_id}/{filename}"
    content_type = req.headers.get("Content-Type") or "application/pdf"

    _object_store.upload(blob_path, file_bytes, content_type)
    _structured_store.save_document(
        DocumentRecord(doc_id=doc_id, user_id=user_id, filename=filename, blob_path=blob_path)
    )

    logging.info("doc=%s user=%s: uploaded via HTTP, blob_path=%s", doc_id, user_id, blob_path)

    return func.HttpResponse(
        json.dumps({"doc_id": doc_id, "status": "pending"}),
        status_code=202,
        mimetype="application/json",
    )


@app.function_name(name="get_document_status")
@app.route(route="documents/{doc_id}", methods=["GET"], auth_level=func.AuthLevel.FUNCTION)
def get_document_status(req: func.HttpRequest) -> func.HttpResponse:
    doc_id = req.route_params.get("doc_id")
    user_id = req.params.get("user_id")

    if not user_id:
        return func.HttpResponse(
            json.dumps({"error": "user_id query parameter is required"}),
            status_code=400,
            mimetype="application/json",
        )

    record = _structured_store.get_document(doc_id)

    if record is None or record.user_id != user_id:
        return func.HttpResponse(
            json.dumps({"error": "document not found"}),
            status_code=404,
            mimetype="application/json",
        )

    return func.HttpResponse(
        json.dumps(
            {
                "doc_id": record.doc_id,
                "filename": record.filename,
                "status": record.status.value,
                "error_message": record.error_message,
            }
        ),
        status_code=200,
        mimetype="application/json",
    )