"""
Notes Backend API (FastAPI).

This service provides CRUD + search endpoints for a notes application, with optional tag support.
It persists data in the shared SQLite database managed by the `notes_database` container.

Database contract:
- SQLite file path is defined by notes_database's db_connection.txt guidance.
- This backend reads DB location from env var SQLITE_DB if provided (recommended for deployments),
  otherwise falls back to the canonical path from db_connection.txt:
  /home/kavia/workspace/code-generation/note-keeper-333122-333138/notes_database/myapp.db

Key API features:
- Create, read, update, delete notes
- Search notes by title/content with optional tag filtering
- Optional tags: list tags, set note tags, fetch note tags

Observability:
- Logs key operations with note_id/tag names for debuggability.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Sequence

from fastapi import FastAPI, HTTPException, Query, Response, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logger = logging.getLogger("notes_backend")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())

DEFAULT_SQLITE_PATH = (
    "/home/kavia/workspace/code-generation/note-keeper-333122-333138/notes_database/myapp.db"
)


def _get_sqlite_db_path() -> str:
    """Resolve SQLite DB path from environment, with a safe, documented fallback."""
    # NOTE: notes_database container advertises SQLITE_DB as its db_env_vars, so this is the
    # primary way to configure DB location across environments.
    return os.getenv("SQLITE_DB") or DEFAULT_SQLITE_PATH


@contextmanager
def _db_conn() -> sqlite3.Connection:
    """Context-managed SQLite connection with required pragmas enabled."""
    db_path = _get_sqlite_db_path()
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        yield conn
        conn.commit()
    except sqlite3.Error as e:
        # Add contextual information for debugging operational issues.
        logger.exception("SQLite operation failed db_path=%s error=%s", db_path, str(e))
        raise
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _row_to_note_dict(row: sqlite3.Row) -> Dict[str, Any]:
    """Convert a notes row to API dict (without tags)."""
    return {
        "id": int(row["id"]),
        "title": str(row["title"]),
        "content": str(row["content"]),
        "created_at": str(row["created_at"]),
        "updated_at": str(row["updated_at"]),
    }


def _normalize_tag_name(name: str) -> str:
    """Normalize tag names (trim + lower)."""
    return name.strip().lower()


def _fetch_tags_for_note(conn: sqlite3.Connection, note_id: int) -> List[str]:
    cur = conn.execute(
        """
        SELECT t.name
        FROM tags t
        JOIN note_tags nt ON nt.tag_id = t.id
        WHERE nt.note_id = ?
        ORDER BY t.name ASC
        """,
        (note_id,),
    )
    return [str(r["name"]) for r in cur.fetchall()]


def _ensure_tag_ids(conn: sqlite3.Connection, tag_names: Sequence[str]) -> List[int]:
    """
    Ensure tags exist and return their IDs.

    Contract:
    - Inputs: list of normalized, non-empty tag names.
    - Output: list of tag_ids in the same order as input names (deduped by name).
    - Errors: sqlite3.Error on DB issues.
    """
    # Deduplicate while preserving order
    seen: set[str] = set()
    uniq_names: List[str] = []
    for n in tag_names:
        if n and n not in seen:
            seen.add(n)
            uniq_names.append(n)

    tag_ids: List[int] = []
    for name in uniq_names:
        conn.execute("INSERT OR IGNORE INTO tags (name) VALUES (?)", (name,))
        row = conn.execute("SELECT id FROM tags WHERE name = ?", (name,)).fetchone()
        if not row:
            # Extremely unlikely unless concurrent deletion; treat as server error.
            raise sqlite3.Error(f"Failed to resolve tag id for tag '{name}'")
        tag_ids.append(int(row["id"]))
    return tag_ids


def _set_note_tags(conn: sqlite3.Connection, note_id: int, tag_names: Sequence[str]) -> List[str]:
    """
    Replace note tags with the provided tag list.

    Contract:
    - Inputs:
      - note_id: must exist
      - tag_names: sequence of raw tag strings (will be normalized, empty removed, deduped)
    - Output:
      - normalized tag names actually set (sorted asc for stable API)
    - Side effects:
      - inserts missing tags into tags table
      - replaces join rows in note_tags for the note_id
    """
    normalized = [_normalize_tag_name(t) for t in tag_names]
    normalized = [t for t in normalized if t]
    # Ensure tag IDs exist
    tag_ids = _ensure_tag_ids(conn, normalized) if normalized else []

    conn.execute("DELETE FROM note_tags WHERE note_id = ?", (note_id,))
    for tag_id in tag_ids:
        conn.execute(
            "INSERT OR IGNORE INTO note_tags (note_id, tag_id) VALUES (?, ?)",
            (note_id, tag_id),
        )

    # Return canonical sorted tag list
    cur = conn.execute(
        """
        SELECT t.name
        FROM tags t
        JOIN note_tags nt ON nt.tag_id = t.id
        WHERE nt.note_id = ?
        ORDER BY t.name ASC
        """,
        (note_id,),
    )
    return [str(r["name"]) for r in cur.fetchall()]


class ApiInfo(BaseModel):
    """Simple health response."""

    status: str = Field(..., description="Service status string.")
    sqlite_db: str = Field(..., description="Resolved SQLite DB path used by the API.")


class NoteBase(BaseModel):
    """Base payload fields for notes."""

    title: str = Field(..., min_length=1, max_length=500, description="Note title.")
    content: str = Field("", description="Note content/body (plain text).")


class NoteCreate(NoteBase):
    """Create note request."""

    tags: Optional[List[str]] = Field(
        default=None,
        description="Optional list of tag names to attach to the note.",
    )


class NoteUpdate(BaseModel):
    """Update note request (partial)."""

    title: Optional[str] = Field(None, min_length=1, max_length=500, description="Updated title.")
    content: Optional[str] = Field(None, description="Updated content/body (plain text).")
    tags: Optional[List[str]] = Field(
        default=None,
        description="Optional list of tag names to replace the note tags. If omitted, tags unchanged.",
    )


class NoteOut(BaseModel):
    """Note response model."""

    id: int = Field(..., description="Note ID.")
    title: str = Field(..., description="Note title.")
    content: str = Field(..., description="Note content/body (plain text).")
    created_at: str = Field(..., description="Creation timestamp (SQLite text).")
    updated_at: str = Field(..., description="Last updated timestamp (SQLite text).")
    tags: Optional[List[str]] = Field(default=None, description="Optional list of tags (if requested).")


class TagOut(BaseModel):
    """Tag response model."""

    name: str = Field(..., description="Tag name (normalized lower-case).")
    count: int = Field(..., description="Number of notes associated with this tag.")


openapi_tags = [
    {"name": "Health", "description": "Service health and diagnostics."},
    {"name": "Notes", "description": "CRUD operations for notes."},
    {"name": "Search", "description": "Search notes by title/content with optional tag filters."},
    {"name": "Tags", "description": "Tag listing and note-tag management."},
]

app = FastAPI(
    title="Notes Backend API",
    description=(
        "CRUD + search API for a notes application, backed by SQLite.\n\n"
        "DB configuration:\n"
        "- Set environment variable SQLITE_DB to the SQLite file path.\n"
        f"- If not set, defaults to: {DEFAULT_SQLITE_PATH}\n"
    ),
    version="1.0.0",
    openapi_tags=openapi_tags,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Frontend may run on a different port in dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# PUBLIC_INTERFACE
@app.get(
    "/",
    response_model=ApiInfo,
    tags=["Health"],
    summary="Health check",
    description="Returns service status and the resolved SQLite DB path used by the backend.",
    operation_id="health_check",
)
def health_check() -> ApiInfo:
    """Health check endpoint."""
    return ApiInfo(status="ok", sqlite_db=_get_sqlite_db_path())


def _get_note_or_404(conn: sqlite3.Connection, note_id: int) -> sqlite3.Row:
    row = conn.execute("SELECT * FROM notes WHERE id = ?", (note_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Note not found")
    return row


def _note_out(conn: sqlite3.Connection, row: sqlite3.Row, include_tags: bool) -> NoteOut:
    base = _row_to_note_dict(row)
    if include_tags:
        base["tags"] = _fetch_tags_for_note(conn, int(row["id"]))
    return NoteOut(**base)


# PUBLIC_INTERFACE
@app.get(
    "/notes",
    response_model=List[NoteOut],
    tags=["Notes"],
    summary="List notes",
    description=(
        "List notes ordered by updated_at desc.\n\n"
        "Optional query params:\n"
        "- include_tags: include tags in each note response.\n"
        "- tag: filter to notes with a given tag (normalized).\n"
        "- limit/offset: pagination."
    ),
    operation_id="list_notes",
)
def list_notes(
    include_tags: bool = Query(False, description="Whether to include tags on each note."),
    tag: Optional[str] = Query(None, description="Optional tag filter (case-insensitive)."),
    limit: int = Query(50, ge=1, le=200, description="Max number of notes to return."),
    offset: int = Query(0, ge=0, description="Pagination offset."),
) -> List[NoteOut]:
    """List notes with optional tag filtering."""
    tag_norm = _normalize_tag_name(tag) if tag else None
    with _db_conn() as conn:
        if tag_norm:
            rows = conn.execute(
                """
                SELECT n.*
                FROM notes n
                JOIN note_tags nt ON nt.note_id = n.id
                JOIN tags t ON t.id = nt.tag_id
                WHERE t.name = ?
                ORDER BY n.updated_at DESC, n.id DESC
                LIMIT ? OFFSET ?
                """,
                (tag_norm, limit, offset),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT *
                FROM notes
                ORDER BY updated_at DESC, id DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            ).fetchall()

        return [_note_out(conn, r, include_tags=include_tags) for r in rows]


# PUBLIC_INTERFACE
@app.post(
    "/notes",
    response_model=NoteOut,
    status_code=status.HTTP_201_CREATED,
    tags=["Notes"],
    summary="Create a note",
    description="Create a new note. Tags are optional and will be normalized (trim + lower).",
    operation_id="create_note",
)
def create_note(payload: NoteCreate) -> NoteOut:
    """Create a note with optional tags."""
    logger.info("create_note title_len=%s", len(payload.title))
    with _db_conn() as conn:
        cur = conn.execute(
            "INSERT INTO notes (title, content) VALUES (?, ?)",
            (payload.title, payload.content or ""),
        )
        note_id = int(cur.lastrowid)

        if payload.tags is not None:
            _set_note_tags(conn, note_id, payload.tags)

        row = _get_note_or_404(conn, note_id)
        return _note_out(conn, row, include_tags=True)


# PUBLIC_INTERFACE
@app.get(
    "/notes/{note_id}",
    response_model=NoteOut,
    tags=["Notes"],
    summary="Get note details",
    description="Fetch a single note by ID. Set include_tags=true to include tags.",
    operation_id="get_note",
)
def get_note(
    note_id: int,
    include_tags: bool = Query(True, description="Whether to include tags on the note."),
) -> NoteOut:
    """Fetch a note by its ID."""
    with _db_conn() as conn:
        row = _get_note_or_404(conn, note_id)
        return _note_out(conn, row, include_tags=include_tags)


# PUBLIC_INTERFACE
@app.put(
    "/notes/{note_id}",
    response_model=NoteOut,
    tags=["Notes"],
    summary="Replace a note",
    description=(
        "Replace title/content and optionally replace tags.\n\n"
        "If tags is omitted, tags are unchanged. If tags is provided (including []), "
        "it replaces the existing tag set."
    ),
    operation_id="replace_note",
)
def replace_note(note_id: int, payload: NoteCreate) -> NoteOut:
    """Replace note contents and optionally tags."""
    logger.info("replace_note note_id=%s", note_id)
    with _db_conn() as conn:
        _get_note_or_404(conn, note_id)
        conn.execute(
            "UPDATE notes SET title = ?, content = ? WHERE id = ?",
            (payload.title, payload.content or "", note_id),
        )
        if payload.tags is not None:
            _set_note_tags(conn, note_id, payload.tags)
        row = _get_note_or_404(conn, note_id)
        return _note_out(conn, row, include_tags=True)


# PUBLIC_INTERFACE
@app.patch(
    "/notes/{note_id}",
    response_model=NoteOut,
    tags=["Notes"],
    summary="Update a note (partial)",
    description=(
        "Partially update title/content and optionally replace tags.\n\n"
        "If tags is omitted, tags are unchanged. If tags is provided (including []), "
        "it replaces the existing tag set."
    ),
    operation_id="update_note",
)
def update_note(note_id: int, payload: NoteUpdate) -> NoteOut:
    """Update a note partially."""
    logger.info("update_note note_id=%s", note_id)
    with _db_conn() as conn:
        existing = _get_note_or_404(conn, note_id)
        new_title = payload.title if payload.title is not None else str(existing["title"])
        new_content = payload.content if payload.content is not None else str(existing["content"])

        conn.execute(
            "UPDATE notes SET title = ?, content = ? WHERE id = ?",
            (new_title, new_content, note_id),
        )

        if payload.tags is not None:
            _set_note_tags(conn, note_id, payload.tags)

        row = _get_note_or_404(conn, note_id)
        return _note_out(conn, row, include_tags=True)


# PUBLIC_INTERFACE
@app.delete(
    "/notes/{note_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["Notes"],
    summary="Delete a note",
    description="Delete a note by ID. Cascades to note_tags via foreign key.",
    operation_id="delete_note",
)
def delete_note(note_id: int) -> Response:
    """Delete note by ID."""
    logger.info("delete_note note_id=%s", note_id)
    with _db_conn() as conn:
        _get_note_or_404(conn, note_id)
        conn.execute("DELETE FROM notes WHERE id = ?", (note_id,))
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# PUBLIC_INTERFACE
@app.get(
    "/notes/search",
    response_model=List[NoteOut],
    tags=["Search"],
    summary="Search notes",
    description=(
        "Search notes by substring match against title/content.\n\n"
        "Implementation uses LIKE for portability (not FTS). For case-insensitivity, SQLite LIKE "
        "is typically case-insensitive for ASCII depending on collation.\n\n"
        "Optional:\n"
        "- tag: restrict results to notes having a particular tag.\n"
        "- include_tags: include tags in each result.\n"
        "- limit/offset: pagination."
    ),
    operation_id="search_notes",
)
def search_notes(
    q: str = Query(..., min_length=1, max_length=200, description="Search query text."),
    tag: Optional[str] = Query(None, description="Optional tag filter (case-insensitive)."),
    include_tags: bool = Query(False, description="Whether to include tags on each note."),
    limit: int = Query(50, ge=1, le=200, description="Max number of notes to return."),
    offset: int = Query(0, ge=0, description="Pagination offset."),
) -> List[NoteOut]:
    """Search notes by title/content with optional tag filter."""
    q_like = f"%{q}%"
    tag_norm = _normalize_tag_name(tag) if tag else None
    logger.info("search_notes q_len=%s tag=%s", len(q), tag_norm)
    with _db_conn() as conn:
        if tag_norm:
            rows = conn.execute(
                """
                SELECT DISTINCT n.*
                FROM notes n
                JOIN note_tags nt ON nt.note_id = n.id
                JOIN tags t ON t.id = nt.tag_id
                WHERE t.name = ?
                  AND (n.title LIKE ? OR n.content LIKE ?)
                ORDER BY n.updated_at DESC, n.id DESC
                LIMIT ? OFFSET ?
                """,
                (tag_norm, q_like, q_like, limit, offset),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT *
                FROM notes
                WHERE title LIKE ? OR content LIKE ?
                ORDER BY updated_at DESC, id DESC
                LIMIT ? OFFSET ?
                """,
                (q_like, q_like, limit, offset),
            ).fetchall()

        return [_note_out(conn, r, include_tags=include_tags) for r in rows]


# PUBLIC_INTERFACE
@app.get(
    "/tags",
    response_model=List[TagOut],
    tags=["Tags"],
    summary="List tags",
    description="List tags with counts of associated notes.",
    operation_id="list_tags",
)
def list_tags() -> List[TagOut]:
    """List all tags with note counts."""
    with _db_conn() as conn:
        rows = conn.execute(
            """
            SELECT t.name as name, COUNT(nt.note_id) as count
            FROM tags t
            LEFT JOIN note_tags nt ON nt.tag_id = t.id
            GROUP BY t.id
            ORDER BY t.name ASC
            """
        ).fetchall()
        return [TagOut(name=str(r["name"]), count=int(r["count"])) for r in rows]


# PUBLIC_INTERFACE
@app.get(
    "/notes/{note_id}/tags",
    response_model=List[str],
    tags=["Tags"],
    summary="Get note tags",
    description="Get the list of tags attached to a note.",
    operation_id="get_note_tags",
)
def get_note_tags(note_id: int) -> List[str]:
    """Return tags for a note."""
    with _db_conn() as conn:
        _get_note_or_404(conn, note_id)
        return _fetch_tags_for_note(conn, note_id)


# PUBLIC_INTERFACE
@app.put(
    "/notes/{note_id}/tags",
    response_model=List[str],
    tags=["Tags"],
    summary="Replace note tags",
    description=(
        "Replace the tag list for a note. Provide an empty list to clear tags.\n\n"
        "Request body is a JSON array of strings."
    ),
    operation_id="replace_note_tags",
)
def replace_note_tags(note_id: int, tags: List[str]) -> List[str]:
    """Replace note tags (idempotent)."""
    logger.info("replace_note_tags note_id=%s tags_count=%s", note_id, len(tags))
    with _db_conn() as conn:
        _get_note_or_404(conn, note_id)
        return _set_note_tags(conn, note_id, tags)
