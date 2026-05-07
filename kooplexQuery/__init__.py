"""kooplexQuery package.

Keep package initialization side-effect free so backend services can import
submodules (e.g. db_chat, db, motor) without pulling Streamlit-only modules.
"""

__all__ = [
	"history",
	"motor",
	"utils",
]