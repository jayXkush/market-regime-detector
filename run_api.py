"""
run_api.py — CLI entry point for Phase 5: FastAPI Backend.

Usage:
    python run_api.py                    # Start on default port 10000
    python run_api.py --port 8080        # Custom port
    python run_api.py --host 0.0.0.0     # Bind to all interfaces
    python run_api.py --reload           # Auto-reload for development

Swagger docs available at: http://localhost:10000/docs
"""

from __future__ import annotations

import argparse
import os
import sys

import uvicorn

from src.utils.logger import get_logger

logger = get_logger("run_api")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Market Regime Detection — Phase 5: FastAPI Backend",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the server to. Default: 0.0.0.0.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("PORT", 10000)),
        help="Port to run the server on. Default: PORT env var or 10000.",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development.",
    )
    return parser.parse_args()


def _log_memory_usage() -> None:
    """Log approximate memory usage at startup."""
    try:
        import resource
        mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        logger.info("  Memory usage: ~%.1f MB", mem_mb)
    except ImportError:
        # Windows — fallback
        try:
            import psutil
            process = psutil.Process(os.getpid())
            mem_mb = process.memory_info().rss / (1024 * 1024)
            logger.info("  Memory usage: ~%.1f MB", mem_mb)
        except ImportError:
            logger.info("  Memory usage: (install psutil for tracking)")


def main() -> None:
    args = parse_args()

    logger.info("=" * 60)
    logger.info("  Market Regime Detection — Phase 7: Production Deployment")
    logger.info("=" * 60)
    logger.info("  Host:   %s", args.host)
    logger.info("  Port:   %d", args.port)
    logger.info("  Reload: %s", args.reload)
    logger.info("  Env:    %s", os.environ.get("PYTHON_ENV", "development"))
    logger.info("  Docs:   http://%s:%d/docs", args.host, args.port)
    _log_memory_usage()
    logger.info("=" * 60)

    # Workers=1 for free tier memory optimization
    uvicorn.run(
        "src.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
        workers=1,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
        sys.exit(0)
    except Exception as exc:
        logger.exception("Fatal error: %s", exc)
        sys.exit(1)
