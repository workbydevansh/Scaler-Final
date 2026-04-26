"""Root OpenEnv server entrypoint.

The implementation lives in supportops_x_env.server.app; this wrapper keeps the
repo compatible with the standard OpenEnv scaffold expected by the validator.
"""

from supportops_x_env.server.app import app


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Run the SupportOps-X server."""

    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if args.host == "0.0.0.0" and args.port == 8000:
        main()
    else:
        main(host=args.host, port=args.port)
