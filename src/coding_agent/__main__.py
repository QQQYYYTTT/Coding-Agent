"""Allow the installed package to run as a Python module."""

from coding_agent.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
