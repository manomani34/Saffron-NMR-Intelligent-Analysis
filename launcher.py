import os
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path


def get_bundle_dir() -> Path:
    """
    Return the directory where PyInstaller stores bundled files.
    With PyInstaller 6 one-folder builds, this is usually _internal.
    In normal Python execution, it is the project directory.
    """

    if getattr(sys, "frozen", False):

        exe_dir = (
            Path(sys.executable)
            .resolve()
            .parent
        )

        internal_dir = exe_dir / "_internal"

        if internal_dir.exists():
            return internal_dir

        return exe_dir

    return (
        Path(__file__)
        .resolve()
        .parent
    )


def find_free_port() -> int:
    """
    Find a free localhost TCP port.
    """

    with socket.socket(
        socket.AF_INET,
        socket.SOCK_STREAM,
    ) as sock:

        sock.bind(
            ("127.0.0.1", 0)
        )

        return int(
            sock.getsockname()[1]
        )


def wait_and_open_browser(
    url: str,
    port: int,
    timeout_seconds: int = 60,
) -> None:
    """
    Wait until Streamlit is listening,
    then open the browser.
    """

    start = time.time()

    while (
        time.time() - start
        < timeout_seconds
    ):

        try:

            with socket.create_connection(
                (
                    "127.0.0.1",
                    port,
                ),
                timeout=1,
            ):

                webbrowser.open(url)
                return

        except OSError:

            time.sleep(0.5)

    webbrowser.open(url)


def main() -> None:

    bundle_dir = get_bundle_dir()

    dashboard_path = (
        bundle_dir / "dashboard.py"
    )

    if not dashboard_path.exists():

        raise FileNotFoundError(
            "Dashboard file not found:\n"
            f"{dashboard_path}"
        )

    port = find_free_port()

    url = (
        f"http://127.0.0.1:{port}"
    )

    browser_thread = threading.Thread(
        target=wait_and_open_browser,
        args=(
            url,
            port,
        ),
        daemon=True,
    )

    browser_thread.start()

    # --------------------------------------------------------
    # Important:
    # dashboard.py uses:
    # Path(__file__).resolve().parent
    #
    # Therefore dashboard.py and all bundled data files
    # must be considered relative to the bundle directory.
    # --------------------------------------------------------

    os.chdir(bundle_dir)

    sys.argv = [
        "streamlit",
        "run",
        str(dashboard_path),

        "--server.headless=true",

        "--server.address=127.0.0.1",

        f"--server.port={port}",

        "--server.fileWatcherType=none",

        "--global.developmentMode=false",

        "--browser.gatherUsageStats=false",
    ]

    from streamlit.web import cli as stcli

    stcli.main()


if __name__ == "__main__":
    main()