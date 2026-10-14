import socket
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


BASE = Path(__file__).resolve().parent
APP_DIR = BASE / "classification"
START_PORT = 8766


class ClassificationAppHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(APP_DIR), **kwargs)


def start_server():
    server = None
    for port in range(START_PORT, START_PORT + 20):
        try:
            server = ThreadingHTTPServer(("127.0.0.1", port), ClassificationAppHandler)
            break
        except OSError:
            continue

    if server is None:
        raise OSError(f"Could not find an available local port from {START_PORT} to {START_PORT + 19}")

    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"Classification comparison is running at {url}")
    print("Press Ctrl+C to stop the server.")
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping classification comparison server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    start_server()
