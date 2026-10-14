import http.server
import socketserver
import webbrowser
import socket
import sys
from pathlib import Path

PORT = 8000
ROOT_DIR = Path(__file__).resolve().parent

class BenchmarkAppHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT_DIR), **kwargs)

def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0

def start_server():
    url = f"http://localhost:{PORT}/app/"
    if is_port_in_use(PORT):
        print("\n" + "=" * 55)
        print(f"  Benchmark Lab server is already active!")
        print(f"  Access URL: {url}")
        print("  Opening browser...")
        print("=" * 55 + "\n")
        webbrowser.open(url)
        return

    try:
        with socketserver.TCPServer(("", PORT), BenchmarkAppHandler) as httpd:
            print("\n" + "=" * 55)
            print("  Quantized PCA Benchmark Lab is running!")
            print(f"  Access URL: {url}")
            print("  Press Ctrl+C to stop the server")
            print("=" * 55 + "\n")
            webbrowser.open(url)
            httpd.serve_forever()
    except OSError as e:
        if "Address already in use" in str(e) or getattr(e, "winerror", None) == 10048:
            print(f"\nPort {PORT} is already in use. Opening browser at {url}\n")
            webbrowser.open(url)
        else:
            raise e

if __name__ == "__main__":
    start_server()
