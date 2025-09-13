import asyncio
import subprocess
import logging
from websockets import serve, WebSocketServerProtocol, ConnectionClosed
from threading import Lock

logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s] %(levelname)s - %(message)s')

# Shared state
active_streams = {}  # {camera_id: {'process': ..., 'clients': set([...]), 'server': ..., 'port': ...}}
lock = Lock()

FFMPEG_PATH = "app/setup_files/ffmpeg/bin/ffmpeg.exe"  

async def start_stream(camera_id: str, camera_url: str, ws_port: int):
    """
    Starts the streaming server for the given camera.
    If another camera is streaming, it will be stopped.
    """
    with lock:
        # Stop other streams
        for cid in list(active_streams.keys()):
            if cid != camera_id:
                asyncio.create_task(stop_stream(cid))

        if camera_id in active_streams:
            logging.info(f"Stream for camera {camera_id} already running.")
            return active_streams[camera_id]["port"]

        logging.info(f"Starting stream for camera {camera_id} on port {ws_port}.")

        try:
            process = subprocess.Popen(
                [
                    FFMPEG_PATH,
                    "-i", camera_url,
                    "-f", "mpegts",
                    "-codec:v", "mpeg1video",
                    "-r", "25",
                    "-"
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
        except Exception as e:
            logging.error(f"Failed to start ffmpeg for camera {camera_id}: {e}")
            return None

        clients = set()

        # Define the server placeholder, will be assigned after creation
        server = None

        # Register the stream
        active_streams[camera_id] = {
            "process": process,
            "clients": clients,
            "server": server,
            "port": ws_port
        }

    # Start WebSocket server
    server = await serve(lambda ws, path: handler(ws, path, camera_id), "0.0.0.0", ws_port)
    with lock:
        active_streams[camera_id]["server"] = server

    # Launch the forward stream task
    asyncio.create_task(forward_stream(camera_id))

    logging.info(f"Stream for camera {camera_id} started.")
    return ws_port


async def handler(ws: WebSocketServerProtocol, path: str, camera_id: str):
    """
    Handles incoming WebSocket connections.
    """
    logging.info(f"Client attempting to connect to camera {camera_id}.")
    with lock:
        stream = active_streams.get(camera_id)
        if not stream:
            logging.error(f"No active stream found for camera {camera_id}. Closing connection.")
            await ws.close()
            return
        stream["clients"].add(ws)

    logging.info(f"Client connected to camera {camera_id}.")

    try:
        async for message in ws:
            # For now, ignore incoming messages
            pass
    except ConnectionClosed:
        logging.info(f"Client disconnected from camera {camera_id}.")
    except Exception as e:
        logging.error(f"Error in WebSocket connection for camera {camera_id}: {e}")
    finally:
        with lock:
            if camera_id in active_streams:
                active_streams[camera_id]["clients"].discard(ws)


async def forward_stream(camera_id: str):
    """
    Reads chunks from ffmpeg and forwards them to all connected clients.
    """
    logging.info(f"Forwarding stream for camera {camera_id}.")
    with lock:
        stream = active_streams.get(camera_id)
        if not stream:
            logging.error(f"No stream found for camera {camera_id}. Cannot forward.")
            return
        process = stream["process"]

    while True:
        chunk = process.stdout.read(1024)
        if not chunk:
            logging.info(f"Stream ended for camera {camera_id}.")
            break
        with lock:
            clients = list(stream["clients"])

        for ws in clients:
            try:
                await ws.send(chunk)
            except Exception as e:
                logging.warning(f"Failed to send data to client for camera {camera_id}: {e}")
                with lock:
                    stream["clients"].discard(ws)


async def stop_stream(camera_id: str):
    """
    Stops the stream for the given camera.
    """
    logging.info(f"Stopping stream for camera {camera_id}.")
    with lock:
        stream = active_streams.pop(camera_id, None)

    if not stream:
        logging.warning(f"No active stream found for camera {camera_id}.")
        return

    # Terminate ffmpeg process
    process = stream["process"]
    if process and process.poll() is None:
        try:
            process.terminate()
            await asyncio.sleep(1)
            process.kill()
        except Exception as e:
            logging.error(f"Error terminating ffmpeg process for camera {camera_id}: {e}")

    # Close WebSocket server
    server = stream["server"]
    if server:
        try:
            server.close()
            await server.wait_closed()
        except Exception as e:
            logging.error(f"Error closing WebSocket server for camera {camera_id}: {e}")

    logging.info(f"Stream for camera {camera_id} stopped.")


async def run_websocket_server():
    """
    Keeps the event loop alive.
    """
    logging.info("WebSocket server loop running.")
    while True:
        await asyncio.sleep(3600)  # Keep the loop alive
