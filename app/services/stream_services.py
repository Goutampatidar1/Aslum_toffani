# import asyncio
# import websockets
# import subprocess
# import threading
# import logging

# # from config import FFMPEG_PATH, BASE_WS_PORT
# from app.services.camera_service import get_camera_by_id

# active_streams = {}
# lock = threading.Lock()


# BASE_WS_PORT = 4440
# FFMPEG_PATH = "app\setup_files\ffmpeg\bin\ffmpeg.exe"


# def get_ws_port(camera_id):
#     return BASE_WS_PORT


# def start_stream(camera_id):
#     with lock:
#         if camera_id in active_streams:
#             return get_ws_port(camera_id), "Already streaming"

#         camera = get_camera_by_id(camera_id)
#         if not camera:
#             raise ValueError("Camera not found")
#         print("caMERA IN sTRAT_STREAM: " , camera)
#         port = get_ws_port(camera_id)
#         loop = asyncio.new_event_loop()
#         t = threading.Thread(
#             target=_run_stream_server,
#             args=(camera["url"], port, loop, camera_id),
#             daemon=True,
#         )
#         t.start()

#         active_streams[camera_id] = {
#             "thread": t,
#             "loop": loop,
#         }

#         return port, "Streaming started"


# def stop_stream(camera_id):
#     with lock:
#         stream = active_streams.get(camera_id)
#         if not stream:
#             return "Stream not active"

#         loop = stream["loop"]
#         if loop and loop.is_running():
#             loop.call_soon_threadsafe(loop.stop)

#         del active_streams[camera_id]
#         return "Streaming stopped"


# def _run_stream_server(rtsp_url, port, loop, camera_id):
#     asyncio.set_event_loop(loop)

#     async def handler(websocket, path):
#         logging.info(f"[Camera {camera_id}] Client connected.")
#         process = subprocess.Popen(
#             [
#                 FFMPEG_PATH,
#                 "-i",
#                 rtsp_url,
#                 "-f",
#                 "mpegts",
#                 "-codec:v",
#                 "mpeg1video",
#                 "-r",
#                 "25",
#                 "-",
#             ],
#             stdout=subprocess.PIPE,
#             stderr=subprocess.PIPE,
#         )

#         try:
#             while True:
#                 data = process.stdout.read(1024)
#                 if not data:
#                     break
#                 await websocket.send(data)
#         except Exception as e:
#             logging.error(f"[Camera {camera_id}] WebSocket error: {e}")
#         finally:
#             logging.info(f"[Camera {camera_id}] Closing FFmpeg process.")
#             process.kill()

#     start_server = websockets.serve(handler, "0.0.0.0", port)

#     try:
#         loop.run_until_complete(start_server)
#         logging.info(f"[Camera {camera_id}] WebSocket server started on port {port}")
#         loop.run_forever()
#     except Exception as e:
#         logging.error(f"[Camera {camera_id}] WebSocket server error: {e}")
#     finally:
#         loop.close()
#         logging.info(f"[Camera {camera_id}] WebSocket server shut down.")


import asyncio
import websockets
import subprocess
import threading
import logging

from app.services.camera_service import get_camera_by_id

active_streams = {}
lock = threading.Lock()

BASE_WS_PORT = 4440
FFMPEG_PATH = "app/setup_files/ffmpeg/bin/ffmpeg.exe"


def get_ws_port(camera_id):
    return BASE_WS_PORT + (abs(hash(camera_id)) % 1000)


def start_stream(camera_id):
    with lock:
        if camera_id in active_streams:
            return get_ws_port(camera_id), "Already streaming"

        camera = get_camera_by_id(camera_id)
        if not camera:
            raise ValueError("Camera not found")

        port = get_ws_port(camera_id)
        loop = asyncio.new_event_loop()

        t = threading.Thread(
            target=_run_stream_server,
            args=(camera["url"], port, loop, camera_id),
            daemon=True,
        )
        t.start()

        active_streams[camera_id] = {"thread": t, "loop": loop, "port": port}

        return port, "Streaming started"


# def stop_stream(camera_id):
#     with lock:
#         stream = active_streams.get(camera_id)
#         if not stream:
#             return "Stream not active"

#         loop = stream["loop"]
#         if loop and loop.is_running():
#             loop.call_soon_threadsafe(loop.stop)

#         del active_streams[camera_id]
#         return "Streaming stopped"

def stop_stream(camera_id):
    with lock:
        stream = active_streams.get(camera_id)
        if not stream:
            return "Stream not active"

        loop = stream["loop"]
        if loop and loop.is_running():
            # Cancel all tasks to avoid "Task was destroyed but is pending"
            for task in asyncio.all_tasks(loop):
                task.cancel()
            loop.call_soon_threadsafe(loop.stop)

        del active_streams[camera_id]
        return "Streaming stopped"

from websockets.server import serve, WebSocketServerProtocol
import asyncio
import subprocess
import logging

def _run_stream_server(rtsp_url, port, loop, camera_id):
    asyncio.set_event_loop(loop)

    async def handler(websocket: WebSocketServerProtocol):
        path = websocket.path
        logging.info(f"[Camera {camera_id}] Client connected. Path: {path}")
        logging.info(f"[RTSP_URL {rtsp_url}] ")

        process = subprocess.Popen(
            [
                FFMPEG_PATH,
                "-rtsp_transport", "tcp",
                "-fflags", "nobuffer",
                "-flags", "low_delay",
                "-i", rtsp_url,
                "-f", "mpegts",
                "-codec:v", "mpeg1video",
                "-r", "25",
                "-"
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            # bufsize=262144,
        )

        try:
            while True:
                data = await loop.run_in_executor(None, process.stdout.read, 1024)
                if not data:
                    break
                await websocket.send(data)
                logging.debug(f"[Camera {camera_id}] Sent {len(data)} bytes to client")
        except Exception as e:
            logging.error(f"[Camera {camera_id}] WebSocket error: {e}")
        finally:
            process.kill()
            logging.info(f"[Camera {camera_id}] Closed FFmpeg process.")

    async def start():
        server = await serve(
            handler,
            "0.0.0.0",
            port,
            create_protocol=WebSocketServerProtocol
        )
        logging.info(f"[Camera {camera_id}] WebSocket server started on port {port}")
        await server.wait_closed()

    try:
        loop.run_until_complete(start())
        loop.run_forever()
    except Exception as e:
        logging.error(f"[Camera {camera_id}] WebSocket server error: {e}")
    finally:
        loop.close()
        logging.info(f"[Camera {camera_id}] WebSocket server shut down.")




