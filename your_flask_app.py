import subprocess
import time
import psutil

def kill_process(process_name):
    for proc in psutil.process_iter():
        if proc.name() == process_name:
            proc.kill()
            print(f"Terminated process: {proc.pid}")

def start_flask_server():
    subprocess.Popen(["python", "your_flask_app.py"])
    print("Started Flask server")

def main():
    flask_process_name = "python.exe"  # Change this to match your Flask process name
    dash_process_name = "python.exe"   # Change this to match your Dash process name

    # Kill existing Flask and Dash processes
    kill_process(flask_process_name)
    kill_process(dash_process_name)

    # Wait for processes to terminate
    time.sleep(2)

    # Start Flask server
    start_flask_server()

if __name__ == "__main__":
    main()