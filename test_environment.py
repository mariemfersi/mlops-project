# Only import what is used
# e.g., if testing Python version or basic ML packages
import subprocess

def test_python():
    result = subprocess.run(["python3", "--version"], capture_output=True)
    print(result.stdout.decode())

if __name__ == "__main__":
    test_python()
