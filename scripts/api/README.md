# Script for running API server
1. Setup virtual environment
```bash
$ bash scripts/venv.sh
```
2. Run API server
```bash
$ bash scripts/api/run.sh <IP> <PORT>
# Example
$ bash scripts/api/run.sh 127.0.0.1 8000 # Change IP to 0.0.0.0 to run all networks
```