@echo off
echo Starting Federated Learning Clients...

:: Start Client 0
start cmd /k python src/client/client.py --client_id 0 --num_clients 3 --data_path "data/bank_transactions_data.csv"

:: Start Client 1
start cmd /k python src/client/client.py --client_id 1 --num_clients 3 --data_path "data/bank_transactions_data.csv"

:: Start Client 2
start cmd /k python src/client/client.py --client_id 2 --num_clients 3 --data_path "data/bank_transactions_data.csv"

echo All clients started. Check the individual windows for progress.