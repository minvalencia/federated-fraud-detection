# Import required libraries for federated learning server
import flwr as fl
from typing import List, Tuple, Dict, Optional
import numpy as np
from flwr.common import Metrics
from flwr.server.client_proxy import ClientProxy

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """
    Calculate weighted average of metrics across all clients.
    Weights are based on number of examples in each client.

    Args:
        metrics (List[Tuple[int, Metrics]]): List of tuples containing
            number of examples and metrics from each client

    Returns:
        Dict: Weighted average metrics
    """
    # Multiply each client's accuracy by number of examples to get weighted sum
    accuracies = [m["accuracy"] * num_examples for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Calculate weighted average accuracy
    return {"accuracy": sum(accuracies) / sum(examples)}

class FraudDetectionServer:
    """
    Federated Learning server for fraud detection.
    Coordinates multiple clients (banks) for collaborative model training.
    """

    def __init__(self, min_fit_clients: int = 2, min_available_clients: int = 2):
        """
        Initialize the federated learning server.

        Args:
            min_fit_clients (int): Minimum number of clients required for training
            min_available_clients (int): Minimum number of available clients required
        """
        self.min_fit_clients = min_fit_clients
        self.min_available_clients = min_available_clients

    def start_server(self, host: str = "0.0.0.0", port: int = 8081):
        """
        Start the federated learning server.

        Args:
            host (str): Server host address
            port (int): Server port number
        """
        # Configure the federated learning strategy
        strategy = fl.server.strategy.FedAvg(
            # Use all available clients for training and evaluation
            fraction_fit=1.0,  # 100% of clients participate in training
            fraction_evaluate=1.0,  # 100% of clients participate in evaluation

            # Set minimum client requirements
            min_fit_clients=self.min_fit_clients,
            min_available_clients=self.min_available_clients,
            min_evaluate_clients=2,

            # Use weighted average for aggregating metrics
            evaluate_metrics_aggregation_fn=weighted_average,
        )

        # Start Flower server
        fl.server.start_server(
            server_address=f"{host}:{port}",
            strategy=strategy,
            # Configure server for 3 rounds of federated learning
            config=fl.server.ServerConfig(num_rounds=3)
        )

if __name__ == "__main__":
    # Create and start server with default configuration
    server = FraudDetectionServer()
    server.start_server()