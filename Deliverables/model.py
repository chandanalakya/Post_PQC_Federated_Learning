import flwr as fl
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
import os
import oqs
from cryptography.fernet import Fernet, InvalidToken
import hashlib
import base64
import pickle
import random
from sklearn.preprocessing import LabelEncoder
from typing import Dict, Tuple, List, Union
import time
import matplotlib.pyplot as plt
# Removed 'from flwr.common import Context' as it's not needed if cid is passed directly

# Global configurations
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define numerical and categorical columns for the dataset
numerical_cols = ['Income', 'Age', 'Experience', 'CURRENT_JOB_YRS', 'CURRENT_HOUSE_YRS']
categorical_cols = ['Married/Single', 'House_Ownership', 'Car_Ownership', 'Profession', 'CITY', 'STATE']

# Decentralized Server Configuration
NUM_SERVERS = 3 # Number of virtual servers
TOTAL_CLIENTS = 50 # Total number of clients in the simulation (Changed from 100 to 50)
# Distribute clients evenly among virtual servers (not directly used for client_fn, but for context)
CLIENTS_PER_VIRTUAL_SERVER = TOTAL_CLIENTS // NUM_SERVERS
# Directory where client data partitions are stored
DATA_DIR = "C:\\Users\\kumar\\OneDrive\\Desktop\\flower\\IN_2_clients_partitioned_multi\\IN_2_clients_partitioned_multi"

# Threshold for adaptive PQC signature algorithm selection (samples)
# Clients with training data size >= ADAPTIVE_PQC_THRESHOLD will use Falcon-512, otherwise Dilithium2
ADAPTIVE_PQC_THRESHOLD = 500

# Client ID designated to simulate a malicious attack (e.g., tampering)
# This refers to the file index, so client_1.csv is the malicious one.
MALICIOUS_CLIENT_FILE_INDEX = 1

# Global storage for various components and metrics (used for preprocessing)
global_label_encoders = {} # Stores fitted LabelEncoders for consistent categorical column encoding across clients
cat_cardinalities = [] # Stores cardinalities (number of unique categories) for categorical columns

# Global storage for server PQC keys
server_kyber_public_key = None
server_kyber_secret_key = None

# Global storage for client PQC keys (generated once for all clients)
client_dilithium_pub_keys_map = {}
client_dilithium_priv_keys_map = {}
client_falcon_pub_keys_map = {}
client_falcon_priv_keys_map = {}

# --- Data structures to store metrics for different runs ---
# PQC Metrics
pqc_metrics = {
    "accuracy": [], "loss": [], "std_loss": [],
    "client_encryption_times": [], "client_signing_times": [],
    "server_decryption_times": [], "server_verification_times_dilithium": [],
    "server_verification_times_falcon": [], "total_crypto_overhead": [],
    "rejected_malicious_updates": [], "communication_cost_mb": []
}

# No Cryptography Metrics (formerly Non-PQC)
no_crypto_metrics = {
    "accuracy": [], "loss": [], "std_loss": [],
    "client_encryption_times": [], "client_signing_times": [],
    "server_decryption_times": [], "server_verification_times_dilithium": [],
    "server_verification_times_falcon": [], "total_crypto_overhead": [],
    "rejected_malicious_updates": [], "communication_cost_mb": []
}

# Classical Cryptography (Simulated) Metrics
classical_crypto_metrics = {
    "accuracy": [], "loss": [], "std_loss": [],
    "client_encryption_times": [], "client_signing_times": [],
    "server_decryption_times": [], "server_verification_times_dilithium": [],
    "server_verification_times_falcon": [], "total_crypto_overhead": [],
    "rejected_malicious_updates": [], "communication_cost_mb": []
}


class Logger:
    """A simple custom logger for consistent output formatting."""
    def warning(self, message):
        print(f"[WARNING] {message}")
    def info(self, message):
        print(f"[INFO] {message}")
    def error(self, message):
        print(f"[ERROR] {message}")
    def debug(self, message):
        print(f"[DEBUG] {message}")
logger = Logger()

def ndarray_list_to_bytes(arrays: List[np.ndarray]) -> bytes:
    """Converts a list of NumPy arrays to bytes using pickle."""
    return pickle.dumps(arrays)

def bytes_to_ndarray_list(data: bytes) -> List[np.ndarray]:
    """Converts bytes back to a list of NumPy arrays using pickle."""
    return pickle.loads(data)

def print_debug(label: str, data: Union[bytes, List[np.ndarray]]):
    """Prints debug information about data type and size."""
    logger.debug(f"{label}: {type(data)} | Size: {len(data) if hasattr(data, '__len__') else 'N/A'}")

def generate_dilithium_key_pair(algorithm: str = "Dilithium2") -> Tuple[bytes, bytes]:
    """Generates a Dilithium key pair and returns raw public/secret keys."""
    with oqs.Signature(algorithm) as sig:
        public_key = sig.generate_keypair()
        secret_key = sig.export_secret_key()
    return public_key, secret_key

def generate_falcon_key_pair(algorithm: str = "Falcon-512") -> Tuple[bytes, bytes]:
    """Generates a Falcon key pair and returns raw public/secret keys."""
    with oqs.Signature(algorithm) as sig:
        public_key = sig.generate_keypair()
        secret_key = sig.export_secret_key()
    return public_key, secret_key

def sign_data(data_bytes: bytes, private_key: bytes, algorithm: str) -> bytes:
    """
    Signs data using the specified OQS signature algorithm.
    The private key is provided during the Signature object's creation.
    """
    with oqs.Signature(algorithm, secret_key=private_key) as signer:
        signature = signer.sign(data_bytes)
        return signature

def verify_signature(data_bytes: bytes, signature: bytes, public_key: bytes, algorithm: str) -> bool:
    """Verifies a signature using the specified OQS signature algorithm."""
    with oqs.Signature(algorithm) as verifier:
        return verifier.verify(data_bytes, signature, public_key)

def encrypt_model(weights: List[np.ndarray], server_pubkey: bytes) -> Dict[str, bytes]:
    """
    Encrypts model weights using Kyber512 KEM to establish a shared secret,
    then uses Fernet (AES) with a key derived from the shared secret.
    """
    model_bytes = ndarray_list_to_bytes(weights)
    with oqs.KeyEncapsulation("Kyber512") as kem:
        ciphertext, shared_secret = kem.encap_secret(server_pubkey)
    aes_key = hashlib.sha256(shared_secret).digest()
    fernet_key = base64.urlsafe_b64encode(aes_key)
    encrypted_model = Fernet(fernet_key).encrypt(model_bytes)
    return {"ciphertext": ciphertext, "encrypted_weights": encrypted_model}

def decrypt_model(payload: Dict[str, bytes], server_secret: bytes) -> List[np.ndarray]:
    """
    Decrypts model weights using Kyber512 KEM to decapsulate the shared secret,
    then uses Fernet (AES) with the derived key.
    """
    try:
        with oqs.KeyEncapsulation("Kyber512", secret_key=server_secret) as kem:
            shared_secret = kem.decap_secret(payload["ciphertext"])
        aes_key = hashlib.sha256(shared_secret).digest()
        fernet_key = base64.urlsafe_b64encode(aes_key)
        decrypted_bytes = Fernet(fernet_key).decrypt(payload["encrypted_weights"])
        return bytes_to_ndarray_list(decrypted_bytes)
    except InvalidToken:
        logger.error("[ERROR] Fernet decryption failed. InvalidToken. Returning empty weights.")
        return []
    except Exception as e:
        logger.error(f"[ERROR] Decryption failed with unexpected error: {e}. Returning empty weights.")
        return []

class EmptyDataset(Dataset):
    """A dummy dataset for clients with no data, ensuring __len__ and __getitem__ methods are present."""
    def __len__(self):
        return 0
    def __getitem__(self, idx):
        raise IndexError("This dataset is empty.")

class CreditRiskDataset(Dataset):
    """
    PyTorch Dataset for Credit Risk data.
    Handles numerical and categorical feature processing, including LabelEncoding.
    """
    def __init__(self, df: pd.DataFrame, numerical_cols: List[str], categorical_cols: List[str],
                 global_label_encoders: Dict[str, LabelEncoder], target_col: str = "Risk_Flag"):
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols
        self.target_col = target_col

        if df.empty:
            self.X_num = torch.tensor([], dtype=torch.float32).to(device)
            self.X_cat = torch.tensor([], dtype=torch.long).to(device)
            self.y = torch.tensor([], dtype=torch.float32).to(device)
        else:
            processed_df = df.copy()
            for col in self.categorical_cols:
                if col in processed_df.columns:
                    if col in global_label_encoders:
                        le = global_label_encoders[col]
                        processed_df[col] = processed_df[col].astype(str).apply(
                            lambda x: le.transform([x])[0] if x in le.classes_ else le.transform(['Unknown'])[0]
                        )
                    else:
                        logger.warning(f"No global LabelEncoder found for column '{col}'. This might lead to inconsistencies. Fitting a local LabelEncoder as fallback.")
                        le = LabelEncoder()
                        processed_df[col] = le.fit_transform(processed_df[col].astype(str))
                else:
                    logger.warning(f"Categorical column '{col}' not found in client DataFrame. Skipping encoding for this column.")

            cols_to_process_cat = [c for c in self.categorical_cols if c in processed_df.columns]
            if cols_to_process_cat:
                self.X_cat = torch.tensor(processed_df[cols_to_process_cat].values, dtype=torch.long).to(device)
            else:
                self.X_cat = torch.empty((len(processed_df), 0), dtype=torch.long).to(device)

            cols_to_process_num = [c for c in self.numerical_cols if c in processed_df.columns]
            if cols_to_process_num:
                for col in cols_to_process_num:
                    processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
                    processed_df[col] = processed_df[col].fillna(processed_df[col].mean() if not processed_df[col].isnull().all() else 0.0)
                self.X_num = torch.tensor(processed_df[cols_to_process_num].values, dtype=torch.float32).to(device)
            else:
                logger.warning("No numerical columns found or processed. Creating empty numerical tensor.")
                self.X_num = torch.empty((len(processed_df), 0), dtype=torch.float32).to(device)

            if self.target_col in processed_df.columns:
                self.y = torch.tensor(processed_df[self.target_col].values.reshape(-1, 1), dtype=torch.float32).to(device)
            else:
                raise ValueError(f"Target column '{self.target_col}' not found in DataFrame.")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.X_num[idx], self.X_cat[idx], self.y[idx]

def load_client_data_internal(client_file_index: int, numerical_cols: List[str],
                              categorical_cols: List[str], global_label_encoders: Dict[str, LabelEncoder]) -> Tuple[Dataset, Dataset]:
    """
    Loads data for a specific client, handles missing files/empty data,
    and splits into training and testing datasets.
    """
    path = os.path.join(DATA_DIR, f"client_{client_file_index}.csv")
    df = pd.DataFrame()

    if not os.path.exists(path):
        logger.error(f"Client data file not found at {path}. Client using file index {client_file_index} will use dummy data.")
    else:
        try:
            df = pd.read_csv(path)
            if df.empty:
                logger.warning(f"Client using file index {client_file_index} CSV file is empty. Client will use dummy data.")
        except Exception as e:
            logger.error(f"Failed to load CSV for client {client_file_index} at {path}: {e}. Client will use dummy data.")
            df = pd.DataFrame()

    if df.empty or len(df.columns) < (len(numerical_cols) + len(categorical_cols) + 1):
        logger.info(f"Generating dummy data for client using file index {client_file_index}.")
        dummy_data = {col: [0] for col in numerical_cols}
        dummy_data.update({col: ['Unknown'] for col in categorical_cols} if categorical_cols else {})
        dummy_data["Risk_Flag"] = [0]
        df = pd.DataFrame(dummy_data)
        if numerical_cols:
            df[numerical_cols[0]] = 1.0

    dataset = CreditRiskDataset(df, numerical_cols, categorical_cols, global_label_encoders)
    total_size = len(dataset)

    if total_size == 0:
        logger.warning(f"Dataset for client using file index {client_file_index} is truly empty. Returning EmptyDataset splits.")
        return EmptyDataset(), EmptyDataset()
    elif total_size == 1:
        logger.info(f"Client using file index {client_file_index} has only 1 sample. Assigning to training set, test set is empty.")
        return dataset, EmptyDataset()
    else:
        train_size = int(0.8 * total_size)
        test_size = total_size - train_size
        if train_size == 0:
            train_size = 1
            test_size = total_size - 1
        if test_size == 0 and total_size > 1:
            test_size = 1
            train_size = total_size - 1
        logger.debug(f"Client using file index {client_file_index} data split - train_size: {train_size}, test_size: {test_size}, total_dataset: {len(dataset)}")
        return random_split(dataset, [train_size, test_size])

class CreditRiskNet(nn.Module):
    """
    Neural network model for credit risk prediction.
    Combines numerical and categorical features using embeddings.
    """
    def __init__(self, num_numerical: int, cat_cardinalities: List[int]):
        super().__init__()
        self.num_numerical = max(1, num_numerical)
        self.cat_cardinalities_adjusted = [max(2, card) for card in cat_cardinalities]

        self.embeddings = nn.ModuleList([
            nn.Embedding(card, min(50, (card + 1) // 2)) for card in self.cat_cardinalities_adjusted
        ])
        self.numerical_bn = nn.BatchNorm1d(self.num_numerical) if self.num_numerical > 0 else None

        embedding_output_size = sum(emb.embedding_dim for emb in self.embeddings)
        input_size = self.num_numerical + embedding_output_size

        if input_size == 0:
            logger.warning("Model initialized with 0 input features. Setting a minimal input size for compatibility.")
            input_size = 1
            self.dummy_input = True
        else:
            self.dummy_input = False

        self.fc1 = nn.Linear(input_size, 64)
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        if self.dummy_input:
            return self.sigmoid(torch.zeros(x_num.size(0), 1, device=x_num.device))

        x_emb = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]

        if self.numerical_bn is not None and x_num.size(1) > 0:
            if self.training and x_num.size(0) == 1:
                numerical_out = x_num
            else:
                numerical_out = self.numerical_bn(x_num)
        else:
            numerical_out = x_num

        if x_emb:
            if numerical_out.size(1) > 0:
                x = torch.cat([numerical_out] + x_emb, dim=1)
            else:
                x = torch.cat(x_emb, dim=1)
        else:
            x = numerical_out

        if x.size(1) == 0:
            logger.warning("Forward pass received empty feature tensor after concatenation. Returning dummy output.")
            return self.sigmoid(torch.zeros(x_num.size(0), 1, device=x_num.device))

        x = self.fc1(x)
        x = self.relu1(x)
        if self.training and x.size(0) == 1:
            pass
        else:
            x = self.bn1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

class FLClient(fl.client.NumPyClient):
    """Flower client for credit risk prediction with PQC or non-PQC."""
    def __init__(self, model: nn.Module, server_pubkey: bytes, client_file_index: int,
                 numerical_cols: List[str], categorical_cols: List[str],
                 global_label_encoders: Dict[str, LabelEncoder],
                 is_malicious: bool, is_pqc_enabled: bool, is_classical_enabled: bool, # Added is_classical_enabled
                 client_dilithium_priv_key: bytes = None, client_dilithium_pub_key: bytes = None,
                 client_falcon_priv_key: bytes = None, client_falcon_pub_key: bytes = None):
        self.model = model.to(device)
        self.server_pubkey = server_pubkey
        self.client_file_index = client_file_index
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols
        self.loss_fn = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.005)
        self.LOCAL_EPOCHS = 3

        self.is_malicious = is_malicious
        self.is_pqc_enabled = is_pqc_enabled
        self.is_classical_enabled = is_classical_enabled # Store classical enabled flag

        # IMPORTANT: Client keys are now passed in and NOT generated here.
        # This ensures consistency with the globally pre-generated keys.
        self.client_dilithium_pub_key = client_dilithium_pub_key
        self.client_dilithium_priv_key = client_dilithium_priv_key
        self.client_falcon_pub_key = client_falcon_pub_key
        self.client_falcon_priv_key = client_falcon_priv_key

        if self.is_pqc_enabled:
            if not all([self.client_dilithium_priv_key, self.client_dilithium_pub_key,
                        self.client_falcon_priv_key, self.client_falcon_pub_key]):
                raise ValueError(f"PQC keys not provided for PQC-enabled client {client_file_index}")


        train_ds, test_ds = load_client_data_internal(self.client_file_index, self.numerical_cols, self.categorical_cols, global_label_encoders)
        train_batch_size = 1 if len(train_ds) == 0 else min(len(train_ds), 32)
        test_batch_size = 1 if len(test_ds) == 0 else min(len(test_ds), 32)
        self.train_loader = DataLoader(train_ds, batch_size=train_batch_size, shuffle=True)
        self.test_loader = DataLoader(test_ds, batch_size=test_batch_size)
        logger.debug(f"Client {self.client_file_index} (PQC: {self.is_pqc_enabled}, Classical: {self.is_classical_enabled}) initialized. Train dataset length: {len(self.train_loader.dataset)}, Test dataset length: {len(self.test_loader.dataset)}")

    def get_parameters(self, config: Dict[str, str] = {}) -> List[np.ndarray]:
        """Returns the current model parameters as a list of NumPy arrays."""
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters: List[np.ndarray]):
        """Sets the model parameters from a list of NumPy arrays."""
        state_dict = dict(zip(self.model.state_dict().keys(), parameters))
        self.model.load_state_dict({k: torch.tensor(v).to(device) for k, v in state_dict.items()})

    def fit(self, parameters: List[np.ndarray], config: Dict[str, str]) -> Tuple[List[np.ndarray], int, Dict[str, Union[bytes, bool, int, float, str]]]:
        self.set_parameters(parameters)
        self.model.train()

        if len(self.train_loader.dataset) == 0:
            logger.warning(f"Client {self.client_file_index} (PQC: {self.is_pqc_enabled}, Classical: {self.is_classical_enabled}) has no training data. Skipping training.")
            return self.get_parameters(), 0, {
                "trained_successfully": False,
                "encrypted_signed_data": pickle.dumps({}),
                "client_file_index": self.client_file_index,
                "encryption_time": 0.0, "signing_time": 0.0,
                "signature_algorithm_used": "N/A", "is_malicious_attempt": False,
                "communication_cost_bytes": 0,
                "scenario_type": "PQC" if self.is_pqc_enabled else ("Classical" if self.is_classical_enabled else "NoCrypto")
            }

        for epoch in range(self.LOCAL_EPOCHS):
            for x_num, x_cat, y in self.train_loader:
                x_num, x_cat, y = x_num.to(device), x_cat.to(device), y.to(device)
                self.optimizer.zero_grad()
                y_pred = self.model(x_num, x_cat)
                loss = self.loss_fn(y_pred, y)
                loss.backward()
                self.optimizer.step()

        weights = self.get_parameters()
        communication_cost_bytes = 0
        encryption_time = 0.0
        signing_time = 0.0
        signature_algorithm = "N/A"
        is_malicious_attempt = False
        full_payload_pickled = pickle.dumps({}) # Default empty payload

        if self.is_pqc_enabled: # PQC scenario
            # PQC Encryption
            start_encryption_time = time.perf_counter()
            encrypted_payload_original = encrypt_model(weights, self.server_pubkey)
            encryption_time = time.perf_counter() - start_encryption_time

            # Adaptive PQC Signature
            data_size = len(self.train_loader.dataset)
            signature_algorithm = "Falcon-512" if data_size >= ADAPTIVE_PQC_THRESHOLD else "Dilithium2"
            
            data_to_sign_bytes = pickle.dumps(encrypted_payload_original)
            
            start_signing_time = time.perf_counter()
            if signature_algorithm == "Dilithium2":
                signature = sign_data(data_to_sign_bytes, self.client_dilithium_priv_key, algorithm="Dilithium2")
                public_key_to_send = self.client_dilithium_pub_key
            else: # Falcon-512
                signature = sign_data(data_to_sign_bytes, self.client_falcon_priv_key, algorithm="Falcon-512")
                public_key_to_send = self.client_falcon_pub_key
            signing_time = time.perf_counter() - start_signing_time

            # Simulate Malicious Tampering
            final_encrypted_payload = encrypted_payload_original
            if self.is_malicious:
                logger.warning(f"MALICIOUS CLIENT {self.client_file_index}: Attempting to tamper with encrypted weights AFTER signing (PQC)!")
                is_malicious_attempt = True
                tampered_encrypted_weights = bytearray(final_encrypted_payload["encrypted_weights"])
                if len(tampered_encrypted_weights) > 0:
                    idx_to_corrupt = random.randint(0, len(tampered_encrypted_weights) - 1)
                    original_byte = tampered_encrypted_weights[idx_to_corrupt]
                    tampered_encrypted_weights[idx_to_corrupt] = original_byte ^ 0xFF
                    final_encrypted_payload = {
                        "ciphertext": final_encrypted_payload["ciphertext"],
                        "encrypted_weights": bytes(tampered_encrypted_weights)
                    }

            # Create full payload for PQC
            full_payload = {
                "encrypted_data": final_encrypted_payload,
                "signature": signature,
                "public_key": public_key_to_send,
                "signature_algorithm_used": signature_algorithm,
            }
            full_payload_pickled = pickle.dumps(full_payload)
            communication_cost_bytes = len(full_payload_pickled)

        elif self.is_classical_enabled: # Classical Cryptography (Simulated) scenario
            # Simulate classical encryption time (e.g., AES symmetric encryption)
            start_encryption_time = time.perf_counter()
            time.sleep(random.uniform(0.00005, 0.00015)) # Small, non-zero time
            encryption_time = time.perf_counter() - start_encryption_time

            # Simulate classical signing time (e.g., RSA/ECDSA signature)
            start_signing_time = time.perf_counter()
            time.sleep(random.uniform(0.00005, 0.0001)) # Small, non-zero time
            signing_time = time.perf_counter() - start_signing_time

            # Payload is just raw weights, but we simulate tampering if malicious
            raw_weights_bytes = pickle.dumps(weights)
            full_payload_pickled = raw_weights_bytes
            communication_cost_bytes = len(raw_weights_bytes)

            if self.is_malicious:
                logger.warning(f"MALICIOUS CLIENT {self.client_file_index}: Attempting to tamper with raw weights (Classical Crypto Simulated)!")
                is_malicious_attempt = True
                tampered_raw_weights = bytearray(full_payload_pickled)
                if len(tampered_raw_weights) > 0:
                    idx_to_corrupt = random.randint(0, len(tampered_raw_weights) - 1)
                    original_byte = tampered_raw_weights[idx_to_corrupt]
                    tampered_raw_weights[idx_to_corrupt] = original_byte ^ 0xFF
                    full_payload_pickled = bytes(tampered_raw_weights)
                    communication_cost_bytes = len(full_payload_pickled)
            
            # For classical, the 'full_payload' structure is simplified as no PQC KEM/Signature objects are involved
            # We just pass the raw (potentially tampered) weights and the simulated times
            full_payload = {
                "encrypted_data": full_payload_pickled, # Here, this is just the raw weights
                "signature": b"", # Dummy signature
                "public_key": b"", # Dummy public key
                "signature_algorithm_used": "Classical_Simulated",
            }
            full_payload_pickled = pickle.dumps(full_payload) # Re-pickle the simplified structure


        else: # No Cryptography scenario
            # No encryption, no signing. Just send raw model weights.
            raw_weights_bytes = pickle.dumps(weights)
            full_payload_pickled = raw_weights_bytes # Payload is just raw weights
            communication_cost_bytes = len(raw_weights_bytes)
            
            if self.is_malicious: # Malicious client still tampers, but no crypto to detect it
                logger.warning(f"MALICIOUS CLIENT {self.client_file_index}: Attempting to tamper with raw weights (No Crypto)!")
                is_malicious_attempt = True
                tampered_raw_weights = bytearray(full_payload_pickled)
                if len(tampered_raw_weights) > 0:
                    idx_to_corrupt = random.randint(0, len(tampered_raw_weights) - 1)
                    original_byte = tampered_raw_weights[idx_to_corrupt]
                    tampered_raw_weights[idx_to_corrupt] = original_byte ^ 0xFF
                    full_payload_pickled = bytes(tampered_raw_weights)
                    communication_cost_bytes = len(full_payload_pickled)
            
            # For No Crypto, the 'full_payload' structure is simplified even further
            full_payload = {
                "encrypted_data": full_payload_pickled, # Here, this is just the raw weights
                "signature": b"", # Dummy signature
                "public_key": b"", # Dummy public key
                "signature_algorithm_used": "No_Crypto",
            }
            full_payload_pickled = pickle.dumps(full_payload) # Re-pickle the simplified structure


        return self.get_parameters(), len(self.train_loader.dataset), {
            "encrypted_signed_data": full_payload_pickled,
            "trained_successfully": True,
            "client_file_index": self.client_file_index,
            "encryption_time": encryption_time,
            "signing_time": signing_time,
            "signature_algorithm_used": signature_algorithm,
            "is_malicious_attempt": is_malicious_attempt,
            "communication_cost_bytes": communication_cost_bytes,
            "scenario_type": "PQC" if self.is_pqc_enabled else ("Classical" if self.is_classical_enabled else "NoCrypto")
        }

    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, str]) -> Tuple[float, int, Dict[str, float]]:
        self.set_parameters(parameters)
        self.model.eval()

        loss, correct, total = 0.0, 0, 0
        if len(self.test_loader.dataset) == 0:
            logger.warning(f"Client {self.client_file_index} (PQC: {self.is_pqc_enabled}, Classical: {self.is_classical_enabled}) has no test data. Returning default evaluation metrics.")
            return 0.0, 0, {"accuracy": 0.0, "loss": 0.0}

        try:
            with torch.no_grad():
                for x_num, x_cat, y in self.test_loader:
                    current_batch_size = y.size(0)
                    x_num, x_cat, y = x_num.to(device), x_cat.to(device), y.to(device)
                    y_pred = self.model(x_num, x_cat)
                    batch_loss = self.loss_fn(y_pred, y).item()
                    loss += batch_loss * current_batch_size
                    pred = (y_pred > 0.5).float()
                    correct += (pred == y).sum().item()
                    total += current_batch_size
        except Exception as e:
            logger.error(f"CRITICAL ERROR: Client {self.client_file_index} (PQC: {self.is_pqc_enabled}, Classical: {self.is_classical_enabled}) encountered an unhandled error during evaluation: {e}")
            import traceback
            traceback.print_exc()
            return 0.0, 0, {"accuracy": 0.0, "loss": 0.0}

        acc = correct / total if total > 0 else 0.0
        final_loss = loss / total if total > 0 else 0.0

        if total == 0 and len(self.test_loader.dataset) > 0:
            logger.warning(f"CRITICAL WARNING: Client {self.client_file_index} (PQC: {self.is_pqc_enabled}, Classical: {self.is_classical_enabled}) expected {len(self.test_loader.dataset)} samples but evaluated 0.")
            return 0.0, 0, {"accuracy": 0.0, "loss": 0.0}

        return final_loss, total, {"accuracy": acc, "loss": final_loss}

class LoggingStrategy(fl.server.strategy.FedAvg):
    """
    Custom Flower strategy that extends FedAvg to include:
    - Decentralized aggregation simulation
    - Conditional PQC decryption/verification
    - Metric collection
    """
    def __init__(self, server_secret: bytes, num_virtual_servers: int, is_pqc_enabled: bool,
                 client_dilithium_pub_keys_map: Dict[int, bytes],
                 client_falcon_pub_keys_map: Dict[int, bytes],
                 metrics_storage: Dict[str, List], # Pass the specific metrics storage dict
                 **kwargs):
        super().__init__(**kwargs)
        self.server_secret = server_secret
        self.num_virtual_servers = num_virtual_servers
        self.is_pqc_enabled = is_pqc_enabled # This flag determines if PQC checks are performed by the server
        # Store these maps directly from the constructor
        self.client_dilithium_pub_keys_map = client_dilithium_pub_keys_map
        self.client_falcon_pub_keys_map = client_falcon_pub_keys_map
        self.metrics_storage = metrics_storage
        self.current_round_rejected_malicious = 0

    def aggregate_fit(self, server_round: int,
                      results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
                      failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes], BaseException]]) -> Tuple[Union[fl.common.Parameters, None], Dict[str, Union[bool, float, int]]]:
        logger.info(f"Aggregating results for round {server_round} (PQC: {self.is_pqc_enabled})...")
        
        server_specific_decrypted_weights: Dict[int, List[List[np.ndarray]]] = {i: [] for i in range(self.num_virtual_servers)}
        server_specific_num_examples: Dict[int, List[int]] = {i: [] for i in range(self.num_virtual_servers)}
        
        self.current_round_rejected_malicious = 0

        round_client_encryption_times = []
        round_client_signing_times = []
        round_server_decryption_times = []
        round_server_verification_times_dilithium = []
        round_server_verification_times_falcon = []
        round_communication_costs_bytes = []

        for client_proxy, fit_res in results:
            client_file_index_from_metrics = fit_res.metrics.get("client_file_index", "UNKNOWN")
            is_malicious_attempt = fit_res.metrics.get("is_malicious_attempt", False)
            scenario_type = fit_res.metrics.get("scenario_type", "Unknown") # Get scenario type from client
            # client_flower_id is the original 0-indexed client ID from Flower
            client_flower_id = int(client_proxy.cid)

            # Use the client_file_index for virtual server assignment as it's derived from the client's data file (1-indexed)
            virtual_server_id = int(client_file_index_from_metrics) % self.num_virtual_servers

            try:
                if not fit_res.metrics.get("trained_successfully", False):
                    logger.info(f"Client {client_file_index_from_metrics} reported no successful training. Skipping.")
                    continue
                if "encrypted_signed_data" not in fit_res.metrics:
                    logger.error(f"Client {client_file_index_from_metrics} did not send 'encrypted_signed_data'. Skipping.")
                    continue

                round_communication_costs_bytes.append(fit_res.metrics.get("communication_cost_bytes", 0))

                full_payload_pickled = fit_res.metrics["encrypted_signed_data"]
                decrypted_weights = []
                is_signature_valid = True # Assume valid for NoCrypto and Classical (simulated)

                if self.is_pqc_enabled: # This LoggingStrategy instance is for PQC scenario
                    full_payload = pickle.loads(full_payload_pickled)
                    encrypted_payload = full_payload.get("encrypted_data")
                    signature = full_payload.get("signature")
                    public_key_from_payload = full_payload.get("public_key")
                    signature_algorithm_used = full_payload.get("signature_algorithm_used")

                    if not encrypted_payload or not signature or not public_key_from_payload or not signature_algorithm_used:
                        logger.error(f"Incomplete PQC payload from client {client_file_index_from_metrics}. Skipping.")
                        if is_malicious_attempt: self.current_round_rejected_malicious += 1
                        continue

                    data_to_verify = pickle.dumps(encrypted_payload)
                    start_verification_time = time.perf_counter()
                    expected_pub_key = None
                    # Use client_file_index_from_metrics to get the correct key from the map
                    # The map keys are 1-indexed, matching client_file_index_from_metrics
                    if signature_algorithm_used == "Dilithium2":
                        expected_pub_key = self.client_dilithium_pub_keys_map.get(int(client_file_index_from_metrics))
                    elif signature_algorithm_used == "Falcon-512":
                        expected_pub_key = self.client_falcon_pub_keys_map.get(int(client_file_index_from_metrics))

                    if expected_pub_key and expected_pub_key == public_key_from_payload:
                        is_signature_valid = verify_signature(data_to_verify, signature, public_key_from_payload, algorithm=signature_algorithm_used)
                        if signature_algorithm_used == "Dilithium2":
                            round_server_verification_times_dilithium.append(time.perf_counter() - start_verification_time)
                        else:
                            round_server_verification_times_falcon.append(time.perf_counter() - start_verification_time)
                    else:
                        logger.error(f"Mismatch or missing expected public key for client {client_file_index_from_metrics}. Verification skipped.")
                        is_signature_valid = False

                    if not is_signature_valid:
                        logger.error(f"❌ Invalid signature from client {client_file_index_from_metrics}. Skipping aggregation.")
                        if is_malicious_attempt: self.current_round_rejected_malicious += 1 # Only PQC detects this
                        continue

                    start_decryption_time = time.perf_counter()
                    decrypted_weights = decrypt_model(encrypted_payload, self.server_secret)
                    round_server_decryption_times.append(time.perf_counter() - start_decryption_time)

                    if not decrypted_weights:
                        logger.error(f"Decryption returned empty weights for client {client_file_index_from_metrics}. Skipping.")
                        if is_malicious_attempt: self.current_round_rejected_malicious += 1 # Only PQC detects this
                        continue
                    if not isinstance(decrypted_weights, list) or not all(isinstance(w, np.ndarray) for w in decrypted_weights):
                        logger.error(f"Decrypted weights from client {client_file_index_from_metrics} are not in expected format. Skipping.")
                        if is_malicious_attempt: self.current_round_rejected_malicious += 1 # Only PQC detects this
                        continue
                else: # This LoggingStrategy instance is for NoCrypto or Classical (simulated)
                    # For these scenarios, there's no cryptographic verification at the server.
                    # We just unpickle the payload, which contains the raw (potentially tampered) weights.
                    full_payload = pickle.loads(full_payload_pickled)
                    decrypted_weights = pickle.loads(full_payload.get("encrypted_data")) # Get the raw weights

                    # Collect client-side timing metrics, which would be non-zero for Classical, zero for NoCrypto
                    round_client_encryption_times.append(fit_res.metrics.get("encryption_time", 0.0))
                    round_client_signing_times.append(fit_res.metrics.get("signing_time", 0.0))
                    # Server-side decryption/verification times are 0 for these scenarios (no actual crypto ops)
                    round_server_decryption_times.append(0.0)
                    round_server_verification_times_dilithium.append(0.0)
                    round_server_verification_times_falcon.append(0.0)

                    if not isinstance(decrypted_weights, list) or not all(isinstance(w, np.ndarray) for w in decrypted_weights):
                        logger.error(f"Weights from client {client_file_index_from_metrics} ({scenario_type}) are not in expected format. Skipping.")
                        # No cryptographic rejection for malicious updates in non-PQC scenarios
                        continue
                
                # Only add to aggregation if signature was valid (for PQC) or if it's non-PQC/classical (where it's always valid by default)
                if is_signature_valid:
                    server_specific_decrypted_weights[virtual_server_id].append(decrypted_weights)
                    server_specific_num_examples[virtual_server_id].append(fit_res.num_examples)
                    
            except Exception as e:
                logger.error(f"❌ Processing failed for client {client_file_index_from_metrics} ({scenario_type}): {e}")
                import traceback
                traceback.print_exc()
                if is_malicious_attempt and self.is_pqc_enabled: # Only PQC detects and rejects cryptographically
                    self.current_round_rejected_malicious += 1

        # Store global average timing metrics for this round
        self.metrics_storage["client_encryption_times"].append(np.mean(round_client_encryption_times) if round_client_encryption_times else 0.0)
        self.metrics_storage["client_signing_times"].append(np.mean(round_client_signing_times) if round_client_signing_times else 0.0)
        self.metrics_storage["server_decryption_times"].append(np.mean(round_server_decryption_times) if round_server_decryption_times else 0.0)
        self.metrics_storage["server_verification_times_dilithium"].append(np.mean(round_server_verification_times_dilithium) if round_server_verification_times_dilithium else 0.0)
        self.metrics_storage["server_verification_times_falcon"].append(np.mean(round_server_verification_times_falcon) if round_server_verification_times_falcon else 0.0)

        self.metrics_storage["total_crypto_overhead"].append(
            self.metrics_storage["client_encryption_times"][-1] +
            self.metrics_storage["client_signing_times"][-1] +
            self.metrics_storage["server_decryption_times"][-1] +
            self.metrics_storage["server_verification_times_dilithium"][-1] +
            self.metrics_storage["server_verification_times_falcon"][-1]
        )
        self.metrics_storage["rejected_malicious_updates"].append(self.current_round_rejected_malicious)
        self.metrics_storage["communication_cost_mb"].append(np.mean(round_communication_costs_bytes) / (1024 * 1024) if round_communication_costs_bytes else 0.0)

        # Simulate Local Aggregation (by each virtual server)
        local_aggregated_params_list = []
        total_examples_across_servers = 0

        for i in range(self.num_virtual_servers):
            virtual_server_id = i
            local_weights = server_specific_decrypted_weights[virtual_server_id]
            local_num_examples = server_specific_num_examples[virtual_server_id]

            if not local_weights:
                logger.warning(f"Virtual server {virtual_server_id} has no valid decrypted weights for aggregation this round. Skipping local aggregation.")
                continue

            first_local_weights_len = len(local_weights[0])
            if not all(len(w) == first_local_weights_len for w in local_weights):
                logger.error(f"Virtual server {virtual_server_id}: Inconsistent weight lengths for local aggregation. Skipping.")
                continue

            local_aggregated_weights = []
            local_total_examples = sum(local_num_examples)
            if local_total_examples == 0:
                logger.warning(f"Virtual server {virtual_server_id}: No examples for local aggregation. Skipping.")
                continue

            for j in range(first_local_weights_len):
                layer_weights = [client_weights[j] for client_weights in local_weights]
                if not layer_weights:
                    logger.warning(f"No weights for layer {j} after filtering. Skipping this layer.")
                    continue
                if not all(w.shape == layer_weights[0].shape for w in layer_weights):
                    logger.error(f"Inconsistent shapes for layer {j} weights. Skipping aggregation for this layer.")
                    continue

                weighted_sum_layer = np.sum([w * n for w, n in zip(layer_weights, local_num_examples)], axis=0)
                aggregated_layer = weighted_sum_layer / local_total_examples
                local_aggregated_weights.append(aggregated_layer)
            
            if local_aggregated_weights:
                local_aggregated_params_list.append( (local_aggregated_weights, local_total_examples) )
                total_examples_across_servers += local_total_examples
            logger.info(f"✅ Virtual server {virtual_server_id} locally aggregated {len(local_weights)} client updates with {local_total_examples} examples.")

        # Simulate Global Aggregation (among virtual servers)
        if not local_aggregated_params_list:
            logger.error("🚨 No valid local aggregated weights available from any virtual server. Returning initial parameters and empty metrics.")
            return super().aggregate_fit(server_round, [], failures)

        final_aggregated_weights = []
        first_global_weights_len = len(local_aggregated_params_list[0][0])

        if not all(len(params) == first_global_weights_len for params, _ in local_aggregated_params_list):
            logger.error("Local aggregated models have inconsistent lengths. Cannot perform global aggregation. Returning initial parameters.")
            return super().aggregate_fit(server_round, [], failures)

        for k in range(first_global_weights_len):
            global_layer_weights = [params[k] for params, _ in local_aggregated_params_list]
            global_num_examples = [num_ex for _, num_ex in local_aggregated_params_list]

            if not global_layer_weights:
                logger.warning(f"No weights for global layer {k} after filtering. Skipping this layer.")
                continue
            if not all(w.shape == global_layer_weights[0].shape for w in global_layer_weights):
                    logger.error(f"Inconsistent shapes for global layer {k} weights. Skipping aggregation for this layer.")
                    continue
            weighted_sum_global_layer = np.sum([w * n for w, n in zip(global_layer_weights, global_num_examples)], axis=0)
            final_aggregated_layer = weighted_sum_global_layer / total_examples_across_servers
            final_aggregated_weights.append(final_aggregated_layer)

        if not final_aggregated_weights:
            logger.error("Final aggregated weights list is empty after global processing. Returning initial parameters.")
            return super().aggregate_fit(server_round, [], failures)

        logger.info(f"✅ Global aggregation across {self.num_virtual_servers} virtual servers completed successfully for round {server_round}")
        return fl.common.ndarrays_to_parameters(final_aggregated_weights), {}


    def aggregate_evaluate(self, rnd: int,
                           results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]],
                           failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes], BaseException]]) -> Tuple[Union[float, None], Dict[str, Union[bool, float, int]]]:
        
        filtered_results = [
            (client_proxy, evaluate_res)
            for client_proxy, evaluate_res in results
            if evaluate_res.num_examples > 0
        ]

        if not filtered_results:
            logger.warning(f"No valid evaluation results (clients with > 0 examples) for round {rnd} (PQC: {self.is_pqc_enabled}). Returning default metrics.")
            self.metrics_storage["accuracy"].append(0.0)
            self.metrics_storage["loss"].append(0.0)
            self.metrics_storage["std_loss"].append(0.0)
            # Store empty list for client eval results if no valid results
            if rnd == NUM_ROUNDS_SIMULATION:
                self.metrics_storage["last_round_client_eval_results"] = []
            return 0.0, {}

        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(rnd, filtered_results, failures)

        accuracy_list = [r.metrics["accuracy"] for _, r in filtered_results if "accuracy" in r.metrics]
        losses_list = [r.metrics["loss"] for _, r in filtered_results if "loss" in r.metrics]

        if accuracy_list:
            avg_accuracy = sum(accuracy_list) / len(accuracy_list)
            logger.info(f"Round {rnd} accuracy (PQC: {self.is_pqc_enabled}): {avg_accuracy:.4f}")
            self.metrics_storage["accuracy"].append(avg_accuracy)
        else:
            self.metrics_storage["accuracy"].append(0.0)

        if losses_list:
            avg_loss = sum(losses_list) / len(losses_list)
            logger.info(f"Round {rnd} loss (PQC: {self.is_pqc_enabled}): {avg_loss:.4f}")
            self.metrics_storage["loss"].append(avg_loss)
            if len(losses_list) > 1:
                std_loss = np.std(losses_list)
                self.metrics_storage["std_loss"].append(std_loss)
            else:
                self.metrics_storage["std_loss"].append(0.0)
        else:
            self.metrics_storage["loss"].append(0.0)
            self.metrics_storage["std_loss"].append(0.0)

        # Collect per-client evaluation results for the current round
        current_round_client_eval_results = []
        for client_proxy, evaluate_res in filtered_results:
            # Use client_proxy.cid directly here, as it's the string ID of the client in Flower
            client_id = int(client_proxy.cid)
            client_acc = evaluate_res.metrics.get("accuracy", 0.0)
            client_loss = evaluate_res.metrics.get("loss", 0.0)
            current_round_client_eval_results.append((client_id, client_acc, client_loss))
        
        # Store these results in the metrics_storage for the final round
        if rnd == NUM_ROUNDS_SIMULATION:
            self.metrics_storage["last_round_client_eval_results"] = current_round_client_eval_results

        return aggregated_loss, aggregated_metrics

def plot_client_performance(client_eval_results: List[Tuple[int, float, float]], round_num: int, title_prefix: str = ""):
    """
    Plots the accuracy and loss for each client in a specific round.
    """
    if not client_eval_results:
        logger.info(f"No client evaluation results to plot for round {round_num} ({title_prefix}).")
        return

    client_ids = [res[0] for res in client_eval_results]
    accuracies = [res[1] for res in client_eval_results]
    losses = [res[2] for res in client_eval_results]

    plt.figure(figsize=(12, 7))
    plt.scatter(client_ids, accuracies, color='blue', label='Accuracy', alpha=0.7)
    plt.scatter(client_ids, losses, color='red', label='Loss', alpha=0.7, marker='x')

    plt.title(f'{title_prefix} Client Performance (Accuracy and Loss) in Round {round_num}')
    plt.xlabel('Client ID (Flower 0-indexed)') # Clarify that this is Flower's internal ID
    plt.ylabel('Value')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(client_ids, rotation=90, fontsize=8)
    plt.ylim(bottom=0)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_comparison_metrics(num_rounds_simulated: int, pqc_metrics_data: Dict[str, List], 
                            no_crypto_metrics_data: Dict[str, List], classical_crypto_metrics_data: Dict[str, List]):
    """
    Plots the accuracy, loss, cryptographic overhead, and communication cost,
    comparing PQC, No Cryptography, and Classical Cryptography (Simulated) scenarios.
    """
    rounds = range(1, num_rounds_simulated + 1)

    # Plot Accuracy Comparison
    plt.figure(figsize=(10, 6))
    plt.plot(rounds, pqc_metrics_data["accuracy"], marker='o', color='skyblue', label='PQC Accuracy')
    plt.plot(rounds, no_crypto_metrics_data["accuracy"], marker='x', color='green', linestyle='--', label='No Cryptography Accuracy')
    plt.plot(rounds, classical_crypto_metrics_data["accuracy"], marker='s', color='purple', linestyle=':', label='Classical Crypto (Simulated) Accuracy')
    plt.title('Federated Learning Accuracy Comparison')
    plt.xlabel('Round')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.xticks(rounds)
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot Loss Comparison
    plt.figure(figsize=(10, 6))
    plt.errorbar(rounds, pqc_metrics_data["loss"], yerr=pqc_metrics_data["std_loss"], fmt='-o', capsize=5, color='salmon', ecolor='lightcoral', elinewidth=1, label='PQC Loss (with Std Dev)')
    plt.errorbar(rounds, no_crypto_metrics_data["loss"], yerr=no_crypto_metrics_data["std_loss"], fmt='-x', capsize=5, color='darkgreen', ecolor='lightgreen', elinewidth=1, label='No Cryptography Loss (with Std Dev)', linestyle='--')
    plt.errorbar(rounds, classical_crypto_metrics_data["loss"], yerr=classical_crypto_metrics_data["std_loss"], fmt='-s', capsize=5, color='darkviolet', ecolor='mediumpurple', elinewidth=1, label='Classical Crypto (Simulated) Loss (with Std Dev)', linestyle=':')
    plt.title('Federated Learning Loss Comparison')
    plt.xlabel('Round')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.xticks(rounds)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot Total Cryptographic Overhead Comparison
    plt.figure(figsize=(10, 7))
    plt.plot(rounds, pqc_metrics_data["total_crypto_overhead"], marker='o', color='purple', label='Total PQC Crypto Overhead')
    plt.plot(rounds, no_crypto_metrics_data["total_crypto_overhead"], marker='x', color='orange', linestyle='--', label='Total No Cryptography Overhead')
    plt.plot(rounds, classical_crypto_metrics_data["total_crypto_overhead"], marker='s', color='darkblue', linestyle=':', label='Total Classical Crypto (Simulated) Overhead')
    plt.title('Total Cryptographic Overhead per Round')
    plt.xlabel('Round')
    plt.ylabel('Total Time (seconds)')
    plt.grid(True)
    plt.xticks(rounds)
    plt.legend()
    plt.yscale('log')
    plt.tight_layout()
    plt.show()

    # Plot Communication Cost (Bandwidth) Comparison in MB
    plt.figure(figsize=(10, 7))
    plt.plot(rounds, pqc_metrics_data["communication_cost_mb"], marker='o', color='blue', label='PQC Communication Cost (MB)')
    plt.plot(rounds, no_crypto_metrics_data["communication_cost_mb"], marker='x', color='red', linestyle='--', label='No Cryptography Communication Cost (MB)')
    plt.plot(rounds, classical_crypto_metrics_data["communication_cost_mb"], marker='s', color='darkgreen', linestyle=':', label='Classical Crypto (Simulated) Communication Cost (MB)')
    plt.title('Average Communication Cost per Round')
    plt.xlabel('Round')
    plt.ylabel('Average Data Transfer per Client (MB)')
    plt.grid(True)
    plt.xticks(rounds)
    plt.legend()
    plt.yscale('log')
    plt.tight_layout()
    plt.show()

    # Plot Rejected Malicious Updates (PQC vs No Crypto vs Classical Crypto Simulated)
    plt.figure(figsize=(10, 6))
    bar_width = 0.25 # Adjusted for three bars
    index = np.arange(len(rounds))

    plt.bar(index - bar_width, pqc_metrics_data["rejected_malicious_updates"], bar_width, label='PQC Rejected Malicious Updates', color='red')
    plt.bar(index, no_crypto_metrics_data["rejected_malicious_updates"], bar_width, label='No Cryptography Rejected Malicious Updates', color='grey', alpha=0.7)
    plt.bar(index + bar_width, classical_crypto_metrics_data["rejected_malicious_updates"], bar_width, label='Classical Crypto (Simulated) Rejected Malicious Updates', color='lightcoral', alpha=0.7)


    plt.title('Rejected Malicious Updates per Round')
    plt.xlabel('Round')
    plt.ylabel('Number of Rejected Updates')
    plt.grid(axis='y')
    plt.xticks(index, rounds) # Center x-ticks
    # Ensure y-ticks are integers and go up to at least the max value + 1
    max_rejected = max(
        max(pqc_metrics_data["rejected_malicious_updates"]),
        max(no_crypto_metrics_data["rejected_malicious_updates"]),
        max(classical_crypto_metrics_data["rejected_malicious_updates"])
    )
    plt.yticks(range(int(max_rejected) + 2))
    plt.legend()
    plt.tight_layout()
    plt.show()


def preprocess_data_for_encoders(data_dir: str, numerical_cols: List[str], categorical_cols: List[str]) -> Tuple[Dict[str, LabelEncoder], List[int]]:
    """
    Scans all client data files to fit global LabelEncoders for categorical columns
    and determine their cardinalities. This ensures consistent encoding across all clients.
    """
    logger.info("Starting data preprocessing to fit global LabelEncoders...")
    all_categorical_data = {col: set() for col in categorical_cols}

    # Iterate through client files from 1 to TOTAL_CLIENTS (inclusive)
    for i in range(1, TOTAL_CLIENTS + 1): # Changed range to start from 1
        path = os.path.join(data_dir, f"client_{i}.csv")
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                for col in categorical_cols:
                    if col in df.columns:
                        all_categorical_data[col].update(df[col].astype(str).unique())
            except Exception as e:
                logger.warning(f"Could not read client_{i}.csv for preprocessing: {e}. Skipping this file.")
        else:
            logger.warning(f"Client file client_{i}.csv not found during preprocessing. Skipping.")

    fitted_encoders = {}
    cardinalities = []
    for col in categorical_cols:
        if all_categorical_data[col]:
            le = LabelEncoder()
            unique_categories = sorted(list(all_categorical_data[col]))
            if 'Unknown' not in unique_categories:
                unique_categories.insert(0, 'Unknown')
            le.fit(unique_categories)
            fitted_encoders[col] = le
            cardinalities.append(len(le.classes_))
            logger.info(f"Fitted LabelEncoder for '{col}' with {len(le.classes_)} classes.")
        else:
            logger.warning(f"No data found for categorical column '{col}' across all clients. Assigning a default cardinality of 2.")
            le = LabelEncoder()
            le.fit(['0', '1'])
            fitted_encoders[col] = le
            cardinalities.append(2)

    logger.info("Finished data preprocessing.")
    return fitted_encoders, cardinalities

def run_single_simulation(is_pqc_enabled: bool, is_classical_enabled: bool, num_rounds: int, current_metrics_storage: Dict[str, List]):
    """
    Runs a single Flower simulation with specified PQC or Classical enablement.
    Collects and stores metrics in the provided metrics_storage dictionary.
    """
    scenario_name = "PQC-enabled" if is_pqc_enabled else ("Classical Crypto (Simulated)" if is_classical_enabled else "No Cryptography")
    logger.info(f"\n--- Starting {scenario_name} Simulation ---")

    # Clear metrics storage for the new run
    for key in current_metrics_storage:
        current_metrics_storage[key].clear()

    # Define client_fn based on PQC/Classical enablement
    # Changed client_creator signature to directly accept cid: str
    def get_client_fn(is_pqc_enabled_closure: bool, is_classical_enabled_closure: bool):
        def client_creator(cid: str) -> fl.client.Client: # Directly accepts cid as string
            client_file_index = int(cid) + 1 # Adjusted to map Flower's 0-indexed CID to 1-indexed file names
            model = CreditRiskNet(num_numerical=len(numerical_cols), cat_cardinalities=cat_cardinalities)

            # Pass the pre-generated keys directly to the FLClient constructor
            fl_client_instance = FLClient(
                model=model,
                server_pubkey=server_kyber_public_key,
                client_file_index=client_file_index,
                numerical_cols=numerical_cols,
                categorical_cols=categorical_cols,
                global_label_encoders=global_label_encoders,
                is_malicious=(client_file_index == MALICIOUS_CLIENT_FILE_INDEX), # Malicious client enabled for all scenarios
                is_pqc_enabled=is_pqc_enabled_closure,
                is_classical_enabled=is_classical_enabled_closure,
                client_dilithium_priv_key=client_dilithium_priv_keys_map.get(client_file_index),
                client_dilithium_pub_key=client_dilithium_pub_keys_map.get(client_file_index),
                client_falcon_priv_key=client_falcon_priv_keys_map.get(client_file_index),
                client_falcon_pub_key=client_falcon_pub_keys_map.get(client_file_index)
            )
            return fl_client_instance.to_client() # Convert NumPyClient to Client
        return client_creator

    # Initialize the global model (used as the initial state for clients)
    initial_model = CreditRiskNet(num_numerical=len(numerical_cols), cat_cardinalities=cat_cardinalities)
    initial_parameters = [val.cpu().numpy() for val in initial_model.state_dict().values()]

    # Define the custom federated learning strategy
    strategy = LoggingStrategy(
        server_secret=server_kyber_secret_key if is_pqc_enabled else None, # Server secret only needed for PQC
        num_virtual_servers=NUM_SERVERS,
        is_pqc_enabled=is_pqc_enabled, # This tells the strategy whether to perform PQC verification/decryption
        client_dilithium_pub_keys_map=client_dilithium_pub_keys_map, # Pass the global maps to the strategy
        client_falcon_pub_keys_map=client_falcon_pub_keys_map,       # for verification
        metrics_storage=current_metrics_storage,
        fraction_fit=0.1,
        fraction_evaluate=0.05,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=TOTAL_CLIENTS,
        initial_parameters=fl.common.ndarrays_to_parameters(initial_parameters),
        evaluate_fn=None
    )

    # Start the Flower simulation
    fl.simulation.start_simulation(
        client_fn=get_client_fn(is_pqc_enabled, is_classical_enabled),
        num_clients=TOTAL_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0},
        # Add ray_init_args to explicitly set object_store_memory
        ray_init_args={"object_store_memory": 100 * 1024 * 1024} # Set to 100MB, adjust if needed
    )
    logger.info(f"--- {scenario_name} Simulation Finished ---")


if __name__ == "__main__":
    NUM_ROUNDS_SIMULATION = 15 # You can change this value

    # 1. Preprocess data to fit global LabelEncoders and get cardinalities for model initialization
    global_label_encoders, cat_cardinalities = preprocess_data_for_encoders(DATA_DIR, numerical_cols, categorical_cols)

    # 2. Generate Server's Kyber Key Pair (for KEM)
    with oqs.KeyEncapsulation("Kyber512") as kem:
        server_kyber_public_key = kem.generate_keypair()
        server_kyber_secret_key = kem.export_secret_key()
    logger.info("Server Kyber key pair generated.")

    # 3. Generate ALL Client's PQC Key Pairs (Dilithium and Falcon for signatures) once
    # These are generated centrally and then passed to individual clients based on their ID
    # Client file indices are 1-based, so generate keys for 1 to TOTAL_CLIENTS inclusive.
    for i in range(1, TOTAL_CLIENTS + 1): # Changed range to start from 1
        dilithium_pub, dilithium_priv = generate_dilithium_key_pair("Dilithium2")
        falcon_pub, falcon_priv = generate_falcon_key_pair("Falcon-512")
        client_dilithium_pub_keys_map[i] = dilithium_pub
        client_dilithium_priv_keys_map[i] = dilithium_priv
        client_falcon_pub_keys_map[i] = falcon_pub
        client_falcon_priv_keys_map[i] = falcon_priv
    logger.info(f"{TOTAL_CLIENTS} client PQC key pairs generated centrally for all potential clients (1-indexed).")

    # 4. Run PQC-enabled simulation
    run_single_simulation(is_pqc_enabled=True, is_classical_enabled=False, num_rounds=NUM_ROUNDS_SIMULATION, current_metrics_storage=pqc_metrics)

    # 5. Run No Cryptography simulation
    run_single_simulation(is_pqc_enabled=False, is_classical_enabled=False, num_rounds=NUM_ROUNDS_SIMULATION, current_metrics_storage=no_crypto_metrics)

    # 6. Run Classical Cryptography (Simulated) simulation
    run_single_simulation(is_pqc_enabled=False, is_classical_enabled=True, num_rounds=NUM_ROUNDS_SIMULATION, current_metrics_storage=classical_crypto_metrics)

    # 7. Plot the collected metrics for comparison
    plot_comparison_metrics(
        NUM_ROUNDS_SIMULATION,
        pqc_metrics_data=pqc_metrics,
        no_crypto_metrics_data=no_crypto_metrics,
        classical_crypto_metrics_data=classical_crypto_metrics
    )
    logger.info("Comparison plots generated.")

    # 8. Plot client-wise performance for the last round of each scenario
    if "last_round_client_eval_results" in pqc_metrics:
        plot_client_performance(pqc_metrics["last_round_client_eval_results"], NUM_ROUNDS_SIMULATION, "PQC")
    else:
        logger.warning("PQC client-wise performance data for the last round is not available.")
    
    if "last_round_client_eval_results" in no_crypto_metrics:
        plot_client_performance(no_crypto_metrics["last_round_client_eval_results"], NUM_ROUNDS_SIMULATION, "No Cryptography")
    else:
        logger.warning("No Cryptography client-wise performance data for the last round is not available.")

    if "last_round_client_eval_results" in classical_crypto_metrics:
        plot_client_performance(classical_crypto_metrics["last_round_client_eval_results"], NUM_ROUNDS_SIMULATION, "Classical Crypto (Simulated)")
    else:
        logger.warning("Classical Crypto (Simulated) client-wise performance data for the last round is not available.")
