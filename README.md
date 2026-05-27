#  Secure Federated Learning Framework with Post-Quantum Cryptography

An advanced research-oriented Federated Learning framework integrating **Post-Quantum Cryptography (PQC)**, secure aggregation mechanisms, adaptive signature schemes, and decentralized client-server training using **Flower, PyTorch, and Open Quantum Safe (liboqs)**.

This project focuses on enhancing the security of Federated Learning systems against future quantum attacks by implementing **Kyber-based encryption** and **Dilithium/Falcon digital signatures** while evaluating system performance, communication overhead, and malicious client detection.

This work was developed as part of a research and academic publication.

---

#  Research Contribution

This project was implemented and evaluated for a published research paper focusing on:

* Secure Federated Learning
* Post-Quantum Cryptography (PQC)
* Adaptive Cryptographic Signature Schemes
* Malicious Client Detection
* Privacy-Preserving Distributed AI Systems

---

#  Key Features

##  Federated Learning Framework

* Distributed model training using Flower (FLWR)
* Multi-client federated simulation
* Decentralized training architecture
* PyTorch-based deep learning integration

##  Post-Quantum Cryptography Integration

* Kyber-based secure key exchange
* Dilithium digital signatures
* Falcon signature algorithm support
* Adaptive PQC selection based on client workload

##  Secure Aggregation Pipeline

* Encrypted model parameter exchange
* Client-side signing of updates
* Server-side signature verification
* Secure communication workflow

##  Malicious Client Detection

* Simulated malicious client attacks
* Tampering detection support
* Rejection of unauthorized model updates
* Security audit metrics

##  Performance Evaluation

* Accuracy and loss tracking
* Cryptographic overhead analysis
* Communication cost evaluation
* Encryption and verification time measurement

##  Comparative Analysis

The framework compares:

* PQC-enabled Federated Learning
* Classical cryptography simulation
* Federated Learning without cryptography

---

#  Tech Stack

| Technology                 | Purpose                      |
| -------------------------- | ---------------------------- |
| Python                     | Core Development             |
| PyTorch                    | Deep Learning Framework      |
| Flower (FLWR)              | Federated Learning Framework |
| Open Quantum Safe (liboqs) | Post-Quantum Cryptography    |
| NumPy                      | Numerical Processing         |
| Pandas                     | Dataset Handling             |
| Matplotlib                 | Visualization & Analysis     |
| Scikit-learn               | Data Preprocessing           |
| Cryptography Library       | Secure Encryption Support    |

---

#  System Architecture

The system follows a secure federated learning pipeline:

1. Dataset is partitioned across multiple clients
2. Clients train local deep learning models
3. Model parameters are encrypted and signed
4. Secure updates are transmitted to the server
5. Server verifies signatures and aggregates updates
6. Malicious or tampered updates are rejected
7. Global model is redistributed to clients

---

#  Project Structure

```bash
Secure-Federated-Learning-PQC/
│
├── model.py                              # Main federated learning implementation
├── requirements.txt                      # Project dependencies
├── README.md                             # Project documentation
│
├── IN_2_clients_partitioned_multi/
│   └── client datasets
│
├── results/
│   ├── accuracy graphs
│   ├── overhead analysis
│   └── communication metrics
│
├── papers/
│   └── research publication PDFs
│
└── visualizations/
```

---

#  Installation & Setup

## 1️ Clone the Repository

```bash
git clone https://github.com/your-username/secure-federated-learning-pqc.git
cd secure-federated-learning-pqc
```

---

## 2️ Create Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3️ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4️ Install Open Quantum Safe (liboqs)

Install liboqs and Python bindings:

```bash
pip install oqs
```

For detailed installation instructions:

urlOpen Quantum Safe Project[https://openquantumsafe.org/](https://openquantumsafe.org/)

---

## 5️ Configure Dataset Path

Update dataset directory path inside:

```python
DATA_DIR = "path_to_client_dataset"
```

---

## 6️ Run the Federated Learning Simulation

```bash
python model.py
```

---

#  Core Modules

## 🤖 Federated Learning Engine

* Multi-client FL training
* Flower framework integration
* Global model aggregation

## 🔐 PQC Security Module

* Kyber encryption
* Dilithium signatures
* Falcon signatures
* Secure key management

## ⚠ Attack Simulation Module

* Malicious client simulation
* Tampered update rejection
* Integrity verification

## 📊 Metrics & Evaluation Module

* Accuracy measurement
* Cryptographic overhead tracking
* Communication cost analysis
* Security benchmarking

## 📈 Visualization Module

* Training graphs
* Performance comparison plots
* Cryptographic analysis charts

---

# 📊 Evaluation Metrics

The framework evaluates:

| Metric                | Description                    |
| --------------------- | ------------------------------ |
| Accuracy              | Global model performance       |
| Loss                  | Training convergence           |
| Encryption Time       | Client-side crypto overhead    |
| Verification Time     | Server-side validation time    |
| Communication Cost    | Model transmission overhead    |
| Rejected Updates      | Malicious detection capability |
| Total Crypto Overhead | Security performance tradeoff  |

---

#  Security Features

* Post-Quantum Secure Encryption
* Digital Signature Verification
* Secure Federated Aggregation
* Tampering Detection
* Adaptive Cryptographic Selection
* Privacy-Preserving Training

---

#  Research Objectives

This project aims to:

* Secure Federated Learning systems against quantum threats
* Evaluate practical PQC integration in distributed AI
* Analyze computational overhead of PQC mechanisms
* Improve trust and integrity in collaborative machine learning

---

#  Future Enhancements

* Real-time federated deployment
* Blockchain-integrated secure aggregation
* Differential privacy support
* Homomorphic encryption integration
* Edge-device optimization
* Kubernetes-based distributed deployment
* Federated learning dashboard
* Advanced anomaly detection for adversarial clients

---

#  Learning Outcomes

This project demonstrates:

* Federated Learning architecture
* Secure distributed AI systems
* Post-Quantum Cryptography implementation
* Privacy-preserving machine learning
* Deep learning with PyTorch
* Security benchmarking and evaluation
* Research-oriented system design

---

#  Publication & Research

This repository accompanies a research publication related to secure federated learning and post-quantum cryptography.

If you use this work in academic research, please cite the corresponding paper appropriately.

---

#  Authors

* Chandana
* Research Team

---

#  License

This project is intended for academic, educational, and research purposes.

---

# 🌟 Acknowledgements

Special thanks to:

* Flower Federated Learning Framework
* Open Quantum Safe (OQS)
* PyTorch Community
* Research mentors and collaborators
