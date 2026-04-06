FUDefense: Towards Robust Recovery from Poisoning Attacks in Federated Learning

FUDefense is a lightweight, plug-and-play framework designed to enhance the robustness and efficiency of Federated Unlearning (FU). By integrating Parameter Hypothesis Testing (HT) with Offline Reinforcement Learning (RL), FUDefense effectively identifies malicious updates and optimizes client selection during the recovery phase.This repository supports the integration of FUDefense with several baseline unlearning algorithms and provides a comprehensive suite for comparison against state-of-the-art online federated defense mechanisms.1. Environment SetupWe recommend using Python 3.10 and a Conda environment for dependency management.

# Create and activate the environment
conda create -n fudefense python=3.10 -y
conda activate fudefense

# Install core dependencies
conda install pytorch==2.0.1 torchvision==0.15.2 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt
2. Experimental ConfigurationTraining HyperparametersLocal Training: Each client performs 5 local epochs per communication round.Optimizer: Stochastic Gradient Descent (SGD) without momentum.Learning Rate: Fixed at 0.005.Batch Size: 64.Dataset GenerationSupported datasets: MNIST, Fashion-MNIST, CIFAR-10/100. Run the preprocessing scripts to partition data:Bash# For IID and balanced scenarios
python generate_mnist.py iid balance -

# For Practical Non-IID (Dirichlet) scenarios
python generate_mnist.py noniid - dir
Note: Please delete existing folders in data/dataset_name/train or test before re-splitting.3. Framework ArchitectureFUDefense is composed of two primary modules:Hypothesis Testing (HT): A statistical module based on the Log-Normal distribution that calculates relative deviations to filter out anomalous client updates.Reinforcement Learning (RL): An offline Q-Learning agent that dynamically selects the optimal client subset based on historical performance (Accuracy/Loss) to accelerate model recovery.Workspace StructureTo facilitate ablation studies and comparative analysis, the repository is divided into two public workspaces:Historical Unlearning Workspace (*_jy.py): Includes crab_jy.py, eraser_jy.py, and recover_jy.py. This area is used to integrate FUDefense modules into baselines (FedEraser, FedRecover, Crab) to create variants.Online Defense Workspace (*_jy_zx.py): Includes crab_jy_zx.py etc. This area is used to benchmark against online defense algorithms (SCC, FedRo, Viceroy).4. Robustness & Attack VectorsWe evaluate the framework against Byzantine adversaries by varying the fraction of malicious clients from 10% to 50%.Attack Scenarios:Backdoor Attacks: 4×4 white-pixel triggers with configurable label injection (Fix, Random, Exclusive).Gradient Poisoning:LIE (Little Is Enough): Noise scaling at 0.8.SF (Sign Flipping): Gradient sign inversion.Pruning Attacks: Gaussian noise injection or 10% parameter corruption.5. Unlearning Parameters & MetricsUnlearning Setup:Time Window Generation Ratio ($\alpha$): 0.1Buffer Window Ratio ($\lambda$): 0.8Client Participation Ratio ($\delta$): 0.8Adaptive Rollback Threshold ($\beta$): 0.3Evaluation Metrics:Accuracy: Overall prediction correctness.Precision/Recall/F1-Score: Evaluated to determine class-specific performance and unlearning effectiveness.Training Time: Measures the computational efficiency and speed of the recovery process.6. Execution GuideUse FedMoss.py to start the end-to-end simulation. Example for running the Crab algorithm integrated with FUDefense:
1.python FedMoss.py -data fmnist -verify -algo Crab -unlearn 10
2.python FedMoss.py -verify -algo Crab -data fmnist -unlearn 5 -backdoor -clamp -gr 20 -robust Krum

For rapid simulations, you may also refer to the provided run.sh script, which contains pre-configured command-line samples.For detailed parameter descriptions and advanced options, please refer to the source code in FedMoss.py.
