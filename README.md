# Deep-Learning-Enhanced-Microfluidic-Biosensing

The prevalence of foodborne bacteria presents a significant public health risk, emphasizing the urgent need for rapid and sensitive detection methods to ensure food safety. This study introduces a novel approach: a deep-learning-assisted microfluidic fluorescence digital analysis technique designed for detecting pathogens at ultra-low concentrations. The Staggered Herringbone Double-Spiral (SHDS) multifunctional microfluidic biosensor integrates processes for capturing, detecting, and releasing bacteria. Concanavalin A (Con A) is employed for specific bacteria capture, while Quantum dot (QD)-Aptamer with signal amplification ensures precise identification and fluorescence detection. The biosensor operates within a linear detection range from 10 to 3×10⁶ CFU/mL (R²=0.990), effectively enriching and detecting Escherichia coli (E. coli) within 1.5 hours, achieving 100% capture efficiency. Moreover, the biosensor can release captured E. coli in a mildly acidic environment for off-chip analysis. Testing on chicken and milk samples yielded consistent results compared to the standard culture method, with recoveries ranging from 96.7% to 106.1% in spiked testing. Integration of deep learning algorithms during image processing further enhances sensitivity for bacteria detection compared to traditional methods such as ImageJ, achieving an ultra-low detection limit (LOD) of 2 CFU/mL, which is 10 times lower than that of ImageJ. Furthermore, direct determination of bacteria concentration from fluorescence images simplifies data processing, reducing the time needed for complex numerical measurements and calculations. These findings highlight microfluidic fluorescence digital analysis as a promising tool for rapid pathogen detection in complex food matrices, crucial for ensuring food safety.

![模型示意图](https://github.com/jay-mini/Deep-Learning-Enhanced-Microfluidic-Biosensing/blob/master/CNN_Detection/model_diagram.jpg)

## Running the Code

### Prerequisites

- Python 3.6
- CUDA 12.1
- PyTorch 1.10.2
- torchvision 0.11.3
- PIL
- - Other dependencies as listed in `requirements.txt`

### Hardware

- NVIDIA A100 GPU

### Setup

1. Clone the repository:
    ```bash
    git clone https://github.com/jay-mini/Deep-Learning-Enhanced-Microfluidic-Biosensing.git
    cd Deep-Learning-Enhanced-Microfluidic-Biosensing
    ```

2. Create a virtual environment and activate it:
    ```bash
    python3.6 -m venv venv
    source venv/bin/activate
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Model

To run the model, use the following command:
```bash
python main.py --input your_input_data

