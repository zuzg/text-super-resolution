# 📝 text-super-resolution
## Done
 - Problem: 3
 - Model: 1+2+1
 - Model additional: 1+1
 - Dataset: 1+1
 - HP tuning: 1
 - Few optimizers: 1
 - Various loss functions: 1
 - Neptune: 1
 - Docker: 1

**Sum: 16**
## Bibliography
1. TextZoom Dataset: https://paperswithcode.com/dataset/textzoom, https://github.com/JasonBoy1/TextZoom, https://arxiv.org/pdf/2005.03341v3.pdf
2. NEOCR Dataset: http://www.iapr-tc11.org/dataset/NEOCR/neocr_metadata_doc.pdf
3. Super-resolution: https://arxiv.org/pdf/2103.02368v1.pdf
4. Models:
    * SRResNet: https://arxiv.org/pdf/1609.04802.pdf
    * Text Gestalt: https://arxiv.org/pdf/2112.08171v1.pdf, https://arxiv.org/pdf/1706.03762v5.pdf
    * ESRGAN: https://arxiv.org/pdf/1809.00219.pdf

## How to use the notebook
1. Download the data first and paste it into `/data` directory
2. In order to use **neptune.ai** you need to provide your api token (paste your token in ./cfg/tokens/api_token.yaml file in the format `token: <your-api-token>`).
3. Prepare environment using one of two options:
   - Install dependencies using `pip install requirements.txt` and set `PYTHONPATH=src`
   - Use Docker:
       - `docker build -t <image-name> .`
       - `docker run -p 8888:8888 <image-name>`

## Runtime environment
We trained our models with cuda, having two independent gpus available:
- NVIDIA GeForce GTX 1060 6GB
- NVIDIA GeForce RTX 2080 Ti 11GB
