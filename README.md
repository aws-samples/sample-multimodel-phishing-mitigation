# 🔒 Multi-Model Phishing Detection on AWS

A three-stage ensemble pipeline combining a LightGBM URL classifier, a DistilBERT DOM classifier, and Claude on Amazon Bedrock — deployed on Amazon SageMaker AI.

## Technical Architecture

**Stage 1a — LightGBM URL Classifier**
- Model: LightGBM gradient-boosted trees
- Deployment: Amazon SageMaker AI real-time endpoint (`ml.m5.large`)
- Input: 43 lexical/structural URL features extracted by `feature_utils.py`
- Output: Multiclass probabilities (benign / phishing / defacement / malware)
- Performance: ~99% accuracy · AUROC 0.9999 · <0.5% FNR · <0.02% FPR on 130k test URLs

**Stage 1b — DistilBERT DOM Classifier**
- Model: DistilBERT fine-tuned for sequence classification
- Deployment: Amazon SageMaker AI HuggingFace endpoint (`ml.m5.xlarge`)
- Input: Structured DOM text (title, forms, inputs, links, visible text), tokenized to 512 tokens
- Output: P(phishing)
- Training data: 1,000 synthetic phishing pages (Claude / Amazon Bedrock) + crawled benign pages

**Stage 2 — Claude VLM Analysis**
- Model: `us.anthropic.claude-sonnet-4-6` via Amazon Bedrock Converse API
- Deployment: Serverless (Amazon Bedrock on-demand)
- Input: Base64 screenshot or rendered HTML + URL context
- Output: `{ phishing_prob, reasoning }` — brand impersonation, urgency tactics, credential harvesting
- Trigger: conditional on preliminary score (see Decision Engine)

**Decision Engine**
- Preliminary score: `0.35 × url_score + 0.45 × dom_score`
- `score ≤ 0.25` → ALLOW (auto-pass)
- `score ≥ 0.85` → AUTO_BLOCK (auto-block)
- `0.25 < score < 0.85` and `score > 0.40` → trigger Stage 2 (Claude VLM)
- Final score (with VLM): `0.35 × url_score + 0.45 × dom_score + 0.20 × vlm_score`
- Final decision: AUTO_BLOCK / HUMAN_REVIEW / ALLOW
- **HUMAN_REVIEW** is intended to feed a downstream analyst queue — it does not automatically block or allow traffic. Integrate with a ticketing or review workflow before deploying in production.

**Dataset — URL Classifier**
- Source: [Kaggle Malicious URLs dataset](https://www.kaggle.com/datasets/sid321axn/malicious-urls-dataset)
- Size: 651k URLs (130k held out for test)
- Labels: benign, phishing, defacement, malware

**Latency**
- Without VLM: 160–200 ms
- With VLM: 3–4 s

## Repository Structure

```
multi-model-phishing-mitigation/
├── 01_urlclassification_lightgbm.ipynb   # Train + deploy LightGBM URL classifier
├── 02_htmlclassification.ipynb           # Fine-tune + deploy DistilBERT DOM classifier
├── 03_phishing_detection_pipeline.ipynb  # Integration pipeline, evaluation, cost estimate
├── feature_utils.py                      # Shared URL feature extraction (43 features)
├── app.py                                # Streamlit UI
├── utils.py                              # Shared crawl utilities + Tranco benign-URL builder (notebook 02)
├── html_source/
│   └── train.py                          # SageMaker AI HuggingFace training script
└── requirements.txt
```

## Notebook Workflow

The notebooks are designed to run sequentially. Endpoint names produced in Steps 1 and 2 are needed as inputs for Step 3.

**`01_urlclassification_lightgbm.ipynb`**
Downloads the Kaggle URL dataset, extracts 43 lexical features via `feature_utils.py`, trains LightGBM with Amazon SageMaker JumpStart, and deploys a real-time endpoint. Note the endpoint name printed at the end.

**`02_htmlclassification.ipynb`**
Generates 1,000 synthetic phishing HTML pages via Claude on Amazon Bedrock (20 brand scenarios × 50 variations), combines them with cached benign pages from `html_source/`, fine-tunes DistilBERT with an Amazon SageMaker AI HuggingFace Estimator, and deploys a real-time endpoint. Note the endpoint name printed at the end.

> **Benign HTML sourcing:** The notebook calls `crawl_benign_pages()` from `utils.py`. On first run it automatically downloads the [Tranco](https://tranco-list.eu/) top sites list, writes the benign URL list to `benign_urls.csv` locally, and crawls those pages into `html_cache/`. No manual steps required before running the notebook.

**`03_phishing_detection_pipeline.ipynb`**
Wires both endpoints and Amazon Bedrock into the full three-stage pipeline. Update the config variables in the first cell before running:

```python
URL_ENDPOINT_NAME = '<YOUR_URL_ENDPOINT_NAME>'   # from Step 1
DOM_ENDPOINT_NAME = '<YOUR_DOM_ENDPOINT_NAME>'   # from Step 2
VLM_MODEL_ID      = 'us.anthropic.claude-sonnet-4-6'
AWS_REGION        = 'us-east-1'
```

Includes four test cases (Chase phishing, benign blog, IRS lure, batch mode), a cost estimate calculator, and a cleanup cell to delete both endpoints.

Once both endpoints are live you can also run the Streamlit UI — see the [Streamlit UI](#streamlit-ui) section below.

## Streamlit UI

`app.py` is a demo UI for testing the pipeline without writing code. Run it from the repo root, since it imports `feature_utils.py`. On the SageMaker AI notebook instance, open a terminal and run:

```bash
streamlit run app.py --server.port 8501
```

Access it via the SageMaker AI notebook proxy URL:

```
https://<notebook-instance-name>.notebook.<region>.sagemaker.aws/proxy/8501/
```

The trailing slash is required. You can find your notebook instance name in the SageMaker AI console under **Notebook instances**.

**Sidebar configuration**
- URL endpoint name (from Step 1)
- DOM endpoint name (from Step 2)
- VLM model ID and AWS region
- Toggle to enable/disable the Claude VLM stage

**Features**
- Paste any URL + raw HTML and click **Analyze**
- Three built-in presets: Chase phishing, IRS tax lure, benign tech blog
- Displays decision badge (AUTO_BLOCK / HUMAN_REVIEW / ALLOW), per-model score bars, Claude VLM reasoning, per-stage latency, and raw model output

The app requires both SageMaker AI endpoints to be running. On first run it downloads the `distilbert-base-uncased` tokenizer from Hugging Face, so it needs internet access.

`app.py` has its own copy of the pipeline logic. If you change the pipeline in `03_phishing_detection_pipeline.ipynb`, make the same change in `app.py`.

**Running locally**

You can also run the app on your own machine instead of the notebook instance:

1. From the repo root, install dependencies: `pip install -r requirements.txt`
2. Configure AWS credentials for the account and region where both endpoints are deployed, for example `aws sso login --profile <profile>` and `export AWS_PROFILE=<profile>`.
3. Start the app: `streamlit run app.py --server.port 8501`
4. Open http://localhost:8501 and set the endpoint names and region in the sidebar.

**IAM permissions for the app**

The identity running the app needs only these permissions, separate from the training and execution role below. The Converse API is authorized with `bedrock:InvokeModel`. The default model ID `us.anthropic.claude-sonnet-4-6` is a cross-region inference profile, so the policy must allow both the inference profile and the foundation model in each region the profile routes to.

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "InvokePhishingEndpoints",
      "Effect": "Allow",
      "Action": "sagemaker:InvokeEndpoint",
      "Resource": [
        "arn:aws:sagemaker:<region>:<account-id>:endpoint/<url-endpoint-name>",
        "arn:aws:sagemaker:<region>:<account-id>:endpoint/<dom-endpoint-name>"
      ]
    },
    {
      "Sid": "InvokeVlm",
      "Effect": "Allow",
      "Action": "bedrock:InvokeModel",
      "Resource": [
        "arn:aws:bedrock:<region>:<account-id>:inference-profile/us.anthropic.claude-sonnet-4-6",
        "arn:aws:bedrock:*::foundation-model/anthropic.claude-sonnet-4-6*"
      ]
    }
  ]
}
```

## Prerequisites

- AWS account with Amazon SageMaker AI and Amazon Bedrock access
- Amazon SageMaker AI execution role with permissions for Amazon SageMaker AI, S3, and Amazon Bedrock
- [Kaggle account](https://www.kaggle.com/) (for the URL training dataset)
- Amazon Bedrock model access enabled for `Claude Sonnet 4.6` in your target region

The entire project runs inside a **SageMaker AI notebook instance**. Open a terminal in the notebook and install dependencies once:

```bash
pip install -r requirements.txt
```

**Instance recommendation:** use at least `ml.t3.medium` (2 vCPU, 4 GB RAM) with **10 GB EBS storage**. If running DistilBERT fine-tuning directly in the notebook rather than as a SageMaker AI training job, use a GPU-backed instance such as `ml.p3.2xlarge`.

## Cost Estimate

| Component | Instance | Approx. cost |
|---|---|---|
| LightGBM endpoint | ml.m5.large | ~$84/mo |
| DistilBERT endpoint | ml.m5.xlarge | ~$168/mo |
| Amazon Bedrock Claude (VLM) | — | ~$0.0015/call |

**Total endpoint baseline:** ~$252/mo (fixed, regardless of traffic)

| URLs/month | VLM trigger rate | Approx. total/mo |
|---|---|---|
| 10,000 | 10% | ~$254 |
| 100,000 | 10% | ~$267 |
| 1,000,000 | 5% | ~$327 |

For low-volume or bursty workloads, replace real-time endpoints with **SageMaker AI Serverless Inference** to eliminate the fixed cost (cold-start tradeoff of ~1–2 s).

## Cleanup

Run the **Cleanup** cell at the bottom of `03_phishing_detection_pipeline.ipynb` to delete both Amazon SageMaker AI endpoints, or delete them from the Amazon SageMaker AI console under **Inference → Endpoints**.

## Security

### Shared Responsibility

AWS is responsible for security *of* the cloud (physical infrastructure, hypervisor, managed service availability). You are responsible for security *in* the cloud: IAM policies, network controls, encryption at rest and in transit, and the data you store.

### Training Data Licensing and Classification

**Kaggle Malicious URLs dataset** — licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Attribution: "Malicious URLs dataset" by Sid321axn on Kaggle. This dataset is used for training only and is not redistributed in this repository.

**Tranco top sites list** — the benign URL list is generated locally at runtime by `build_benign_urls()` in `utils.py`, which downloads the [Tranco list](https://tranco-list.eu/). Tranco is a research-oriented ranking aggregated from several upstream providers, each under its own terms; this repository does **not** redistribute the list or any derived file. Review the Tranco site for the terms that apply to the underlying data before redistributing any list you generate.

**Synthetic phishing HTML** — generated by Claude via Amazon Bedrock for training purposes only. All generated URLs use `.invalid` or `.example` TLDs (non-routable per RFC 2606) and are not contacted during normal operation.

**Data classification:** training data (phishing URLs, HTML content, model artifacts) should be treated as **internal/confidential**. Do not store real end-user URLs or page content in the training bucket without appropriate data handling controls.

### IAM Permissions

The Amazon SageMaker AI execution role requires the following minimum permissions:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "SageMakerTrainAndDeploy",
      "Effect": "Allow",
      "Action": [
        "sagemaker:CreateTrainingJob",
        "sagemaker:DescribeTrainingJob",
        "sagemaker:CreateModel",
        "sagemaker:CreateEndpointConfig",
        "sagemaker:CreateEndpoint",
        "sagemaker:InvokeEndpoint",
        "sagemaker:DeleteEndpoint",
        "sagemaker:DeleteEndpointConfig",
        "sagemaker:DeleteModel"
      ],
      "Resource": [
        "arn:aws:sagemaker:<region>:<account-id>:training-job/phishing-*",
        "arn:aws:sagemaker:<region>:<account-id>:model/phishing-*",
        "arn:aws:sagemaker:<region>:<account-id>:endpoint-config/phishing-*",
        "arn:aws:sagemaker:<region>:<account-id>:endpoint/phishing-*"
      ]
    },
    {
      "Sid": "BedrockInvoke",
      "Effect": "Allow",
      "Action": "bedrock:InvokeModel",
      "Resource": "arn:aws:bedrock:<region>::foundation-model/anthropic.claude-*"
    },
    {
      "Sid": "S3TrainingData",
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:PutObject", "s3:ListBucket"],
      "Resource": [
        "arn:aws:s3:::sagemaker-<region>-<account-id>",
        "arn:aws:s3:::sagemaker-<region>-<account-id>/phishing-detection/*"
      ],
      "Condition": {
        "Bool": {
          "aws:SecureTransport": "true"
        }
      }
    }
  ]
}
```

Replace `<region>` and `<account-id>` with your values. Do not attach `AdministratorAccess` or `AmazonS3FullAccess`. Review and rotate IAM credentials periodically using AWS IAM Access Analyzer.

### S3 Security

The Amazon SageMaker AI default bucket (`sagemaker-<region>-<account-id>`) stores training data, model artifacts, and output CSVs. Recommended configuration:

- **Block Public Access** — enable all four Block Public Access settings on the bucket.
- **Server-side encryption** — enable SSE-S3 (AES-256) or SSE-KMS as the default encryption policy. When using SSE-KMS, grant the Amazon SageMaker AI execution role `kms:GenerateDataKey` and `kms:Decrypt` on the KMS key. Enable automatic annual key rotation on any customer-managed KMS key used for this project.
- **TLS enforcement** — add a bucket policy that denies requests where `aws:SecureTransport` is `false`.
- **Access logging** — enable S3 server access logging on the bucket and route logs to a separate audit bucket. Enable Amazon CloudWatch Logs on the SageMaker AI endpoints to record invocation metadata.
- **Versioning** — enable bucket versioning to allow recovery from accidental overwrites of model artifacts.
- **Deny unencrypted uploads** — add a bucket policy condition denying `s3:PutObject` requests that do not include `x-amz-server-side-encryption`.

### Amazon SageMaker AI Hardening

- Deploy endpoints inside a VPC by setting `VpcConfig` on `CreateEndpointConfig` and disabling internet access (`EnableNetworkIsolation: true`) where feasible.
- Enable KMS encryption for the endpoint storage volume (`KmsKeyId` on the production variant).


### Amazon Bedrock

Claude is accessed via the Amazon Bedrock Converse API and is subject to the [AWS Responsible AI Policy](https://aws.amazon.com/machine-learning/responsible-machine-learning/) and [Amazon Bedrock Acceptable Use Policy](https://aws.amazon.com/aup/). Prompts and completions are not stored or used to train AWS models. Review the [Amazon Bedrock data privacy FAQ](https://aws.amazon.com/bedrock/faqs/) for your region.

### Model Bias and Fairness

The LightGBM model is trained on the [Kaggle Malicious URLs dataset](https://www.kaggle.com/datasets/sid321axn/malicious-urls-dataset), which reflects the URL distribution at the time of collection. The DistilBERT DOM classifier is trained on synthetic phishing pages generated by Claude and crawled benign pages, which may not reflect the full diversity of real-world phishing campaigns. Evaluate both models on your own traffic distribution before relying on them for production blocking decisions.

## License

This library is licensed under the MIT-0 License. See the [LICENSE](LICENSE) file.
