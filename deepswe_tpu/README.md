# How to Prepare GKE and Run the DeepSWE Recipe in `verl`

To run the DeepSWE training recipe ([`run_qwen3_5_35b_deepswe.sh`](./run_qwen3_5_35b_deepswe.sh)), **your TPU machines and your sandbox CPU node pool must live in the same GKE cluster**. During rollout, the agent framework running on the Ray workers talks directly to the in-cluster Kubernetes API (`kubernetes.default.svc`) to claim R2E-Gym container sandboxes and stream bash/editor tool calls into those pods via `kubectl exec` (`pods/exec`).

Below is each resource you need to have in place, what it is used for, how it should be configured, and the commands to set it up.

---

## 1. GKE Cluster with GCSFuse and Workload Identity

### What it is for
The GKE cluster hosts both the Ray TPU job and the container sandboxes. It needs the **Cloud Storage FUSE CSI driver** (`GcsFuseCsiDriver`) so TPU worker pods can mount your GCS bucket at `/data` to read the model checkpoint and dataset, and **Workload Identity** so pods can authenticate to GCS and Artifact Registry without static keys.

### How it should look
- A Regional or Zonal GKE cluster in the same region as your TPU capacity and Artifact Registry mirror (for example, `us-central1` for TPU v7x).
- Addons enabled: `GcsFuseCsiDriver` (and optionally `RayOperator` if using `RayCluster` CRDs).
- Workload Identity pool enabled: `<PROJECT>.svc.id.goog`.
- A small default/system CPU node pool (e.g., 1–2 `n2-standard-16` nodes) to host the Ray Head pod, which coordinates the cluster and requires ~12 vCPUs and 48 GiB of memory.

### Suggested commands
```bash
export PROJECT="<GCP_PROJECT>"
export REGION="<REGION>"          # e.g., europe-west4 or us-central1
export ZONE="<ZONE>"              # e.g., europe-west4-b
export CLUSTER="<CLUSTER_NAME>"
export NAMESPACE="<NAMESPACE>"    # e.g., "default"

gcloud container clusters create "${CLUSTER}" \
  --project="${PROJECT}" \
  --region="${REGION}" \
  --node-locations="${ZONE}" \
  --addons=RayOperator,GcsFuseCsiDriver \
  --workload-pool="${PROJECT}.svc.id.goog" \
  --machine-type=n2-standard-16 \
  --num-nodes=2

gcloud container clusters get-credentials "${CLUSTER}" --region="${REGION}" --project="${PROJECT}"
kubectl create namespace "${NAMESPACE}" || true
```

---

## 2. Dedicated Sandbox CPU Node Pool (`sandbox-cpu-pool`)

### What it is for
Each R2E-Gym rollout episode spins up an isolated repository container (such as `numpy`, `pillow`, `pandas`, or `tornado`) where the ReAct agent runs bash commands, edits code, and executes `pytest` suites. Running dozens of concurrent container sandboxes on a dedicated CPU node pool prevents their CPU and disk I/O bursts from starving the Ray head pod or interfering with the TPU host VMs.

### How it should look
- **Machine type & sizing**: CPU nodes with plenty of vCPUs and local disk (e.g., `n2-standard-32` or `n2-standard-64` with `200GB+` `pd-balanced` boot disks to cache multiple R2E-Gym container images).
- **Node label**: `cloud.google.com/gke-nodepool=sandbox-cpu-pool` so `GKEAgentSandbox` can target this pool via `NODE_SELECTOR_KEY` and `NODE_SELECTOR_VAL`.
- **Node taint**: `cloud.google.com/gke-nodepool=sandbox-cpu-pool:NoSchedule` so regular cluster workloads do not land on sandbox nodes (`GKEAgentSandbox` automatically adds a matching toleration to every sandbox pod it creates).

### How to size the number of nodes

**Mental model:** Each training run puts two kinds of sandbox pods on this CPU node pool:
1. **Working sandboxes** — containers actively running shell commands and `pytest` for the trajectories currently in flight. These are what actually consume CPU and RAM (typically bursting to ~`1–2` vCPUs and `2–4 GiB` RAM each).
2. **Warm standby sandboxes** — idle containers pre-started ahead of time (2-slot lookahead for the current + next batch, plus just-in-time eval containers on validation steps) so rollouts do not block on multi-gigabyte image pulls. Standby pods use almost no CPU, so they mainly count toward Kubernetes' per-node pod limit and local disk cache.

If `K` training runs share the same sandbox pool in parallel, both buckets scale linearly by `K`.

**Simple sizing rule (for `n2-standard-32`, 32 vCPU / 128 GiB RAM):**
1. **How many working sandboxes run at once per job?**
   Each rollout batch has `TRAIN_BATCH_SIZE * ROLLOUT_N` trajectories, and with `NUM_WARMUP_BATCHES=1` up to two batches overlap, capped by `CONCURRENCY`:
   ```text
   active_per_run = min(CONCURRENCY, (1 + NUM_WARMUP_BATCHES) * TRAIN_BATCH_SIZE * ROLLOUT_N)
   ```
   *(Default TPU v7x: `min(384, 2 * 16 * 16) = 384` working sandboxes.)*
2. **Convert working sandboxes to CPU nodes:**
   One `n2-standard-32` node comfortably handles about **`16–24` working sandboxes** (~20 as a rule of thumb) without CPU throttling or OOM kills during parallel `pytest` bursts. Multiply by the number of parallel runs `K`:
   ```text
   nodes = ceil(K * active_per_run / 20)   # conservative: divide by 16; tighter: divide by 24
   ```
3. **Quick sanity checks for pod limit and disk:**
   - **Pod cap (`110` pods/node on GKE, ~`100` usable):** Standby warm pools add roughly `2 * TRAIN_BATCH_SIZE * min(ROLLOUT_N, SANDBOX_MAX_WARMPOOL_SIZE)` pods (plus `VAL_MAX_SAMPLES * SANDBOX_EVAL_REPLICAS` during validation) — around `~710` total pods at peak for one default v7x run. Make sure `nodes >= ceil(K * peak_pods / 100)` (compute dominates on `n2-standard-32`, but the pod cap can bind on `n2-standard-64+`).
   - **Boot disk (`200–300 GB`):** R2E-Gym images are ~`2–5 GB` uncompressed. Replicas of the same image co-locate onto one node by default (`colocate_replicas=True`) to reuse cached containerd layers; `200 GB+` disks keep ~`30–50` images cached per node.

**Quick-reference examples (`n2-standard-32`):**

| Workload | `TRAIN_BATCH_SIZE` x `ROLLOUT_N` | `CONCURRENCY` | Working sandboxes (`K * active`) | Recommended `n2-standard-32` nodes |
| --- | --- | --- | --- | --- |
| 1 small / dev run | 8 x 8 | 64 | 64 | **4–6** |
| 1 default TPU v7x run | 16 x 16 | 384 | 384 | **16–24** (autoscale `4–32`) |
| 2–3 parallel TPU v7x runs | 16 x 16 (x `K`) | 384 (x `K`) | 768–1,152 | **32–64+** |

### Suggested command
```bash
gcloud container node-pools create sandbox-cpu-pool \
  --cluster="${CLUSTER}" \
  --project="${PROJECT}" \
  --region="${REGION}" \
  --node-locations="${ZONE}" \
  --machine-type=n2-standard-32 \
  --num-nodes=16 \
  --enable-autoscaling --min-nodes=4 --max-nodes=64 \
  --disk-type=pd-balanced --disk-size=200GB \
  --node-labels="cloud.google.com/gke-nodepool=sandbox-cpu-pool" \
  --node-taints="cloud.google.com/gke-nodepool=sandbox-cpu-pool:NoSchedule" \
  --workload-metadata=GKE_METADATA \
  --scopes=cloud-platform
```

---

## 3. TPU Node Pools (1 Trainer Slice + N Rollout Slices)

### What it is for
`verl`'s `separate_async` trainer runs TorchTitan FSDP2 training on a dedicated trainer TPU slice while `vLLM` generates rollouts on separate rollout TPU slice(s), syncing weights between them after each step via `tpu-sync`. Keeping trainer and rollout on separate TPU slices prevents training and inference from competing for HBM on the same chips.

### How they should look
- **Default optimized TPU v7x topology**: One `2x2x2` trainer slice (`tpu7x-standard-4t`, 2 hosts x 8 TensorCores = 16 TensorCores, `tpu-group-0`) plus three `2x2x1` rollout slices (1 host x 8 TensorCores each = 3 replicas of TP8/EP8, `tpu-group-1`..`tpu-group-3`).
- Smaller topologies (such as one `2x2x1` trainer slice + one `2x2x1` rollout slice) also work by overriding `NNODES_TRAINER`, `NNODES_ROLLOUT`, and `PARAM_DTYPE`/`OPTIMIZER_HOST_OFFLOAD`.

### Suggested commands
```bash
# Trainer slice (2 hosts x 8 TensorCores, 2x2x2 topology)
gcloud container node-pools create trainer-v7x-slice \
  --cluster="${CLUSTER}" \
  --project="${PROJECT}" \
  --region="${REGION}" \
  --node-locations="${ZONE}" \
  --machine-type=tpu7x-standard-4t \
  --tpu-topology=2x2x2 \
  --num-nodes=2 \
  --workload-metadata=GKE_METADATA \
  --scopes=cloud-platform

# Three rollout slices (1 host x 8 TensorCores each, 2x2x1 topology)
for i in 1 2 3; do
  gcloud container node-pools create "rollout-v7x-slice-${i}" \
    --cluster="${CLUSTER}" \
    --project="${PROJECT}" \
    --region="${REGION}" \
    --node-locations="${ZONE}" \
    --machine-type=tpu7x-standard-4t \
    --tpu-topology=2x2x1 \
    --num-nodes=1 \
    --workload-metadata=GKE_METADATA \
    --scopes=cloud-platform
done
```

---

## 4. Agent Sandbox CRDs, Controller, and Namespace RBAC

### What it is for
`uni-agent`'s `GKEAgentSandbox` (`uni_agent/sandbox/gke_agent_sandbox.py`) uses `agent_sandbox_rl` (`SandboxFleet`) to manage pre-warmed pools of R2E-Gym containers in Kubernetes. Under the hood, this relies on the `kubernetes-sigs/agent-sandbox` controller and four custom resources (`Sandbox`, `SandboxTemplate`, `SandboxWarmPool`, and `SandboxClaim`), plus RBAC permissions for the Ray worker's Kubernetes ServiceAccount to create those resources and exec into the claimed sandbox pods.

By default, `GKEAgentSandbox` configures every `SandboxTemplate` with `networkPolicyManagement: Managed` and an empty `networkPolicy` (`{ingress: [], egress: []}`), blocking all ingress and egress traffic (including public internet, internal cluster services, and cloud metadata endpoints) while keeping pod loopback (`localhost`) and Kubernetes API `pods/exec` working.

### How it should look
- **CRDs & Controller**: `sandboxes.agents.x-k8s.io`, `sandboxtemplates.extensions.agents.x-k8s.io`, `sandboxwarmpools.extensions.agents.x-k8s.io`, and `sandboxclaims.extensions.agents.x-k8s.io` installed in the cluster, with `agent-sandbox-controller` running in `agent-sandbox-system`.
- **RBAC (`Role` + `RoleBinding`)**: In the namespace where your Ray pods run (e.g., `default`), the ServiceAccount used by your Ray pods (`${RAY_SERVICE_ACCOUNT:-default}`) must have `get`, `list`, `watch`, `create`, `update`, `patch`, `delete` on `pods`, `services`, `sandboxes`, `sandboxtemplates`, `sandboxwarmpools`, and `sandboxclaims`, plus `get` and `create` on `pods/exec` and `pods/log`.

### Suggested commands
Check if the CRDs already exist, and install them if missing:
```bash
kubectl get crd sandboxes.agents.x-k8s.io \
  sandboxtemplates.extensions.agents.x-k8s.io \
  sandboxwarmpools.extensions.agents.x-k8s.io \
  sandboxclaims.extensions.agents.x-k8s.io || {
  export AGENT_SANDBOX_VERSION="v0.2.1"
  kubectl apply -f "https://github.com/kubernetes-sigs/agent-sandbox/releases/download/${AGENT_SANDBOX_VERSION}/manifest.yaml"
  kubectl apply -f "https://github.com/kubernetes-sigs/agent-sandbox/releases/download/${AGENT_SANDBOX_VERSION}/extensions.yaml"
}
```

Apply the RBAC permissions in your target namespace:
```bash
export RAY_SERVICE_ACCOUNT="${RAY_SERVICE_ACCOUNT:-default}"

cat <<EOF | kubectl apply -f -
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: agent-sandbox-manager
  namespace: ${NAMESPACE}
rules:
  - apiGroups: [""]
    resources: ["pods", "services"]
    verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
  - apiGroups: [""]
    resources: ["pods/exec", "pods/log"]
    verbs: ["get", "create"]
  - apiGroups: ["agents.x-k8s.io"]
    resources: ["sandboxes"]
    verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
  - apiGroups: ["extensions.agents.x-k8s.io"]
    resources: ["sandboxclaims", "sandboxtemplates", "sandboxwarmpools"]
    verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: agent-sandbox-manager-binding
  namespace: ${NAMESPACE}
subjects:
  - kind: ServiceAccount
    name: ${RAY_SERVICE_ACCOUNT}
    namespace: ${NAMESPACE}
roleRef:
  kind: Role
  name: agent-sandbox-manager
  apiGroup: rbac.authorization.k8s.io
EOF
```

---

## 5. Regional Artifact Registry Mirror for R2E-Gym Images

### What it is for
Each row in the R2E-Gym dataset specifies a Docker Hub image (`namanjain12/<repo>_final:<commit>`). If dozens of GKE nodes pull those multi-gigabyte images directly from Docker Hub at once, Docker Hub blocks the pulls with `429 Too Many Requests` rate-limit errors. Hosting a mirror of the 8 R2E-Gym repositories (`aiohttp_final`, `datalad_final`, `numpy_final`, `pandas_final`, `pillow_final`, `pyramid_final`, `scrapy_final`, `tornado_final`) in a regional Artifact Registry repository lets `GKEAgentSandbox` rewrite dataset images to your regional mirror via `sandbox.image_map` in `task_config_react_gke.yaml` or via `SANDBOX_IMAGE_MAP`.

### How it should look
- An Artifact Registry Docker repository in the same region as your cluster (for example, `us-central1-docker.pkg.dev/<PROJECT>/<REPO>`) containing the mirrored `<repo>_final:<commit>` tags.
- Either configure `sandbox.image_map` in `task_config_react_gke.yaml`, or pass `SANDBOX_IMAGE_MAP="namanjain12/**=<REGION>-docker.pkg.dev/<PROJECT>/<REPO>/**"` when launching the job.

---

## 6. GCS Bucket with Model Checkpoint, Preprocessed Dataset, and Code

### What it is for
The TPU worker pods mount a GCS bucket at `/data` via GCSFuse. Storing the text-only `Qwen3.5-35B-A3B` checkpoint, the preprocessed R2E-Gym Parquet files (`train.parquet` and `test.parquet`), and the three code repositories on GCS makes them immediately available to every TPU node.

### How it should look
Under your GCS bucket `gs://<BUCKET>/<USER_DIR>` (which maps to `RAY_DATA_HOME=/data/<USER_DIR>` inside the pods):
- `assets/hf/Qwen3.5-35B-A3B/`: Text-only Hugging Face checkpoint (`*.safetensors` + tokenizer files).
- `data/r2e_gym_easy/train.parquet` and `data/r2e_gym_easy/test.parquet`: Generated by `uni_agent.tasks.r2e_gym.preprocess`.
- `src/deepswe_bundle.tar.gz`: Tarball of the `uni-agent`, `verl`, and `verl-hardware-plugin` checkouts so pods can extract them to `/tmp/deepswe_src` on startup.

### Suggested commands
```bash
export BUCKET="<GCS_BUCKET>"
export USER_DIR="<USER_DIR>"

# 1. Clone verl main, verl-hardware-plugin main, uni-agent (GKEAgentSandbox + R2E-Gym), and this recipe
git clone https://github.com/verl-project/verl.git
git clone https://github.com/verl-project/verl-hardware-plugin.git
git clone https://github.com/verl-project/uni-agent.git
git clone https://github.com/verl-project/verl-recipe.git
mkdir -p verl/recipe && cp -r verl-recipe/deepswe_tpu verl/recipe/deepswe_tpu

# 2. Preprocess R2E-Gym into Parquet files and upload to GCS
PYTHONPATH=uni-agent:verl python3 -m uni_agent.tasks.r2e_gym.preprocess \
  --local-save-dir /tmp/r2e_gym_easy
gcloud storage cp -r /tmp/r2e_gym_easy "gs://${BUCKET}/${USER_DIR}/data/"

# 3. Upload the text-only Qwen3.5-35B-A3B checkpoint to GCS
gcloud storage cp -r /path/to/Qwen3.5-35B-A3B "gs://${BUCKET}/${USER_DIR}/assets/hf/"

# 4. Bundle and upload the repositories (the recipe is inside verl/recipe/deepswe_tpu)
tar -czf /tmp/deepswe_bundle.tar.gz \
  --exclude='.git' --exclude='__pycache__' \
  uni-agent verl verl-hardware-plugin
gcloud storage cp /tmp/deepswe_bundle.tar.gz "gs://${BUCKET}/${USER_DIR}/src/deepswe_bundle.tar.gz"
```

---

## 7. Multi-Slice Ray `JobSet` on GKE

### What it is for
The `JobSet` starts the Ray cluster across your pods: 1 Ray head pod on the CPU pool, 1 `trainer` replicated job on the first TPU slice, and 1 `rollout` replicated job on the second TPU slice.

### How it should look
- **Head pod**: Runs `ray start --head --port=6379 --dashboard-host=0.0.0.0 --dashboard-port=8265 --num-cpus=0 --num-gpus=0 --block` on the system CPU pool and extracts `/tmp/deepswe_src`.
- **Trainer and Rollout TPU worker pods**:
  1. Mount `gs://<BUCKET>` at `/data` via `gcsfuse.csi.storage.gke.io` and a `64Gi` in-memory `emptyDir` at `/dev/shm`.
  2. **Unset `MEGASCALE_*` variables before `ray start`**: GKE's TPU `JobSet` webhook automatically injects `MEGASCALE_COORDINATOR_ADDRESS` and `MEGASCALE_NUM_SLICES` across all jobs in a `JobSet`. Unsetting `MEGASCALE_*`, `TPU_WORKER_HOSTNAMES`, and `TPU_WORKER_ID` before launching `ray start` ensures the trainer slice and rollout slice form two independent `libtpu` meshes instead of waiting for a single combined multi-slice barrier.
  3. **Pre-warm checkpoint shards**: Sequentially read `*.safetensors` from `/data/.../Qwen3.5-35B-A3B` into the OS page cache before starting Ray so multi-process `mmap` page faults do not trigger slow random reads over GCSFuse.
  4. **Advertise slice affinity**: Pass `--resources='{"TPU": <CHIPS_PER_HOST>, "tpu-group-0": 1}'` on the trainer pods and `"tpu-group-1": 1` on the rollout pods so `verl-hardware-plugin` places the trainer and rollout worker groups on separate slices.

### Suggested commands
Apply your `JobSet` manifest and verify all pods join Ray:
```bash
kubectl apply -f verl-deepswe-jobset.yaml
kubectl get pods -n "${NAMESPACE}" -l jobset.sigs.k8s.io/jobset-name=verl-deepswe

HEAD_POD=$(kubectl get pods -n "${NAMESPACE}" \
  -l jobset.sigs.k8s.io/jobset-name=verl-deepswe,role=head \
  -o jsonpath='{.items[0].metadata.name}')
kubectl exec -n "${NAMESPACE}" "${HEAD_POD}" -- ray status
```

---

## 8. Running the Recipe in `verl`

### Required `verl` version
This recipe targets upstream `verl` `main` together with `verl-hardware-plugin` `main` (for TPU support) and `uni-agent`. See [`REQUIRED_VERL.txt`](./REQUIRED_VERL.txt).

### What it is for
[`run_qwen3_5_35b_deepswe.sh`](./run_qwen3_5_35b_deepswe.sh) wires `uni-agent`'s `AgentFrameworkRolloutAdapter` (`ReActAgent` + `r2e_gym` + `GKEAgentSandbox`) into `verl`'s V1 async PPO/RLOO trainer (`verl/examples/tpu/grpo/run_qwen3_5_torchtitan.sh`).

### How the launch command should look
- Because `ray job submit` starts in `/app` with a clean environment on the head pod, pass your `export` statements inside `bash -c "..."` so both the shell script and `ray.init(runtime_env=...)` inherit them.
- Mount or copy `verl-recipe` at `${VERL_ROOT}/recipe` (or set `VERL_ROOT` and include `verl-recipe` on `PYTHONPATH`).
- Set `SANDBOX_NAMESPACE`, `SANDBOX_IMAGE_MAP`, `NODE_SELECTOR_KEY=cloud.google.com/gke-nodepool`, and `NODE_SELECTOR_VAL=sandbox-cpu-pool` so `GKEAgentSandbox` provisions warm pools in your namespace on the sandbox CPU pool using the mirrored images.
- If your cluster uses custom pod annotations (for example, admission-webhook ownership metadata), pass them via `SANDBOX_POD_ANNOTATIONS='{"key":"value"}'`.
- Defaults are optimized for **TPU v7x**: `2x2x2` trainer (`NNODES_TRAINER=2`, `N_CHIPS_TRAINER=8`, FSDP16, on-device `PARAM_DTYPE=float32`, fused AdamW, SparseCore async FSDP collective offload, `TRAINER_SCOPED_VMEM_LIMIT_KIB=65472`, `AC_SAVE_SPLASH_AND_ROUTING=True`, `SPLASH_MAX_BLOCK_SIZE=1024`, `FROZEN_PARAM_REGEX=moe.router.gate`) and 3 rollout replicas of `2x2x1` (`NNODES_ROLLOUT=3`, `N_CHIPS_ROLLOUT=8`, TP8/EP8, `USE_MOE_FUSED_EP_KERNEL=1`, `TPU_MOE_SKIP_PADDED_TOKENS=1`, `TPU_ENABLE_GDN_DYNAMIC_TILING=1`, `ENABLE_PREFIX_CACHING=True`, `VLLM_BLOCK_SIZE=512`, `VLLM_ASYNC_SCHEDULING=False`, `MAX_NUM_SEQS=128`, `CONCURRENCY=384`), 32k context (`MAX_PROMPT_LEN=8192`, `MAX_RESPONSE_LEN=24576`, `STATIC_MICRO_BATCH_LEN=32768`), and `ROLLOUT_RS_THRESHOLD=0.998_1.002`.
- **Optional `GDN_ROW_SLOT_LEN=32768` speedup**: For an image built from `verl/examples/tpu/grpo/Dockerfile` with `--build-arg TORCHTITAN_COMMIT=4e16e67aed8c1e702e6f440b1ff81619bb6873d4` (`google-pytorch/torchtitan#142`), export `GDN_ROW_SLOT_LEN=32768` to put one sequence per GatedDeltaNet row slot. It is left unset by default so the recipe runs out of the box on `verl main`'s default image.

### Suggested command (TPU v7x)
```bash
kubectl exec -n "${NAMESPACE}" "${HEAD_POD}" -- \
  ray job submit --address=http://127.0.0.1:8265 --no-wait -- \
  bash -c "
    cd /tmp/deepswe_src/verl &&
    export PYTHONPATH='/tmp/deepswe_src/uni-agent:/tmp/deepswe_src/verl:/tmp/deepswe_src/verl-hardware-plugin:/torchtitan:/torchtitan/torchtitan' &&
    export VERL_ROOT='/tmp/deepswe_src/verl' &&
    export RAY_DATA_HOME='/data/${USER_DIR}' &&
    export UNI_AGENT_HOME='/tmp/deepswe_src/uni-agent' &&
    export PROJECT_NAME='verl-tpu-deepswe' &&
    export EXPERIMENT_NAME='qwen3_5_35b_deepswe_r2e_gym_v7x_50steps' &&
    export TOTAL_TRAINING_STEPS=50 &&
    export TEST_FREQ=10 &&
    export VAL_BEFORE_TRAIN=False &&
    export SANDBOX_NAMESPACE='${NAMESPACE}' &&
    export SANDBOX_IMAGE_MAP='namanjain12/**=<REGION>-docker.pkg.dev/<PROJECT>/<REPO>/**' &&
    export NODE_SELECTOR_KEY='cloud.google.com/gke-nodepool' &&
    export NODE_SELECTOR_VAL='sandbox-cpu-pool' &&
    # Optional (only when image was built with TORCHTITAN_COMMIT=4e16e67aed8c1e702e6f440b1ff81619bb6873d4):
    # export GDN_ROW_SLOT_LEN=32768 &&
    exec bash /tmp/deepswe_src/verl/recipe/deepswe_tpu/run_qwen3_5_35b_deepswe.sh \
      \"trainer.logger=['console','wandb']\" \
      \"+ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH='\${PYTHONPATH}'\"
  "
```

### Suggested commands to monitor the run
```bash
# Follow job logs (prints the WandB URL once workers initialize)
kubectl exec -n "${NAMESPACE}" "${HEAD_POD}" -- \
  ray job logs --address=http://127.0.0.1:8265 -f <JOB_ID>

# Check warm pools and active sandbox claims in your namespace
kubectl get sandboxwarmpools,sandboxclaims -n "${NAMESPACE}"
```
