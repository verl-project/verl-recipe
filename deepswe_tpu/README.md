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

Grant the Kubernetes ServiceAccount your Ray pods run as read access to the bucket (set `BUCKET` as in section 6). Without it the pods stay in `Init` with `FailedMount ... PermissionDenied ... storage.objects.list`:
```bash
export PROJECT_NUMBER="$(gcloud projects describe "${PROJECT}" --format='value(projectNumber)')"
gcloud storage buckets add-iam-policy-binding "gs://${BUCKET}" \
  --role=roles/storage.objectViewer \
  --member="principal://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${PROJECT}.svc.id.goog/subject/ns/${NAMESPACE}/sa/${RAY_SERVICE_ACCOUNT:-default}"
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
2. **Warm standby sandboxes** — idle containers pre-started ahead of time (for the current and the next batch, plus just-in-time eval containers on validation steps) so rollouts do not block on multi-gigabyte image pulls. Standby pods use almost no CPU, so they mainly count toward Kubernetes' per-node pod limit and local disk cache.

If `K` training runs share the same sandbox pool in parallel, both buckets scale linearly by `K`.

**Simple sizing rule (for `n2-standard-32`, 32 vCPU / 128 GiB RAM):**
1. **How many working sandboxes run at once per job?**
   Each rollout batch has `TRAIN_BATCH_SIZE * ROLLOUT_N` trajectories, and with `NUM_WARMUP_BATCHES=2` up to three batches overlap, capped by `CONCURRENCY`:
   ```text
   active_per_run = min(CONCURRENCY, (1 + NUM_WARMUP_BATCHES) * TRAIN_BATCH_SIZE * ROLLOUT_N)
   ```
   *(Default TPU v7x: `min(384, 3 * 16 * 16) = 384` working sandboxes.)*
2. **Convert working sandboxes to CPU nodes:**
   One `n2-standard-32` node comfortably handles about **`16–24` working sandboxes** (~20 as a rule of thumb) without CPU throttling or OOM kills during parallel `pytest` bursts. Multiply by the number of parallel runs `K`:
   ```text
   nodes = ceil(K * active_per_run / 20)   # conservative: divide by 16; tighter: divide by 24
   ```
3. **Quick sanity checks for pod limit and disk:**
   - **Pod cap (`110` pods/node on GKE, ~`100` usable):** Standby warm pools add roughly `2 * TRAIN_BATCH_SIZE * ROLLOUT_N` pods (plus `VAL_MAX_SAMPLES * SANDBOX_EVAL_REPLICAS` during validation) — around `~740` total pods at peak for one default v7x run (`VAL_MAX_SAMPLES=100`). Make sure `nodes >= ceil(K * peak_pods / 100)` (compute dominates on `n2-standard-32`, but the pod cap can bind on `n2-standard-64+`).
   - **Boot disk (`200–300 GB`):** R2E-Gym images are ~`2–5 GB` uncompressed. Replicas of the same image co-locate onto one node where possible to reuse cached containerd layers; `200 GB+` disks keep ~`30–50` images cached per node.

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
`uni-agent`'s `GKEAgentSandbox` (`uni_agent/sandbox/gke.py`) uses `agent_sandbox_rl` (`SandboxFleet`) to run each task's container as a Kubernetes pod. The provider can keep pods started ahead of time in warm pools; this recipe decides which ones in [`sandbox_lookahead.py`](./sandbox_lookahead.py), warming every batch's images one training step ahead. Under the hood, this relies on the `kubernetes-sigs/agent-sandbox` controller and four custom resources (`Sandbox`, `SandboxTemplate`, `SandboxWarmPool`, and `SandboxClaim`), plus RBAC permissions for the Ray worker's Kubernetes ServiceAccount to create those resources and exec into the claimed sandbox pods.

By default, `GKEAgentSandbox` configures every `SandboxTemplate` with `networkPolicyManagement: Managed` and an empty `networkPolicy` (`{ingress: [], egress: []}`), blocking all ingress and egress traffic (including public internet, internal cluster services, and cloud metadata endpoints) while keeping pod loopback (`localhost`) and Kubernetes API `pods/exec` working. To use a different policy, or `Unmanaged` mode on a cluster whose namespace NetworkPolicy already isolates the sandboxes, set `sandbox.sandbox_kwargs` in [`task_config.yaml`](./task_config.yaml).

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
    # deletecollection: the cleanup reaper (section 8) deletes a run's pods as a collection before falling back to
    # one-by-one deletes.
    verbs: ["get", "list", "watch", "create", "update", "patch", "delete", "deletecollection"]
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

If the sandboxes run in a different namespace from the Ray pods (`SANDBOX_NAMESPACE`), create the `Role` and `RoleBinding` in the sandbox namespace and keep the Ray namespace in `subjects[].namespace`.

Build the training image from `verl/examples/tpu/grpo/Dockerfile` and add the sandbox client packages on top: `uni-agent`'s runtime dependencies plus the two `kubernetes-sigs/agent-sandbox` client packages (`agent-sandbox-rl` is not on PyPI yet). Use the pinned `git+https` install commands in the GKE Agent Sandbox section of `uni-agent/docs/source/concepts/sandbox.md`, which is the single place that pin is kept. Install `uni-agent` itself from your checkout (`pip install -e 'path/to/uni-agent[gke]'`) or put the checkout on `PYTHONPATH` as the launch command in section 8 does; do not `pip install uni-agent` by name, which resolves to whatever package of that name is on PyPI rather than your checkout.

---

## 5. Regional Artifact Registry Mirror for R2E-Gym Images

### What it is for
Each row in the R2E-Gym dataset specifies a Docker Hub image (`namanjain12/<repo>_final:<commit>`). If dozens of GKE nodes pull those multi-gigabyte images directly from Docker Hub at once, Docker Hub blocks the pulls with `429 Too Many Requests` rate-limit errors. Hosting a mirror of the R2E-Gym repositories (`aiohttp_final`, `coveragepy_final`, `datalad_final`, `numpy_final`, `orange3_final`, `pandas_final`, `pillow_final`, `pyramid_final`, `scrapy_final`, `tornado_final`) in a regional Artifact Registry repository lets the run pull them from your region instead: `sandbox.image_map` in [`task_config.yaml`](./task_config.yaml) rewrites the dataset images to the mirror (or pass `--repos` to `prepare_data.py` if you mirror only a subset).

### How it should look
- An Artifact Registry Docker repository in the same region as your cluster (for example, `us-central1-docker.pkg.dev/<PROJECT>/<REPO>`) containing the mirrored `<repo>_final:<commit>` tags.
- `sandbox.image_map` uncommented in [`task_config.yaml`](./task_config.yaml), with `to:` set to `<REGION>-docker.pkg.dev/<PROJECT>/<REPO>/**`. Every Ray node reads this file; to keep it elsewhere, point `TASK_CONFIG` at a copy that exists at the same path on all of them.

---

## 6. GCS Bucket with Model Checkpoint, Preprocessed Dataset, and Code

### What it is for
The TPU worker pods mount a GCS bucket at `/data` via GCSFuse. Storing the text-only `Qwen3.5-35B-A3B` checkpoint, the preprocessed R2E-Gym Parquet files (`train.parquet` and `test.parquet`), and the three code repositories on GCS makes them immediately available to every TPU node.

### How it should look
Under your GCS bucket `gs://<BUCKET>/<USER_DIR>` (which maps to `RAY_DATA_HOME=/data/<USER_DIR>` inside the pods):
- `assets/hf/Qwen3.5-35B-A3B/`: Text-only Hugging Face checkpoint (`*.safetensors` + tokenizer files).
- `data/r2e_gym/train.parquet` and `data/r2e_gym/test.parquet`: Non-overlapping train (default `700`) and eval (default `100`) splits generated by [`prepare_data.py`](./prepare_data.py). It leaves out the instances listed in [`excluded_instances.txt`](./excluded_instances.txt): their gold patch does not reliably pass the dataset's own tests in a sandbox without network access, so a correct fix would not be rewarded.
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

# 2. Download R2E-Gym, build non-overlapping train (700) and eval (100) Parquet splits, and upload to GCS
PYTHONPATH=uni-agent python3 verl/recipe/deepswe_tpu/prepare_data.py \
  --local-save-dir /tmp/r2e_gym \
  --num-train 700 \
  --num-eval 100 \
  --seed 42
gcloud storage cp -r /tmp/r2e_gym "gs://${BUCKET}/${USER_DIR}/data/"

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
  2. **Unset the TPU webhook's variables before `ray start`**: GKE's TPU `JobSet` webhook injects `MEGASCALE_COORDINATOR_ADDRESS` and `MEGASCALE_NUM_SLICES` across all jobs in a `JobSet`, plus per-pod `TPU_WORKER_HOSTNAMES`, `TPU_WORKER_ID`, `TPU_VISIBLE_CHIPS` and `TORCH_TPU_TOPOLOGY`. Unset all of them. Unsetting `MEGASCALE_*` and `TPU_WORKER_*` lets the trainer slice and each rollout slice form independent `libtpu` meshes instead of waiting for one combined multi-slice barrier. `TPU_VISIBLE_CHIPS` must go because it lists physical chips (4 per TPU v7x host) while the recipe counts TensorCores (8 per host): with it set, `ray start --resources='{"TPU": 8, ...}'` aborts with `ValueError: Attempting to start raylet with 8 TPU, but TPU_VISIBLE_CHIPS contains ['2', '3', '0', '1']`.
  3. **Pre-warm checkpoint shards**: Sequentially read `*.safetensors` from `/data/.../Qwen3.5-35B-A3B` into the OS page cache before starting Ray so multi-process `mmap` page faults do not trigger slow random reads over GCSFuse.
  4. **Advertise slice affinity**: Pass `--resources='{"TPU": <N_CHIPS>, "tpu-group-0": 1}'` on the trainer pods (`<N_CHIPS>` is `N_CHIPS_TRAINER` / `N_CHIPS_ROLLOUT`, i.e. TensorCores per host: `8` on TPU v7x) and `"tpu-group-<i>": 1` on the pods of rollout slice `i` (`1`..`NNODES_ROLLOUT`) so `verl-hardware-plugin` places the trainer and rollout worker groups on separate slices.

### Suggested commands
The recipe does not ship a manifest: clusters differ in node selectors, placement policies and queueing. Start from `verl/examples/tpu/grpo/ray-tpu-v6e-2slice.yaml` (one `JobSet` per slice works too) and use this as the container command of every TPU worker pod (trainer shown; rollout slice `i` uses `tpu-group-<i>`; `RAY_HEAD_HOST` is the DNS name of the head pod's Service):
```bash
set -e
rm -rf /tmp/deepswe_src && mkdir -p /tmp/deepswe_src
tar -xzf "/data/${USER_DIR}/src/deepswe_bundle.tar.gz" -C /tmp/deepswe_src
unset $(compgen -e | grep -E '^(MEGASCALE_|TPU_WORKER_HOSTNAMES$|TPU_WORKER_ID$|TPU_VISIBLE_CHIPS$|TORCH_TPU_TOPOLOGY$)' || true)
for f in "/data/${USER_DIR}/assets/hf/Qwen3.5-35B-A3B"/*.safetensors; do cat "$f" > /dev/null; done
until ray health-check --address "${RAY_HEAD_HOST}:6379" >/dev/null 2>&1; do sleep 2; done
exec ray start --address="${RAY_HEAD_HOST}:6379" --resources='{"TPU": 8, "tpu-group-0": 1}' --block
```
The head pod runs the same two extraction lines before its `ray start --head ...` command. Reading the 65 GB checkpoint into the page cache takes several minutes per host.

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
This recipe targets upstream `verl` `main` together with `verl-hardware-plugin` `main` (for TPU support) and `uni-agent`. See [`REQUIRED_VERL.txt`](./REQUIRED_VERL.txt). Rollout correction defaults to `seq_mean_k1` sequence rejection with no IS weights; `ROLLOUT_IS=seq_mask_tis` / `ROLLOUT_RS=seq_mean_abs_exp` (`ROLLOUT_RS_THRESHOLD=2.0`) need verl-project/verl#8677 and will become the defaults once it is merged.

### What it is for
[`run_qwen3_5_35b_deepswe.sh`](./run_qwen3_5_35b_deepswe.sh) wires `uni-agent`'s `AgentFrameworkRolloutAdapter` (`ReActAgent` + `r2e_gym` + `GKEAgentSandbox`) into `verl`'s V1 async PPO/RLOO trainer (`verl/examples/tpu/grpo/run_qwen3_5_torchtitan.sh`).

### How the launch command should look
- Because `ray job submit` starts in `/app` with a clean environment on the head pod, pass your `export` statements inside `bash -c "..."` so both the shell script and `ray.init(runtime_env=...)` inherit them.
- Mount or copy `verl-recipe` at `${VERL_ROOT}/recipe` (or set `VERL_ROOT` and include `verl-recipe` on `PYTHONPATH`).
- Set `SANDBOX_NAMESPACE`, `NODE_SELECTOR_KEY=cloud.google.com/gke-nodepool`, and `NODE_SELECTOR_VAL=sandbox-cpu-pool` so `GKEAgentSandbox` starts the sandboxes in your namespace on the sandbox CPU pool.
- Agent settings (`max_steps`, tools, prompt) and sandbox options (image mirror, network policy) come from [`task_config.yaml`](./task_config.yaml); set `TASK_CONFIG` to use another file.
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
    export PROJECT_NAME='verl-tpu-deepswe' &&
    export EXPERIMENT_NAME='qwen3_5_35b_deepswe_r2e_gym_v7x_50steps' &&
    export TOTAL_TRAINING_STEPS=50 &&
    export TEST_FREQ=10 &&
    export VAL_BEFORE_TRAIN=False &&
    export SANDBOX_NAMESPACE='${NAMESPACE}' &&
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

# If a job is preempted or killed before on_train_end() tears down its sandboxes, reap that run by its run-id label.
# SANDBOX_RUN_ID is the value you exported, or the generated "deepswe-<MMDD-HHMMSS>" printed at the top of the job
# log. The ":?" guard aborts if it is unset; never match on names with grep, an empty pattern matches every job.
kubectl exec -n "${NAMESPACE}" "${HEAD_POD}" -- \
  python3 -m agent_sandbox_rl.reaper --in-cluster --namespace "${SANDBOX_NAMESPACE:-${NAMESPACE}}" \
    --run-id "${SANDBOX_RUN_ID:?set SANDBOX_RUN_ID to the run to clean up}"
```
