Configuration
=============

A deployment config is one YAML file. ``reef serve -c <file>`` reads it, starts
every process in its ``services`` list in dependency order, and hands the
``reef`` section to the HTTP service.

.. code:: yaml

   reef:
     host: 0.0.0.0
     port: 8900
     recipe: recipe
     token: ${REEF_TOKEN}
     upstream_url: ${REEF_UPSTREAM_URL}
     upstream_api_key: ${REEF_UPSTREAM_API_KEY}

   services:
     - name: reef
       command: python -m reef.service
       ready: curl -sf http://127.0.0.1:${reef.port}/healthz

Values interpolate from the environment with ``${VAR}`` and from the config
itself with ``${dotted.path}``. Any value can be overridden on the command line:
a bare ``--model_path /models/demo`` targets the ``reef`` section, and a dotted
``--training.checkpoint_dir /tmp/ckpt`` targets any other. Each process writes a
log under ``/tmp/reef-stack/``; set ``run_dir`` to move it.

Start from a bundled stack
--------------------------

Every runnable stack lives under ``recipes/``: the learn-nothing ones on the
base ``recipe`` kind in ``recipes/basic/``, and each method's with its
examples.

+-------------------------------------------------------------+----------------------------------------------------------+
| File                                                        | What it starts                                           |
+=============================================================+==========================================================+
| ``recipes/basic/external-provider.yaml``                    | no GPU, no local model: one Reef process proxying an     |
|                                                             | HTTP provider                                            |
+-------------------------------------------------------------+----------------------------------------------------------+
| ``recipes/basic/local-sglang.yaml``                         | local inference: an SGLang server plus Reef, no training |
+-------------------------------------------------------------+----------------------------------------------------------+
| ``recipes/<method>/examples/<example>/serve.yaml``          | weight training: Ray head, Slime driver, Reef, and the   |
|                                                             | method's own services                                    |
+-------------------------------------------------------------+----------------------------------------------------------+

Each weight-training example ships its stack as ``serve.yaml``.
``recipes/sao/examples/sao/serve.yaml`` is the smallest, two GPUs for one
actor and one rollout engine; ``recipes/tttd/examples/tttd/serve.yaml`` adds
LoRA training, and ``recipes/openclawrl/examples/openclawrl/serve.yaml`` adds
a PRM engine and a student model.

The ``reef`` section
--------------------

.. config::

   reef.recipe | the recipe this deployment serves. Required.
   reef.host | 0.0.0.0 | bind address
   reef.port | 8900 | bind port
   reef.token | the bearer token the service accepts. Use ``tokens: [...]`` to accept several while rotating.
   reef.model_path | a local HF model directory or a repo id, downloaded on start
   reef.upstream_url | the OpenAI-compatible provider, with no ``/v1`` suffix
   reef.upstream_api_key | its credential. Reef is the only party that sees it.
   reef.upstream_model | the model to request upstream
   reef.upstream_api | openai | the provider dialect; ``anthropic`` for an Anthropic-style endpoint
   reef.inference_url | the address the training backend reports | the local engine; set only to front the engines with something else
   reef.inference_timeout_s | 300.0 | per-request timeout
   reef.allow_implicit_scenario_creation | true | when false, an unknown scenario is HTTP 404
   reef.checkpoint_every_n_versions | 1 | how often a version becomes durable

Storage paths default under ``.reef/``, but every bundled stack overrides them
to ``/var/lib/reef``. Point them somewhere persistent.

.. config::

   reef.artifact_repository | .reef/artifacts.git | the Git-backed version chain
   reef.artifact_work_dir | .reef/artifact-work | materialization scratch
   reef.artifact_cache_dir | .reef/artifact-cache | fetched artifact cache
   reef.agent_record_dir | .reef/agent-record | the record store

.. warning::

   On ephemeral storage, a restart loses the record store, the commit logs, and
   every version.

Recipe settings sit beside these in the same section — ``batch_size``,
``min_score``, and whatever else the recipe declares with ``config_field``. Keys
the service does not recognize are handed to the recipe.

Recipe configuration
--------------------

A recipe is named three ways:

- **A bundled kind** — ``recipe: sao``
- **A dotted class** — ``recipe: "my_pkg.my_method:MyMethodRecipe"``
- **A named preset** — ``recipe: my-preset``, resolved to ``my-preset.yaml``
  under ``REEF_RECIPE_CONFIG_DIR``

``REEF_RECIPE_CONFIG_DIR`` is the directory preset YAML is read from, and it has
**no default**: a bare recipe name resolves to a preset only when it is set.

A preset is read as-is — ``${VAR}`` interpolates in a deployment config, never
in a preset. A preset carries its own ``kind``, ``model``, ``data``, and — for
harness evolution — ``evolution`` sections:

.. code:: yaml

   kind: harness_evolve
   model:
     path: qwen3-8b
   data:
     batch_size: 1
     max_score: 0.0
   evolution:
     adapter: pi
     propose: methods.mine:propose
     evaluate: methods.mine:evaluate
     tasks: ["..."]

Weight recipes accept their fields as flat ``reef.<name>`` keys *or* as
``data.<name>`` in a preset. Other kinds are configured only by a preset, where
``data`` holds the batching fields and a kind-specific section holds the rest.

The ``services`` list
---------------------

Each entry is one process.

.. config::

   services[].name | the service's id, used by ``depends_on``
   services[].command | the command line to run
   services[].ready | a shell command that succeeds once the service is up
   services[].depends_on | services that must be ready first
   services[].cuda | the value of ``CUDA_VISIBLE_DEVICES`` for this process
   services[].env | extra environment variables

The ``training`` section
------------------------

Read by the weight-training stack. See `Evolve your model
<../user-guide/evolve-your-model.rst>`__ for how to size it.

.. config::

   training.num_gpus | GPUs handed to the Ray head
   training.cuda_visible_devices | the devices Ray and Slime may use
   training.global_batch_size | samples in one optimizer step. Must equal the recipe's ``batch_size``.
   training.checkpoint_dir | where Megatron and HF checkpoints are written
   training.megatron_checkpoint_path | optional pre-converted torch_dist checkpoint, to skip HF conversion on every start
   training.checkpoint_retention | storage-fraction bounds and the retention policy
   training.slime_flags | GPU layout, optimizer, sequence length, and loss settings, as one literal string

Architecture flags — layer counts, hidden sizes — are auto-filled by Slime from
``reef.model_path`` and do not belong in the config.

The ``evaluation`` section
--------------------------

Absent by default, in which case a successful training step publishes without a
gate. When present, Reef calls the named factory once per scenario and hands the
plugin the exported but unpublished checkpoint.

.. config::

   evaluation.module | a ``package.module:factory`` reference to the plugin factory. Required.
   evaluation.config | opaque mapping handed to the factory; Reef never reads it

.. code:: yaml

   evaluation:
     module: my_pkg.evaluation:build_evaluator
     config:
       benchmark: gsm8k
       threshold: 0.8

The plugin interface is in `Write a recipe
<../developer-guide/write-a-recipe.rst#gate-a-candidate>`__.

Experiment tracking
-------------------

Tracking is optional, off by default, and belongs to a Reef *scenario* rather
than to one training backend. The same provider-neutral logger is shared by the
recipe, the processor, backend results, and the commit lifecycle. Install
``reef[wandb]`` when the training extra does not already provide it.

.. code:: yaml

   training:
     wandb:
       enabled: true
       project: reef
       entity: your-team             # optional
       group_prefix: prod-us-east    # optional scenario-group namespace
       name_prefix: baseline         # optional run-name prefix
       tags: [openclawrl, qwen]
       mode: online                  # online, offline, or disabled
       directory: /var/lib/reef/wandb
       upload_checkpoints: false

Export ``WANDB_API_KEY`` before starting, or log in once with the credential
store on the cluster.

.. warning::

   There is no API-key field here. Reef rejects Slime's ``--wandb-key`` flag and
   never writes a credential into metrics or run config. Do not put one in the
   YAML, in ``slime_flags``, in a tag, or in a run name.

``online`` sends data to the project. ``offline`` makes no network calls and
writes syncable data below ``directory`` for a later ``wandb sync``.
``disabled`` makes no calls even when ``enabled`` is true.

Each scenario maps to one group — the scenario name, or
``<group_prefix>/<scenario>``. Within it, Reef opens one run when the scenario
binds and another after each rollback. The deterministic run id includes those
identities, so restarting resumes the same run with ``resume=allow``. A rollback
finishes the current run, marks its summary with the source and target, and
resets ``train/step`` to zero; the globally monotonic ``reef/step`` stays
attached for joining a run back to the commit log.

Recipe and processor code logs through the same object without importing W&B:

.. code:: python

   experiment_logger.log({"temperature": 0.6}, namespace="recipe")
   self.experiment_logger.log({"accepted": 12}, namespace="processor")

Those become ``recipe/*`` and ``processor/*``, each namespace on its own
``<namespace>/event`` axis. Only finite numeric values are sent.

Durable commit metrics carry ``experiment/provider``, ``experiment/project``,
``experiment/group``, and ``experiment/run_id`` — use them to open the run from
a Reef version, and the run's ``reef/training_job_id`` to go the other way.
Checkpoint paths are metadata only unless ``upload_checkpoints: true``.

Import, initialization, logging, summary, and upload failures are reported in
the service log and never fail a training step or its commit.

Inference tracing with LangSmith
--------------------------------

LangSmith tracing is an optional service-edge observer and is disabled by
default. Install the extra, export the credential, and select a project in the
deployment config:

.. code:: bash

   pip install 'reef[langsmith]'
   export LANGSMITH_API_KEY=lsv2_...

.. code:: yaml

   observability:
     langsmith:
       enabled: true
       project: reef-production
       # endpoint: https://eu.api.smith.langchain.com
       include_inputs: true
       include_outputs: true
       include_metadata: true
       redact_keys: [authorization, x-api-key, api_key, password, secret, token]
       queue_size: 1024
       batch_size: 32
       flush_timeout_s: 2.0

``endpoint`` is optional. Without it, the SDK uses ``LANGSMITH_ENDPOINT`` or
the LangSmith SaaS default; set either form for an EU/APAC region, self-hosted
installation, or custom gateway. ``LANGSMITH_WORKSPACE_ID`` is also forwarded
when set. Credentials are environment-only: Reef has no API-key config field,
and never adds a credential to trace payloads, logs, records, or commit metrics.

Enabling the observer exports request inputs, response outputs, and client
metadata by default. Set any ``include_*`` switch to ``false`` to omit that
class of data entirely. ``redact_keys`` is case-insensitive and recursively
replaces matching object fields before export; its defaults cover common
credential names. Reef correlation fields (scenario, recipe, receipt,
artifact/weight versions, retry count, completion/delivery state) remain even
when ``include_metadata`` is false. Review Reef tags and structured report
feedback for private data before enabling a project.

Each inference receipt maps to a root LangSmith run UUID as
``UUIDv5(da12da5a-abb9-5c19-a395-6f93f23f25ee, agent_record_id)``. This mapping
accepts arbitrary client ids, is stable across processes and restarts, and is
the compatibility contract used to attach a later ``/reef/report`` to every
receipt in ``references``. No LangSmith identifier is stored in Reef's record
schema. Search for ``reef.agent_record_id`` to open a trace from a receipt;
``reef.scenario``, ``reef.recipe``, ``reef.artifact_version``,
``reef.serving_weight_version``, and ``reef.tag.<name>`` provide the other
filter dimensions.

The exporter uses a bounded background queue and a bounded shutdown flush.
Initialization, queue pressure, rate limits, network/export errors, flush
errors, and SDK absence are logged by error type only and never alter request
responses, durable record acceptance, report acceptance, training, commit, or
shutdown. An interrupted stream is still recorded according to Reef's normal
rules, but its trace says ``incomplete``/``disconnected`` and is never made
training-eligible merely by the observer.

See also
--------

- `Choosing a recipe <../user-guide/recipes.rst>`__ — what to put in ``reef.recipe``.
