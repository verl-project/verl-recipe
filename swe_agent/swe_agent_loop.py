# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
SWE Agent Loop - External Control Multi-turn Interaction Mode.

This implementation intercepts SWE-Agent's model calls through ModelProxy,
allowing VERL to control the generation process and collect training trajectories.

SWE-Agent is invoked directly via subprocess and manages its own execution environment
(which may include Docker containers based on SWE-Agent's own configuration).
"""

import asyncio
import logging
import os
import shutil
import tempfile
import time
import uuid
from typing import Any, Dict, Optional

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopMetrics,
    AgentLoopOutput,
    register,
)

# Import from model_proxy package
from recipe.swe_agent.model_proxy import ModelProxy

# Import from config package
from recipe.swe_agent.config.config_builder import SWEAgentConfigBuilder
from recipe.swe_agent.config.config_validator import AgentConfigValidator

# Import from utils package
from recipe.swe_agent.utils.patch_extractor import PatchExtractor

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


@register("swe_agent")
class SWEAgentLoop(AgentLoopBase):
    """SWE Agent Loop - External control multi-turn interaction mode.

    This loop intercepts SWE-Agent's model calls through ModelProxy,
    generates responses using VERL's ChatModel, and collects trajectories.
    """

    def __init__(
        self,
        trainer_config,
        server_manager,
        tokenizer,
        processor,
        dataset_cls,
        dataset_config,
        **kwargs,
    ):
        """Initialize SWE Agent Loop.

        Args:
            trainer_config: Trainer configuration wrapper.
            server_manager: AsyncLLMServerManager for model inference.
            tokenizer: Tokenizer for tokenization.
            processor: Processor for multi-modal data.
            dataset_cls: Dataset class.
            dataset_config: Dataset configuration wrapper.
        """
        super().__init__(
            trainer_config=trainer_config,
            server_manager=server_manager,
            tokenizer=tokenizer,
            processor=processor,
            dataset_cls=dataset_cls,
            dataset_config=dataset_config,
            **kwargs,
        )

        # Get agent config (from trainer config)
        agent_config = self.config.actor_rollout_ref.rollout.agent

        # Merge hydra kwargs from YAML config file (these keys are passed via hydra.utils.instantiate)
        # YAML config takes priority over trainer config since YAML is agent-specific
        yaml_proxy_config = kwargs.get("proxy_config", {})
        yaml_sandbox_config = kwargs.get("sandbox_config", {})
        yaml_agent_config = kwargs.get("agent", {})

        # Validate required config
        self._validate_config(agent_config)

        # Initialize ModelProxy
        # ModelProxy auto-handles port conflicts by trying next port when occupied
        proxy_config = agent_config.get("proxy_config", {})
        # YAML config overrides trainer config
        if yaml_proxy_config:
            proxy_config.update(yaml_proxy_config)
        base_port = proxy_config.get("port", 8080)

        logger.info(f"Initializing ModelProxy with base port={base_port} (will auto-increment if needed)")
        self.model_proxy = ModelProxy(port=base_port)

        # Initialize sandbox configuration (merge YAML config on top)
        self.sandbox_config = agent_config.get("sandbox_config", {})
        if yaml_sandbox_config:
            # YAML sandbox_config overrides trainer config
            for k, v in yaml_sandbox_config.items():
                self.sandbox_config[k] = v

        # Store agent-level overrides for generated SWE-Agent runtime config
        # Merge YAML agent config (templates/tools) on top of trainer config
        self.agent_templates = agent_config.get("templates", {})
        if yaml_agent_config.get("templates"):
            self.agent_templates.update(yaml_agent_config["templates"])
        
        tools_config = agent_config.get("tools", {})
        if yaml_agent_config.get("tools"):
            for k, v in yaml_agent_config["tools"].items():
                tools_config[k] = v
        self.agent_tool_env_variables = tools_config.get("env_variables", {})
        self.agent_tool_registry_variables = tools_config.get("registry_variables", {})
        self.agent_tool_bundles = tools_config.get("bundles", None)
        parse_function = tools_config.get("parse_function", {})
        self.agent_parse_function_type = parse_function.get("type", "thought_action")
        self.agent_enable_bash_tool = tools_config.get("enable_bash_tool", True)
        # max_requeries: retries after format errors (smaller models need more retries)
        # Try YAML first, then trainer config, then default=5
        self.agent_max_requeries = yaml_agent_config.get(
            "max_requeries",
            agent_config.get("max_requeries", 5)
        )

        # Validate sandbox_config has required keys (after merging YAML + trainer config).
        # If YAML kwargs were not passed (e.g., re-instantiation by Ray), the sandbox_config
        # may be incomplete. In that case, fall back to sensible defaults to avoid crash.
        _sandbox_defaults = {
            "swe_agent_timeout": 1800,
            "max_steps": 30,
            "execution_timeout": 300,
            "output_dir": os.path.join(os.getcwd(), "swe_agent_outputs"),
            "max_turns": 15,
            "deployment_type": "docker",
        }
        for key, default_val in _sandbox_defaults.items():
            if key not in self.sandbox_config:
                logger.warning(
                    f"sandbox_config missing '{key}', using default: {default_val}. "
                    "Ensure swe_agent_config.yaml is complete."
                )
                self.sandbox_config[key] = default_val

        logger.info(
            f"SWE Agent Loop initialized ("
            f"deployment_type={self.sandbox_config['deployment_type']}, "
            f"max_turns={self.sandbox_config['max_turns']}, "
            f"max_steps={self.sandbox_config['max_steps']})"
        )
    
    def _validate_config(self, agent_config: dict) -> None:
        """Validate agent configuration.

        Args:
            agent_config: Agent configuration dictionary.

        Raises:
            ValueError: If configuration is invalid.
        """
        validator = AgentConfigValidator(agent_config)
        validator.validate()

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        """Run SWE Agent Loop.

        Main loop:
        1. Start ModelProxy server
        2. Start SWE-Agent (background task)
        3. Loop: intercept model calls, generate responses, collect trajectories
        4. Wait for SWE-Agent completion, get patch
        5. Return AgentLoopOutput

        Args:
            sampling_params: Sampling parameters for generation.
            **kwargs: Dataset fields including problem_statement, repo_path, repo_content, etc.

        Returns:
            AgentLoopOutput: Agent loop output with trajectory data.
        """
        run_start_time = time.time()

        # Extract input data
        # Note: problem_statement, repo_content, repo_path may be in extra_info dict
        logger.warning(f"SWE Agent Loop run() called with kwargs keys: {list(kwargs.keys())}")
        
        extra_info = kwargs.get("extra_info", {})
        logger.warning(f"extra_info type: {type(extra_info)}, value preview: {str(extra_info)[:200]}")
        
        if isinstance(extra_info, str):
            import json
            try:
                extra_info = json.loads(extra_info)
            except:
                extra_info = {}
        
        # Try to get from extra_info first, then from kwargs directly
        problem_statement = extra_info.get("problem_statement", "") or kwargs.get("problem_statement", "")
        repo_path = extra_info.get("repo_path", None) or kwargs.get("repo_path", None)
        repo_content = extra_info.get("repo_content", None) or kwargs.get("repo_content", None)
        raw_prompt = kwargs.get("raw_prompt", [])
        
        logger.warning(f"Extracted: problem_statement={problem_statement[:50] if problem_statement else 'None'}..., "
                      f"repo_path={repo_path}, repo_content type={type(repo_content)}")
        
        # Handle repo_content: create temporary repo if needed
        temp_repo_dir = None
        if repo_content and not repo_path:
            repo_create_start = time.time()
            temp_repo_dir = await self._create_temp_repo(repo_content)
            repo_path = temp_repo_dir
            logger.info(f"Created temporary repo at: {repo_path} (took {time.time() - repo_create_start:.1f}s)")
        elif not repo_path:
            repo_path = "/workspace/repo"

        logger.info(f"Starting SWE Agent Loop for problem: {problem_statement[:100]}...")

        # Start ModelProxy server
        proxy_start_time = time.time()
        await self.model_proxy.start_server()
        logger.info(f"ModelProxy started on port {self.model_proxy.port} (took {time.time() - proxy_start_time:.1f}s)")

        try:
            # Start SWE-Agent in background
            agent_start_time = time.time()
            agent_task = asyncio.create_task(
                self._run_swe_agent(problem_statement, repo_path)
            )

            # Collect trajectory data
            # response_ids and response_mask are built together:
            #   - model-generated tokens → mask=1 (participate in gradient)
            #   - tool/observation tokens → mask=0 (excluded from gradient)
            # This mirrors tool_agent_loop.py's approach.
            initial_prompt_ids = None
            all_response_ids: list[int] = []
            all_response_mask: list[int] = []
            all_response_logprobs: list[float] = []
            prev_messages: list[dict] | None = None  # track previous turn's messages for diff
            num_turns = 0
            patch = None
            max_turns = self.sandbox_config["max_turns"]
            max_turns_reached = False
            logger.info(f"Agent loop started with max_turns={max_turns}")

            # Main interaction loop
            # Use asyncio.wait to race get_request() vs agent_task completion.
            # This avoids the previous 5-minute timeout delay when SWE-Agent
            # completes without making model calls (e.g., on failure or fast tasks).
            while True:
                # Check if agent task already completed
                if agent_task.done():
                    try:
                        patch = await agent_task
                        logger.info(f"SWE-Agent completed (pre-check), patch length: {len(patch) if patch else 0}, "
                                   f"agent ran for {time.time() - agent_start_time:.1f}s")
                    except Exception as e:
                        logger.error(f"SWE-Agent task failed: {e}", exc_info=True)
                        patch = None
                    break

                # Create a task for getting the next model request
                request_task = asyncio.create_task(self.model_proxy.get_request())

                # Race: wait for EITHER a model request OR agent task completion
                done, pending = await asyncio.wait(
                    {request_task, agent_task},
                    timeout=300.0,  # 5 minute overall safety timeout
                    return_when=asyncio.FIRST_COMPLETED,
                )

                if not done:
                    # Both timed out - something is seriously wrong
                    logger.error(f"Both request and agent tasks timed out after 300s, "
                                f"agent running for {time.time() - agent_start_time:.1f}s")
                    request_task.cancel()
                    try:
                        await request_task
                    except (asyncio.CancelledError, Exception):
                        pass
                    break

                # If agent_task completed (with or without request_task)
                if agent_task in done:
                    # Cancel the pending request_task
                    if request_task in pending:
                        request_task.cancel()
                        try:
                            await request_task
                        except (asyncio.CancelledError, Exception):
                            pass

                    try:
                        patch = await agent_task
                        logger.info(f"SWE-Agent completed (detected via race), patch length: {len(patch) if patch else 0}, "
                                   f"agent ran for {time.time() - agent_start_time:.1f}s, total turns: {num_turns}")
                    except Exception as e:
                        logger.error(f"SWE-Agent task failed: {e}", exc_info=True)
                        patch = None
                    break

                # request_task completed - process the model request
                if request_task in done:
                    try:
                        model_request = request_task.result()
                    except Exception as e:
                        logger.error(f"Error getting model request: {e}", exc_info=True)
                        continue

                    turn_start_time = time.time()

                    # Normalize OpenAI messages for HF chat template
                    messages = self._normalize_openai_messages(model_request.messages)

                    # ── Extract tool/observation tokens (mask=0) ──
                    # SWE-Agent sends the *full* conversation history each turn.
                    # By diffing with the previous turn's messages, we can extract
                    # the new messages appended since last time:
                    #   - assistant reply (already captured as model output)
                    #   - tool observation (new — needs mask=0)
                    # On the first turn there is no diff to compute.
                    if prev_messages is not None and len(messages) > len(prev_messages):
                        # New messages = messages[len(prev_messages):]
                        # Typically: [assistant_reply, observation, ...]
                        # We only need the non-assistant messages (observations/tool outputs).
                        new_msgs = messages[len(prev_messages):]
                        tool_msgs = [m for m in new_msgs if m.get("role") != "assistant"]
                        if tool_msgs:
                            tool_token_ids = self.tokenizer.apply_chat_template(
                                tool_msgs,
                                add_generation_prompt=False,
                                tokenize=True,
                            )
                            all_response_ids.extend(tool_token_ids)
                            all_response_mask.extend([0] * len(tool_token_ids))
                            all_response_logprobs.extend([0.0] * len(tool_token_ids))
                            logger.debug(
                                f"Turn {num_turns + 1}: appended {len(tool_token_ids)} "
                                f"tool/observation tokens (mask=0)"
                            )

                    # Generate response using VERL's server_manager
                    prompt_ids = await self.apply_chat_template(messages)
                    request_id = str(uuid.uuid4())

                    # Truncate prompt_ids to fit within vLLM max_model_len.
                    # SWE-Agent sends the *full* conversation history each turn,
                    # so later turns can exceed max_model_len. We truncate from
                    # the LEFT (drop oldest context) to keep the most recent
                    # messages, preserving at least min_gen_tokens for generation.
                    rollout_cfg = self.config.actor_rollout_ref.rollout
                    max_model_len = getattr(rollout_cfg, 'max_model_len', None)
                    if max_model_len and len(prompt_ids) >= max_model_len:
                        min_gen_tokens = 256  # minimum space for generation
                        max_prompt_tokens = max_model_len - min_gen_tokens
                        original_len = len(prompt_ids)
                        prompt_ids = prompt_ids[-max_prompt_tokens:]
                        logger.warning(
                            f"Turn {num_turns + 1}: prompt truncated from {original_len} "
                            f"to {len(prompt_ids)} tokens (max_model_len={max_model_len})"
                        )

                    # Store initial prompt (first turn only)
                    if initial_prompt_ids is None:
                        initial_prompt_ids = prompt_ids

                    # Generate with logprobs if requested
                    sampling_params_with_logprobs = {
                        **sampling_params,
                        "logprobs": sampling_params.get("logprobs", True),
                    }

                    gen_start_time = time.time()
                    output = await self.server_manager.generate(
                        request_id=request_id,
                        prompt_ids=prompt_ids,
                        sampling_params=sampling_params_with_logprobs,
                    )
                    gen_elapsed = time.time() - gen_start_time

                    response_ids = output.token_ids
                    # Skip special tokens to avoid confusing SWE-Agent's parser
                    # (e.g., <|im_end|> would break the DISCUSSION/command format parsing)
                    response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)

                    # ── Collect model-generated tokens (mask=1) ──
                    all_response_ids.extend(response_ids)
                    all_response_mask.extend([1] * len(response_ids))
                    if output.log_probs is not None:
                        all_response_logprobs.extend(output.log_probs)
                    else:
                        all_response_logprobs.extend([0.0] * len(response_ids))

                    num_turns += 1

                    # Save current messages for next turn's diff
                    prev_messages = messages

                    # Send response back to SWE-Agent
                    await self.model_proxy.send_response(
                        response_text,
                        request=model_request,
                    )

                    turn_elapsed = time.time() - turn_start_time
                    n_model_tokens = len(response_ids)
                    n_total = len(all_response_ids)
                    n_mask1 = sum(all_response_mask)
                    logger.info(
                        f"Turn {num_turns}: {n_model_tokens} model tokens, "
                        f"total_seq={n_total} (mask=1: {n_mask1}, mask=0: {n_total - n_mask1}), "
                        f"gen={gen_elapsed:.1f}s, turn_total={turn_elapsed:.1f}s"
                    )

                    if max_turns is not None and num_turns >= int(max_turns):
                        max_turns_reached = True
                        logger.warning(
                            f"Max turns reached ({num_turns}/{max_turns}); "
                            "stopping further interaction."
                        )
                        break

            # Ensure agent task is complete
            if not agent_task.done():
                if max_turns_reached:
                    logger.warning("Cancelling SWE-Agent task due to max_turns limit")
                    agent_task.cancel()
                    try:
                        patch = await asyncio.wait_for(agent_task, timeout=10.0)
                    except asyncio.TimeoutError:
                        logger.warning("Timeout waiting for agent task after cancel")
                        patch = None
                    except asyncio.CancelledError:
                        patch = None
                else:
                    try:
                        patch = await asyncio.wait_for(agent_task, timeout=60.0)
                    except asyncio.TimeoutError:
                        logger.warning("Timeout waiting for agent task completion")
                        patch = None

            total_elapsed = time.time() - run_start_time
            logger.info(f"SWE Agent Loop completed: {num_turns} turns, "
                       f"patch={'yes' if patch else 'no'}, total={total_elapsed:.1f}s")

            # Build AgentLoopOutput
            # Structure (mirrors tool_agent_loop.py):
            # - prompt_ids: initial prompt (from first turn)
            # - response_ids: interleaved model tokens + tool observation tokens
            # - response_mask: 1 for model-generated tokens, 0 for tool/observation
            # - response_logprobs: real logprobs for model tokens, 0.0 for tool tokens
            
            # Truncate to max response length
            rollout_config = self.config.actor_rollout_ref.rollout
            max_response_length = rollout_config.response_length
            
            # Use initial prompt from first turn.
            # If no turns happened, use a minimal prompt with just the pad token
            # to ensure tokenizer.pad doesn't receive an empty list.
            if initial_prompt_ids is not None and len(initial_prompt_ids) > 0:
                final_prompt_ids = initial_prompt_ids
            else:
                pad_token_id = getattr(self.tokenizer, 'pad_token_id', 0) or 0
                final_prompt_ids = [pad_token_id]
            
            # Truncate response_ids + mask + logprobs to max length (they stay aligned).
            if all_response_ids:
                final_response_ids = all_response_ids[:max_response_length]
                final_response_mask = all_response_mask[:max_response_length]
            else:
                pad_token_id = getattr(self.tokenizer, 'pad_token_id', 0) or 0
                final_response_ids = [pad_token_id]
                final_response_mask = [0]
            
            # Log mask statistics for debugging
            n_mask1 = sum(final_response_mask)
            n_mask0 = len(final_response_mask) - n_mask1
            logger.info(
                f"Response mask stats: total={len(final_response_mask)}, "
                f"model(mask=1)={n_mask1}, tool(mask=0)={n_mask0}"
            )
            
            # IMPORTANT: Always provide response_logprobs (even if all zeros).
            # The verl core _postprocess checks inputs[0].response_logprobs to decide
            # whether to include rollout_log_probs in the batch TensorDict.
            # If some workers return None and others return logprobs, the cross-worker
            # DataProto.concat will fail with a KeyError due to mismatched keys.
            if all_response_logprobs:
                final_response_logprobs = all_response_logprobs[:len(final_response_ids)]
            else:
                final_response_logprobs = [0.0] * len(final_response_ids)

            metrics = AgentLoopMetrics(
                generate_sequences=0.0,  # Will be calculated by post-processing
                tool_calls=0.0,
                num_preempted=-1,
            )

            output = AgentLoopOutput(
                prompt_ids=final_prompt_ids,
                response_ids=final_response_ids,
                response_mask=final_response_mask,
                response_logprobs=final_response_logprobs,
                num_turns=num_turns,
                metrics=metrics,
                extra_fields={
                    "patch": patch,
                    "problem_statement": problem_statement,
                    "repo_path": repo_path,
                },
            )

            return output

        finally:
            # Cleanup
            await self.model_proxy.stop_server()
            
            # Cleanup temporary repo if created
            if temp_repo_dir:
                try:
                    import shutil
                    shutil.rmtree(temp_repo_dir, ignore_errors=True)
                    logger.debug(f"Cleaned up temporary repo: {temp_repo_dir}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup temp repo: {e}")

    async def _create_temp_repo(self, repo_content: Dict[str, Optional[str]]) -> str:
        """Create a temporary git repository from repo_content.

        Args:
            repo_content: Dictionary mapping file paths to their content.
                         If value is None, a default content is used.

        Returns:
            Path to the temporary repository.
        """
        # Create temporary directory
        temp_dir = tempfile.mkdtemp(prefix="swe_repo_")
        
        try:
            # Write files
            for file_path, content in repo_content.items():
                full_path = os.path.join(temp_dir, file_path)
                os.makedirs(os.path.dirname(full_path), exist_ok=True) if os.path.dirname(full_path) else None
                
                if content is None:
                    # Default content for None values
                    content = f"# {file_path}\n"
                
                with open(full_path, "w", encoding="utf-8") as f:
                    f.write(content)
            
            # Initialize git repository
            cmds = [
                f"cd {temp_dir} && git init",
                f"cd {temp_dir} && git config user.email 'verl@swe-agent.local'",
                f"cd {temp_dir} && git config user.name 'VERL SWE-Agent'",
                f"cd {temp_dir} && git add -A",
                f"cd {temp_dir} && git commit -m 'Initial commit'",
            ]
            
            for cmd in cmds:
                proc = await asyncio.create_subprocess_shell(
                    cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                await proc.communicate()
            
            logger.debug(f"Created temp repo with {len(repo_content)} files at {temp_dir}")
            return temp_dir
            
        except Exception as e:
            logger.error(f"Failed to create temp repo: {e}", exc_info=True)
            # Cleanup on failure
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise

    def _normalize_openai_messages(self, openai_messages: list[dict]) -> list[dict]:
        """Normalize OpenAI-format messages for tokenizer.apply_chat_template."""
        messages: list[dict] = []
        for msg in openai_messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            # OpenAI may send content blocks (list[dict]); flatten to plain text
            if isinstance(content, list):
                text_parts = []
                for part in content:
                    if isinstance(part, dict):
                        if part.get("type") == "text" and "text" in part:
                            text_parts.append(part["text"])
                        else:
                            text_parts.append(str(part.get("text", part)))
                    else:
                        text_parts.append(str(part))
                content = "\n".join(text_parts)
            elif content is None:
                content = ""
            else:
                content = str(content)

            messages.append({"role": role, "content": content})

        return messages

    async def _run_swe_agent(self, problem_statement: str, repo_path: str) -> Optional[str]:
        """Run SWE-Agent via subprocess and return the generated patch.

        SWE-Agent is invoked directly using the sweagent CLI command.
        SWE-Agent may create its own Docker containers based on its configuration.

        IMPORTANT: Docker containers are launched with --rm flag, so they auto-remove
        when stopped. We do NOT use before/after snapshot-based cleanup because that
        approach is fundamentally broken for parallel execution: when worker A finishes
        first, its cleanup would kill Docker containers belonging to workers B-H that
        are still running. SWE-Agent's built-in remove_container=True also handles
        container lifecycle.

        Args:
            problem_statement: The problem statement for SWE-Agent.
            repo_path: Path to the repository.

        Returns:
            Generated patch string or None.
        """
        swe_agent_start_time = time.time()

        # Use full UUID + timestamp to avoid ID collisions under high concurrency
        # Format: {uuid[:12]}-{timestamp} e.g. a1b2c3d4e5f6-1707123456
        instance_id = f"{uuid.uuid4().hex[:12]}-{int(time.time())}"
        output_dir = self.sandbox_config["output_dir"]
        instance_output_dir = os.path.join(output_dir, instance_id)
        
        # Ensure output directory exists
        os.makedirs(instance_output_dir, exist_ok=True)
        
        # Create dedicated temp execution directory to avoid potential issues with cwd="/tmp"
        exec_dir = tempfile.mkdtemp(prefix=f"swe_exec_{instance_id}_")
        
        # Generate SWE-Agent configuration
        config_start = time.time()
        config_path = await self._generate_swe_agent_config(
            instance_id=instance_id,
            repo_path=repo_path,
            output_dir=instance_output_dir,
        )
        logger.info(f"[{instance_id}] Config generated in {time.time() - config_start:.1f}s")
        
        try:
            # Run SWE-Agent via subprocess
            exec_start = time.time()
            patch = await self._execute_swe_agent(
                config_path=config_path,
                problem_statement=problem_statement,
                instance_id=instance_id,
                output_dir=instance_output_dir,
                repo_path=repo_path,
                exec_dir=exec_dir,
            )
            logger.info(f"[{instance_id}] SWE-Agent execution took {time.time() - exec_start:.1f}s, "
                       f"total _run_swe_agent={time.time() - swe_agent_start_time:.1f}s, "
                       f"patch={'yes' if patch else 'no'}")
            return patch
        except Exception as e:
            logger.error(f"[{instance_id}] SWE-Agent execution failed after {time.time() - swe_agent_start_time:.1f}s: {e}", exc_info=True)
            return None
        finally:
            # NOTE: We intentionally do NOT clean up swerex containers here.
            # Docker --rm flag ensures containers auto-remove when they stop.
            # SWE-Agent's remove_container=True config also handles cleanup.
            # The previous before/after snapshot approach was killing OTHER workers'
            # containers in parallel execution, causing "Server disconnected" errors.
            
            # Cleanup temporary config
            try:
                if os.path.exists(config_path):
                    os.unlink(config_path)
            except Exception as e:
                logger.warning(f"Failed to cleanup config file: {e}")
            
            # Cleanup execution directory
            try:
                shutil.rmtree(exec_dir, ignore_errors=True)
            except Exception as e:
                logger.warning(f"Failed to cleanup exec dir: {e}")

    async def _generate_swe_agent_config(
        self,
        instance_id: str,
        repo_path: str,
        output_dir: str,
    ) -> str:
        """Generate SWE-Agent configuration file.

        Args:
            instance_id: Unique instance identifier.
            repo_path: Path to the repository.
            output_dir: Directory for SWE-Agent output.

        Returns:
            Path to the generated config file.
        """
        builder = SWEAgentConfigBuilder(
            instance_id=instance_id,
            repo_path=repo_path,
            output_dir=output_dir,
            model_proxy_port=self.model_proxy.port,
            max_steps=self.sandbox_config["max_steps"],
            execution_timeout=self.sandbox_config["execution_timeout"],
            custom_templates=self.agent_templates,
            custom_env_variables=self.agent_tool_env_variables,
            custom_registry_variables=self.agent_tool_registry_variables,
            custom_tool_bundles=self.agent_tool_bundles,
            parse_function_type=self.agent_parse_function_type,
            max_requeries=self.agent_max_requeries,
            max_observation_length=self.agent_templates.get("max_observation_length", 85000),
            enable_bash_tool=self.agent_enable_bash_tool,
            deployment_type=self.sandbox_config["deployment_type"],
            docker_image=self.sandbox_config.get("docker_image", "swerex-python:3.11"),
            docker_memory_limit=self.sandbox_config.get("docker_memory_limit", "8g"),
            docker_startup_timeout=self.sandbox_config.get("docker_startup_timeout", 180.0),
            docker_remove_container=self.sandbox_config.get("docker_remove_container", True),
            # SWE-Agent checks token count before each LLM call.
            # Exceeding → ContextWindowExceededError → clean exit with auto-submit.
            max_input_tokens=int(getattr(
                self.config.actor_rollout_ref.rollout, 'max_model_len', 0
            ) or 0),
        )

        # Write config to temporary file
        config_file = tempfile.NamedTemporaryFile(
            mode="w",
            suffix=f"_swe_config_{instance_id}.yaml",
            delete=False,
            encoding="utf-8",
        )
        config_file.write(builder.to_yaml())
        config_file.close()

        logger.debug(f"Generated SWE-Agent config at: {config_file.name}")
        return config_file.name

    async def _execute_swe_agent(
        self,
        config_path: str,
        problem_statement: str,
        instance_id: str,
        output_dir: str,
        repo_path: str,
        exec_dir: str,
    ) -> Optional[str]:
        """Execute SWE-Agent CLI and return the generated patch.

        Args:
            config_path: Path to SWE-Agent config file.
            problem_statement: The problem statement.
            instance_id: Unique instance identifier.
            output_dir: Directory for SWE-Agent output.
            repo_path: Path to the repository (used for git diff fallback).
            exec_dir: Temporary directory for command execution (avoids YAML parsing issues).

        Returns:
            Generated patch string or None.
        """
        # Build sweagent command
        swe_agent_timeout = self.sandbox_config.get("swe_agent_timeout", 1800)
        
        cmd = [
            "sweagent",
            "run",
            "--config", config_path,
            "--problem_statement.text", problem_statement,
            "--problem_statement.id", instance_id,
        ]
        
        logger.info(f"[{instance_id}] Executing SWE-Agent (proxy port={self.model_proxy.port})...")
        
        # api_base and api_key are set in the generated SWE-Agent YAML config
        # (config_builder.py). No need to duplicate via environment variables.
        env = os.environ.copy()
        
        # Run SWE-Agent in subprocess
        # IMPORTANT: Use dedicated exec_dir to avoid YAML parsing issues
        # The verl directory has a 'docker' subdir which causes SWE-Agent to
        # misparse "type: docker" as a path instead of a string literal
        process = None
        try:
            subprocess_start = time.time()
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=exec_dir,  # Use dedicated temp dir to avoid potential /tmp issues
            )
            logger.info(f"[{instance_id}] Subprocess created (pid={process.pid}), waiting for completion...")
            
            # Wait for completion with timeout
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=swe_agent_timeout,
                )
            except asyncio.TimeoutError:
                elapsed = time.time() - subprocess_start
                logger.error(f"[{instance_id}] SWE-Agent timed out after {elapsed:.1f}s (limit={swe_agent_timeout}s)")
                process.kill()
                await process.wait()
                return None
            
            subprocess_elapsed = time.time() - subprocess_start
            
            if process.returncode != 0:
                logger.error(f"[{instance_id}] SWE-Agent failed (rc={process.returncode}) after {subprocess_elapsed:.1f}s")
                # Log stderr (last 2000 chars to avoid flooding)
                stderr_text = stderr.decode(errors='replace')
                stdout_text = stdout.decode(errors='replace')
                logger.error(f"[{instance_id}] stderr (last 2000): {stderr_text[-2000:]}")
                logger.error(f"[{instance_id}] stdout (last 1000): {stdout_text[-1000:]}")
                # Still try to extract patch even if process failed
            else:
                logger.info(f"[{instance_id}] SWE-Agent subprocess completed successfully in {subprocess_elapsed:.1f}s")
            
            # Extract patch from output (supports git diff fallback)
            extract_start = time.time()
            patch = await self._extract_patch(output_dir, instance_id, repo_path)
            logger.info(f"[{instance_id}] Patch extraction took {time.time() - extract_start:.1f}s")
            
            if patch:
                logger.info(f"[{instance_id}] Successfully extracted patch ({len(patch)} chars)")
            else:
                logger.warning(f"[{instance_id}] No patch found in SWE-Agent output or git diff")
                
            return patch
            
        except asyncio.CancelledError:
            logger.warning(f"[{instance_id}] SWE-Agent task cancelled, terminating subprocess...")
            if process is not None and process.returncode is None:
                try:
                    process.kill()
                    await process.wait()
                except Exception:
                    pass
            raise
        except FileNotFoundError:
            logger.error("SWE-Agent not found. Please install it with: pip install sweagent")
            return None
        except Exception as e:
            logger.error(f"[{instance_id}] Error running SWE-Agent after {time.time() - subprocess_start:.1f}s: {e}", exc_info=True)
            return None
    
    async def _extract_patch(self, output_dir: str, instance_id: str, repo_path: str) -> Optional[str]:
        """Extract patch from SWE-Agent output directory.

        Uses PatchExtractor for unified patch extraction with file → git diff fallback.

        Args:
            output_dir: SWE-Agent output directory.
            instance_id: Instance identifier.
            repo_path: Path to the repository.

        Returns:
            Patch content string or None.
        """
        extractor = PatchExtractor(
            output_dir=output_dir,
            instance_id=instance_id,
            repo_path=repo_path,
        )

        patch = await extractor.extract()

        if patch:
            logger.info(f"Successfully extracted patch ({len(patch)} chars)")
        else:
            logger.warning("No patch found")

        return patch
