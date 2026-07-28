# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

import pytest
from verl_tinker.model_resource_manager import (
    ModelResourceManager,
    ModelResourceManagerError,
    SamplingResource,
    StaleSamplerError,
    UnknownSamplerError,
    UnknownSamplerPathError,
)


def _tracker() -> ModelResourceManager:
    return ModelResourceManager(actor_model_identifiers=("actor", "/models/actor"))


def test_initial_actor_and_rollout_are_version_zero():
    tracker = _tracker()

    assert tracker.actor_id == 0
    assert tracker.rollout_id == 0
    assert tracker.sampler_ids() == []


def test_sessions_models_and_samplers_have_distinct_retry_stable_ids():
    tracker = _tracker()

    first_session = tracker.create_session()
    second_session = tracker.create_session()
    first_model = tracker.register_model(first_session, 0)
    second_model = tracker.register_model(second_session, 0)
    first_sampler = tracker.sampler_id(first_session, 0)
    second_sampler = tracker.sampler_id(second_session, 0)

    assert first_session == "verl-tinker-session:0"
    assert second_session == "verl-tinker-session:1"
    assert first_model == "verl-tinker-session:0:train:0"
    assert second_model == "verl-tinker-session:1:train:0"
    assert tracker.register_model(first_session, 0) == first_model
    assert tracker.session_id_for_model(first_model) == first_session
    assert first_sampler == "verl-tinker-session:0:sample:0"
    assert second_sampler == "verl-tinker-session:1:sample:0"


def test_unknown_session_and_model_are_rejected():
    tracker = _tracker()

    with pytest.raises(ModelResourceManagerError, match="Unknown Tinker session"):
        tracker.register_model("missing", 0)
    with pytest.raises(ModelResourceManagerError, match="Unknown Tinker session"):
        tracker.sampler_id("missing", 0)
    with pytest.raises(ModelResourceManagerError, match="Unknown training model"):
        tracker.session_id_for_model("missing")


def test_loading_old_actor_id_does_not_reuse_allocated_ids():
    tracker = _tracker()
    tracker.state_saved("tinker://state/base")
    first_update = tracker.actor_updated()
    second_update = tracker.actor_updated()

    tracker.state_loaded("tinker://state/base")
    next_update = tracker.actor_updated()

    assert (first_update, second_update, next_update) == (1, 2, 3)


def test_matching_state_path_skips_and_untracked_load_gets_identity():
    tracker = _tracker()
    tracker.state_saved("tinker://state/zero")

    assert tracker.should_skip_state_load("tinker://state/zero") is True
    assert tracker.should_skip_state_load("tinker://state/external") is False

    external_id = tracker.state_loaded("tinker://state/external")
    assert external_id == 1
    assert tracker.should_skip_state_load("tinker://state/external") is True


def test_sampler_binding_becomes_stale_after_new_rollout_sync():
    tracker = _tracker()
    v0 = tracker.register_actor_sampler("v0", base_model="actor", actor_id=0)
    tracker.actor_updated()

    # Actor changed, but rollout still contains v0.
    assert tracker.resolve_resource(v0, scalar_prompt_logprobs=False) is SamplingResource.ROLLOUT

    tracker.rollout_synchronized()
    with pytest.raises(StaleSamplerError, match="bound to actor ID 0"):
        tracker.resolve_resource(v0, scalar_prompt_logprobs=False)

    v1 = tracker.register_actor_sampler("v1", base_model="actor", actor_id=1)
    assert tracker.resolve_resource(v1, scalar_prompt_logprobs=False) is SamplingResource.ROLLOUT


def test_sampler_path_and_teacher_bindings():
    tracker = _tracker()
    tracker.actor_updated()
    tracker.rollout_synchronized()
    tracker.sampler_path_saved("tinker://sampler/v1")

    assert tracker.actor_id_for_sampler_path("tinker://sampler/v1") == 1
    with pytest.raises(UnknownSamplerPathError):
        tracker.actor_id_for_sampler_path("tinker://sampler/missing")

    tracker.register_teacher_sampler(
        "teacher",
        teacher_model_path="/models/teacher",
        base_model="teacher-model",
    )
    tracker.actor_updated()
    tracker.rollout_synchronized()
    teacher = tracker.get_sampler("teacher")
    assert tracker.resolve_resource(teacher, scalar_prompt_logprobs=False) is SamplingResource.TEACHER


def test_sampler_registration_is_idempotent_but_cannot_retarget():
    tracker = _tracker()
    first = tracker.register_actor_sampler("same", base_model="actor", actor_id=0)
    second = tracker.register_actor_sampler("same", base_model="actor", actor_id=0)

    assert first == second
    with pytest.raises(ModelResourceManagerError, match="reused for a different target"):
        tracker.register_actor_sampler("same", base_model="actor", actor_id=1)


def test_failed_rollout_sync_makes_all_actor_samplers_stale():
    tracker = _tracker()
    initial = tracker.register_actor_sampler("initial", base_model="actor", actor_id=0)
    tracker.rollout_synchronization_failed()

    with pytest.raises(StaleSamplerError, match="rollout_id=None"):
        tracker.resolve_resource(initial, scalar_prompt_logprobs=False)
    assert tracker.resolve_resource(initial, scalar_prompt_logprobs=True) is SamplingResource.ACTOR


def test_sampler_target_resolution_prefers_teacher_then_actor_aliases():
    tracker = ModelResourceManager(
        actor_model_identifiers=("shared", "/models/actor"),
        teacher_models=(("shared", "/models/teacher"), ("/models/teacher", "/models/teacher")),
    )

    teacher = tracker.resolve_sampler_target(base_model="shared", model_path=None)
    actor = tracker.resolve_sampler_target(base_model="/models/actor", model_path=None)

    assert teacher.is_teacher
    assert teacher.teacher_model_path == "/models/teacher"
    assert not actor.is_teacher
    assert actor.actor_id == 0


def test_duplicate_teacher_alias_for_different_paths_is_rejected():
    with pytest.raises(ModelResourceManagerError, match="ambiguous"):
        ModelResourceManager(
            actor_model_identifiers=("actor",),
            teacher_models=(("teacher", "/models/one"), ("teacher", "/models/two")),
        )


def test_direct_sampling_target_uses_known_sampler_path():
    tracker = _tracker()
    tracker.sampler_path_saved("tinker://sampler/base")

    binding = tracker.resolve_sampling_request(
        sampling_session_id=None,
        base_model=None,
        model_path="tinker://sampler/base",
    )

    assert binding.sampler_id is None
    assert binding.actor_id == 0


def test_actor_model_path_binds_to_initial_actor_id():
    tracker = _tracker()

    binding = tracker.resolve_sampler_target(
        base_model="actor",
        model_path="/models/actor",
    )

    assert not binding.is_teacher
    assert binding.actor_id == 0


def test_sampling_session_id_is_resolved_without_direct_target_fallback():
    tracker = _tracker()
    expected = tracker.register_actor_sampler("known", base_model="actor", actor_id=0)

    assert (
        tracker.resolve_sampling_request(
            sampling_session_id="known",
            base_model="unknown",
            model_path="tinker://unknown",
        )
        == expected
    )
    with pytest.raises(UnknownSamplerError):
        tracker.resolve_sampling_request(
            sampling_session_id="missing",
            base_model="actor",
            model_path=None,
        )


def test_valid_sampler_ids_excludes_stale_actor_bindings():
    tracker = _tracker()
    tracker.register_actor_sampler("actor-v0", base_model="actor", actor_id=0)
    tracker.register_teacher_sampler(
        "teacher",
        teacher_model_path="/models/teacher",
        base_model="teacher",
    )
    tracker.actor_updated()
    tracker.rollout_synchronized()
    tracker.register_actor_sampler("actor-v1", base_model="actor", actor_id=1)

    assert tracker.valid_sampler_ids() == ["teacher", "actor-v1"]


def test_late_binding_prefers_rollout_then_actor_then_reference():
    tracker = _tracker()
    v0 = tracker.register_actor_sampler("v0", base_model="actor", actor_id=0)

    assert tracker.resolve_resource(v0, scalar_prompt_logprobs=True) is SamplingResource.ROLLOUT

    tracker.actor_updated()
    assert tracker.resolve_resource(v0, scalar_prompt_logprobs=True) is SamplingResource.ROLLOUT

    tracker.rollout_synchronized()
    current = tracker.register_actor_sampler("current", base_model="actor", actor_id=1)
    assert tracker.resolve_resource(current, scalar_prompt_logprobs=True) is SamplingResource.ROLLOUT

    tracker.rollout_synchronization_failed()
    assert tracker.resolve_resource(current, scalar_prompt_logprobs=True) is SamplingResource.ACTOR

    tracker.configure_resources(rollout_enabled=True, reference_enabled=True)
    assert tracker.resolve_resource(v0, scalar_prompt_logprobs=True) is SamplingResource.REFERENCE


def test_actor_and_reference_fallback_require_scalar_prompt_logprobs():
    tracker = _tracker()
    v0 = tracker.register_actor_sampler("v0", base_model="actor", actor_id=0)
    tracker.actor_updated()
    tracker.rollout_synchronized()
    tracker.configure_resources(rollout_enabled=True, reference_enabled=True)

    with pytest.raises(StaleSamplerError, match="autoregressive sampling"):
        tracker.resolve_resource(v0, scalar_prompt_logprobs=False)
    assert tracker.resolve_resource(v0, scalar_prompt_logprobs=True) is SamplingResource.REFERENCE


def test_no_rollout_server_uses_actor_for_logprobs_only():
    tracker = _tracker()
    tracker.configure_resources(rollout_enabled=False, reference_enabled=False)
    initial = tracker.resolve_sampler_target(base_model="actor", model_path=None)

    assert tracker.rollout_id is None
    assert tracker.resolve_resource(initial, scalar_prompt_logprobs=True) is SamplingResource.ACTOR
    with pytest.raises(StaleSamplerError, match="autoregressive sampling"):
        tracker.resolve_resource(initial, scalar_prompt_logprobs=False)


def test_teacher_model_path_takes_precedence_over_actor_base_model():
    tracker = ModelResourceManager(
        actor_model_identifiers=("shared", "/models/actor"),
        teacher_models=(("/models/teacher", "/models/teacher"),),
    )

    binding = tracker.resolve_sampler_target(
        base_model="shared",
        model_path="/models/teacher",
    )

    assert binding.is_teacher
    assert binding.teacher_model_path == "/models/teacher"
