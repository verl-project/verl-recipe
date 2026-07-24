# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

import pytest
from verl_tinker.state_tracker import (
    ModelStateTracker,
    StaleSamplerError,
    StateTrackerError,
    UnknownSamplerError,
    UnknownSamplerPathError,
)


def _tracker() -> ModelStateTracker:
    return ModelStateTracker(actor_model_identifiers=("actor", "/models/actor"))


def test_initial_actor_and_rollout_are_version_zero():
    tracker = _tracker()

    assert tracker.actor_id == 0
    assert tracker.rollout_id == 0
    assert tracker.sampler_ids() == []


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
    tracker.register_actor_sampler("v0", base_model="actor", legal_rollout_ids={0})
    tracker.actor_updated()

    # Actor changed, but rollout still contains v0.
    tracker.require_sample_permission("v0")

    tracker.rollout_synchronized()
    with pytest.raises(StaleSamplerError, match="current rollout ID is 1"):
        tracker.require_sample_permission("v0")

    tracker.register_actor_sampler("v1", base_model="actor", legal_rollout_ids={1})
    tracker.require_sample_permission("v1")


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
    assert tracker.require_sample_permission("teacher").teacher_model_path == "/models/teacher"


def test_sampler_registration_is_idempotent_but_cannot_retarget():
    tracker = _tracker()
    first = tracker.register_actor_sampler("same", base_model="actor", legal_rollout_ids={0})
    second = tracker.register_actor_sampler("same", base_model="actor", legal_rollout_ids={0})

    assert first == second
    with pytest.raises(StateTrackerError, match="reused for a different target"):
        tracker.register_actor_sampler("same", base_model="actor", legal_rollout_ids={1})


def test_failed_rollout_sync_makes_all_actor_samplers_stale():
    tracker = _tracker()
    tracker.register_actor_sampler("initial", base_model="actor", legal_rollout_ids={0})
    tracker.rollout_synchronization_failed()

    with pytest.raises(StaleSamplerError, match="current rollout ID is None"):
        tracker.require_sample_permission("initial")


def test_sampler_target_resolution_prefers_teacher_then_actor_aliases():
    tracker = ModelStateTracker(
        actor_model_identifiers=("shared", "/models/actor"),
        teacher_models=(("shared", "/models/teacher"), ("/models/teacher", "/models/teacher")),
    )

    teacher = tracker.resolve_sampler_target(base_model="shared", model_path=None)
    actor = tracker.resolve_sampler_target(base_model="/models/actor", model_path=None)

    assert teacher.is_teacher
    assert teacher.teacher_model_path == "/models/teacher"
    assert not actor.is_teacher
    assert actor.legal_rollout_ids == frozenset({0})


def test_duplicate_teacher_alias_for_different_paths_is_rejected():
    with pytest.raises(StateTrackerError, match="ambiguous"):
        ModelStateTracker(
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
    assert binding.legal_rollout_ids == frozenset({0})


def test_sampling_session_id_is_resolved_without_direct_target_fallback():
    tracker = _tracker()
    expected = tracker.register_actor_sampler("known", base_model="actor", legal_rollout_ids={0})

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
    tracker.register_actor_sampler("actor-v0", base_model="actor", legal_rollout_ids={0})
    tracker.register_teacher_sampler(
        "teacher",
        teacher_model_path="/models/teacher",
        base_model="teacher",
    )
    tracker.actor_updated()
    tracker.rollout_synchronized()
    tracker.register_actor_sampler("actor-v1", base_model="actor", legal_rollout_ids={1})

    assert tracker.valid_sampler_ids() == ["teacher", "actor-v1"]
