# Atropos Reward Registry

Pluggable reward bridge that complements the Atropos RL environment integration.

## Problem

verl raises NotImplementedError for unknown data sources when computing rewards.
This causes crashes when using custom environments or datasets not built into verl
reward scoring system (see issue #5558).

## Solution

RewardRegistry provides graceful fallback handling:
1. Tries verl built-in reward functions first
2. Falls back to registered custom handlers
3. Falls back to Atropos scoring via HTTP
4. Returns 0.0 gracefully instead of crashing

## Built-in Handlers

- openai/gsm8k: extracts #### answer with flexible fallback

## Addresses

- Issue #5558: NotImplementedError for unknown data sources
- Issue #5531: Code RL recipe with execution-based rewards  
- Issue #5536: Custom environment support

## Tests

5/5 unit tests passing.
