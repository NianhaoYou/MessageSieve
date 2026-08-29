# MessageSieve: Runtime Message Pruning for Accurate and Efficient LLM-Based Multi-Agent Communication

Official repository for **MessageSieve**, a runtime message-pruning method for LLM-based multi-agent systems.

> **Paper and code are coming soon.** We are currently preparing the camera-ready paper and organizing the implementation and reproducibility materials.

## Overview

LLM-based multi-agent systems improve reasoning through collaboration, but dense multi-round communication can introduce redundant, outdated, or noisy information and substantially increase prompt costs. Existing communication-efficiency methods often prune agents, edges, or graph structures, making decisions at a relatively coarse structural level.

MessageSieve instead operates on individual message instances before each receiver prompt is assembled. For every incoming message, it selects one of three views:

- **Full:** preserve the original message;
- **Lite:** retain short extractive evidence;
- **Mute:** omit the message from the prompt.

The method keeps the original agents, communication graph, prompts, and round schedule unchanged. It uses lightweight receiver-local comparisons and requires neither a trained selector nor an additional LLM call.

## Highlights

- Runtime pruning at the message level rather than the agent or edge level.
- Receiver-aware decisions that adapt across queries, agents, and rounds.
- Training-free operation without modifying the underlying multi-agent workflow.

## Release

The source code, configurations, and reproduction instructions will be released in this repository after they have been cleaned and organized. Please stay tuned.
