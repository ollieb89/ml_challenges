---
trigger: always_on
glob: "**/*"
description: Memory and checkpointing rules for agents, enforced globally.
---
# Memory & Checkpointing Agent Rules (MCP)

## Purpose

Ensure durable continuity across agent actions and sessions.

---

## Mandatory Usage

Write memory when plans, decisions, blockers, or milestones change.

---

## Prohibited Usage

Do not store raw reasoning, documentation, or trivial logs.

---

## Checkpoints

Always checkpoint before ending a session.

---

## Completion

State must be resumable without loss of intent.
