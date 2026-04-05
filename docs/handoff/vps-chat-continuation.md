# VPS Chat Continuation Handoff

This project includes a handoff pack so a new chat on VPS can resume with context.

## What to transfer
1. Repository itself (push/pull via git)
2. Exported memory files in docs/handoff/memory-export/
3. Optional local VS Code chat storage archive (if you created one): ~/copilot-chat-session.tar.gz

## Memory files included
- docs/handoff/memory-export/qwen-model-facts.md
- docs/handoff/memory-export/flux2-klein-api.md
- docs/handoff/memory-export/prompt-pipeline-design.md
- docs/handoff/memory-export/diffusion-text-tuner-plan.md

## Recommended flow
1. Commit and push this repo from local machine.
2. Pull the repo on VPS.
3. Start a new Copilot chat on VPS and paste:
   - Short task you want next
   - Contents of docs/handoff/memory-export/qwen-model-facts.md
   - High-level summary from docs/handoff/memory-export/diffusion-text-tuner-plan.md
4. Attach recent logs/errors if any.

## Optional: include workspace chat archive
If you want to carry your local chat artifact archive as a file:
- Create archive locally from workspaceStorage (already done in your shell history)
- Copy it to VPS via scp
- Keep it as historical backup; Copilot usually cannot directly restore a session from this file

## Suggested first prompt on VPS
Project repo contains a handoff in docs/handoff/. Use these files as source of truth:
- docs/handoff/memory-export/qwen-model-facts.md
- docs/handoff/memory-export/diffusion-text-tuner-plan.md
- docs/handoff/memory-export/flux2-klein-api.md
Continue by validating baseline generation and reward scripts, then run small smoke tests.
