# Security policy

Latent Fovea is maintained by KnowDrive.ai (Recourse Software Inc.).

## Reporting a vulnerability

Please email **security@recoursellm.com** rather than opening a public issue. Include what you found, how to reproduce it, and the version or commit. We will acknowledge your report within three business days and keep you informed until it is resolved. Please give us a reasonable time to release a fix before you disclose it.

## Scope

Latent Fovea is a static web app. It runs entirely in your browser and has no server of its own. The areas that matter most:

- **Tokens.** A Hugging Face token you enter is stored only in that browser's storage and sent only to huggingface.co. Any path that leaks it elsewhere is a vulnerability.
- **Loading models and data.** Weights, tokenizers and data sets are parsed in the browser, in WebAssembly and on WebGPU. A crafted file that escapes those parsers or corrupts memory is in scope.
- **The tools in `tools/`.** They read model files and data you point them at.

## Supported versions

Security fixes go into the latest release.
