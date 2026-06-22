# Third-party licenses and attribution

This project (ARC Prize 2026 entry) is released under MIT-0 (see `LICENSE`). It builds on the
third-party work below, each used under its own license. The fine-tuned model we ship is a
derivative of Qwen3-4B (Apache-2.0); when that model is redistributed (for example as a Kaggle
dataset), the full Apache-2.0 license text (`LICENSE-APACHE-2.0.txt`, the canonical text from
http://www.apache.org/licenses/LICENSE-2.0) and the `NOTICE` file travel with it.

## Components

| Component | Use | License | Source |
| --- | --- | --- | --- |
| RE-ARC (Michael Hodel) | training-data generators | MIT | https://github.com/michaelhodel/re-arc |
| ARC-AGI-2 (ARC Prize Foundation) | official training and eval tasks | Apache-2.0 | https://github.com/arcprize/ARC-AGI-2 |
| Qwen3-4B (Alibaba, Qwen team) | base model | Apache-2.0 | https://huggingface.co/Qwen |
| NVARC `qwen3-4b-grids15-sft139` (NVIDIA) | ARC-specialized base checkpoint | see model card (verify) | Kaggle (ARC Prize 2025, 1st place) |

## Citations

- M. Hodel. "Addressing the Abstraction and Reasoning Corpus via Procedural Example Generation."
  arXiv:2404.07353, 2024.
- F. Chollet. "On the Measure of Intelligence." arXiv:1911.01547, 2019. ARC-AGI-2 Technical Report,
  ARC Prize Foundation, 2025 (https://arcprize.org).
- Qwen Team. "Qwen3 Technical Report." 2025.
- NVIDIA Kaggle Grandmasters (NVARC). ARC Prize 2025 (ARC-AGI-2) 1st-place solution.

## License texts

### RE-ARC (MIT)

```
MIT License

Copyright (c) 2024 Michael Hodel

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### ARC-AGI-2 and Qwen3-4B (Apache-2.0)

Both are licensed under the Apache License, Version 2.0. The full text is at
http://www.apache.org/licenses/LICENSE-2.0 and is bundled as `LICENSE-APACHE-2.0.txt` alongside any
redistributed Apache-licensed artifact (the fine-tuned model). Per Apache-2.0 section 4, the license
copy and the `NOTICE` content are retained with redistribution; significant modifications are
disclosed (we continue-fine-tune the base, documented in the paper's "Data and model provenance").
