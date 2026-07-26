# Copyright 2023-2025 Marigold Team, ETH Zürich. All rights reserved.
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
# --------------------------------------------------------------------------
# More information about Marigold:
#   https://marigoldmonodepth.github.io
#   https://marigoldcomputervision.github.io
# Efficient inference pipelines are now part of diffusers:
#   https://huggingface.co/docs/diffusers/using-diffusers/marigold_usage
#   https://huggingface.co/docs/diffusers/api/pipelines/marigold
# Examples of trained models and live demos:
#   https://huggingface.co/prs-eth
# Related projects:
#   https://rollingdepth.github.io/
#   https://marigolddepthcompletion.github.io/
# Citation (BibTeX):
#   https://github.com/prs-eth/Marigold#-citation
# If you find Marigold useful, we kindly ask you to cite our papers.
# --------------------------------------------------------------------------

from .marigold_depth_pipeline import (
    MarigoldDepthPipeline,
    MarigoldDepthOutput,  # noqa: F401
)
from .marigold_iid_pipeline import MarigoldIIDPipeline, MarigoldIIDOutput  # noqa: F401
from .marigold_normals_pipeline import (
    MarigoldNormalsPipeline,  # noqa: F401
    MarigoldNormalsOutput,  # noqa: F401
)
from .marigold_restoration_pipeline_base import (
    MarigoldRestorationPipelineBase,  # noqa: F401
    MarigoldRestorationOutput,  # noqa: F401
)
from .marigold_restoration_pipeline_patched import (
    MarigoldRestorationPipelinePatched,  # noqa: F401
)
from .marigold_controlnet_restoration_pipeline import (
    MarigoldControlNetRestorationPipeline,  # noqa: F401
)
from .marigold_controlnet_restoration_pipeline_patched import (
    MarigoldControlNetRestorationPipelinePatched,  # noqa: F401
)
from .marigold_hybrid_controlnet_restoration_pipeline import (
    MarigoldHybridControlNetRestorationPipeline,  # noqa: F401
)
from .marigold_hybrid_controlnet_arniqa_003_pipeline import (
    MarigoldHybridControlNetArniqa003Pipeline,  # noqa: F401
)
from .marigold_hybrid_controlnet_arniqa_003_pipeline_patched import (
    MarigoldHybridControlNetArniqa003PipelinePatched,  # noqa: F401
)
from .marigold_hybrid_wavelet_controlnet_pipeline import (
    MarigoldHybridWaveletControlNetPipeline,  # noqa: F401
)

# Backward compatibility alias - points to Patched which includes all functionality
MarigoldRestorationPipeline = MarigoldRestorationPipelinePatched

MarigoldPipeline = MarigoldDepthPipeline  # for backward compatibility
