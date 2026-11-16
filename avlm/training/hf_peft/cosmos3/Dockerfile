# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# nemo-automodel:26.06 plus the transformers release that ships Cosmos3
# (Cosmos3OmniForConditionalGeneration and its auto-class mapping are absent from the
# image's transformers). Installed with --no-deps so the image's torch and PEFT stay as is.
FROM nvcr.io/nvidia/nemo-automodel:26.06

RUN /opt/venv/bin/python3 -m pip install --no-deps "transformers==5.22.1" \
 && /opt/venv/bin/python3 -m pip install -U "huggingface_hub>=1.3" "tokenizers>=0.22" \
 && /opt/venv/bin/python3 -c "from transformers import Cosmos3OmniForConditionalGeneration, AutoModelForImageTextToText, AutoProcessor; import peft, torch; print('transformers ok, peft', peft.__version__, 'torch', torch.__version__)"
