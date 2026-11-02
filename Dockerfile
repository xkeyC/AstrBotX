FROM node:lts-bookworm-slim AS dashboard-builder

WORKDIR /build

COPY dashboard/package.json dashboard/pnpm-lock.yaml dashboard/pnpm-workspace.yaml ./dashboard/
WORKDIR /build/dashboard
RUN npm install -g pnpm@9 \
    && pnpm install --frozen-lockfile

WORKDIR /build
COPY dashboard ./dashboard
COPY astrbot/core/utils/t2i/template/shiki_runtime.iife.js ./astrbot/core/utils/t2i/template/shiki_runtime.iife.js
WORKDIR /build/dashboard
RUN pnpm run build-local

FROM python:3.12-slim
WORKDIR /AstrBot

ENV ASTRBOT_USE_BUNDLED_DASHBOARD=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    build-essential \
    python3-dev \
    libffi-dev \
    libssl-dev \
    ca-certificates \
    bash \
    ffmpeg \
    libavcodec-extra \
    fonts-noto-cjk \
    curl \
    gnupg \
    git \
    ripgrep \
    && curl -fsSL https://deb.nodesource.com/setup_lts.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

COPY . /AstrBot/

# Bootstrap pip from the hash-verified wheel before downloading other packages.
RUN python -m pip install --no-cache-dir \
    "pip @ https://files.pythonhosted.org/packages/3f/06/3e4420302363fbb71be3c7f35ce042f54bfca330e3e66b3a2e3c4cca579f/pip-26.2.2-py3-none-any.whl#sha256=fd7f4dfa8190e014013b9f08f3e973f4e4c26ffd179a204895533a7254813a70" \
    && python -m pip install --upgrade --no-cache-dir "uv>=0.13.4" \
    && echo "3.12" > .python-version \
    && uv lock \
    && uv export --format requirements.txt --output-file requirements.txt --frozen \
    && uv pip install -r requirements.txt --no-cache-dir --system \
    && uv pip install socksio uv pilk --no-cache-dir --system \
    && uv pip install --upgrade "certifi>=2026.10.19" "pip>=26.2.2" --no-cache-dir --system \
    && uv pip install -e . --no-deps --system \
    && touch /AstrBot/.astrbot

COPY --from=dashboard-builder /build/dashboard/dist /AstrBot/astrbot/dashboard/dist

RUN python - <<'EOF'
from pathlib import Path

from astrbot.core.config.default import VERSION

version_file = Path("/AstrBot/astrbot/dashboard/dist/assets/version")
version_file.parent.mkdir(parents=True, exist_ok=True)
version_file.write_text(VERSION, encoding="utf-8")
EOF

RUN rm -rf /AstrBot/dashboard

EXPOSE 6185

CMD ["python", "main.py"]
