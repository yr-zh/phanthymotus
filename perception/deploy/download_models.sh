#!/usr/bin/env bash
# download_models.sh — 评测环境启动时从 juicefs HTTP 网关拉取 OCR 模型。
#
# 榜单提交约定：仓库/镜像不放模型（>1MB 禁止入 git），模型放 juicefs，
# 容器启动时由本脚本下载到约定路径。幂等：已存在且大小一致则跳过；
# MODEL_BASE_URL 为空或不可达时警告并继续（兼容挂载注入的本地模型）。
#
# 可用环境变量覆盖：
#   MODEL_BASE_URL   模型基地址（默认 http://172.28.4.81:34567/zhaoyiru/phanthymotus/ocr/ocr-models）
#   OCR_MODEL_DIR    OCR bundle 目标目录（默认 /models/ocr/ppocrv6-small-trt）
#   LAYOUT_MODEL_DIR layout 引擎目标目录（默认 /models/layout）
set -u

BASE="${MODEL_BASE_URL:-http://172.28.4.81:34567/zhaoyiru/phanthymotus/ocr/ocr-models}"
OCR_DIR="${OCR_MODEL_DIR:-/models/ocr/ppocrv6-small-trt}"
LAYOUT_DIR="${LAYOUT_MODEL_DIR:-/models/layout}"

if [ "${MODEL_BASE_URL:-default}" = "" ]; then
    echo "[download_models] MODEL_BASE_URL 显式置空，跳过下载（使用挂载/本地模型）"
    exit 0
fi

if ! curl -sf -m 10 -o /dev/null "${BASE}/keys.txt"; then
    echo "[download_models] 警告: ${BASE} 不可达，跳过下载（使用挂载/本地模型）"
    exit 0
fi

fetch() {
    # $1=远程文件名 $2=目标目录
    local name="$1" dir="$2"
    local url="${BASE}/${name}" target="${dir}/${name}" tmp
    mkdir -p "${dir}"
    if [ -f "${target}" ]; then
        local remote_size local_size
        remote_size=$(curl -sfI -m 15 "${url}" | awk 'tolower($1)=="content-length:"{print $2}' | tr -d '\r')
        local_size=$(stat -c %s "${target}" 2>/dev/null || echo 0)
        if [ -n "${remote_size}" ] && [ "${remote_size}" = "${local_size}" ]; then
            echo "[download_models] 已存在，跳过: ${target}"
            return 0
        fi
    fi
    tmp="${target}.tmp.$$"
    echo "[download_models] 下载 ${url} -> ${target}"
    if curl -sfL -m 600 -o "${tmp}" "${url}" && [ -s "${tmp}" ]; then
        mv "${tmp}" "${target}"
    else
        rm -f "${tmp}"
        echo "[download_models] 错误: ${name} 下载失败" >&2
        return 1
    fi
}

rc=0
fetch det.engine   "${OCR_DIR}"   || rc=1
fetch rec.engine   "${OCR_DIR}"   || rc=1
fetch cls.engine   "${OCR_DIR}"   || rc=1
fetch keys.txt     "${OCR_DIR}"   || rc=1
fetch layout.engine "${LAYOUT_DIR}" || rc=1

if [ ${rc} -eq 0 ]; then
    echo "[download_models] 全部模型就绪"
else
    echo "[download_models] 警告: 部分模型下载失败，OCR 将尝试自动下载或降级" >&2
fi
exit 0
