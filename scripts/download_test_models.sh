#!/usr/bin/env bash
# Download test GGUF models for EIF-Runtime test suite
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_DIR="${ROOT_DIR}/tests/data"

mkdir -p "${DATA_DIR}"

echo "=========================================================="
echo "  EIF-Runtime: Downloading Test GGUF Models"
echo "=========================================================="

# 1. Jina Embeddings v2 Base Code (ALiBi Model)
# Used for ALiBi verification and benchmark
JINA_TARGET="${DATA_DIR}/jina-embeddings-v2-base-code.gguf"
JINA_URL="https://huggingface.co/second-state/jina-embeddings-v2-base-code-GGUF/resolve/main/jina-embeddings-v2-base-code-Q8_0.gguf"

if [ -f "${JINA_TARGET}" ]; then
    echo "✓ Jina ALiBi model already present: ${JINA_TARGET}"
elif [ -f "${DATA_DIR}/test_jina_alibi.gguf" ]; then
    echo "✓ Local ALiBi test model already present: ${DATA_DIR}/test_jina_alibi.gguf"
else
    echo "⬇ Downloading Jina Embeddings v2 (ALiBi Q8_0)..."
    if curl -L --fail --progress-bar -o "${JINA_TARGET}.tmp" "${JINA_URL}"; then
        mv "${JINA_TARGET}.tmp" "${JINA_TARGET}"
        echo "✓ Successfully downloaded: ${JINA_TARGET}"
    else
        echo "⚠ Warning: Could not download from ${JINA_URL}."
        echo "  If offline, tests will skip live ALiBi downloads."
        rm -f "${JINA_TARGET}.tmp"
    fi
fi

# 2. MiniLM / Granite 30M class Embedding Model
MINILM_TARGET="${DATA_DIR}/minilm.gguf"
MINILM_URL="https://huggingface.co/second-state/all-MiniLM-L6-v2-Embedding-GGUF/resolve/main/all-MiniLM-L6-v2-Q8_0.gguf"

if [ -f "${MINILM_TARGET}" ]; then
    echo "✓ MiniLM model already present: ${MINILM_TARGET}"
elif [ -f "${ROOT_DIR}/../../models/dist/minilm.gguf" ]; then
    echo "✓ Found local workspace model: ${ROOT_DIR}/../../models/dist/minilm.gguf"
else
    echo "⬇ Downloading MiniLM Q8_0..."
    if curl -L --fail --progress-bar -o "${MINILM_TARGET}.tmp" "${MINILM_URL}"; then
        mv "${MINILM_TARGET}.tmp" "${MINILM_TARGET}"
        echo "✓ Successfully downloaded: ${MINILM_TARGET}"
    else
        echo "⚠ Warning: Could not download MiniLM."
        rm -f "${MINILM_TARGET}.tmp"
    fi
fi

# 3. Minimal Qwen2 Test Model (Architecture Detection)
QWEN2_TARGET="${DATA_DIR}/test_qwen2.gguf"
if [ -f "${QWEN2_TARGET}" ]; then
    echo "✓ Qwen2 test model already present: ${QWEN2_TARGET}"
else
    echo "⚙ Synthesizing minimal test_qwen2.gguf..."
    PYTHON_CMD=""
    if command -v python3 >/dev/null 2>&1; then
        PYTHON_CMD="python3"
    elif command -v python >/dev/null 2>&1; then
        PYTHON_CMD="python"
    fi

    if [ -n "${PYTHON_CMD}" ]; then
        ${PYTHON_CMD} -c "
import gguf, numpy as np, sys
try:
    writer = gguf.GGUFWriter('${QWEN2_TARGET}', 'qwen2')
    writer.add_architecture()
    writer.add_uint32('qwen2.embedding_length', 64)
    writer.add_uint32('qwen2.feed_forward_length', 128)
    writer.add_uint32('qwen2.block_count', 1)
    writer.add_uint32('qwen2.attention.head_count', 2)
    writer.add_uint32('qwen2.attention.head_count_kv', 2)
    writer.add_uint32('qwen2.context_length', 128)
    writer.add_float32('qwen2.rope.freq_base', 1000000.0)
    tokens = ['<pad>', '<eos>', 'hello', 'world']
    writer.add_tokenizer_model('gpt2')
    writer.add_token_list(tokens)
    w = np.zeros((64, 64), dtype=np.float32)
    writer.add_tensor('token_embd.weight', np.zeros((4, 64), dtype=np.float32))
    writer.add_tensor('blk.0.attn_norm.weight', np.ones(64, dtype=np.float32))
    writer.add_tensor('blk.0.attn_q.weight', w, raw_dtype=gguf.GGMLQuantizationType.Q4_K)
    writer.add_tensor('blk.0.attn_k.weight', w)
    writer.add_tensor('blk.0.attn_v.weight', w)
    writer.add_tensor('blk.0.attn_output.weight', w)
    writer.add_tensor('blk.0.ffn_norm.weight', np.ones(64, dtype=np.float32))
    writer.add_tensor('blk.0.ffn_gate.weight', np.zeros((128, 64), dtype=np.float32))
    writer.add_tensor('blk.0.ffn_up.weight', np.zeros((128, 64), dtype=np.float32))
    writer.add_tensor('blk.0.ffn_down.weight', np.zeros((64, 128), dtype=np.float32))
    writer.add_tensor('output_norm.weight', np.ones(64, dtype=np.float32))
    writer.add_tensor('output.weight', np.zeros((4, 64), dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print('✓ Generated ${QWEN2_TARGET}')
except Exception as e:
    print(f'⚠ Warning: Failed to synthesize Qwen2 model: {e}')
" 2>/dev/null || true
    fi
fi

echo "All test models setup complete."
