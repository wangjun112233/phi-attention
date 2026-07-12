#!/bin/bash
# 五动驱动桥 一键部署脚本
# 在ECS上运行: bash deploy_bridge.sh

set -e

echo "=========================================="
echo "  五动驱动桥 一键部署"
echo "=========================================="

# 1. 安装依赖
echo ""
echo "[1/5] 安装系统依赖..."
sudo apt-get update -qq
sudo apt-get install -y -qq git cmake gcc g++ wget python3 python3-pip

# 2. 编译llama.cpp
echo ""
echo "[2/5] 编译llama.cpp..."
if [ ! -d "llama.cpp" ]; then
    git clone https://github.com/ggerganov/llama.cpp
fi
cd llama.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release 2>/dev/null
cmake --build build --config Release -j$(nproc) 2>/dev/null
cd ..
echo "  llama.cpp编译完成"

# 3. 下载模型
echo ""
echo "[3/5] 下载Qwen2.5-0.5B-Instruct (Q4_K_M, ~400MB)..."
MODEL_FILE="qwen2.5-0.5b-instruct-q4_k_m.gguf"
if [ ! -f "$MODEL_FILE" ]; then
    wget -q --show-progress \
        "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf" \
        -O "$MODEL_FILE"
fi
echo "  模型就绪: $MODEL_FILE"

# 4. 复制桥接脚本
echo ""
echo "[4/5] 部署桥接脚本..."
mkdir -p ~/five_motion_bridge
cp five_motion_bridge.py ~/five_motion_bridge/ 2>/dev/null || true
echo "  桥接脚本就位"

# 5. 创建启动脚本
echo ""
echo "[5/5] 创建启动脚本..."

cat > ~/start_llm.sh << 'EOF'
#!/bin/bash
# 启动LLM推理服务
echo "启动LLM服务 (端口8080)..."
./llama.cpp/build/bin/llama-server \
    -m qwen2.5-0.5b-instruct-q4_k_m.gguf \
    --host 0.0.0.0 \
    --port 8080 \
    -ngl 0 \
    -c 2048 \
    --threads $(nproc)
EOF
chmod +x ~/start_llm.sh

cat > ~/start_bridge.sh << 'BEOF'
#!/bin/bash
# 启动五动驱动桥
echo "启动五动驱动桥 (dabin)..."
cd ~/five_motion_bridge
python3 five_motion_bridge.py \
    --organ-id dabin \
    --organ-url http://localhost:9000 \
    --llm-url http://localhost:8080 \
    --interval 30
BEOF
chmod +x ~/start_bridge.sh

cat > ~/start_bridge_dry.sh << 'BEOF'
#!/bin/bash
# 启动五动驱动桥 (dry-run模式，只看不发言)
echo "启动五动驱动桥 (dry-run)..."
cd ~/five_motion_bridge
python3 five_motion_bridge.py \
    --organ-id dabin \
    --organ-url http://localhost:9000 \
    --llm-url http://localhost:8080 \
    --interval 30 \
    --dry-run
BEOF
chmod +x ~/start_bridge_dry.sh

echo ""
echo "=========================================="
echo "  部署完成！"
echo "=========================================="
echo ""
echo "启动方式（需要2个终端）："
echo ""
echo "  终端1 - 启动LLM服务："
echo "    cd ~ && bash start_llm.sh"
echo ""
echo "  终端2 - 启动桥接（先试dry-run）："
echo "    cd ~ && bash start_bridge_dry.sh"
echo ""
echo "  确认正常后切换到live模式："
echo "    cd ~ && bash start_bridge.sh"
echo ""
echo "  也可以给weiwei/zaowu启动："
echo "    python3 ~/five_motion_bridge/five_motion_bridge.py --organ-id weiwei"
echo "    python3 ~/five_motion_bridge/five_motion_bridge.py --organ-id zaowu"
